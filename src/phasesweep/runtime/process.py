"""Child process lifecycle management.

Every child subprocess created by phasesweep launches through this module's
``run_supervised``. Shutdown-signal handling and the live-child registry are in
:mod:`phasesweep.runtime.shutdown`; process-group probes, termination, and
stale-trial cleanup are in :mod:`phasesweep.runtime.reaper`.

Design:
  - Children run in their own process group (start_new_session=True) so we
    can kill the whole tree with os.killpg, not just the shell.
  - A global registry of live children lets us clean up on orchestrator death.
  - PID files in each trial_dir let operators identify orphans manually.

Launch barrier (review v0.5.15 / blocker 1): every trial launches through a
two-phase supervisor. Before the parent acknowledges, the supervisor is a
stdlib-only helper script (``phasesweep.runtime.supervisor``) run directly
(never ``-m``) under ``python -I -S`` with a minimal sanitized environment —
just ``PATH``. It cannot import the phasesweep package, a poisoned/shadowed
package from the trainer's ``PYTHONPATH``, or a trainer-composed
``sitecustomize``, and it starts in ~30ms instead of paying phasesweep's
package-import cost. Only after this process durably persists
``process_identity.json`` does it send the trainer's shell command and full
environment to the supervisor as a framed JSON payload. The supervisor stays
outside the trainer's process group as a trusted GPU-lease guardian; its
blocked child becomes the recorded process-group leader before executing
``/bin/sh -c cmd``.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import math
import os
import select
import subprocess
import sys
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.runtime import supervisor as _supervisor
from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import strict_json_loads
from phasesweep.runtime.reaper import (
    _KILL_GRACE_SECONDS,
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    _process_group_alive,
    _terminate_process_group,
    read_boot_id,
    read_proc_starttime,
)
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    _launch_lock,
    _register,
    _unregister,
    defer_shutdown_signals,
)

log = logging.getLogger("phasesweep.runtime.process")

_DIRECT_CHILD_REAP_TIMEOUT_SECONDS = 5.0
_GUARDIAN_EXIT_TIMEOUT_SECONDS = (
    _supervisor._GROUP_TERM_GRACE_SECONDS + _DIRECT_CHILD_REAP_TIMEOUT_SECONDS
)
# With the -I -S stdlib-only launch (review v0.5.15 / blocker 1), supervisor
# startup no longer pays phasesweep's package-import cost (~0.55s pre-fix) —
# real-world readiness lands in ~30ms. 10s stays generous headroom for a
# loaded host, not a tight bound.
_SUPERVISOR_READY_TIMEOUT_SECONDS = 10.0


def _supervisor_script_path() -> str:
    """Return the absolute on-disk path to the stdlib-only supervisor script.

    Importing :mod:`phasesweep.runtime.supervisor` here (in the PARENT) is
    safe — the parent already has the full ``phasesweep`` package loaded.
    Only the supervisor *subprocess* must stay import-free before the
    ready/ack barrier (review v0.5.15 / blocker 1); this import never runs in
    that subprocess, which launches the file directly via its path instead.

    :return str: Absolute path to ``supervisor.py`` on disk.
    :raises RuntimeError: If the imported module has no ``__file__`` (e.g. a
        namespace package or frozen build), making a subprocess launch by
        path impossible.
    """
    module_file = _supervisor.__file__
    if module_file is None:
        raise RuntimeError("phasesweep.runtime.supervisor has no __file__; cannot launch it.")
    return str(Path(module_file).resolve())


_SUPERVISOR_SCRIPT_PATH = _supervisor_script_path()


def _kill_group(pgid: int, proc: subprocess.Popen[bytes]) -> bool:
    """Terminate the trial process group and return whether cleanup is confirmed.

    Returns ``True`` when the group is confirmed gone, ``False`` when cleanup
    is uncertain (survived SIGKILL, permission denied, etc.). Callers must
    propagate uncertainty so the orchestrator can refuse to schedule more work
    onto a potentially-leaked GPU (review v0.5.9 / blocker 3).

    Pre-v0.5.15 this only reaped the direct child (``proc``) when a single
    nonblocking ``poll()`` already observed it exited — a race where the
    child died right after ``poll()`` returned ``None`` left it an unreaped
    zombie despite ``cleanup_confirmed`` being reported ``True`` (review
    v0.5.15 / blocker 4). Once :func:`_terminate_process_group` confirms every
    group member is gone, this now unconditionally does one bounded blocking
    ``wait`` on the direct child so it is provably reaped (or cleanup is
    reported unconfirmed) before returning. This never calls ``proc.wait()``
    from ``_shutdown_handler`` itself — that path documents why it must not
    block (see :func:`_shutdown_handler`); this function only runs on normal
    (non-signal-handler) cleanup paths.

    Worst case this blocks for roughly ``_KILL_GRACE_SECONDS`` (10s) SIGTERM
    grace + ~2s SIGKILL confirm inside :func:`_terminate_process_group`, plus
    ``_DIRECT_CHILD_REAP_TIMEOUT_SECONDS`` (5s) for the direct-child reap
    above — up to ~17s total. Before one caller,
    ``_spawn_blocked_supervisor``'s except-block, reaches this cleanup it can
    spend up to ``_SUPERVISOR_READY_TIMEOUT_SECONDS`` (10s) awaiting
    readiness. Because ``run_supervised`` still holds ``_launch_lock`` with
    shutdown signals deferred throughout, that combined path can delay the
    global shutdown handler by up to ~27s (see the ``_launch_lock`` comment
    block).

    Args:
        pgid: Process-group ID of the trial subprocess.
        proc: The root subprocess's :class:`subprocess.Popen` handle. Reaped
            via a bounded ``wait`` once the group is confirmed terminated.

    Returns:
        ``True`` if every process in the group is gone after the SIGTERM →
        SIGKILL escalation AND the direct child was reaped within
        ``_DIRECT_CHILD_REAP_TIMEOUT_SECONDS``; ``False`` if at least one
        group member survived, cleanup status was inconclusive, or the direct
        child failed to reap in time.

    """
    cleanup_confirmed = _terminate_process_group(pgid, grace_seconds=_KILL_GRACE_SECONDS)
    try:
        proc.wait(timeout=_DIRECT_CHILD_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log.error(
            "Direct child PID %d (pgid %d) did not reap within %.1fs after group "
            "termination; treating cleanup as unconfirmed",
            proc.pid,
            pgid,
            _DIRECT_CHILD_REAP_TIMEOUT_SECONDS,
        )
        cleanup_confirmed = False
    except ChildProcessError:
        pass  # Already reaped elsewhere (e.g. a concurrent wait()).
    return cleanup_confirmed


def _wait_for_guardian_exit(proc: subprocess.Popen[bytes]) -> bool:
    """Bound the in-band wait for the post-root lease guardian.

    The guardian deliberately remains alive after SIGKILL when it cannot prove
    the trainer group is gone, retaining inherited GPU lease descriptors. The
    orchestrator must not wait for that fail-closed lease holder forever: it
    returns cleanup uncertainty so the phase aborts without scheduling more
    work, while the detached guardian continues protecting the device.

    :param subprocess.Popen[bytes] proc: Guardian process whose trainer root already exited.
    :return bool: Whether the guardian exited inside its cleanup allowance.
    """
    try:
        proc.wait(timeout=_GUARDIAN_EXIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log.error(
            "Lease guardian PID %d did not finish descendant cleanup within %.1fs; "
            "returning cleanup uncertainty while it retains the GPU lease",
            proc.pid,
            _GUARDIAN_EXIT_TIMEOUT_SECONDS,
        )
        return False
    return True


def _abort_launch(proc: subprocess.Popen[bytes], pgid: int | None) -> bool:
    """Kill and unregister a subprocess after launch or supervision fails.

    Shared by the readiness-wait failure in
    :func:`_spawn_blocked_supervisor` and the identity, payload-delivery, and
    post-launch wait failures in :func:`run_supervised`. Resolves the target
    process group (the registered ``pgid``, or the bare PID when registration
    never happened), kills it, and unregisters it from the global registry if
    it was registered.

    Args:
        proc: The subprocess whose launch or supervision is being aborted.
        pgid: The process-group ID it was registered under, or ``None`` if
            registration never completed.

    Returns:
        Whatever :func:`_kill_group` returns for the resolved target group —
        ``True`` only if the group is confirmed terminated.

    """
    target_pgid = pgid if pgid is not None else proc.pid
    cleanup_confirmed = _kill_group(target_pgid, proc)
    if pgid is not None:
        _unregister(pgid)
    return cleanup_confirmed


# ---------------------------------------------------------------------------
# Supervised subprocess execution
# ---------------------------------------------------------------------------


@dataclass
class ProcessResult:
    """Result of a supervised subprocess execution."""

    return_code: int
    timed_out: bool
    pid: int
    duration_seconds: float
    failure_reason: str | None = None
    cleanup_confirmed: bool = True
    timeout_capped_by_wallclock: bool = False


def _choose_process_deadline(
    *,
    started: float,
    timeout: float | None,
    wallclock_deadline: float | None,
) -> tuple[float | None, bool]:
    """Choose the earlier subprocess or phase/run deadline.

    :param float started: Monotonic timestamp at supervised-launch entry.
    :param float | None timeout: Relative per-trial subprocess limit.
    :param float | None wallclock_deadline: Absolute phase/run deadline.
    :return tuple[float | None, bool]: Effective absolute deadline and whether
        the phase/run deadline limits it.
    """
    trial_deadline = None if timeout is None else started + timeout
    if wallclock_deadline is not None and (
        trial_deadline is None or wallclock_deadline <= trial_deadline
    ):
        return wallclock_deadline, True
    return trial_deadline, False


class _LaunchDeadlineExpired(Exception):
    """The launch deadline expired before the trainer payload was delivered.

    Raised inside the supervised-launch critical section (review v0.5.16 /
    blocker 6) so :func:`run_supervised` can classify the outcome as a
    timeout — the trainer never started — instead of a generic launch
    failure. Carries the cleanup confirmation and pid when the raising site
    already aborted the blocked supervisor itself.
    """

    def __init__(
        self,
        message: str,
        *,
        cleanup_confirmed: bool = True,
        pid: int = -1,
    ) -> None:
        """Record the expiry context for :func:`run_supervised`'s timeout result.

        :param str message: Human-readable description of which launch stage expired.
        :param bool cleanup_confirmed: Whether the aborted supervisor's group
            is confirmed gone (set by the abort site when it already ran).
        :param int pid: PID of the aborted supervisor, or ``-1`` when unknown.
        """
        super().__init__(message)
        self.cleanup_confirmed = cleanup_confirmed
        self.pid = pid


def _trial_process_identity(
    *,
    attempt_id: str,
    pid: int,
    pgid: int,
) -> StaleProcessIdentity:
    """Build the identity persisted before a blocked supervisor may exec the trainer.

    Args:
        attempt_id: Immutable attempt identity already persisted in Optuna;
            binds this identity record to exactly one attempt so a later
            reader cannot mistake it for a different trial's process.
        pid: PID of the just-launched supervisor process.
        pgid: Process-group ID the supervisor was registered under.

    Returns:
        A :class:`StaleProcessIdentity` combining the given fields with the
        current schema version, this process's ``/proc`` start time (``None``
        off-Linux or if unreadable), and the current boot id (``None`` when
        unavailable).

    """
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id=attempt_id,
        pid=pid,
        pgid=pgid,
        proc_starttime=read_proc_starttime(pid),
        boot_id=read_boot_id(),
    )


def _write_process_identity(path: Path, identity: StaleProcessIdentity) -> None:
    """Atomically persist one complete process identity record.

    Args:
        path: Destination ``process_identity.json`` path under the trial dir.
        identity: Complete identity record to serialize as sorted-key JSON.

    """
    atomic_write_text(
        path,
        json.dumps(
            {
                "schema_version": identity.schema_version,
                "attempt_id": identity.attempt_id,
                "pid": identity.pid,
                "pgid": identity.pgid,
                "proc_starttime": identity.proc_starttime,
                "boot_id": identity.boot_id,
            },
            sort_keys=True,
        )
        + "\n",
    )


ATTEMPT_LIFECYCLE_FILE = "attempt_lifecycle.json"
ATTEMPT_LIFECYCLE_SCHEMA_VERSION = 1
_ATTEMPT_LIFECYCLE_STATES = frozenset({"allocated", "launching", "exited"})


@dataclass(frozen=True)
class AttemptLifecycle:
    """Durable coarse lifecycle state for one trial attempt.

    Closes the two recovery windows the transient identity file could not
    represent (review v0.5.17 / blocker 2): ``allocated`` says a durable
    Optuna ``RUNNING`` trial exists but no process launch was attempted (the
    worker may still be queued for a GPU), ``launching`` is committed before
    ``Popen`` so a failed identity write cannot masquerade as that pre-launch
    state, and ``exited`` says the supervised process group is confirmed gone
    even though evidence extraction and the Optuna terminal commit may not
    have happened yet.
    """

    state: str
    return_code: int | None
    cleanup_confirmed: bool | None


def write_attempt_lifecycle(
    trial_dir: Path,
    *,
    attempt_id: str,
    state: str,
    return_code: int | None = None,
    cleanup_confirmed: bool | None = None,
) -> None:
    """Atomically persist the attempt's coarse lifecycle state.

    Args:
        trial_dir: Per-trial directory (attempt-scoped, so states from
            different attempts can never collide).
        attempt_id: Immutable attempt identity binding the record.
        state: One of ``"allocated"``, ``"launching"``, or ``"exited"``.
        return_code: Root return code; only meaningful for ``"exited"``.
        cleanup_confirmed: Whether the whole process group was confirmed
            gone; only meaningful for ``"exited"``.

    Raises:
        ValueError: ``state`` is not a known lifecycle state.
        OSError: The record could not be written.

    """
    if state not in _ATTEMPT_LIFECYCLE_STATES:
        raise ValueError(f"Unknown attempt lifecycle state {state!r}.")
    atomic_write_text(
        trial_dir / ATTEMPT_LIFECYCLE_FILE,
        json.dumps(
            {
                "schema_version": ATTEMPT_LIFECYCLE_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "state": state,
                "return_code": return_code,
                "cleanup_confirmed": cleanup_confirmed,
            },
            sort_keys=True,
        )
        + "\n",
    )


def read_attempt_lifecycle(
    trial_dir: Path,
    *,
    expected_attempt_id: str,
) -> AttemptLifecycle | None:
    """Load and validate the attempt lifecycle record, if one exists.

    Args:
        trial_dir: Trial directory that may contain ``attempt_lifecycle.json``.
        expected_attempt_id: Attempt identity stored on the Optuna trial.

    Returns:
        The validated record, or ``None`` when no record exists. Current
        recovery callers decide whether absence is safe for their state.

    Raises:
        ValueError: The record is malformed, uses an unknown schema or state,
            or belongs to another attempt. Callers must treat this as
            cleanup uncertainty, not as absence.

    """
    path = trial_dir / ATTEMPT_LIFECYCLE_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Attempt lifecycle record at {path} is unreadable.") from exc
    try:
        payload = strict_json_loads(raw)
    except ValueError as exc:
        raise ValueError(f"Malformed attempt lifecycle record at {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Attempt lifecycle record at {path} must be a JSON object.")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != ATTEMPT_LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported attempt lifecycle schema at {path}.")
    if payload.get("attempt_id") != expected_attempt_id:
        raise ValueError(f"Attempt lifecycle record at {path} belongs to another attempt.")
    state = payload.get("state")
    if state not in _ATTEMPT_LIFECYCLE_STATES:
        raise ValueError(f"Unknown attempt lifecycle state at {path}: {state!r}.")
    return_code = payload.get("return_code")
    if return_code is not None and (isinstance(return_code, bool) or type(return_code) is not int):
        raise ValueError(f"Attempt lifecycle field 'return_code' is invalid at {path}.")
    cleanup_confirmed = payload.get("cleanup_confirmed")
    if cleanup_confirmed is not None and not isinstance(cleanup_confirmed, bool):
        raise ValueError(f"Attempt lifecycle field 'cleanup_confirmed' is invalid at {path}.")
    return AttemptLifecycle(
        state=state,
        return_code=return_code,
        cleanup_confirmed=cleanup_confirmed,
    )


def _record_attempt_exited(
    trial_dir: Path,
    *,
    attempt_id: str,
    return_code: int | None,
) -> None:
    """Best-effort durable transition to the ``exited`` lifecycle state.

    Called after the supervised group is confirmed gone. A write failure must
    not replace the primary trial outcome because the retained process identity
    still lets recovery verify the process the slow way.

    Args:
        trial_dir: Per-trial directory holding the lifecycle record.
        attempt_id: Immutable attempt identity binding the record.
        return_code: Root process return code observed by the supervisor wait.

    """
    try:
        write_attempt_lifecycle(
            trial_dir,
            attempt_id=attempt_id,
            state="exited",
            return_code=return_code,
            cleanup_confirmed=True,
        )
    except OSError:
        log.warning(
            "Could not persist the 'exited' lifecycle transition for attempt %s; "
            "recovery will fall back to verifying the retained process identity.",
            attempt_id,
        )


def _sanitized_supervisor_env() -> dict[str, str]:
    """Build the minimal environment for the pre-ACK supervisor interpreter.

    Only ``PATH`` is passed through. Everything else — the full trainer
    environment, ``PYTHONPATH``/``PYTHONHOME``, and any ``CUDA_*`` var
    composed for this trial — is intentionally withheld: the interpreter
    that runs ``supervisor.py`` has not yet passed the ready/ack barrier, so
    it must not see anything a trainer-composed environment could use to
    execute code before phasesweep's process identity is durable (review
    v0.5.15 / blocker 1). The trainer environment is delivered separately,
    over the ack pipe, only after identity persistence succeeds.

    :return dict[str, str]: ``{"PATH": ...}`` using the parent's ``PATH``, or
        a conservative fallback when the parent has none.
    """
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def _encode_launch_payload(cmd: str, env: dict[str, str], cwd: str | None = None) -> bytes:
    """Frame the trainer command, environment, and cwd for the supervisor's ack pipe.

    Decoded on the other end by
    :func:`phasesweep.runtime.supervisor._read_launch_payload` (via
    ``_read_exact``); keep both sides in sync if the wire format changes.

    :param str cmd: Shell command string the supervisor execs with ``/bin/sh``.
    :param dict[str, str] env: Full trainer process environment.
    :param str | None cwd: Optional working directory the supervisor changes
        into before exec'ing the trainer (review v0.5.17 / blocker 4).
    :return bytes: An ASCII decimal length header (byte length of the UTF-8
        JSON body) immediately followed by that many body bytes.
    """
    payload: dict[str, Any] = {"cmd": cmd, "env": env}
    if cwd is not None:
        payload["cwd"] = cwd
    body = json.dumps(payload).encode("utf-8")
    # supervisor.py's _HEADER_LEN is the single source of truth for the frame
    # header width; derive the format width from it rather than hardcoding
    # the digit count here.
    return f"{len(body):0{_supervisor._HEADER_LEN}d}".encode("ascii") + body


def fd_ready(fd: int, *, timeout: float, write: bool = False) -> bool:
    """Wait until a descriptor can be read, or written, for at most ``timeout`` seconds.

    ``select.select`` refuses descriptors at or above ``FD_SETSIZE`` (1024 on
    Linux), which a process holding many open files reaches; ``poll`` has no
    such limit. A hang-up or error counts as ready, as it does for ``select``,
    so the caller's read or write then reports the EOF or the error itself.

    :param int fd: Open file descriptor to wait on.
    :param float timeout: Longest wait in seconds; zero or less checks once.
    :param bool write: Wait for writability instead of readability.
    :return bool: Whether ``fd`` became ready within ``timeout``.
    :raises OSError: If ``fd`` is not an open descriptor.
    """
    poller = select.poll()
    poller.register(fd, select.POLLOUT if write else select.POLLIN)
    events = poller.poll(max(0, math.ceil(timeout * 1000)))
    if any(mask & select.POLLNVAL for _, mask in events):
        raise OSError(errno.EBADF, os.strerror(errno.EBADF))
    return bool(events)


def _write_all(
    fd: int,
    data: bytes,
    *,
    deadline: float | None = None,
    pid: int = -1,
) -> None:
    """Write every byte of ``data`` without exceeding a launch deadline.

    :param int fd: Open file descriptor to write to.
    :param bytes data: Bytes to write in full.
    :param float | None deadline: Optional absolute ``time.monotonic`` deadline.
    :param int pid: Supervisor PID reported when ``deadline`` expires.
    :raises _LaunchDeadlineExpired: If the pipe cannot accept the payload before
        ``deadline``.
    :raises OSError: If the underlying ``os.write`` call fails.
    """
    import time

    view = memoryview(data)
    if deadline is not None:
        os.set_blocking(fd, False)
    while view:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _LaunchDeadlineExpired(
                    "trial launch deadline expired while delivering the trainer payload",
                    pid=pid,
                )
            if not fd_ready(fd, timeout=remaining, write=True) or time.monotonic() >= deadline:
                raise _LaunchDeadlineExpired(
                    "trial launch deadline expired while delivering the trainer payload",
                    pid=pid,
                )
        try:
            written = os.write(fd, view)
        except BlockingIOError:
            continue
        view = view[written:]


def _read_pipe_frame(fd: int, size: int, *, deadline: float) -> bytes | None:
    """Read one fixed-size pipe frame without exceeding ``deadline``.

    :param int fd: Pipe descriptor to read.
    :param int size: Required frame size in bytes.
    :param float deadline: Absolute ``time.monotonic`` deadline.
    :return bytes | None: Complete frame, or ``None`` on timeout or early EOF.
    """
    import time

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        if not fd_ready(fd, timeout=deadline - time.monotonic()) or time.monotonic() >= deadline:
            return None
        chunk = os.read(fd, remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _spawn_blocked_supervisor(
    *,
    stdout: IO[str],
    stderr: IO[str],
    deadline: float | None = None,
    gpu_lease_fds: Collection[int] = (),
) -> tuple[subprocess.Popen[bytes], int, int, int]:
    """Spawn a supervisor that cannot exec the trainer until its parent delivers a payload.

    Launches the stdlib-only ``phasesweep.runtime.supervisor`` script
    directly (never ``-m phasesweep.runtime.supervisor``) under
    ``python -I -S`` with a minimal sanitized environment — see
    :func:`_sanitized_supervisor_env`. Passes it a readiness pipe and an
    acknowledgement pipe. Blocks (via ``select``) until the supervisor
    forks a blocked trainer session and reports that child's PID, or
    ``_SUPERVISOR_READY_TIMEOUT_SECONDS`` elapses —
    capped by the remaining launch ``deadline`` when one is given (review
    v0.5.16 / blocker 6), so a slow supervisor startup can never outlive the
    trial's own wallclock budget. Then registers the new process group. On
    any failure — timeout, an unexpected readiness byte, or an exception
    from ``Popen`` itself — any spawned process group is killed and
    unregistered before the exception propagates.

    Args:
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.
        deadline: Optional ``time.monotonic()`` launch deadline. When it
            expires during the readiness wait, the spawn is aborted and
            :class:`_LaunchDeadlineExpired` is raised so the caller reports a
            timeout instead of a generic launch failure.
        gpu_lease_fds: Open host GPU-lock descriptors deliberately inherited
            by the trusted supervisor guardian so the kernel lease outlives an
            orchestrator hard exit. The guardian does not pass them to the
            trainer child.

    Returns:
        A ``(proc, pgid, ack_write, status_read)`` tuple: the guardian's
        ``Popen`` handle, the trainer's registered process-group id, the write
        end of the acknowledgement pipe, and the read end of the guardian
        status pipe. The caller owns both descriptors. It must send the framed
        launch payload (see :func:`_encode_launch_payload`) and close
        ``ack_write`` once the trainer identity is durably persisted.
        ``status_read`` yields ``b"X"`` as soon as the trainer root exits and
        then ``b"D"`` when the guardian had to reap descendants, otherwise
        EOF. Waiting for descendant cleanup after ``b"X"`` is outside the
        trainer wallclock budget.

    Raises:
        _LaunchDeadlineExpired: The launch deadline expired before the
            supervisor became ready; the spawned group is already aborted and
            the exception carries the cleanup confirmation.
        RuntimeError: If the supervisor does not signal readiness within
            ``_SUPERVISOR_READY_TIMEOUT_SECONDS``, or signals something other
            than ``b"R"``.
        UnsafeProcessCleanupError: A supervisor was spawned, launch failed or
            was interrupted, and terminating that process could not be confirmed.

    """
    import time

    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    proc: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                _SUPERVISOR_SCRIPT_PATH,
                str(ready_write),
                str(ack_read),
            ],
            env=_sanitized_supervisor_env(),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            pass_fds=(ready_write, ack_read, *gpu_lease_fds),
        )
        closed_fd = ready_write
        ready_write = -1
        os.close(closed_fd)
        closed_fd = ack_read
        ack_read = -1
        os.close(closed_fd)

        ready_deadline = time.monotonic() + _SUPERVISOR_READY_TIMEOUT_SECONDS
        if deadline is not None:
            ready_deadline = min(ready_deadline, deadline)
        ready_frame = _read_pipe_frame(
            ready_read,
            1 + _supervisor._READY_PID_WIDTH,
            deadline=ready_deadline,
        )
        if ready_frame is None and deadline is not None and time.monotonic() >= deadline:
            raise _LaunchDeadlineExpired(
                "trial launch deadline expired while waiting for the supervisor",
                pid=proc.pid,
            )
        if ready_frame is None or not ready_frame.startswith(b"R"):
            raise RuntimeError("trial supervisor did not become ready before launch")
        try:
            trainer_pid = int(ready_frame[1:].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("trial supervisor reported an invalid trainer PID") from exc
        if trainer_pid <= 0:
            raise RuntimeError("trial supervisor reported an invalid trainer PID")
        pgid = trainer_pid
        _register(proc, pgid=pgid)
        status_read = ready_read
        ready_read = -1
        return proc, pgid, ack_write, status_read
    except BaseException as exc:
        if ack_write >= 0:
            closed_fd = ack_write
            ack_write = -1
            os.close(closed_fd)
        if proc is not None:
            confirmed = _abort_launch(proc, pgid)
            if isinstance(exc, _LaunchDeadlineExpired):
                exc.cleanup_confirmed = confirmed
            elif not confirmed:
                raise UnsafeProcessCleanupError(
                    "Trial supervisor launch was interrupted after Popen, and cleanup "
                    "could not be confirmed."
                ) from exc
        raise
    finally:
        for fd in (ready_read, ready_write, ack_read):
            if fd >= 0:
                os.close(fd)


def run_supervised(
    cmd: str,
    *,
    env: dict[str, str],
    stdout: IO[str],
    stderr: IO[str],
    timeout: float | None,
    trial_dir: Path,
    attempt_id: str,
    wallclock_deadline: float | None = None,
    cwd: str | None = None,
    gpu_lease_fds: Collection[int] = (),
) -> ProcessResult:
    """Launch a shell command in its own process group with full lifecycle management.

    Launches a small stdlib-only supervisor first (see module docstring for
    the pre-ACK/post-ACK launch boundary; review v0.5.15 / blocker 1). The
    supervisor waits on an inherited acknowledgement pipe while the parent
    atomically persists ``process_identity.json``. Only after the identity is
    durable does the parent send the trainer command and full trainer
    environment to the supervisor as a framed JSON payload. The supervisor
    then stays alive outside the trainer's process group as a lease guardian
    while a descriptor-scrubbed child executes the trainer command. If the parent dies
    before delivering that payload, pipe EOF makes the supervisor exit
    without starting training.

    The identity record is always retained. On a fully clean exit — root
    returned 0 and no descendant processes were left alive — the attempt
    lifecycle is durably advanced to ``exited``. If the root exits cleanly
    but leaves GPU-holding descendants running, we treat that as a lifecycle
    failure and kill the group (review v0.5.5 / blocker 1).

    On timeout: SIGTERM -> grace -> SIGKILL on the entire group.

    ``timeout`` is the per-trial launch-plus-execution budget, converted to an
    absolute ``time.monotonic()`` deadline at entry (review v0.5.16 /
    blocker 6). ``wallclock_deadline`` carries the already-running phase/run
    budget through work performed before this function. The earlier deadline
    wins. Every supervised stage consumes it: the supervisor readiness wait
    is capped to the remainder, the deadline is re-checked after identity
    persistence and *before* the trainer payload crosses the ack pipe — an
    expired deadline aborts the blocked supervisor and returns a timeout
    without ever starting the trainer — and the wait for the trainer root
    uses the recomputed remainder. Cleanup grace after a timeout or an
    in-budget root exit is explicitly post-deadline time.

    Once the supervisor is spawned, launch failures — an expired deadline, a
    failed identity write, a failed payload delivery — never propagate: the
    group is terminated and the failure is reported through the returned
    :class:`ProcessResult`.

    Args:
        cmd: Shell command string the acknowledged supervisor runs with ``/bin/sh``.
        env: Full process environment for the subprocess.
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.
        timeout: Total wall-clock budget in seconds measured from this call's
            entry (launch overhead included), or ``None`` for no timeout.
        trial_dir: Per-trial directory where ``process_identity.json`` is written.
        attempt_id: Immutable attempt identity already persisted in Optuna.
        wallclock_deadline: Optional absolute ``time.monotonic()`` phase/run
            deadline. Work before this call has already consumed it.
        cwd: Optional working directory the supervisor changes into before
            exec'ing the trainer, delivered over the ack pipe with the rest
            of the launch payload (review v0.5.17 / blocker 4). ``None``
            keeps the invocation cwd.
        gpu_lease_fds: Open host GPU-lock descriptors inherited by the trusted
            supervisor guardian, not its trainer child. The orchestrator
            closes only its copies; the guardian retains each lock until the
            entire trainer process group is gone even if trainer code closes
            every unknown descriptor or its root shell exits before a worker.

    Returns:
        :class:`ProcessResult` capturing return code, wall-clock duration,
        timeout flag, ``failure_reason`` (set on timeout or descendant
        survival), and ``cleanup_confirmed`` (``False`` when SIGKILL did not
        confirm the group is gone).

    Raises:
        RuntimeError: The supervisor never signalled readiness, or signalled an
            unexpected byte, so no process group was ever created.
        OSError: The supervisor process or its pipes could not be created, or
            an unexpected post-launch I/O failure occurred after its process
            group was cleaned and reaped.
        UnsafeProcessCleanupError: An unexpected failure after launch occurred
            and cleanup of the trainer process group could not be confirmed.

    """
    import time

    started = time.monotonic()
    deadline, timeout_capped_by_wallclock = _choose_process_deadline(
        started=started,
        timeout=timeout,
        wallclock_deadline=wallclock_deadline,
    )
    timeout_reason = (
        "phase/run wallclock deadline exceeded"
        if timeout_capped_by_wallclock
        else f"timeout after {timeout}s"
    )

    # The launch + register + identity write must be atomic from the
    # signal handler's perspective. Signal deferral MUST come first so that
    # SIGTERM/SIGINT cannot land between ``_launch_lock`` acquisition and
    # signal masking (review v0.5.9 / blocker 2). Reversing the order
    # (``_launch_lock`` first, then ``defer_shutdown_signals()``) creates
    # two deadlock windows:
    #
    # 1. Signal lands after lock acquired but before mask set: handler runs
    #    in same thread and blocks on the lock it already holds.
    # 2. On exit, mask is restored before lock released: pending signal is
    #    delivered while the thread still owns the lock -> same deadlock.
    #
    # Correct ordering: block signals -> take lock -> work -> release lock
    # -> unblock signals. Any pending signal is delivered after the lock is
    # released, so the handler can safely acquire it.
    proc: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    ack_write: int | None = None
    status_read: int | None = None
    identity_path = trial_dir / PROCESS_IDENTITY_FILE

    try:
        with defer_shutdown_signals(), _launch_lock:
            # Advance durably before Popen. Without this boundary, a spawned
            # supervisor whose identity write and cleanup both fail is
            # indistinguishable on recovery from a worker killed while still
            # queued for a GPU (both otherwise leave ``allocated`` and no
            # process_identity.json).
            write_attempt_lifecycle(
                trial_dir,
                attempt_id=attempt_id,
                state="launching",
            )
            proc, pgid, ack_write, status_read = _spawn_blocked_supervisor(
                stdout=stdout,
                stderr=stderr,
                deadline=deadline,
                gpu_lease_fds=gpu_lease_fds,
            )
            identity = _trial_process_identity(
                attempt_id=attempt_id,
                pid=pgid,
                pgid=pgid,
            )
            _write_process_identity(identity_path, identity)
            if deadline is not None and time.monotonic() >= deadline:
                # The budget expired during launch bookkeeping. The payload
                # has NOT been delivered, so the blocked supervisor can be
                # aborted without any trainer work ever starting (review
                # v0.5.16 / blocker 6).
                raise _LaunchDeadlineExpired(
                    "trial launch deadline expired before the trainer payload was delivered",
                    pid=pgid,
                )
            # Only now does the trainer command and full trainer environment
            # cross into the supervisor — after identity is durable, over the
            # ack pipe as a framed JSON payload (review v0.5.15 / blocker 1).
            _write_all(
                ack_write,
                _encode_launch_payload(cmd, env, cwd),
                deadline=deadline,
                pid=pgid,
            )
            os.close(ack_write)
            ack_write = None
    except _LaunchDeadlineExpired as exc:
        if ack_write is not None:
            os.close(ack_write)
        if status_read is not None:
            os.close(status_read)
            status_read = None
        cleanup_confirmed = exc.cleanup_confirmed
        pid = exc.pid
        return_code = -9
        if proc is not None:
            # Raised at the pre-payload recheck: the supervisor is still
            # blocked on its ack pipe; kill and reap it here.
            cleanup_confirmed = _abort_launch(proc, pgid)
            pid = pgid if pgid is not None else proc.pid
            if proc.returncode is not None:
                return_code = proc.returncode
        if cleanup_confirmed:
            _record_attempt_exited(trial_dir, attempt_id=attempt_id, return_code=return_code)
        log.warning(
            "Trial launch deadline expired before the trainer started (pid %d): %s",
            pid,
            exc,
        )
        return ProcessResult(
            return_code=return_code,
            timed_out=True,
            pid=pid,
            duration_seconds=time.monotonic() - started,
            failure_reason=f"{timeout_reason} before trainer launch",
            cleanup_confirmed=cleanup_confirmed,
            timeout_capped_by_wallclock=timeout_capped_by_wallclock,
        )
    except BaseException as exc:
        if ack_write is not None:
            os.close(ack_write)
        if status_read is not None:
            os.close(status_read)
            status_read = None
        if proc is None:
            if isinstance(exc, UnsafeProcessCleanupError):
                # The spawn helper owned a child before it could return the
                # Popen handle. Its failed cleanup is intentionally not
                # rewritten as a childless terminal attempt.
                raise
            # No child identity exists to support the best-effort fallback used
            # after a spawned group exits. This transition is the sole durable
            # proof that recovery may settle the childless attempt.
            try:
                write_attempt_lifecycle(
                    trial_dir,
                    attempt_id=attempt_id,
                    state="exited",
                    return_code=None,
                    cleanup_confirmed=True,
                )
            except OSError as write_error:
                raise write_error from exc
            raise
        if isinstance(exc, PhaseSweepShutdown):
            # The installed handler already made the authoritative cleanup
            # attempt. Remove its now-settled registry entry before preserving
            # the structured shutdown evidence unchanged.
            if pgid is not None:
                _unregister(pgid)
            raise
        control_flow_exception = not isinstance(exc, Exception)
        target_pgid = pgid if pgid is not None else proc.pid
        # Covers both a failed identity write and a failed payload delivery;
        # the substring "failed to persist process identity" is kept stable
        # because it is asserted on by existing tests.
        launch_failure_reason = (
            f"failed to persist process identity or deliver launch payload: {exc}"
        )
        log.exception(
            "Trial PID %d (pgid %d) launched but identity persistence or launch payload "
            "delivery failed; terminating group",
            proc.pid,
            target_pgid,
        )
        cleanup_confirmed = _abort_launch(proc, pgid)
        if cleanup_confirmed:
            _record_attempt_exited(
                trial_dir,
                attempt_id=attempt_id,
                return_code=proc.returncode if proc.returncode is not None else -9,
            )
        if control_flow_exception:
            if not cleanup_confirmed:
                raise UnsafeProcessCleanupError(
                    f"Trial launch was interrupted after starting process group "
                    f"{target_pgid}, and cleanup could not be confirmed."
                ) from exc
            raise
        duration = time.monotonic() - started
        return ProcessResult(
            return_code=proc.returncode if proc.returncode is not None else -9,
            timed_out=False,
            pid=pgid if pgid is not None else proc.pid,
            duration_seconds=duration,
            failure_reason=launch_failure_reason,
            cleanup_confirmed=cleanup_confirmed,
            timeout_capped_by_wallclock=timeout_capped_by_wallclock,
        )

    assert proc is not None
    assert pgid is not None
    assert status_read is not None

    timed_out = False
    failure_reason: str | None = None
    cleanup_confirmed = True

    try:
        root_status: bytes | None
        if deadline is not None and time.monotonic() >= deadline:
            # Payload delivery itself consumes the total trial budget. Do not
            # let an immediately available exit byte turn post-deadline work
            # into an in-budget success.
            root_status = None
        elif deadline is None:
            root_status = os.read(status_read, 1)
        else:
            root_status = _read_pipe_frame(status_read, 1, deadline=deadline)
        if root_status is None and deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            failure_reason = timeout_reason
            log.warning("Trial PID %d (pgid %d) timed out — terminating group", pgid, pgid)
            cleanup_confirmed = _kill_group(pgid, proc)
        else:
            # The trainer root exited within its budget. Descendant cleanup is
            # lifecycle teardown, so the guardian's SIGTERM/SIGKILL grace must
            # not retroactively turn this into a wallclock timeout.
            guardian_exited = _wait_for_guardian_exit(proc)
            guardian_status = None
            if guardian_exited:
                guardian_status = (
                    os.read(status_read, 1)
                    if root_status == _supervisor._TRAINER_ROOT_EXITED
                    else root_status
                )
                os.close(status_read)
                status_read = None
            else:
                cleanup_confirmed = False
                failure_reason = (
                    f"trainer root exited, but lease guardian {proc.pid} could not "
                    "confirm descendant cleanup within its bounded allowance"
                )
            if guardian_status == _supervisor._DESCENDANTS_REAPED:
                failure_reason = (
                    f"root process exited with code {proc.returncode}, "
                    f"but process group {pgid} still had live descendants"
                )
                log.warning(
                    "Trial PID %d exited with code %s but process group %d "
                    "still had live descendants; the lease guardian terminated them",
                    pgid,
                    proc.returncode,
                    pgid,
                )
            # Root process exited normally. That is not sufficient — the trial
            # is only clean once the entire process group is gone. A common
            # pathological case: `python launcher.py &` exits immediately while
            # the training worker stays alive holding GPU memory.
            if failure_reason is None and _process_group_alive(pgid):
                failure_reason = (
                    f"root process exited with code {proc.returncode}, "
                    f"but process group {pgid} still had live descendants"
                )
                log.warning(
                    "Trial PID %d exited with code %s but process group %d "
                    "still has live descendants — terminating group",
                    pgid,
                    proc.returncode,
                    pgid,
                )
                cleanup_confirmed = _kill_group(pgid, proc)

    except PhaseSweepShutdown:
        # The signal handler already made the authoritative cleanup attempt and
        # attached its evidence to this control-flow exception. Preserve it
        # unchanged; the engine uses that report when recording cancellation.
        raise
    except BaseException as exc:
        # Once the trainer payload crossed the pipe, every exit from this wait
        # owns a live process group until cleanup proves otherwise. An I/O or
        # runtime failure must not unregister the group and let it continue
        # outside both normal supervision and shutdown-handler tracking.
        try:
            cleanup_confirmed = _abort_launch(proc, pgid)
        except PhaseSweepShutdown:
            raise
        except BaseException:
            log.exception(
                "Unexpected failure while waiting for trial process group %d, followed by "
                "an exception during cleanup",
                pgid,
            )
            raise UnsafeProcessCleanupError(
                f"Unexpected failure while waiting for trial process group {pgid}; "
                "cleanup could not be confirmed."
            ) from exc
        if not cleanup_confirmed:
            raise UnsafeProcessCleanupError(
                f"Unexpected failure while waiting for trial process group {pgid}; "
                "cleanup could not be confirmed."
            ) from exc
        try:
            _record_attempt_exited(
                trial_dir,
                attempt_id=attempt_id,
                return_code=proc.returncode if proc.returncode is not None else -9,
            )
        except Exception:
            log.exception(
                "Trial process group %d was cleaned after an unexpected wait failure, but "
                "its exited lifecycle could not be persisted",
                pgid,
            )
        raise
    finally:
        if status_read is not None:
            with contextlib.suppress(OSError):
                os.close(status_read)
        _unregister(pgid)

    # The identity record is deliberately RETAINED on clean exit (review
    # v0.5.17 / blocker 2 gap B): the orchestrator can still die between
    # here and the Optuna terminal commit (evidence extraction, gates), and
    # recovery must be able to distinguish "safely exited" from "identity
    # missing". Both this normal path and the failed-wait cleanup path above
    # write the durable 'exited' transition only after confirming that the
    # whole group is gone; a wait exception alone is never proof of exit.
    if cleanup_confirmed:
        _record_attempt_exited(
            trial_dir,
            attempt_id=attempt_id,
            return_code=proc.returncode if proc.returncode is not None else -9,
        )

    duration = time.monotonic() - started
    return ProcessResult(
        return_code=proc.returncode if proc.returncode is not None else -9,
        timed_out=timed_out,
        pid=pgid,
        duration_seconds=duration,
        failure_reason=failure_reason,
        cleanup_confirmed=cleanup_confirmed,
        timeout_capped_by_wallclock=timeout_capped_by_wallclock,
    )
