"""Child process lifecycle management.

Trainers run in separate process groups tracked by :mod:`phasesweep.runtime.shutdown`.
Probing and termination live in :mod:`phasesweep.runtime.reaper`.
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
import time
from collections.abc import Collection
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

from phasesweep.errors import OperatorAction, PhaseSweepError, UnsafeProcessCleanupError
from phasesweep.runtime import supervisor as _supervisor
from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import strict_json_loads
from phasesweep.runtime.reaper import (
    _KILL_GRACE_SECONDS,
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    _process_group_alive,
    _terminate_process_groups,
    read_boot_id,
    read_pid_namespace,
    read_proc_starttime,
)
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    ShutdownLaunchRefusedError,
    _launch_lock,
    _register,
    _unregister,
    defer_shutdown_signals,
    servicing_shutdown_signal,
)

log = logging.getLogger("phasesweep.runtime.process")

_DIRECT_CHILD_REAP_TIMEOUT_SECONDS = 5.0
_GUARDIAN_EXIT_TIMEOUT_SECONDS = (
    _supervisor._GROUP_TERM_GRACE_SECONDS + _DIRECT_CHILD_REAP_TIMEOUT_SECONDS
)
_SUPERVISOR_READY_TIMEOUT_SECONDS = 10.0
# poll() takes its timeout as a C int of milliseconds, about 24.8 days, and a
# trial timeout may be longer.
_POLL_MAX_MILLISECONDS = 2**31 - 1


def _supervisor_script_path() -> str:
    """Return the absolute on-disk path to the stdlib-only supervisor script.

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
    """Request guardian cancellation and confirm the trial group is gone.

    A known trainer group is terminated by its guardian after durable cancellation intent.
    Before trainer readiness, only the guardian's own group is terminated directly.
    A bounded wait then reaps the guardian; uncertain cleanup blocks new work.

    :param int pgid: Process-group ID of the trial subprocess.
    :param subprocess.Popen[bytes] proc: Guardian subprocess, reaped with a bounded wait.
    :return bool: ``True`` if every process in the group is gone and the guardian was reaped within
        ``_DIRECT_CHILD_REAP_TIMEOUT_SECONDS``; ``False`` if at least one group member
        survived, cleanup status was inconclusive, or the direct child failed to reap in time.
    """
    cleanup_confirmed = _terminate_process_groups(
        (pgid,),
        grace_seconds=_KILL_GRACE_SECONDS,
        guardians=None if pgid == proc.pid else {pgid: proc.pid},
    )[pgid]
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
    """Wait a bounded time for the GPU-lease guardian to exit.

    An uncertain group cleanup leaves the guardian holding its leases; a wait
    that expires must preserve that uncertainty.

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

    :param subprocess.Popen[bytes] proc: The subprocess whose launch or supervision is being
        aborted.
    :param int | None pgid: The process-group ID it was registered under, or ``None`` if
        registration never completed, in which case ``proc.pid`` is the target group.
    :return bool: Whatever :func:`_kill_group` returns for the resolved target group,
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

    Raised inside the supervised-launch critical section so
    :func:`run_supervised` can classify the outcome as a timeout — the trainer
    never started — instead of a generic launch failure. Carries the cleanup
    confirmation and pid when the raising site already aborted the blocked
    supervisor itself.
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
    pid_namespace: str,
) -> StaleProcessIdentity:
    """Build the identity persisted before a blocked supervisor may exec the trainer.

    :param str attempt_id: Immutable attempt identity already persisted in Optuna; binds this
        identity record to exactly one attempt so a later reader cannot mistake it for a
        different trial's process.
    :param int pid: PID of the just-launched blocked child that becomes the trainer group leader.
    :param int pgid: Process-group ID that child was registered under.
    :param str pid_namespace: PID namespace ``pid`` and ``pgid`` are numbered in, from
        :func:`read_pid_namespace` before the launch.
    :return StaleProcessIdentity: The given fields combined with the current schema version,
        ``pid``'s kernel start time (``None`` if unreadable), and the current
        boot id (``None`` when unavailable).
    """
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id=attempt_id,
        pid=pid,
        pgid=pgid,
        proc_starttime=read_proc_starttime(pid),
        boot_id=read_boot_id(),
        pid_namespace=pid_namespace,
    )


def _write_process_identity(path: Path, identity: StaleProcessIdentity) -> None:
    """Atomically persist one complete process identity record.

    :param Path path: Destination ``process_identity.json`` path under the trial dir.
    :param StaleProcessIdentity identity: Complete identity record to serialize as sorted-key
        JSON.
    """
    atomic_write_text(
        path,
        json.dumps(asdict(identity), sort_keys=True) + "\n",
    )


ATTEMPT_LIFECYCLE_FILE = "attempt_lifecycle.json"
ATTEMPT_LIFECYCLE_SCHEMA_VERSION = _supervisor.ATTEMPT_LIFECYCLE_SCHEMA_VERSION
_ATTEMPT_LIFECYCLE_STATES = frozenset({"allocated", "launching", "terminating", "exited"})


@dataclass(frozen=True)
class AttemptLifecycle:
    """Durable launch state for one attempt.

    ``allocated`` identifies a durable RUNNING trial before launch, possibly queued
    for a GPU. ``launching`` is recorded before spawning, so a missing identity
    then requires recovery. ``terminating`` preserves cancellation before the guardian
    signals the trainer. ``exited`` confirms group cleanup before Optuna completion.
    """

    state: str
    return_code: int | None
    cleanup_confirmed: bool | None
    duration_seconds: float | None


def write_attempt_lifecycle(
    trial_dir: Path,
    *,
    attempt_id: str,
    state: str,
    return_code: int | None = None,
    cleanup_confirmed: bool | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Atomically persist the attempt's coarse lifecycle state.

    :param Path trial_dir: Per-trial directory (attempt-scoped, so states from different
        attempts can never collide).
    :param str attempt_id: Immutable attempt identity binding the record.
    :param str state: Allocated, launching, terminating, or exited lifecycle state.
    :param int | None return_code: Root return code or a guardian's nonzero cancellation code.
    :param bool | None cleanup_confirmed: Whether the whole process group was confirmed gone; false while cancellation is in progress.
    :param float | None duration_seconds: Supervised elapsed time when the group exited.
    :raises ValueError: ``state`` is not a known lifecycle state.
    :raises OSError: The record could not be written.
    """
    if state not in _ATTEMPT_LIFECYCLE_STATES:
        raise ValueError(f"Unknown attempt lifecycle state {state!r}.")
    _supervisor.write_lifecycle_record(
        trial_dir / ATTEMPT_LIFECYCLE_FILE,
        attempt_id=attempt_id,
        state=state,
        return_code=return_code,
        cleanup_confirmed=cleanup_confirmed,
        duration_seconds=duration_seconds,
    )


def read_attempt_lifecycle(
    trial_dir: Path,
    *,
    expected_attempt_id: str,
) -> AttemptLifecycle | None:
    """Load and validate the attempt lifecycle record, if one exists.

    :param Path trial_dir: Trial directory that may contain ``attempt_lifecycle.json``.
    :param str expected_attempt_id: Attempt identity stored on the Optuna trial.
    :raises ValueError: The record is unreadable, malformed, uses an unknown schema or state,
        or belongs to another attempt. Callers must treat this as cleanup uncertainty, not as
        absence.
    :return AttemptLifecycle | None: The validated record, or ``None`` when no record exists.
        Current recovery callers decide whether absence is safe for their state.
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
    if not isinstance(state, str) or state not in _ATTEMPT_LIFECYCLE_STATES:
        raise ValueError(f"Unknown attempt lifecycle state at {path}: {state!r}.")
    return_code = payload.get("return_code")
    if return_code is not None and (isinstance(return_code, bool) or type(return_code) is not int):
        raise ValueError(f"Attempt lifecycle field 'return_code' is invalid at {path}.")
    cleanup_confirmed = payload.get("cleanup_confirmed")
    if cleanup_confirmed is not None and not isinstance(cleanup_confirmed, bool):
        raise ValueError(f"Attempt lifecycle field 'cleanup_confirmed' is invalid at {path}.")
    duration = payload.get("duration_seconds")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValueError(f"Attempt lifecycle field 'duration_seconds' is invalid at {path}.")
    return AttemptLifecycle(
        state=state,
        return_code=return_code,
        cleanup_confirmed=cleanup_confirmed,
        duration_seconds=duration,
    )


def _record_attempt_exited(
    trial_dir: Path,
    *,
    attempt_id: str,
    return_code: int | None,
) -> None:
    """Best-effort persistence of confirmed process cleanup.

    Recorded exit or cancellation status survives repeated cleanup confirmations.
    A write failure preserves the primary error; the retained identity permits recovery.

    :param Path trial_dir: Per-trial directory holding the lifecycle record.
    :param str attempt_id: Immutable attempt identity binding the record.
    :param int | None return_code: Root process return code observed by the supervisor wait.
    """
    try:
        lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=attempt_id)
        if (
            lifecycle is not None
            and lifecycle.state == "exited"
            and lifecycle.cleanup_confirmed
            and lifecycle.duration_seconds is not None
        ):
            return
        write_attempt_lifecycle(
            trial_dir,
            attempt_id=attempt_id,
            state="exited",
            return_code=(
                lifecycle.return_code
                if lifecycle is not None
                and lifecycle.state in {"terminating", "exited"}
                and lifecycle.return_code is not None
                else return_code
            ),
            cleanup_confirmed=True,
            duration_seconds=None if lifecycle is None else lifecycle.duration_seconds,
        )
    except (OSError, ValueError):
        log.warning(
            "Could not persist the 'exited' lifecycle transition for attempt %s; "
            "recovery will fall back to verifying the retained process identity.",
            attempt_id,
        )


def _sanitized_supervisor_env() -> dict[str, str]:
    """Return the pre-ACK supervisor environment.

    Only ``PATH`` is inherited; trainer environment arrives after identity persistence.

    :return dict[str, str]: ``{"PATH": ...}`` using the parent's ``PATH``, or
        a conservative fallback when the parent has none.
    """
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def _encode_launch_payload(cmd: str, env: dict[str, str], cwd: str | None = None) -> bytes:
    """Encode the length-prefixed trainer command, environment, and optional cwd.

    :param str cmd: Shell command string the supervisor execs with ``/bin/sh``.
    :param dict[str, str] env: Full trainer process environment.
    :param str | None cwd: Optional working directory the supervisor changes
        into before exec'ing the trainer.
    :return bytes: An ASCII decimal length header (byte length of the UTF-8
        JSON body) immediately followed by that many body bytes.
    """
    payload: dict[str, Any] = {"cmd": cmd, "env": env}
    if cwd is not None:
        payload["cwd"] = cwd
    body = json.dumps(payload).encode("utf-8")
    return f"{len(body):0{_supervisor._HEADER_LEN}d}".encode("ascii") + body


def fd_ready(fd: int, *, timeout: float, write: bool = False) -> bool:
    """Wait for a descriptor without a ``select`` FD_SETSIZE limit.

    EOF and errors count as ready. Long waits share one deadline across poll calls.

    :param int fd: Open file descriptor to wait on.
    :param float timeout: Longest wait in seconds; zero or less checks once.
    :param bool write: Wait for writability instead of readability.
    :return bool: Whether ``fd`` became ready within ``timeout``.
    :raises OSError: If ``fd`` is not an open descriptor.
    """
    deadline = time.monotonic() + timeout
    poller = select.poll()
    poller.register(fd, select.POLLOUT if write else select.POLLIN)
    while True:
        remaining_ms = max(0, math.ceil((deadline - time.monotonic()) * 1000))
        events = poller.poll(min(remaining_ms, _POLL_MAX_MILLISECONDS))
        if events:
            if any(mask & select.POLLNVAL for _, mask in events):
                raise OSError(errno.EBADF, os.strerror(errno.EBADF))
            return True
        if time.monotonic() >= deadline:
            return False


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
    lifecycle_args: tuple[str, str, str] | tuple[()] = (),
) -> tuple[subprocess.Popen[bytes], int, int, int]:
    """Spawn and register a supervisor whose trainer waits for a launch payload.

    Readiness is bounded by both the supervisor timeout and the launch deadline.
    Failures abort and unregister any spawned group before propagating.

    :param IO[str] stdout: Already-open file handle that receives the subprocess stdout.
    :param IO[str] stderr: Already-open file handle that receives the subprocess stderr.
    :param float | None deadline: Absolute ``time.monotonic()`` launch deadline, or ``None``.
    :param Collection[int] gpu_lease_fds: Host GPU-lock descriptors inherited only by the
        guardian, keeping leases held after an orchestrator exit until cleanup finishes.
    :param tuple[str, str, str] | tuple[()] lifecycle_args: Optional attempt directory, identity, and monotonic start.
    :raises _LaunchDeadlineExpired: Readiness exceeded the launch deadline; the group was
        aborted and the exception carries cleanup confirmation.
    :raises RuntimeError: Supervisor readiness timed out or returned an invalid ready frame
        (unexpected tag or trainer PID).
    :raises UnsafeProcessCleanupError: Failed or interrupted launch left cleanup unconfirmed.
    :return tuple[subprocess.Popen[bytes], int, int, int]: ``(proc, pgid, ack_write,
        status_read)``: guardian process, registered trainer group, launch pipe and status
        pipe. The caller owns both descriptors; after persisting trainer identity, send the
        framed payload and close ``ack_write``. ``status_read`` yields ``b"X"`` when the
        trainer root exits, then ``b"D"`` if descendants were reaped or EOF otherwise.
        Cleanup after ``b"X"`` is outside the trainer wallclock budget.
    """
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
                *lifecycle_args,
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
        if ready_frame is None or not ready_frame.startswith(_supervisor._READY_TAG):
            raise RuntimeError("trial supervisor did not become ready before launch")
        try:
            trainer_pid = int(ready_frame[1:].decode("ascii"))
        except ValueError as exc:
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
    """Launch a shell command in its own process group with durable ownership.

    Persist identity before delivering the trainer payload; earlier EOF prevents execution.
    Identity files remain; ``exited`` requires confirmed cleanup. Surviving descendants cause
    failure and termination even if their root succeeded. The guardian holds GPU leases
    until the entire group is gone, independently of the trainer's file descriptors.

    The earlier timeout or absolute deadline covers launch and execution; cleanup is
    outside that budget. Expected post-spawn deadline, identity-write and payload failures
    return a ProcessResult after cleanup.

    :param str cmd: Shell command run with ``/bin/sh``.
    :param dict[str, str] env: Full trainer environment.
    :param IO[str] stdout: Open stdout destination.
    :param IO[str] stderr: Open stderr destination.
    :param float | None timeout: Seconds from call entry, or ``None`` for no relative deadline.
    :param Path trial_dir: Directory for ``process_identity.json`` and lifecycle records.
    :param str attempt_id: Attempt identity already persisted in Optuna.
    :param float | None wallclock_deadline: Absolute ``time.monotonic()`` phase/run deadline.
    :param str | None cwd: Trainer working directory; ``None`` retains invocation cwd.
    :param Collection[int] gpu_lease_fds: GPU-lock descriptors inherited by the guardian alone.
    :raises PhaseSweepError: PID namespace is unreadable; nothing is recorded or launched.
    :raises ShutdownLaunchRefusedError: Shutdown blocks launch; the attempt is recorded exited.
    :raises RuntimeError: Supervisor readiness failed before creating a process group.
    :raises OSError: Spawn/pipe creation failed, or unexpected I/O failed after confirmed
        cleanup.
    :raises UnsafeProcessCleanupError: Unexpected post-launch failure with uncertain cleanup.
    :return ProcessResult: Return code, elapsed time, timeout attribution, failure reason and
        cleanup confirmation.
    """
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

    # Defer signals before taking the launch lock, and release it before unmasking.
    # The handler takes the same lock; either reversed transition can self-deadlock.
    proc: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    ack_write: int | None = None
    status_read: int | None = None
    identity_path = trial_dir / PROCESS_IDENTITY_FILE
    # Read the inherited PID namespace before writing or spawning; unreadable
    # identity must refuse launch without side effects.
    pid_namespace = read_pid_namespace()
    if pid_namespace is None:
        raise PhaseSweepError(
            "cannot read this process's host PID namespace. Refused to launch a trial "
            "whose process identity could not be tied to it.",
            action=OperatorAction.FIX_CONFIG,
        )

    try:
        with defer_shutdown_signals(), _launch_lock:
            # The handler sets the servicing signal before it takes this lock
            # to snapshot children, so a launch past this check is in its
            # snapshot and one after it never starts a process.
            signum = servicing_shutdown_signal()
            if signum is not None:
                raise ShutdownLaunchRefusedError(signum)
            # Persist launching before Popen so missing identity requires recovery rather
            # than being mistaken for a worker still queued for a GPU.
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
                lifecycle_args=(str(trial_dir), attempt_id, str(started)),
            )
            identity = _trial_process_identity(
                attempt_id=attempt_id,
                pid=pgid,
                pgid=pgid,
                pid_namespace=pid_namespace,
            )
            _write_process_identity(identity_path, identity)
            if deadline is not None and time.monotonic() >= deadline:
                # No payload was delivered; abort the blocked supervisor before execution.
                raise _LaunchDeadlineExpired(
                    "trial launch deadline expired before the trainer payload was delivered",
                    pid=pgid,
                )
            # Deliver the trainer payload only after _write_process_identity commits.
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
            # Raised at the pre-payload recheck or while the payload is written:
            # the supervisor is still blocked on its ack pipe; kill and reap it here.
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
                # Popen handle. Its failed cleanup is not
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
        launch_failure_reason = (
            f"failed to persist process identity or deliver launch payload: {exc}"
        )
        log.exception(
            "Process PID %d (pgid %d) launched but identity persistence or launch payload "
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
            log.warning("Process PID %d (pgid %d) timed out — terminating group", pgid, pgid)
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
                    "Process PID %d exited with code %s but process group %d "
                    "still had live descendants; the lease guardian terminated them",
                    pgid,
                    proc.returncode,
                    pgid,
                )
            # Root exit is insufficient: surviving descendants still require group cleanup.
            if failure_reason is None and _process_group_alive(pgid):
                failure_reason = (
                    f"root process exited with code {proc.returncode}, "
                    f"but process group {pgid} still had live descendants"
                )
                log.warning(
                    "Process PID %d exited with code %s but process group %d "
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
        # After payload delivery, retain ownership and shutdown tracking until cleanup
        # proves the group gone, including when waiting raises.
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

    # Retain identity through the Optuna commit crash window. Persist exited only
    # after confirming the whole group is gone.
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
