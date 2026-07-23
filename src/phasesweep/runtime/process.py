"""Child process lifecycle management.

Owns the dangerous parts: process groups, signal forwarding, PID tracking,
and graceful + forceful termination. Every child subprocess created by
phasesweep goes through this module.

Design:
  - Children run in their own process group (start_new_session=True) so we
    can kill the whole tree with os.killpg, not just the shell.
  - A global registry of live children lets us clean up on orchestrator death.
  - PID files in each trial_dir let operators identify orphans manually.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO

from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import strict_json_loads

log = logging.getLogger("phasesweep.runtime.process")

_KILL_GRACE_SECONDS = 10.0
_SUPERVISOR_READY_TIMEOUT_SECONDS = 10.0
PROCESS_IDENTITY_FILE = "process_identity.json"
PROCESS_IDENTITY_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Global child registry + signal handler
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_active_children: dict[int, subprocess.Popen] = {}  # pgid -> Popen
_installed = False

# The launch lock guards the Popen() -> _register() critical section so the
# shutdown handler cannot snapshot _active_children while a child has been
# spawned but not yet registered (review v0.5.7 / blocker 3). The handler
# acquires the same lock before snapshotting, which forces it to wait until
# every in-flight launch has either registered its PGID or failed.
_launch_lock = threading.Lock()
_shutdown_handler_lock = threading.Lock()
_SHUTDOWN_SIGNALS: tuple[int, ...] = tuple(
    sig
    for sig in (
        signal.SIGTERM,
        signal.SIGINT,
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
)


@dataclass(frozen=True)
class ShutdownCleanupReport:
    """Cleanup evidence captured when the orchestrator handles a shutdown signal."""

    signum: int
    cleanup_confirmed: bool
    child_pgids: tuple[int, ...]


class PhaseSweepShutdown(SystemExit):
    """SystemExit carrying child process-group cleanup evidence."""

    def __init__(self, signum: int, report: ShutdownCleanupReport) -> None:
        """Create a POSIX-style signaled exit with structured cleanup evidence.

        Args:
            signum: Shutdown signal that triggered the exit; the exit code is
                ``128 + signum`` per the POSIX signaled-exit convention.
            report: Cleanup evidence captured by the shutdown handler for the
                child process groups it terminated.

        """
        super().__init__(128 + signum)
        self.signum = signum
        self.report = report


# Python-level shutdown deferral. Kernel signal masks are per-thread, but
# CPython runs Python signal handlers only in the main thread — and it does so
# whenever ANY thread's C-level handler tripped the pending flag, regardless of
# the main thread's kernel mask. Library thread pools (e.g. BLAS workers pulled
# in via numpy/optuna) keep shutdown signals unblocked, so a signal sent while
# the main thread is inside a ``_defer_shutdown_signals()`` window can still
# execute ``_shutdown_handler`` in the main thread mid-critical-section, where
# re-acquiring ``_launch_lock``/``_lock`` self-deadlocks. These state variables
# extend the deferral to the Python level: while the main thread is inside a
# window, the handler records the signal and returns; the outermost window exit
# services it. Both are touched only from the main thread (window bookkeeping
# by construction, the handler by CPython's main-thread guarantee), so no lock
# is needed.
_deferred_shutdown_signum: int | None = None
_main_thread_defer_depth = 0


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    """Kill all tracked child process groups, then exit.

    CRITICAL: we must NOT call proc.wait() here because the main thread may
    already be inside proc.wait() on the same Popen object, and Python's
    internal _waitpid_lock is non-reentrant. Calling wait() from the signal
    handler would deadlock.

    Instead: send SIGTERM to all groups, brief sleep, SIGKILL for stragglers,
    then raise SystemExit. The original wait() call unblocks when the child dies.

    We snapshot the PGID set once under the lock and use that snapshot for both
    the SIGTERM and SIGKILL phases (review v0.5.5 / blocker 1). Without this,
    a worker thread can unregister a PGID after the root process exits on
    SIGTERM but before we escalate to SIGKILL — leaving descendants alive.

    The handler also acquires ``_launch_lock`` before snapshotting (review
    v0.5.7 / blocker 3) so a child that was just ``Popen()``-ed but not yet
    ``_register()``-ed cannot escape the snapshot. When the launcher is a
    worker thread, this handler (main thread) blocks until the launch site
    releases the lock, then sees the new PGID in ``_active_children``. When
    the launcher IS the main thread, blocking on the lock would self-deadlock:
    kernel masking cannot prevent that (a signal delivered to any unblocked
    library thread still runs this handler in the main thread), so if a
    main-thread ``_defer_shutdown_signals()`` window is open the handler
    records the signal and returns; the window exit re-invokes it.

    A second signal delivered while this handler is already running returns
    immediately. The first invocation remains responsible for the complete
    PGID snapshot and its cleanup evidence instead of re-entering the
    non-reentrant registry locks or interrupting the kill loop.

    Args:
        signum: The signal number that fired (``SIGTERM``, ``SIGINT``, or
            ``SIGHUP`` where available).
        _frame: The interrupted stack frame at signal-delivery time; unused
            but required by the ``signal.signal`` handler protocol.

    Raises:
        SystemExit: With exit code ``128 + signum`` (POSIX ``signaled-exit``
            convention) — always, except when deferred mid-critical-section
            or ignored during an active handler invocation as described above.

    """
    global _deferred_shutdown_signum  # noqa: PLW0603

    # Python signal handlers can interrupt an earlier invocation of this
    # handler. Let the first signal finish the authoritative cleanup pass;
    # re-entering could deadlock on the non-reentrant registry locks or replace
    # the first signal's cleanup evidence partway through the kill loop.
    if not _shutdown_handler_lock.acquire(blocking=False):
        return

    try:
        if _main_thread_defer_depth > 0:
            # The main thread is inside a launch/unregister critical section and
            # may already hold the locks below. Record and return; the outermost
            # window exit services the shutdown.
            _deferred_shutdown_signum = signum
            return
        # Any recorded-but-unserviced signal is superseded by this invocation.
        _deferred_shutdown_signum = None

        with _launch_lock, _lock:
            pgids = tuple(_active_children)

        log.warning("Received signal %d — killing %d active child group(s)", signum, len(pgids))

        confirmed_by_pgid: dict[int, bool] = {}
        for pgid in pgids:
            try:
                confirmed_by_pgid[pgid] = _terminate_process_group(
                    pgid,
                    grace_seconds=_KILL_GRACE_SECONDS,
                )
            except Exception:
                log.exception(
                    "Failed while cleaning child process group %d after signal %d", pgid, signum
                )
                confirmed_by_pgid[pgid] = False

        cleanup_confirmed = all(confirmed_by_pgid.get(pgid, False) for pgid in pgids)
        report = ShutdownCleanupReport(
            signum=signum,
            cleanup_confirmed=cleanup_confirmed,
            child_pgids=pgids,
        )
        if cleanup_confirmed:
            log.warning(
                "Received signal %d; confirmed cleanup for %d child group(s)",
                signum,
                len(pgids),
            )
        else:
            uncertain = [pgid for pgid in pgids if not confirmed_by_pgid.get(pgid, False)]
            log.error(
                "Received signal %d; cleanup is uncertain for child group(s): %s",
                signum,
                uncertain,
            )

        raise PhaseSweepShutdown(signum, report)
    finally:
        _shutdown_handler_lock.release()


def _unblock_shutdown_signals() -> None:
    """Ensure shutdown signals can reach the main-thread handler."""
    if not hasattr(signal, "pthread_sigmask"):
        return
    if threading.current_thread() is not threading.main_thread():
        return
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _SHUTDOWN_SIGNALS)


def install_signal_handlers() -> None:
    """Install shutdown handlers that clean up child process groups.

    Handles SIGTERM/SIGINT and SIGHUP where the platform exposes it. Safe to
    call multiple times; only installs once. The main thread's shutdown signals
    are unblocked on each call so an inherited signal mask cannot prevent the
    handlers from running. Must be called from the main thread.
    """
    global _installed  # noqa: PLW0603
    _unblock_shutdown_signals()
    if _installed:
        return
    try:
        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _shutdown_handler)
        _installed = True
    except ValueError:
        log.debug("Cannot install signal handlers (not on main thread)")


@contextlib.contextmanager
def _defer_shutdown_signals() -> Iterator[None]:
    """Defer shutdown handling while the calling thread is in a critical section.

    Used to keep the ``Popen() -> _register()`` window atomic from the
    perspective of the signal handler (review v0.5.7 / blocker 3). CPython
    runs Python signal handlers in the main thread, so when the launcher is
    the main thread, taking ``_launch_lock`` from the handler would deadlock
    against the launcher's own lock acquisition. Two layers close that:

    1. Kernel mask: the calling thread blocks shutdown signals, so a signal
       aimed at it queues until the critical section ends.
    2. Python-level deferral (main thread only): the kernel mask is per-thread
       and cannot stop a signal delivered to some other unblocked thread (e.g.
       a BLAS pool worker) from tripping CPython's pending flag — the Python
       handler then runs in the main thread mid-window anyway. While a
       main-thread window is open, ``_shutdown_handler`` records the signal
       and returns; the outermost window exit re-invokes it after the kernel
       mask is restored.

    For worker-thread launchers (``n_jobs > 1``) the handler still runs on the
    main thread, which is NOT inside the window, so it simply blocks on
    ``_launch_lock`` until the worker finishes registration — the designed
    behavior, with no self-deadlock possible.

    Kernel masking is skipped on platforms without ``signal.pthread_sigmask``
    (Windows); the Python-level deferral still applies.

    Yields:
        ``None``. Use as ``with _defer_shutdown_signals(): ...``.

    """
    global _main_thread_defer_depth  # noqa: PLW0603
    is_main = threading.current_thread() is threading.main_thread()
    old_mask = None
    if hasattr(signal, "pthread_sigmask"):
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
    if is_main:
        _main_thread_defer_depth += 1
    try:
        yield
    finally:
        if is_main:
            _main_thread_defer_depth -= 1
        if old_mask is not None:
            # Restoring the mask delivers any kernel-queued signal right here;
            # its handler runs normally (depth is already back to zero) and
            # clears any deferred marker before raising.
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if is_main and _main_thread_defer_depth == 0 and _deferred_shutdown_signum is not None:
            _service_deferred_shutdown()


def _service_deferred_shutdown() -> None:
    """Run the shutdown handler for a signal recorded mid-critical-section.

    Raises:
        PhaseSweepShutdown: Via ``_shutdown_handler``, carrying the cleanup
            evidence for the recorded signal.

    """
    global _deferred_shutdown_signum  # noqa: PLW0603
    signum = _deferred_shutdown_signum
    _deferred_shutdown_signum = None
    if signum is not None:
        _shutdown_handler(signum, None)


def _register(proc: subprocess.Popen) -> int:
    """Add a freshly-launched subprocess to the global child registry.

    Args:
        proc: The ``Popen`` object returned by a just-completed ``Popen()`` call.

    Returns:
        The process-group ID (``pgid``) the subprocess was registered under.
        Callers store this for later ``_unregister`` and signal targeting.

    """
    pgid = os.getpgid(proc.pid)
    with _lock:
        _active_children[pgid] = proc
    return pgid


def _unregister(pgid: int) -> None:
    """Remove a finished process group from the global child registry.

    Defers shutdown signals while holding ``_lock`` so the signal handler
    (which also acquires ``_lock``) cannot interrupt and deadlock against
    the same thread (review v0.5.9 / blocker 2).

    Args:
        pgid: The process-group ID returned by :func:`_register`. A pgid not
            currently registered is silently ignored.

    """
    with _defer_shutdown_signals(), _lock:
        _active_children.pop(pgid, None)


def _kill_group(pgid: int, proc: subprocess.Popen) -> bool:
    """Terminate the trial process group and return whether cleanup is confirmed.

    Returns ``True`` when the group is confirmed gone, ``False`` when cleanup
    is uncertain (survived SIGKILL, permission denied, etc.). Callers must
    propagate uncertainty so the orchestrator can refuse to schedule more work
    onto a potentially-leaked GPU (review v0.5.9 / blocker 3).

    Args:
        pgid: Process-group ID of the trial subprocess.
        proc: The root subprocess's :class:`subprocess.Popen` handle. Used
            for a final non-blocking ``wait`` to reap the zombie root.

    Returns:
        ``True`` if every process in the group is gone after the SIGTERM →
        SIGKILL escalation; ``False`` if at least one survived or cleanup
        status was inconclusive.

    """
    cleanup_confirmed = _terminate_process_group(pgid, grace_seconds=_KILL_GRACE_SECONDS)
    if proc.poll() is not None:
        with contextlib.suppress(Exception):
            proc.wait(timeout=0)
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


@dataclass(frozen=True)
class StaleProcessIdentity:
    """Durable identity of one launched trial process group."""

    schema_version: int
    attempt_id: str
    pid: int
    pgid: int
    proc_starttime: int | None
    boot_id: str | None


def read_boot_id() -> str | None:
    """Return the current Linux boot identity, or ``None`` when unavailable."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


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


def _spawn_blocked_supervisor(
    cmd: str,
    *,
    env: dict[str, str],
    stdout: IO[str],
    stderr: IO[str],
) -> tuple[subprocess.Popen, int, int]:
    """Spawn a supervisor that cannot exec ``cmd`` until its parent acknowledges it.

    Launches ``phasesweep.runtime.supervisor`` in its own session, passing it
    a readiness pipe and an acknowledgement pipe. Blocks (via ``select``) until
    the supervisor signals readiness or ``_SUPERVISOR_READY_TIMEOUT_SECONDS``
    elapses, then registers the new process group. On any failure — timeout,
    an unexpected readiness byte, or an exception from ``Popen`` itself — any
    spawned process group is killed and unregistered before the exception
    propagates.

    Args:
        cmd: Shell command string the acknowledged supervisor execs with ``/bin/sh``.
        env: Full process environment for the supervisor subprocess.
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.

    Returns:
        A ``(proc, pgid, ack_write)`` tuple: the supervisor's ``Popen`` handle,
        its registered process-group id, and the write end of the
        acknowledgement pipe. The caller owns ``ack_write`` and must write one
        acknowledgement byte and close it once the trial's process identity is
        durably persisted.

    Raises:
        RuntimeError: If the supervisor does not signal readiness within
            ``_SUPERVISOR_READY_TIMEOUT_SECONDS``, or signals something other
            than ``b"R"``.

    """
    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    proc: subprocess.Popen | None = None
    pgid: int | None = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "phasesweep.runtime.supervisor",
                str(ready_write),
                str(ack_read),
                cmd,
            ],
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            pass_fds=(ready_write, ack_read),
        )
        os.close(ready_write)
        ready_write = -1
        os.close(ack_read)
        ack_read = -1

        pgid = _register(proc)
        readable, _, _ = select.select(
            [ready_read],
            [],
            [],
            _SUPERVISOR_READY_TIMEOUT_SECONDS,
        )
        if not readable or os.read(ready_read, 1) != b"R":
            raise RuntimeError("trial supervisor did not become ready before launch")
        os.close(ready_read)
        ready_read = -1
        return proc, pgid, ack_write
    except Exception:
        if ack_write >= 0:
            os.close(ack_write)
            ack_write = -1
        if proc is not None:
            target_pgid = pgid if pgid is not None else proc.pid
            _kill_group(target_pgid, proc)
        if pgid is not None:
            _unregister(pgid)
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
) -> ProcessResult:
    """Launch a shell command in its own process group with full lifecycle management.

    Launches a small supervisor first. The supervisor waits on an inherited
    acknowledgement pipe while the parent atomically persists
    ``process_identity.json``. Only after the identity is durable does the
    parent acknowledge the supervisor, which then execs the trainer command.
    If the parent dies before acknowledgement, pipe EOF makes the supervisor
    exit without starting training.

    The identity record is removed only on a fully clean exit: root returned 0
    **and** no descendant processes were left alive. If the root exits cleanly
    but leaves GPU-holding descendants running, we treat that as a lifecycle
    failure, kill the group, and preserve the identity record for forensics (review
    v0.5.5 / blocker 1).

    On timeout: SIGTERM -> grace -> SIGKILL on the entire group.

    Args:
        cmd: Shell command string the acknowledged supervisor execs with ``/bin/sh``.
        env: Full process environment for the subprocess.
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.
        timeout: Wall-clock timeout in seconds, or ``None`` for no timeout.
        trial_dir: Per-trial directory where ``process_identity.json`` is written.
        attempt_id: Immutable attempt identity already persisted in Optuna.

    Returns:
        :class:`ProcessResult` capturing return code, wall-clock duration,
        timeout flag, ``failure_reason`` (set on timeout or descendant
        survival), and ``cleanup_confirmed`` (``False`` when SIGKILL did not
        confirm the group is gone).

    """
    import time

    started = time.monotonic()

    # The launch + register + identity write must be atomic from the
    # signal handler's perspective. Signal deferral MUST come first so that
    # SIGTERM/SIGINT cannot land between ``_launch_lock`` acquisition and
    # signal masking (review v0.5.9 / blocker 2). Reversing the order
    # (``_launch_lock`` first, then ``_defer_shutdown_signals()``) creates
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
    proc: subprocess.Popen | None = None
    pgid: int | None = None
    ack_write: int | None = None
    identity_path = trial_dir / PROCESS_IDENTITY_FILE

    try:
        with _defer_shutdown_signals(), _launch_lock:
            proc, pgid, ack_write = _spawn_blocked_supervisor(
                cmd,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            identity = _trial_process_identity(
                attempt_id=attempt_id,
                pid=proc.pid,
                pgid=pgid,
            )
            _write_process_identity(identity_path, identity)
            if os.write(ack_write, b"A") != 1:
                raise RuntimeError("trial supervisor acknowledgement was not delivered")
            os.close(ack_write)
            ack_write = None
    except Exception as exc:
        if ack_write is not None:
            os.close(ack_write)
        if proc is None:
            raise
        target_pgid = pgid if pgid is not None else proc.pid
        identity_failure_reason = f"failed to persist process identity: {exc}"
        log.exception(
            "Trial PID %d (pgid %d) launched but identity persistence failed; terminating group",
            proc.pid,
            target_pgid,
        )
        cleanup_confirmed = _kill_group(target_pgid, proc)
        if pgid is not None:
            _unregister(pgid)
        duration = time.monotonic() - started
        return ProcessResult(
            return_code=proc.returncode if proc.returncode is not None else -9,
            timed_out=False,
            pid=proc.pid,
            duration_seconds=duration,
            failure_reason=identity_failure_reason,
            cleanup_confirmed=cleanup_confirmed,
        )

    assert proc is not None
    assert pgid is not None

    timed_out = False
    failure_reason: str | None = None
    cleanup_confirmed = True

    try:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            failure_reason = f"timeout after {timeout}s"
            log.warning("Trial PID %d (pgid %d) timed out — terminating group", proc.pid, pgid)
            cleanup_confirmed = _kill_group(pgid, proc)
        else:
            # Root process exited normally. That is not sufficient — the trial
            # is only clean once the entire process group is gone. A common
            # pathological case: `python launcher.py &` exits immediately while
            # the training worker stays alive holding GPU memory.
            if _process_group_alive(pgid):
                failure_reason = (
                    f"root process exited with code {proc.returncode}, "
                    f"but process group {pgid} still had live descendants"
                )
                log.warning(
                    "Trial PID %d exited with code %s but process group %d "
                    "still has live descendants — terminating group",
                    proc.pid,
                    proc.returncode,
                    pgid,
                )
                cleanup_confirmed = _kill_group(pgid, proc)

    finally:
        _unregister(pgid)

        # Only clean the identity record when the trial exited cleanly AND no
        # descendant cleanup was required.
        if failure_reason is None and proc.returncode == 0:
            identity_path.unlink(missing_ok=True)

    duration = time.monotonic() - started
    return ProcessResult(
        return_code=proc.returncode if proc.returncode is not None else -9,
        timed_out=timed_out,
        pid=proc.pid,
        duration_seconds=duration,
        failure_reason=failure_reason,
        cleanup_confirmed=cleanup_confirmed,
    )


# ---------------------------------------------------------------------------
# Stale process utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProcStat:
    """Parsed fields from one Linux ``/proc/<pid>/stat`` record."""

    state: str
    pgrp: int
    starttime: int


def _read_proc_stat(proc_entry: Path) -> _ProcStat | None:
    """Parse the proc stat fields phasesweep uses for liveness checks.

    :param Path proc_entry: ``/proc/<pid>`` directory to inspect.
    :return _ProcStat | None: Parsed state, process group, and starttime, or ``None`` when
        unreadable.
    """
    try:
        data = (proc_entry / "stat").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    rparen = data.rfind(b")")
    if rparen < 0:
        return None
    rest = data[rparen + 1 :].strip().split()
    if len(rest) < 20:
        return None
    try:
        return _ProcStat(state=rest[0].decode("ascii"), pgrp=int(rest[2]), starttime=int(rest[19]))
    except (UnicodeDecodeError, ValueError):
        return None


def read_proc_starttime(pid: int) -> int | None:
    """Read the start time of a process from /proc/<pid>/stat.

    On Linux, (pid, starttime) uniquely identifies a process across its
    lifetime. This is the only reliable way to avoid PID-reuse hazards
    when killing stale processes from a prior orchestrator run.

    Args:
        pid: The process ID to inspect.

    Returns:
        The starttime in clock ticks (``/proc/<pid>/stat`` field 22), or
        ``None`` on non-Linux systems and when the proc entry is unreadable.

    """
    stat = _read_proc_stat(Path("/proc") / str(pid))
    return None if stat is None else stat.starttime


def is_pid_alive(pid: int) -> bool:
    """Check if a PID exists (best-effort; race-free check is impossible).

    Args:
        pid: The process ID to probe via ``kill(pid, 0)``.

    Returns:
        ``True`` if the PID exists (or exists but is owned by another user);
        ``False`` if the kernel reports ``ProcessLookupError``.

    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user


def is_same_process(pid: int, saved_starttime: int | None) -> bool:
    """Check whether `pid` is the same process that recorded `saved_starttime`.

    When no starttime was recorded, this falls back to a PID-alive check. When
    a starttime was recorded but the current proc entry is unreadable, identity
    is unknown and this fails closed instead of treating the PID as a match.

    Args:
        pid: PID read from a stale ``trial_dir/pid`` file.
        saved_starttime: Starttime read from the matching ``pid_starttime``
            file, or ``None`` if unavailable.

    Returns:
        ``True`` if ``pid`` is alive AND (no saved starttime, OR the current
        ``/proc`` starttime matches the saved value). ``False`` if the PID is
        dead, has been reused by an unrelated process, or cannot be verified.

    """
    if not is_pid_alive(pid):
        return False
    if saved_starttime is None:
        # No starttime to verify — fall back to alive-only (best effort).
        return True
    current_starttime = read_proc_starttime(pid)
    if current_starttime is None:
        return False
    return current_starttime == saved_starttime


def is_pid_zombie(pid: int) -> bool:
    """Return whether ``pid`` is a zombie (exited but not yet reaped by its parent).

    A zombie still answers ``kill(pid, 0)`` because it occupies the PID table,
    so :func:`is_pid_alive` and :func:`is_same_process` both report it as alive.
    For a liveness decision it is effectively dead — it holds no resources and
    is doing no work. This reads ``/proc/<pid>/stat`` (state is the first field
    after the ``)`` that closes ``comm``) and returns ``True`` only for state
    ``Z``. On non-Linux (no ``/proc``) it returns ``False``, preserving the
    legacy alive-only semantics used elsewhere.

    Args:
        pid: Process ID to probe.

    Returns:
        ``True`` if the process exists and is a zombie; ``False`` if it is live,
        gone, or undeterminable (non-Linux).

    """
    stat = _read_proc_stat(Path("/proc") / str(pid))
    return stat is not None and stat.state == "Z"


def is_same_live_process(pid: int | None, saved_starttime: int | None) -> bool:
    """Return whether a PID identifies the same live, non-zombie process.

    :param int | None pid: Process identifier, if one was recorded.
    :param int | None saved_starttime: Recorded Linux process start time.
    :return bool: Whether the identity still names a live non-zombie process.
    """
    return pid is not None and is_same_process(pid, saved_starttime) and not is_pid_zombie(pid)


def reap_child(pid: int) -> bool:
    """Best-effort non-blocking reap of one exited child.

    A long-lived parent (the MCP server) that spawns detached runners and never
    waits on them accumulates a zombie per runner as each one exits. Call this
    for a known runner pid to reap it if it has already exited; it is a no-op if
    the process is still running, was never our child, or has already been reaped.
    This runs on status-read paths, so it must stay strictly non-blocking. A
    single ``waitpid(WNOHANG)`` reduces zombie buildup; it does not guarantee
    that a child exiting immediately after this call is reaped before the next
    status scan.

    Args:
        pid: PID of a runner this process spawned.

    Returns:
        ``True`` when this call reaped ``pid``; ``False`` otherwise.

    """
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except OSError:
        return False
    return waited == pid


def read_stale_process_identity(
    trial_dir: Path,
    *,
    expected_attempt_id: str,
) -> StaleProcessIdentity:
    """Load and validate the atomic identity for a possibly-stale trial.

    Args:
        trial_dir: Trial directory containing ``process_identity.json``.
        expected_attempt_id: Attempt identity stored on the Optuna trial.

    Returns:
        A complete, attempt-bound :class:`StaleProcessIdentity`.

    Raises:
        OSError: The identity record is missing or unreadable.
        ValueError: The identity record is malformed or belongs to another attempt.

    """
    path = trial_dir / PROCESS_IDENTITY_FILE
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Malformed trial process identity at {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Trial process identity at {path} must be a JSON object.")
    required_fields = {
        "schema_version",
        "attempt_id",
        "pid",
        "pgid",
        "proc_starttime",
        "boot_id",
    }
    if not required_fields.issubset(payload):
        raise ValueError(f"Trial process identity at {path} is partial.")
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != PROCESS_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trial process identity schema at {path}.")
    attempt_id = payload.get("attempt_id")
    if attempt_id != expected_attempt_id:
        raise ValueError(f"Trial process identity at {path} belongs to another attempt.")

    def positive_int(field: str) -> int:
        """Validate and return one required positive-int identity field.

        Args:
            field: Name of the top-level identity field to validate, used
                only to build the error message.

        Returns:
            The field's value, guaranteed to be a non-bool ``int`` greater
            than zero.

        Raises:
            ValueError: If the field is missing, not an ``int`` (or is a
                ``bool``), or not strictly positive.

        """
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Trial process identity field {field!r} is invalid at {path}.")
        return value

    proc_starttime = payload.get("proc_starttime")
    if proc_starttime is not None and (
        isinstance(proc_starttime, bool)
        or not isinstance(proc_starttime, int)
        or proc_starttime <= 0
    ):
        raise ValueError(f"Trial process identity field 'proc_starttime' is invalid at {path}.")
    boot_id = payload.get("boot_id")
    if boot_id is not None and (not isinstance(boot_id, str) or not boot_id):
        raise ValueError(f"Trial process identity field 'boot_id' is invalid at {path}.")
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id=attempt_id,
        pid=positive_int("pid"),
        pgid=positive_int("pgid"),
        proc_starttime=proc_starttime,
        boot_id=boot_id,
    )


def cleanup_stale_trial_process(
    identity: StaleProcessIdentity,
    *,
    grace_seconds: float = _KILL_GRACE_SECONDS,
) -> bool:
    """Safely clean a stale trial group using boot- and process-bound identity.

    Refuses to act when ``identity`` lacks a recorded boot id or start time, or
    the current boot id cannot be read, since PID-reuse safety cannot be
    verified in that case. When ``identity.boot_id`` differs from the current
    boot id, the host has rebooted since launch, so no process from that boot
    can still be alive and cleanup is trivially complete. Otherwise delegates
    to :func:`kill_stale_group` with the identity's ``pid``/``pgid``/starttime.

    Args:
        identity: Durable process identity read from ``process_identity.json``.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL; forwarded to :func:`kill_stale_group`.

    Returns:
        ``True`` when it is safe to mark the trial ``FAIL`` (nothing to clean,
        the host rebooted, or cleanup was confirmed). ``False`` when identity
        cannot be verified or :func:`kill_stale_group` reports cleanup is
        uncertain; callers must not advance state in that case.

    """
    current_boot_id = read_boot_id()
    if identity.boot_id is None or identity.proc_starttime is None or current_boot_id is None:
        log.warning(
            "Refusing automatic cleanup for attempt %s because robust process-birth identity "
            "is unavailable on this platform.",
            identity.attempt_id,
        )
        return False
    if identity.boot_id != current_boot_id:
        log.warning(
            "Attempt %s belongs to an earlier host boot; no process from that boot remains.",
            identity.attempt_id,
        )
        return True
    return kill_stale_group(
        identity.pid,
        identity.proc_starttime,
        pgid=identity.pgid,
        grace_seconds=grace_seconds,
    )


def _terminate_process_group(pgid: int, *, grace_seconds: float) -> bool:
    """Send SIGTERM, wait, then SIGKILL — and confirm the group is actually gone.

    Returns ``True`` only when the process group is confirmed gone. Returns
    ``False`` when delivery fails (permission denied, OS error) or the group
    is still alive after SIGKILL. Pre-v0.5.8 this function returned ``True``
    even when the group survived SIGKILL — callers then marked the trial
    ``FAIL`` and proceeded, potentially launching new trials onto a GPU still
    held by the leaked process (review v0.5.7 / blocker 2).

    ``ProcessLookupError`` from ``killpg`` means the group is already gone, so
    those branches return ``True``.

    Args:
        pgid: Target process-group ID.
        grace_seconds: Seconds to wait after SIGTERM before escalating to SIGKILL.

    Returns:
        ``True`` if the group is confirmed dead (already gone, or died within
        the SIGTERM grace, or died within 2s after SIGKILL). ``False`` if
        signal delivery failed for non-``ProcessLookupError`` reasons, or the
        group is still alive 2s after SIGKILL.

    """
    import time

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError) as exc:
        log.error("Failed to send SIGTERM to process group %d: %s", pgid, exc)
        return False

    member_pids = set(_group_member_pids(pgid))
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_alive_with_members(pgid, member_pids):
            return True
        time.sleep(0.1)

    log.warning("Stale process group %d survived SIGTERM — sending SIGKILL", pgid)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError) as exc:
        log.error("Failed to send SIGKILL to process group %d: %s", pgid, exc)
        return False

    # Wait briefly for the kernel to actually reap descendants so the reaper's
    # "marked FAIL" state means cleanup completed, not requested.
    kill_deadline = time.monotonic() + 2.0
    while time.monotonic() < kill_deadline:
        if not _process_group_alive_with_members(pgid, member_pids):
            return True
        time.sleep(0.05)

    log.error("Process group %d still appears alive after SIGKILL", pgid)
    return False


def _process_group_alive(pgid: int) -> bool:
    """Check whether any non-zombie process in the group ``pgid`` is alive.

    ``os.killpg(pgid, 0)`` returns success for zombie processes too, because
    they still occupy the PID table. For our purposes a zombie is dead — it
    holds no GPU memory, no file descriptors, no shared resources. Skipping
    zombies stops the cleanup escalation from looping after SIGKILL when the
    parent hasn't reaped its child yet (review v0.5.7 / blocker 2 follow-up).

    On non-Linux, ``/proc/<pid>/stat`` doesn't exist; we fall back to the
    previous behavior (``killpg(0)`` semantics).

    Args:
        pgid: Process-group ID to probe.

    Returns:
        ``True`` if at least one non-zombie member of the group exists.
        ``False`` if the group is gone, or every remaining member is a zombie
        (state ``Z``/``X``).

    """
    return _process_group_alive_with_members(pgid, None)


def _process_group_exists(pgid: int) -> bool:
    """Return whether the process group has any PID-table entry.

    :param int pgid: Process-group ID to probe.
    :return bool: ``True`` when the group exists or exists but is not inspectable.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Group exists but is owned by a different user. Treat as alive.
        return True
    return True


def _stored_pgid_is_reused_group_leader(pgid: int, saved_starttime: int) -> bool | None:
    """Return whether ``pgid`` identifies a new group leader, or is unverifiable.

    A PGID number can later be reused as an ordinary PID in another process
    group; that does not make ``killpg(pgid, ...)`` unsafe because the process
    is not a member of the target group. The unsafe case is narrower: a live
    ``/proc/<pgid>`` entry belongs to process group ``pgid`` but has a different
    starttime, meaning the saved group ID is now led by an unrelated process.

    :param int pgid: Stored process-group ID whose leader identity is checked.
    :param int saved_starttime: Starttime recorded for the original group leader.
    :return bool | None: ``True`` for a reused leader, ``False`` when no live
        leader exists or its identity is compatible, and ``None`` when a live
        leader's proc identity is unreadable.
    """
    stat = _read_proc_stat(Path("/proc") / str(pgid))
    if stat is None:
        return None if is_pid_alive(pgid) else False
    return stat.pgrp == pgid and stat.starttime != saved_starttime


def _process_group_alive_with_members(pgid: int, member_pids: set[int] | None) -> bool:
    """Check group liveness, optionally using and refreshing a cached member set.

    :param int pgid: Process-group ID to probe.
    :param set[int] | None member_pids: Cached group members, or ``None`` to scan once.
    :return bool: ``True`` when a non-zombie member of the group is still alive.
    """
    if not _process_group_exists(pgid):
        return False
    proc_root = Path("/proc")
    if not proc_root.exists():
        return True
    if member_pids is None:
        return _member_pids_alive(pgid, _group_member_pids(pgid))
    if _member_pids_alive(pgid, member_pids):
        return True
    refreshed = set(_group_member_pids(pgid))
    member_pids.clear()
    member_pids.update(refreshed)
    return _member_pids_alive(pgid, member_pids)


def _group_member_pids(pgid: int) -> list[int]:
    """Return current ``/proc`` PIDs that belong to process group ``pgid``.

    :param int pgid: Process-group ID to find under ``/proc``.
    :return list[int]: PIDs currently reporting membership in ``pgid``.
    """
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    member_pids: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        stat = _read_proc_stat(entry)
        if stat is None:
            continue
        if stat.pgrp == pgid:
            member_pids.append(int(entry.name))
    return member_pids


def _member_pids_alive(pgid: int, member_pids: set[int] | list[int]) -> bool:
    """Return whether any known member PID is still live and in ``pgid``.

    :param int pgid: Process-group ID each PID must still belong to.
    :param set[int] | list[int] member_pids: Candidate member PIDs to inspect.
    :return bool: ``True`` when any candidate is a live, non-zombie member.
    """
    for pid in member_pids:
        stat = _read_proc_stat(Path("/proc") / str(pid))
        if stat is None:
            continue
        if stat.pgrp != pgid:
            continue
        if stat.state in {"Z", "X"}:
            continue
        return True
    return False


def kill_stale_group(
    pid: int | None,
    saved_starttime: int | None,
    *,
    pgid: int | None = None,
    grace_seconds: float = _KILL_GRACE_SECONDS,
) -> bool:
    """Terminate a stale trial process group, escalating SIGTERM -> SIGKILL.

    Returns ``True`` only when it is safe to mark the trial ``FAIL``: either
    nothing was alive to clean up, or cleanup ran and the process group is
    confirmed gone. Returns ``False`` when cleanup is uncertain — PID/PGID
    identity was reused unsafely, permission was denied, or the group survived
    SIGKILL. Callers must not advance state when this returns ``False``
    (review v0.5.7 / blocker 2): a leaked training process can still hold a
    GPU, scribble over W&B runs, or starve the host scheduler.

    Recovery order (review v0.5.3 / blocker 2):

    1. ``pid`` is alive AND saved starttime matches: derive PGID from the live
       PID and kill the group. Starttime check guards against PID reuse.
    2. ``pid`` is alive but starttime mismatches: this is PID reuse by an
       unrelated process. If the stored PGID proves the old group is gone,
       cleanup is complete. If the PGID still exists, use it only when the
       reused PID is not the leader of that group; otherwise fail closed to
       avoid killing an unrelated process group.
    3. ``pid`` is dead but its verified same-boot identity includes ``pgid``:
       use the group only when its leader identity has not been reused. The
       root shell may have exited while descendants still hold GPU memory.
    4. No PID and no PGID: nothing to clean up, return ``True``.

    Args:
        pid: Root PID from a durable process identity, or ``None`` if absent.
        saved_starttime: Saved Linux process start time, or ``None`` when identity
            cannot be verified. A live PID/PGID is never signalled in that case.
        pgid: Process-group ID from the same durable identity.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL; passed through to :func:`_terminate_process_group`.

    Returns:
        ``True`` when it is safe to mark the trial ``FAIL`` (cleanup
        confirmed or nothing to clean). ``False`` when cleanup is uncertain
        and callers must NOT advance state.

    """
    target_pgid: int | None = None

    if saved_starttime is None:
        pid_alive = pid is not None and is_pid_alive(pid)
        pgid_alive = pgid is not None and _process_group_exists(pgid)
        if pid_alive or pgid_alive:
            log.warning(
                "Refusing to signal stale pid=%s pgid=%s without a saved process start time.",
                pid,
                pgid,
            )
            return False
        return True

    if pid is not None:
        pid_alive = is_pid_alive(pid)
        same_process = False
        if pid_alive:
            current_starttime = read_proc_starttime(pid)
            if current_starttime is None:
                log.warning(
                    "PID %d is alive but its /proc start time is unreadable; "
                    "refusing cleanup because process identity is unknown.",
                    pid,
                )
                return False
            same_process = current_starttime == saved_starttime
        if same_process:
            try:
                target_pgid = os.getpgid(pid)
            except ProcessLookupError:
                target_pgid = None
            except (PermissionError, OSError) as exc:
                log.error("Failed reading PGID for stale PID %d: %s", pid, exc)
                return False
        elif pid_alive:
            # PID reuse detected. A persisted PGID can still prove cleanup is
            # complete (group gone) or target original descendants (group alive
            # but not led by the reused PID). Refuse only when the stored PGID
            # itself appears to be a reused group leader.
            if pgid is not None and not _process_group_exists(pgid):
                log.warning(
                    "PID %d is alive but starttime does not match saved value; "
                    "PID was reused, but stored process group %d no longer "
                    "exists.",
                    pid,
                    pgid,
                )
                return True
            if pgid is None:
                log.warning(
                    "PID %d is alive but starttime does not match saved value; "
                    "PID was reused and no stored PGID is available. Cleanup "
                    "status is uncertain.",
                    pid,
                )
                return False
            pgid_reused = _stored_pgid_is_reused_group_leader(pgid, saved_starttime)
            if pgid_reused is None:
                log.warning(
                    "Stored PGID %d has a live but unreadable leader; refusing "
                    "PGID fallback because process identity is unknown.",
                    pgid,
                )
                return False
            if pgid_reused:
                log.warning(
                    "PID %d is alive but starttime does not match saved value; "
                    "stored PGID %d is led by a different process. Refusing "
                    "PGID fallback to avoid killing an unrelated process group.",
                    pid,
                    pgid,
                )
                return False

    if target_pgid is None and pgid is not None and saved_starttime is not None:
        if not _process_group_exists(pgid):
            return True
        pgid_reused = _stored_pgid_is_reused_group_leader(pgid, saved_starttime)
        if pgid_reused is None:
            log.warning(
                "Stored PGID %d has a live but unreadable leader; refusing PGID "
                "fallback because process identity is unknown.",
                pgid,
            )
            return False
        if pgid_reused:
            log.warning(
                "Stored PGID %d is now led by a different process. Refusing "
                "PGID fallback to avoid killing an unrelated process group "
                "(saved starttime %s).",
                pgid,
                saved_starttime,
            )
            return False

    if target_pgid is None:
        if pgid is None:
            # No identity at all → nothing alive to clean up.
            return True
        log.warning(
            "Root PID is gone, reused, or unrecoverable; using stored PGID %d for "
            "best-effort cleanup of stale trial process group.",
            pgid,
        )
        target_pgid = pgid

    if not _process_group_alive(target_pgid):
        return True

    log.warning("Terminating stale training process group pgid=%d (pid=%s)", target_pgid, pid)
    return _terminate_process_group(target_pgid, grace_seconds=grace_seconds)
