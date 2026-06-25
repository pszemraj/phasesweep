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
import logging
import os
import signal
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO

from phasesweep.runtime.files import atomic_write_text

log = logging.getLogger("phasesweep.runtime.process")

_KILL_GRACE_SECONDS = 10.0

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
_SHUTDOWN_SIGNALS: tuple[int, ...] = tuple(
    sig
    for sig in (
        signal.SIGTERM,
        signal.SIGINT,
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
)


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
    ``_register()``-ed cannot escape the snapshot. The launch path holds
    ``_launch_lock`` across both calls; this handler will block until the
    launch site releases it, then see the new PGID in ``_active_children``.

    Args:
        signum: The signal number that fired (``SIGTERM``, ``SIGINT``, or
            ``SIGHUP`` where available).
        _frame: The interrupted stack frame at signal-delivery time; unused
            but required by the ``signal.signal`` handler protocol.

    Raises:
        SystemExit: Always, with exit code ``128 + signum`` (POSIX
            ``signaled-exit`` convention).

    """
    import time

    with _launch_lock, _lock:
        pgids = list(_active_children)

    log.warning("Received signal %d — killing %d active child group(s)", signum, len(pgids))

    # Phase 1: SIGTERM all groups.
    for pgid in pgids:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGTERM)

    if pgids:
        time.sleep(0.5)  # Brief grace for clean shutdown.

    # Phase 2: SIGKILL any survivors from the *same snapshot*.
    for pgid in pgids:
        if _process_group_exists(pgid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pgid, signal.SIGKILL)

    raise SystemExit(128 + signum)


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
    """Temporarily block shutdown signals in the calling thread.

    Used to keep the ``Popen() -> _register()`` window atomic from the
    perspective of the signal handler (review v0.5.7 / blocker 3). CPython
    delivers signals to the main thread, so when ``n_jobs == 1`` the same
    thread is both the launcher and the handler target — taking
    ``_launch_lock`` from the handler would deadlock against the launcher's
    own lock acquisition. Blocking the signal at the kernel level instead
    queues it until the launcher exits the critical section.

    For worker threads (``n_jobs > 1``) the handler still runs on the main
    thread, so the signal-mask state of the worker is irrelevant. The
    ``_launch_lock`` acquired by the launcher closes the race in that case.

    No-op on platforms without ``signal.pthread_sigmask`` (Windows).

    Yields:
        ``None``. Use as ``with _defer_shutdown_signals(): ...``.

    """
    if hasattr(signal, "pthread_sigmask"):
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    else:
        yield


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


def run_supervised(
    cmd: str,
    *,
    env: dict[str, str],
    stdout: IO[str],
    stderr: IO[str],
    timeout: float | None,
    trial_dir: Path,
) -> ProcessResult:
    """Launch a shell command in its own process group with full lifecycle management.

    On launch, writes three identity files into ``trial_dir``:

    * ``pid`` — root subprocess PID, for forensic identification.
    * ``pgid`` — process-group ID, used by the stale reaper as a fallback when
      the root PID has exited but descendants are still alive (review v0.5.2 /
      blocker 7). Without this, a reaper that only knows the root PID cannot
      recover the PGID via ``os.getpgid`` once the shell has exited.
    * ``pid_starttime`` — ``/proc/<pid>/stat`` field 22, used to detect PID
      reuse so the reaper never kills an unrelated process that recycled the PID.

    Identity files are removed only on a fully clean exit: root returned 0
    **and** no descendant processes were left alive. If the root exits cleanly
    but leaves GPU-holding descendants running, we treat that as a lifecycle
    failure, kill the group, and preserve identity files for forensics (review
    v0.5.5 / blocker 1).

    On timeout: SIGTERM -> grace -> SIGKILL on the entire group.

    Args:
        cmd: Shell command string to execute (passed to ``Popen(shell=True)``).
        env: Full process environment for the subprocess.
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.
        timeout: Wall-clock timeout in seconds, or ``None`` for no timeout.
        trial_dir: Per-trial directory; identity files (``pid``, ``pgid``,
            ``pid_starttime``) are written here.

    Returns:
        :class:`ProcessResult` capturing return code, wall-clock duration,
        timeout flag, ``failure_reason`` (set on timeout or descendant
        survival), and ``cleanup_confirmed`` (``False`` when SIGKILL did not
        confirm the group is gone).

    """
    import time

    started = time.time()

    # The launch + register + identity-file writes must be atomic from the
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
    pid_path = trial_dir / "pid"
    pgid_path = trial_dir / "pgid"
    starttime_path = trial_dir / "pid_starttime"

    try:
        with _defer_shutdown_signals(), _launch_lock:
            proc = subprocess.Popen(
                cmd,
                shell=True,  # noqa: S602
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )

            # Register in the global child registry immediately so the signal handler
            # can reach this group even if we haven't written files yet.
            pgid = _register(proc)

            atomic_write_text(pid_path, f"{proc.pid}\n")
            atomic_write_text(pgid_path, f"{pgid}\n")

            starttime = read_proc_starttime(proc.pid)
            if starttime is not None:
                atomic_write_text(starttime_path, f"{starttime}\n")
    except Exception as exc:
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
        duration = time.time() - started
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

        # Only clean identity files when the trial exited cleanly AND no
        # descendant cleanup was required.
        if failure_reason is None and proc.returncode == 0:
            for path in (pid_path, pgid_path, starttime_path):
                path.unlink(missing_ok=True)

    duration = time.time() - started
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
        text = (proc_entry / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    rparen = text.rfind(")")
    if rparen < 0:
        return None
    rest = text[rparen + 1 :].strip().split()
    if len(rest) < 20:
        return None
    try:
        return _ProcStat(state=rest[0], pgrp=int(rest[2]), starttime=int(rest[19]))
    except ValueError:
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

    If starttime verification is unavailable (non-Linux, or no saved starttime),
    falls back to pid-alive check only (the pre-v0.4 behavior).

    Args:
        pid: PID read from a stale ``trial_dir/pid`` file.
        saved_starttime: Starttime read from the matching ``pid_starttime``
            file, or ``None`` if unavailable.

    Returns:
        ``True`` if ``pid`` is alive AND (no saved starttime, OR the current
        ``/proc`` starttime matches the saved value). ``False`` if the PID is
        dead or has been reused by an unrelated process.

    """
    if not is_pid_alive(pid):
        return False
    if saved_starttime is None:
        # No starttime to verify — fall back to alive-only (best effort).
        return True
    current_starttime = read_proc_starttime(pid)
    if current_starttime is None:
        # Can't read /proc — non-Linux or proc vanished. Fall back.
        return True
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


def reap_child(pid: int) -> None:
    """Best-effort non-blocking reap of an exited child to prevent zombie buildup.

    A long-lived parent (the MCP server) that spawns detached runners and never
    waits on them accumulates a zombie per runner as each one exits. Call this
    for a known runner pid to reap it if it has already exited; it is a no-op if
    the process is still running, was never our child, or has already been reaped.
    This runs on status-read paths, so it must stay strictly non-blocking.

    Args:
        pid: PID of a runner this process spawned.

    """
    with contextlib.suppress(OSError):
        os.waitpid(pid, os.WNOHANG)


@dataclass
class StaleProcessIdentity:
    """Forensic identity files left in a trial directory after launch.

    Persisted by ``run_supervised`` and read by the orchestrator's stale-trial
    reaper (review v0.5.2 / blocker 7). PGID is the fallback when the root PID
    has exited but descendants are still alive.
    """

    pid: int | None
    pgid: int | None
    starttime: int | None


def read_stale_process_identity(trial_dir: Path) -> StaleProcessIdentity:
    """Load all available identity files for a possibly-stale trial.

    Args:
        trial_dir: A trial directory possibly containing ``pid``, ``pgid``,
            and ``pid_starttime`` files written by :func:`run_supervised`.

    Returns:
        :class:`StaleProcessIdentity` with whichever fields could be read.
        Missing or malformed files surface as ``None`` on the respective
        attributes; this is the input the stale reaper uses to decide
        whether to kill the group.

    """
    pid: int | None = None
    pgid: int | None = None
    starttime: int | None = None

    pid_file = trial_dir / "pid"
    pgid_file = trial_dir / "pgid"
    starttime_file = trial_dir / "pid_starttime"

    if pid_file.is_file():
        with contextlib.suppress(ValueError, OSError):
            pid = int(pid_file.read_text().strip())
    if pgid_file.is_file():
        with contextlib.suppress(ValueError, OSError):
            pgid = int(pgid_file.read_text().strip())
    if starttime_file.is_file():
        with contextlib.suppress(ValueError, OSError):
            starttime = int(starttime_file.read_text().strip())

    return StaleProcessIdentity(pid=pid, pgid=pgid, starttime=starttime)


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
        if not _tracked_process_group_alive(pgid, member_pids):
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
        if not _tracked_process_group_alive(pgid, member_pids):
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


def _stored_pgid_is_reused_group_leader(pgid: int, saved_starttime: int) -> bool:
    """Return whether ``pgid`` appears to identify a new group leader.

    A PGID number can later be reused as an ordinary PID in another process
    group; that does not make ``killpg(pgid, ...)`` unsafe because the process
    is not a member of the target group. The unsafe case is narrower: a live
    ``/proc/<pgid>`` entry belongs to process group ``pgid`` but has a different
    starttime, meaning the saved group ID is now led by an unrelated process.
    """
    stat = _read_proc_stat(Path("/proc") / str(pgid))
    return stat is not None and stat.pgrp == pgid and stat.starttime != saved_starttime


def _tracked_process_group_alive(pgid: int, member_pids: set[int]) -> bool:
    """Check group liveness using a cached PID set, refreshing only if needed.

    :param int pgid: Process-group ID to probe.
    :param set[int] member_pids: Cached group member PIDs, refreshed in place on apparent
        death.
    :return bool: ``True`` when any non-zombie group member still appears live.
    """
    return _process_group_alive_with_members(pgid, member_pids)


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
    3. ``pid`` is unrecoverable (dead or never recorded) but ``pgid`` was
       persisted at launch: best-effort PGID kill with a loud warning. The
       root PID may have exited (``shell=True`` shells often do) while
       descendants are still holding GPU memory.
    4. No PID and no PGID: nothing to clean up, return ``True``.

    Args:
        pid: Root PID read from ``trial_dir/pid``, or ``None`` if unrecorded.
        saved_starttime: Starttime read from ``trial_dir/pid_starttime``, or
            ``None`` if unavailable / non-Linux.
        pgid: Process-group ID read from ``trial_dir/pgid`` (the fallback
            target when ``pid`` cannot be trusted).
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL; passed through to :func:`_terminate_process_group`.

    Returns:
        ``True`` when it is safe to mark the trial ``FAIL`` (cleanup
        confirmed or nothing to clean). ``False`` when cleanup is uncertain
        and callers must NOT advance state.

    """
    target_pgid: int | None = None

    if pid is not None:
        if is_same_process(pid, saved_starttime):
            try:
                target_pgid = os.getpgid(pid)
            except ProcessLookupError:
                target_pgid = None
            except (PermissionError, OSError) as exc:
                log.error("Failed reading PGID for stale PID %d: %s", pid, exc)
                return False
        elif is_pid_alive(pid) and saved_starttime is not None:
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
            if _stored_pgid_is_reused_group_leader(pgid, saved_starttime):
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
        if _stored_pgid_is_reused_group_leader(pgid, saved_starttime):
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


def terminate_group(pgid: int, *, grace_seconds: float = _KILL_GRACE_SECONDS) -> bool:
    """Public SIGTERM -> grace -> SIGKILL of a process group; confirm it is gone.

    Thin wrapper over the internal escalation used by trial cleanup, exposed so
    callers outside ``runtime`` (the MCP cancel path) do not reach into a
    private helper. Same contract: returns ``True`` only when the group is
    confirmed dead (already gone, died in the SIGTERM grace, or died within 2s
    of SIGKILL), ``False`` when cleanup is uncertain.

    Args:
        pgid: Target process-group ID.
        grace_seconds: Seconds to wait after SIGTERM before escalating.

    Returns:
        Whether the group is confirmed gone.

    """
    return _terminate_process_group(pgid, grace_seconds=grace_seconds)
