"""Block a trial command until the orchestrator commits its process identity.

Run directly under ``python -I -S`` with a minimal pre-ACK environment and stdlib
imports only. The blocked child becomes the trainer group leader; the guardian
stays outside that group and retains GPU leases until cleanup.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from types import FrameType

# Wire frame constants shared with phasesweep.runtime.process.
_HEADER_LEN = 10
_READY_PID_WIDTH = 10
_READY_TAG = b"R"
_TRAINER_ROOT_EXITED = b"X"
_DESCENDANTS_REAPED = b"D"
_GROUP_POLL_SECONDS = 0.05
_GROUP_MAX_POLL_SECONDS = 0.5
_GROUP_TERM_GRACE_SECONDS = 10.0
# Helpers that watch for their parent's death exit within milliseconds (CPython's
# resource tracker and forkserver exit on pipe end-of-file) or within one 5 s poll
# (an orphaned PyTorch DataLoader worker checks getppid() every
# MP_STATUS_CHECK_INTERVAL). Ten seconds is twice the slowest of them.
_HELPER_SETTLE_SECONDS = 10.0
# The survivor list follows the D byte in one write, kept under the POSIX minimum
# PIPE_BUF of 512 bytes: at most four ASCII entries of 100 characters each.
_SURVIVOR_ENTRY_CHARS = 100
_SURVIVOR_ENTRIES = 4
ATTEMPT_LIFECYCLE_SCHEMA_VERSION = 3


def write_lifecycle_record(
    path: Path,
    *,
    attempt_id: str,
    state: str,
    return_code: int | None,
    cleanup_confirmed: bool | None,
    duration_seconds: float | None,
) -> None:
    """Atomically persist lifecycle evidence from either supervisor or orchestrator.

    :param Path path: Existing attempt directory's lifecycle file.
    :param str attempt_id: Immutable attempt identity.
    :param str state: Allocated, launching, terminating, or exited lifecycle state.
    :param int | None return_code: Supervised process exit code, when known.
    :param bool | None cleanup_confirmed: Whether the whole group has exited.
    :param float | None duration_seconds: Supervised elapsed time, when known.
    :raises OSError: The record could not be durably replaced.
    """
    payload = {
        "schema_version": ATTEMPT_LIFECYCLE_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "state": state,
        "return_code": return_code,
        "cleanup_confirmed": cleanup_confirmed,
        "duration_seconds": duration_seconds,
    }
    fd, stage = tempfile.mkstemp(prefix=".attempt-lifecycle-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(stage)


def _read_exact(fd: int, size: int) -> bytes | None:
    """Read exactly ``size`` bytes from ``fd``, looping past partial reads.

    :param int fd: File descriptor to read from.
    :param int size: Exact number of bytes required.
    :return bytes | None: The bytes read, or ``None`` if ``fd`` hit EOF before
        ``size`` bytes were available.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_launch_payload(ack_fd: int) -> tuple[str, dict[str, str], str | None] | None:
    """Read the length-prefixed launch payload from the acknowledgement pipe.

    ``_HEADER_LEN`` ASCII decimal bytes specify the UTF-8 JSON body's byte length.
    The body contains ``{"cmd": <str>, "env": {<str>: <str>}}`` and optional
    ``"cwd": <str>``.

    :param int ack_fd: Read end of the acknowledgement pipe.
    :return tuple[str, dict[str, str], str | None] | None: ``(cmd, env, cwd)``
        on a well-formed payload; ``None`` on EOF, a malformed header, a
        truncated body, or a body of the wrong shape.
    """
    header = _read_exact(ack_fd, _HEADER_LEN)
    if header is None:
        return None
    try:
        length = int(header.decode("ascii"))
    except ValueError:
        return None
    if length < 0:
        return None
    body = _read_exact(ack_fd, length)
    if body is None:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    cmd = payload.get("cmd")
    env = payload.get("env")
    cwd = payload.get("cwd")
    if not isinstance(cmd, str) or not isinstance(env, dict):
        return None
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        return None
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
    return cmd, env, cwd


def _has_exited(pid: int, state: bytes) -> bool:
    """Return whether a process whose ``stat`` reads ``state`` can no longer run.

    This is a stdlib-only copy of ``phasesweep.runtime.reaper._has_exited``,
    because the guardian runs under ``python -I -S`` without the package; change
    both together. ``X`` always means exited. ``Z`` means exited only once
    ``/proc/<pid>/task`` lists no thread but the leader's own: a leader that
    ends with ``pthread_exit`` reads ``Z`` while its other threads keep running.
    An exiting thread leaves that listing without ever reading ``Z``, so the
    other entries' own states are not read. A vanished entry has exited; an
    unreadable listing keeps the lease.

    :param int pid: Process whose ``stat`` was read.
    :param bytes state: Its state field from ``/proc/<pid>/stat``.
    :return bool: Whether every thread of the process has exited.
    """
    if state == b"X":
        return True
    if state != b"Z":
        return False
    try:
        with os.scandir(f"/proc/{pid}/task") as tasks:
            return all(task.name == str(pid) for task in tasks)
    except (FileNotFoundError, ProcessLookupError):
        return True
    except OSError:
        return False


def _trainer_group_alive(pgid: int, members: list[int] | None = None) -> bool:
    """Return whether ``pgid`` still has a member that can run.

    ``killpg(..., 0)`` is the primary existence probe. On Linux a procfs scan
    checks for group members that have not exited, by :func:`_has_exited`.
    Unreadable possible members keep the lease.

    :param int pgid: Trainer process-group identifier.
    :param list[int] | None members: When given, the scan continues past the first
        live member and appends every live or unreadable member's PID here; the
        verdict is the same either way.
    :return bool: ``True`` while a live member exists or absence is uncertain.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True

    try:
        entries = os.scandir("/proc")
    except OSError:
        return True

    complete = True
    found_member = False
    live_member = False
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as stat_file:
                    data = stat_file.read()
                right_paren = data.rfind(b")")
                fields = data[right_paren + 2 :].split()
                state = fields[0]
                member_pgid = int(fields[2])
            except FileNotFoundError:
                continue
            except (IndexError, OSError, ValueError):
                # Only an unreadable possible group member must retain the lease.
                try:
                    unreadable_pgid = os.getpgid(pid)
                except ProcessLookupError:
                    continue
                except OSError:
                    complete = False
                    continue
                if unreadable_pgid != pgid:
                    continue
            else:
                if member_pgid != pgid:
                    continue
                found_member = True
                if _has_exited(pid, state):
                    continue
            live_member = True
            if members is None:
                return True
            members.append(pid)

    # A group that exists but had no readable member, or a scan with missing
    # evidence, is not enough proof to release a host-wide GPU lock.
    return live_member or not (found_member and complete)


def _describe_members(pids: list[int]) -> str:
    """Name surviving group members for the trial's failure reason.

    Each entry is the PID and its ``/proc/<pid>/cmdline`` with NUL separators read as
    spaces, or its ``comm`` when the command line is empty or unreadable, cut to
    ``_SURVIVOR_ENTRY_CHARS``. Control bytes read as spaces and non-ASCII bytes as
    ``?``, so each character is one byte of the status frame. Without procfs the scan
    that produced ``pids`` finds none, so the result is empty.

    :param list[int] pids: Live or unreadable members found by :func:`_trainer_group_alive`.
    :return str: Up to ``_SURVIVOR_ENTRIES`` entries joined by ``", "``, then ``+N more``
        for the rest, or ``""`` when ``pids`` is empty.
    """
    entries: list[str] = []
    for pid in pids[:_SURVIVOR_ENTRIES]:
        command = ""
        for name in ("cmdline", "comm"):
            try:
                with open(f"/proc/{pid}/{name}", "rb") as source:
                    raw = source.read(_SURVIVOR_ENTRY_CHARS * 4)
            except OSError:
                continue
            text = "".join(
                " " if byte < 0x20 else chr(byte) if byte < 0x7F else "?" for byte in raw
            )
            command = " ".join(text.split())
            if command:
                break
        entry = str(pid)
        if command:
            room = _SURVIVOR_ENTRY_CHARS - len(entry) - len(" ()")
            if len(command) > room:
                command = command[: room - 3] + "..."
            entry = f"{entry} ({command})"
        entries.append(entry)
    if len(pids) > _SURVIVOR_ENTRIES:
        entries.append(f"+{len(pids) - _SURVIVOR_ENTRIES} more")
    return ", ".join(entries)


def _wait_for_group_exit(pgid: int, timeout: float | None) -> bool:
    """Reap the trainer root while waiting for its whole group to disappear.

    :param int pgid: Group led by the guardian's child, which may already be reaped.
    :param float | None timeout: Maximum seconds to wait, or ``None`` for no deadline.
    :return bool: ``True`` once the group is gone, or ``False`` when the deadline expires.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    poll_seconds = _GROUP_POLL_SECONDS
    while True:
        # Without procfs, killpg still sees our exited child until it is reaped.
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pgid, os.WNOHANG)
        if not _trainer_group_alive(pgid):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)
        poll_seconds = min(poll_seconds * 2, _GROUP_MAX_POLL_SECONDS)


def _reap_remaining_group(pgid: int) -> str | None:
    """Terminate a trainer group while retaining inherited leases.

    :param int pgid: Trainer process group being cancelled or surviving its root.
    :return str | None: ``None`` when no member was left; otherwise the members as
        :func:`_describe_members` named them before SIGTERM, which is ``""`` when none
        could be named.
    """
    members: list[int] = []
    if not _trainer_group_alive(pgid, members):
        return None
    survivors = _describe_members(members)

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return survivors
    except OSError:
        pass

    if _wait_for_group_exit(pgid, _GROUP_TERM_GRACE_SECONDS):
        return survivors

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return survivors
    except OSError:
        pass

    # There is nobody left upstream to protect this lease after an
    # orchestrator hard exit. If SIGKILL cannot be confirmed,
    # wait without a deadline rather than reopening the GPU to another run.
    _wait_for_group_exit(pgid, None)
    return survivors


def _close_trainer_fds() -> None:
    """Close every non-stdio descriptor before executing trainer code."""
    try:
        max_fd = int(os.sysconf("SC_OPEN_MAX"))
    except (OSError, TypeError, ValueError):
        max_fd = 1_048_576
    os.closerange(3, max_fd)


def _run_trainer(ready_fd: int, ack_fd: int) -> int:
    """Create the trainer session, cross the launch barrier, and exec it.

    :param int ready_fd: Descriptor used to publish the trainer process identity.
    :param int ack_fd: Descriptor carrying the committed launch payload.
    :return int: Supervisor error status when setup or ``execve`` fails; a
        successful call replaces the process and does not return.
    """
    try:
        os.setsid()
    except OSError:
        return 74

    ready_frame = _READY_TAG + f"{os.getpid():0{_READY_PID_WIDTH}d}".encode("ascii")
    try:
        os.write(ready_fd, ready_frame)
    finally:
        os.close(ready_fd)

    try:
        payload = _read_launch_payload(ack_fd)
    finally:
        os.close(ack_fd)
    if payload is None:
        return 75
    cmd, env, cwd = payload

    if cwd is not None:
        try:
            os.chdir(cwd)
        except OSError:
            return 76

    _close_trainer_fds()
    try:
        os.execve("/bin/sh", ["/bin/sh", "-c", cmd], env)
    except OSError:
        return 77


def _return_child_status(status: int) -> int:
    """Return an exit status or reproduce the trainer root's signal.

    :param int status: Raw wait status returned for the trainer root.
    :return int: Root exit status, conventional signal status if signal delivery
        returns, or ``70`` for an unrecognized wait state.
    """
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        terminating_signal = os.WTERMSIG(status)
        # SIGKILL and SIGSTOP already have immutable default dispositions;
        # asking signal.signal() to change either one raises EINVAL and would
        # corrupt the trainer's ``-signal`` return code into guardian exit 1.
        if terminating_signal not in {signal.SIGKILL, signal.SIGSTOP}:
            signal.signal(terminating_signal, signal.SIG_DFL)
        os.kill(os.getpid(), terminating_signal)
        return 128 + terminating_signal
    return 70


def main(argv: list[str] | None = None) -> int:
    """Fork a blocked trainer and guard its whole process group.

    :param list[str] | None argv: Optional ``[ready_fd, ack_fd, trial_dir, attempt_id, started]`` argument
        vector; defaults to ``sys.argv[1:]`` when omitted.
    :return int: ``64`` if ``argv`` has neither two nor five elements; ``75``
        if the acknowledgement pipe does not deliver a well-formed
        ``{"cmd": str, "env": {str: str}}`` payload before EOF or a shape
        violation; ``76`` if the payload's optional working directory cannot
        be entered. On success, the supervisor remains outside the trainer's
        process group, gives members that outlive the root ``_HELPER_SETTLE_SECONDS``
        to exit before terminating them, holds inherited GPU leases until every group
        member is gone, and returns the trainer root's exit status or terminating signal.
        Guardian cancellation preserves SIGTERM as the result even if the trainer handles it by exiting zero.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in {2, 5}:
        return 64
    ready_fd = int(args[0])
    ack_fd = int(args[1])

    # A cancellation request must wait until the guardian can preserve its intent.
    signal.pthread_sigmask(signal.SIG_SETMASK, {signal.SIGTERM})

    child_pid = os.fork()
    if child_pid == 0:
        signal.pthread_sigmask(signal.SIG_SETMASK, set())
        os._exit(_run_trainer(ready_fd, ack_fd))

    # cancel() reaps a root that had already exited and leaves its wait status here.
    reaped_root_status: int | None = None

    def cancel(signum: int, _frame: FrameType | None) -> None:
        """Interrupt supervision only while the trainer root still runs.

        A request that lands after the root exited but before ``main`` stops listening
        does not apply: the root is reaped here if ``main`` has not yet, and supervision
        goes on to the settle and verdict, so a finished trainer is not recorded as cancelled.
        Reaping with ``WNOHANG`` works on every supported platform; ``os.waitid``, which
        could test the zombie without reaping it, is missing on macOS before Python 3.13.

        :param int signum: Cancellation request delivered to the guardian.
        :param FrameType | None _frame: Unused signal frame.
        :raises SystemExit: A running root's group needs durably recorded cancellation.
        """
        nonlocal reaped_root_status
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            reaped_pid, wait_status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            return  # main() has just reaped the exited root.
        if reaped_pid == child_pid:
            reaped_root_status = wait_status
            return
        if _trainer_group_alive(child_pid):
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)

    # The guardian is outside the trainer's new session and
    # process group. Group-directed SIGKILL can therefore never release its
    # inherited GPU leases before every trainer descendant is confirmed gone.
    os.close(ack_fd)
    cancelled = False
    try:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, set())
            try:
                _, status = os.waitpid(child_pid, 0)
            except ChildProcessError:
                # cancel() reaped the exited root while this wait was interrupted.
                if reaped_root_status is None:
                    raise
                status = reaped_root_status
            # Once the root has exited, cancellation no longer applies: the settle
            # and the verdict below run to completion, keeping a finished result.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except SystemExit:
            cancelled = True
            try:
                if len(args) == 5:
                    write_lifecycle_record(
                        Path(args[2]) / "attempt_lifecycle.json",
                        attempt_id=args[3],
                        state="terminating",
                        return_code=-signal.SIGTERM,
                        cleanup_confirmed=False,
                        duration_seconds=None,
                    )
            except OSError:
                # Without durable intent, retain the lease and let the trainer finish.
                _, status = os.waitpid(child_pid, 0)
                _wait_for_group_exit(child_pid, None)
            else:
                _reap_remaining_group(child_pid)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(child_pid, 0)
                status = signal.SIGTERM
        with contextlib.suppress(OSError):
            os.write(ready_fd, _TRAINER_ROOT_EXITED)
        if not cancelled:
            # Helpers that watch for their parent's death exit by themselves; only a
            # member still running after the settle is the trainer's leak. The settle
            # follows a nonzero exit too, so a crash keeps its own failure reason.
            _wait_for_group_exit(child_pid, _HELPER_SETTLE_SECONDS)
        survivors = _reap_remaining_group(child_pid)
        if survivors is not None:
            with contextlib.suppress(OSError):
                os.write(ready_fd, _DESCENDANTS_REAPED + survivors.encode("ascii", "replace"))
            # A closed pipe is expected after an orchestrator hard exit: there
            # is no parent left to consume the diagnostic, but cleanup continues.
        if len(args) == 5:
            return_code = os.waitstatus_to_exitcode(status)
            # Descendant cleanup is a failed trial even when its root returned zero.
            if survivors is not None and return_code == 0:
                return_code = 1
            with contextlib.suppress(OSError):
                write_lifecycle_record(
                    Path(args[2]) / "attempt_lifecycle.json",
                    attempt_id=args[3],
                    state="exited",
                    return_code=return_code,
                    cleanup_confirmed=True,
                    duration_seconds=time.monotonic() - float(args[4]),
                )
        return _return_child_status(status)
    finally:
        os.close(ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
