"""Block a trial command until the orchestrator commits its process identity.

Run directly under ``python -I -S`` with a minimal pre-ACK environment and stdlib
imports only. The blocked child becomes the trainer group leader; the guardian
stays outside that group, adopts the trainer's orphaned descendants where Linux
allows, and retains GPU leases until cleanup.
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
# Read far enough into a command line to reach the script behind long path arguments.
_SURVIVOR_CMDLINE_BYTES = 4096
# prctl(2) option from <linux/prctl.h>.
_PR_SET_CHILD_SUBREAPER = 36
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
    :param bool | None cleanup_confirmed: Whether the whole group and every adopted descendant have exited.
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


def _unreadable_member(pid: int, pgid: int) -> bool | None:
    """Decide whether a process whose ``stat`` could not be read is in the trainer group.

    ``getpgid`` answers without procfs. It is asked only while ``killpg`` finds the group,
    which cannot gain members once it is gone.

    :param int pid: Process whose ``stat`` was unreadable.
    :param int pgid: Trainer process group.
    :return bool | None: ``True`` for a member, ``False`` for none, ``None`` when unknown.
    """
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False
    except OSError:
        return None


def _trainer_group_alive(pgid: int, members: list[int] | None = None) -> bool:
    """Return whether the trainer root's process group still has a member that can run.

    ``killpg(..., 0)`` is the primary existence probe. On Linux a procfs scan checks for
    group members that have not exited, by :func:`_has_exited`. Unreadable possible members
    keep the lease, as :func:`_unreadable_member` decides. Processes that left the group are
    :func:`_trainer_alive`'s concern.

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
                verdict = _unreadable_member(pid, pgid)
                if verdict is None:
                    complete = False
                if not verdict:
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


def _children_remain() -> bool:
    """Reap every exited child of this process and return whether any child is left.

    The kernel answers exactly, without procfs or permission checks: ``waitpid`` with
    ``WNOHANG`` returns ``0`` while some child has not exited and raises
    ``ChildProcessError`` once there is none.

    :return bool: Whether at least one child has not exited.
    """
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False
        if pid == 0:
            return True


def _adopted_children(root: int) -> list[int]:
    """List this process's children other than the trainer root, from procfs.

    The list only names and signals adopted processes; :func:`_children_remain` decides
    whether any is left. A child the scan misses, such as one adopted during it, is found by
    the next poll.

    :param int root: The trainer root's PID, which the group scan covers.
    :return list[int]: PIDs whose ``stat`` names this process as their parent.
    """
    guardian = os.getpid()
    children: list[int] = []
    try:
        entries = os.scandir("/proc")
    except OSError:
        return children
    with entries:
        for entry in entries:
            if not entry.name.isdigit() or entry.name == str(root):
                continue
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as stat_file:
                    data = stat_file.read()
                parent = int(data[data.rfind(b")") + 2 :].split()[1])
            except (IndexError, OSError, ValueError):
                continue
            if parent == guardian:
                children.append(int(entry.name))
    return children


def _trainer_alive(pgid: int, members: list[int] | None = None, *, adopts: bool) -> bool:
    """Return whether anything the trainer started can still run.

    Without adoption that is the root's group, by :func:`_trainer_group_alive`. A child
    subreaper also owns every process that left the group: each is its child or a
    descendant of one, so after the group check :func:`_children_remain` reaps the exited
    children and says whether one is left. That order is race-free. A group with no live
    member cannot gain one, and the kernel re-parents a dying process's children to the
    guardian before that process reads as exited, so nothing leaves the group unseen
    between the two checks. The children check also runs while the group lives, so an
    exited child is reaped before it could be named or signalled as a survivor.

    :param int pgid: Trainer process group, which is also the root's PID.
    :param list[int] | None members: When given, every live or unreadable group member and,
        while adopted children remain, each of them is appended here for naming and
        signalling; the verdict is the same either way.
    :param bool adopts: Whether this process adopts the trainer's orphaned descendants.
    :return bool: ``True`` while something the trainer started may still run.
    """
    group_alive = _trainer_group_alive(pgid, members)
    if not adopts:
        return group_alive
    children_alive = _children_remain()
    if children_alive and members is not None:
        members.extend(pid for pid in _adopted_children(pgid) if pid not in members)
    return group_alive or children_alive


def _short_argument(argument: str) -> str:
    """Cut one command-line argument to the last component of the path it names.

    The survivors are terminated before anyone reads their names, so the command line is
    all that identifies them, and an interpreter or checkout prefix would otherwise fill
    the entry before the script's name. An option keeps its name: ``--config=/a/b.yaml``
    reads ``--config=b.yaml``. An argument with whitespace or a URL scheme, such as inline
    code after ``-c`` or a tracking URI, is no path and is kept whole.

    :param str argument: One argument of a process's command line.
    :return str: The argument with any directory prefix removed.
    """
    option, separator, value = (
        argument.partition("=") if argument.startswith("-") else ("", "", argument)
    )
    if "://" in value or len(value.split()) != 1:
        return argument
    return option + separator + (value.rstrip("/").rsplit("/", 1)[-1] or value)


def _describe_members(pids: list[int]) -> str:
    """Name surviving group members for the trial's failure reason.

    Each entry is the PID and the arguments of its ``/proc/<pid>/cmdline``, each cut by
    :func:`_short_argument`, or its ``comm`` when the command line is empty or
    unreadable, cut to ``_SURVIVOR_ENTRY_CHARS``. Control bytes read as spaces and
    non-ASCII bytes as ``?``, so each character is one byte of the status frame. Without
    procfs the scan that produced ``pids`` finds none, so the result is empty.

    :param list[int] pids: Live or unreadable members found by :func:`_trainer_alive`.
    :return str: Up to ``_SURVIVOR_ENTRIES`` entries joined by ``", "``, then ``+N more``
        for the rest, or ``""`` when ``pids`` is empty.
    """
    entries: list[str] = []
    for pid in pids[:_SURVIVOR_ENTRIES]:
        command = ""
        for name in ("cmdline", "comm"):
            try:
                with open(f"/proc/{pid}/{name}", "rb") as source:
                    raw = source.read(_SURVIVOR_CMDLINE_BYTES)
            except OSError:
                continue
            arguments = (
                "".join(" " if byte < 0x20 else chr(byte) if byte < 0x7F else "?" for byte in part)
                for part in raw.split(b"\0")
            )
            command = " ".join(" ".join(map(_short_argument, arguments)).split())
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


def _signal_members(pgid: int, members: list[int], signum: int, signalled: set[int]) -> None:
    """Send ``signum`` once to the trainer group and to each member's own group.

    A member outside the trainer group, such as a rank its launcher started in a new
    session, belongs to a group of its own, and signalling that group reaches the rank's
    children too. The guardian's own group is never signalled: no trainer process can
    join it, because the root's session is a different one.

    :param int pgid: Trainer process group.
    :param list[int] members: Live members found by :func:`_trainer_alive`.
    :param int signum: Signal to send.
    :param set[int] signalled: Groups already sent ``signum``, extended here.
    """
    groups = {pgid}
    for pid in members:
        with contextlib.suppress(OSError):
            groups.add(os.getpgid(pid))
    for group in groups - signalled - {os.getpgrp()}:
        signalled.add(group)
        with contextlib.suppress(OSError):
            os.killpg(group, signum)


def _wait_for_group_exit(
    pgid: int, timeout: float | None, *, adopts: bool, signum: int | None = None
) -> bool:
    """Reap the trainer root while waiting for everything the trainer started to exit.

    :param int pgid: Group led by the guardian's child, which may already be reaped.
    :param float | None timeout: Maximum seconds to wait, or ``None`` for no deadline.
    :param bool adopts: Whether this process adopts the trainer's orphaned descendants.
    :param int | None signum: Signal sent once to the trainer group and to the group of
        every member outside it, including members adopted while waiting; ``None`` only waits.
    :return bool: ``True`` once nothing the trainer started is left, or ``False`` when the
        deadline expires.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    poll_seconds = _GROUP_POLL_SECONDS
    signalled: set[int] = set()
    while True:
        # Without procfs, killpg still sees our exited child until it is reaped.
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pgid, os.WNOHANG)
        members: list[int] | None = None if signum is None else []
        if not _trainer_alive(pgid, members, adopts=adopts):
            return True
        if signum is not None and members is not None:
            _signal_members(pgid, members, signum, signalled)
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)
        poll_seconds = min(poll_seconds * 2, _GROUP_MAX_POLL_SECONDS)


def _reap_remaining_group(pgid: int, *, adopts: bool) -> str | None:
    """Terminate everything the trainer started while retaining inherited leases.

    SIGTERM and SIGKILL share one grace, so a member adopted only once its launcher dies
    is killed at first sight rather than given a grace of its own.

    :param int pgid: Trainer process group being cancelled or surviving its root.
    :param bool adopts: Whether this process adopts the trainer's orphaned descendants.
    :return str | None: ``None`` when no member was left; otherwise the members as
        :func:`_describe_members` named them before SIGTERM, which is ``""`` when none
        could be named.
    """
    members: list[int] = []
    if not _trainer_alive(pgid, members, adopts=adopts):
        return None
    survivors = _describe_members(members)

    if _wait_for_group_exit(pgid, _GROUP_TERM_GRACE_SECONDS, adopts=adopts, signum=signal.SIGTERM):
        return survivors

    # There is nobody left upstream to protect this lease after an
    # orchestrator hard exit. If SIGKILL cannot be confirmed,
    # wait without a deadline rather than reopening the GPU to another run.
    _wait_for_group_exit(pgid, None, adopts=adopts, signum=signal.SIGKILL)
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


def _wait_for_root(root: int) -> int:
    """Block until the trainer root exits, reaping every other child that exits first.

    A child subreaper gains children while the root runs, and each one that exits would
    otherwise wait as a zombie until the root does.

    :param int root: The trainer root, an unreaped child of this process.
    :return int: The root's raw wait status.
    """
    while True:
        pid, status = os.waitpid(-1, 0)
        if pid == root:
            return status


def _become_subreaper() -> bool:
    """Make the trainer's orphaned descendants re-parent to this process.

    A launcher such as torchrun starts each rank in a session of its own, outside the
    trainer's group; if the launcher dies first, its ranks would re-parent to init and
    outlive the lease. A child subreaper adopts them instead: ``waitpid`` says whether any
    is left, and the procfs scan finds their PIDs to signal. The standard library has no
    ``prctl`` binding, so the call goes through ``ctypes``. An adopted process that cannot
    be found cannot be signalled, and would hold the lease until it exits by itself, so
    adoption is skipped when ``/proc`` cannot be listed.

    :return bool: Whether the kernel accepted ``PR_SET_CHILD_SUBREAPER``.
    """
    if sys.platform != "linux":
        return False
    try:
        with os.scandir("/proc"):
            pass
        import ctypes

        result = ctypes.CDLL(None).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (AttributeError, ImportError, OSError):
        return False
    return isinstance(result, int) and result == 0


def main(argv: list[str] | None = None) -> int:
    """Fork a blocked trainer and guard its whole process group.

    :param list[str] | None argv: Optional ``[ready_fd, ack_fd, trial_dir, attempt_id, started]`` argument
        vector; defaults to ``sys.argv[1:]`` when omitted.
    :return int: ``64`` if ``argv`` has neither two nor five elements; ``75``
        if the acknowledgement pipe does not deliver a well-formed
        ``{"cmd": str, "env": {str: str}}`` payload before EOF or a shape
        violation; ``76`` if the payload's optional working directory cannot
        be entered. On success, the supervisor remains outside the trainer's
        process group, adopts the trainer's orphaned descendants where the kernel allows,
        gives members that outlive the root ``_HELPER_SETTLE_SECONDS`` to exit before
        terminating them, holds inherited GPU leases until every group member and adopted
        descendant is gone, and returns the trainer root's exit status or terminating signal.
        Guardian cancellation preserves SIGTERM as the result even if the trainer handles it by exiting zero.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in {2, 5}:
        return 64
    ready_fd = int(args[0])
    ack_fd = int(args[1])
    adopts = _become_subreaper()

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
        The handler then ends ``main``'s wait with ``ChildProcessError``, because a wait
        resumed after a normal return would block on adopted children that still run.
        Reaping with ``WNOHANG`` works on every supported platform; ``os.waitid``, which
        could test the zombie without reaping it, is missing on macOS before Python 3.13.

        :param int signum: Cancellation request delivered to the guardian.
        :param FrameType | None _frame: Unused signal frame.
        :raises ChildProcessError: The root had exited and is reaped here, its status kept.
        :raises SystemExit: A running root needs durably recorded cancellation.
        """
        nonlocal reaped_root_status
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            reaped_pid, wait_status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            return  # main() has just reaped the exited root.
        if reaped_pid == child_pid:
            reaped_root_status = wait_status
            raise ChildProcessError(f"trainer root {child_pid} was reaped by cancel()")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)

    # The guardian is outside the trainer's new session and
    # process group. Group-directed SIGKILL can therefore never release its
    # inherited GPU leases before every trainer descendant is confirmed gone.
    os.close(ack_fd)
    cancelled = False
    try:
        try:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, set())
                status = _wait_for_root(child_pid)
            except ChildProcessError:
                # cancel() reaped the exited root and ended this wait.
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
                status = _wait_for_root(child_pid)
                _wait_for_group_exit(child_pid, None, adopts=adopts)
            else:
                _reap_remaining_group(child_pid, adopts=adopts)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(child_pid, 0)
                status = signal.SIGTERM
        with contextlib.suppress(OSError):
            os.write(ready_fd, _TRAINER_ROOT_EXITED)
        if not cancelled:
            # Helpers that watch for their parent's death exit by themselves; only a
            # member still running after the settle is the trainer's leak. The settle
            # follows a nonzero exit too, so a crash keeps its own failure reason.
            _wait_for_group_exit(child_pid, _HELPER_SETTLE_SECONDS, adopts=adopts)
        survivors = _reap_remaining_group(child_pid, adopts=adopts)
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
