"""Block a trial command until the orchestrator commits its process identity.

STDLIB-ONLY, NO PHASESWEEP IMPORTS (review v0.5.15 / blocker 1): this script
runs in its own interpreter BEFORE PhaseSweep's durable process identity
exists and BEFORE the parent has applied the trainer's environment. Anything
importable from that environment — a poisoned or shadowed ``phasesweep``
package via a trainer-composed ``PYTHONPATH``, a ``sitecustomize``/
``usercustomize`` module, or heavy eager imports in ``phasesweep/__init__.py``
— must never get a chance to run here: it could fork and ``setsid``-escape
undetected before any identity record is written. The parent therefore
launches this file directly (never ``-m phasesweep.runtime.supervisor``)
under ``python -I -S`` with a minimal sanitized environment. ``-S`` skips
site initialization (no ``sitecustomize``/``usercustomize``); ``-I``
(isolated mode) implies ``-P`` on Python >= 3.11, so this script's own
directory — which contains the sibling module ``phasesweep/runtime/json.py``
— is never prepended to ``sys.path``, keeping ``import json`` resolved to the
stdlib module every time. Keep imports here limited to the stdlib modules
``contextlib``, ``os``, ``sys``, ``json``, ``signal``, and ``time``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time

# Single source of truth for the wire frame's header width:
# phasesweep.runtime.process._encode_launch_payload reads this constant
# directly (the parent already imports this module).
_HEADER_LEN = 10
_READY_PID_WIDTH = 10
_TRAINER_ROOT_EXITED = b"X"
_DESCENDANTS_REAPED = b"D"
_GROUP_POLL_SECONDS = 0.05
_GROUP_MAX_POLL_SECONDS = 0.5
_GROUP_TERM_GRACE_SECONDS = 10.0


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
    """Read and parse the length-prefixed launch payload from the ack pipe.

    The wire format is ``_HEADER_LEN`` ASCII decimal bytes giving the byte
    length of a UTF-8 JSON body, immediately followed by that many body
    bytes. The body must decode to ``{"cmd": <str>, "env": {<str>: <str>}}``
    with an optional ``"cwd": <str>`` naming the trainer's working directory
    (review v0.5.17 / blocker 4).

    Encoded on the other end by ``phasesweep.runtime.process._encode_launch_payload``
    (prose reference only — this module is stdlib-only and must never import
    phasesweep); keep both sides in sync if the wire format changes.

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
    except (UnicodeDecodeError, ValueError):
        return None
    if length < 0:
        return None
    body = _read_exact(ack_fd, length)
    if body is None:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
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


def _trainer_group_alive(pgid: int) -> bool:
    """Return whether ``pgid`` still has a non-zombie member.

    ``killpg(..., 0)`` is the primary existence probe. Linux keeps a process
    group visible while its last member is a zombie, though, so a complete
    procfs scan may prove that no process capable of holding GPU state remains.
    Any unreadable evidence fails closed: the guardian keeps the lease.

    :param int pgid: Trainer process-group identifier.
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
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
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
                # hidepid/ProtectProc can hide unrelated users' stat files.
                # getpgid still lets the kernel exclude those entries from
                # this trainer group; only an unreadable possible member keeps
                # the lease fail-closed.
                try:
                    unreadable_pgid = os.getpgid(int(entry.name))
                except ProcessLookupError:
                    continue
                except OSError:
                    complete = False
                    continue
                if unreadable_pgid != pgid:
                    continue
                return True
            if member_pgid != pgid:
                continue
            found_member = True
            if state != b"Z":
                return True

    # A group that exists but had no readable member, or a scan with missing
    # evidence, is not enough proof to release a host-wide GPU lock.
    return not (found_member and complete)


def _wait_for_group_exit(pgid: int, timeout: float | None) -> bool:
    """Wait until ``pgid`` is gone, bounded only when ``timeout`` is set.

    :param int pgid: Process group whose exit is awaited.
    :param float | None timeout: Maximum seconds to wait, or ``None`` for no deadline.
    :return bool: ``True`` once the group is gone, or ``False`` when the deadline expires.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    poll_seconds = _GROUP_POLL_SECONDS
    while _trainer_group_alive(pgid):
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)
        poll_seconds = min(poll_seconds * 2, _GROUP_MAX_POLL_SECONDS)
    return True


def _reap_remaining_group(pgid: int) -> bool:
    """Terminate post-root descendants while retaining inherited leases.

    :param int pgid: Trainer process group whose root was already reaped.
    :return bool: Whether at least one descendant remained after root exit.
    """
    if not _trainer_group_alive(pgid):
        return False

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        pass

    if _wait_for_group_exit(pgid, _GROUP_TERM_GRACE_SECONDS):
        return True

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        pass

    # There is nobody left upstream to protect this lease after an
    # orchestrator hard exit. If SIGKILL cannot be confirmed, deliberately
    # wait without a deadline rather than reopening the GPU to another run.
    _wait_for_group_exit(pgid, None)
    return True


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

    ready_frame = b"R" + f"{os.getpid():0{_READY_PID_WIDTH}d}".encode("ascii")
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

    :param list[str] | None argv: Optional ``[ready_fd, ack_fd]`` argument
        vector; defaults to ``sys.argv[1:]`` when omitted.
    :return int: ``64`` if ``argv`` does not have exactly two elements; ``75``
        if the acknowledgement pipe does not deliver a well-formed
        ``{"cmd": str, "env": {str: str}}`` payload before EOF or a shape
        violation; ``76`` if the payload's optional working directory cannot
        be entered. On success, the supervisor remains outside the trainer's
        process group, holds inherited GPU leases until every group member is
        gone, and returns the trainer root's exit status or terminating signal.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        return 64
    ready_fd = int(args[0])
    ack_fd = int(args[1])

    # The parent spawns this process from inside its shutdown-signal deferral
    # window, and the blocked signal mask survives fork AND exec — without
    # this reset, the supervisor and every trainer exec'd from it would run
    # with SIGTERM/SIGINT/SIGHUP permanently blocked, making the documented
    # SIGTERM -> grace -> SIGKILL escalation a dead letter (trainers could
    # never shut down gracefully and every kill burned the full grace before
    # SIGKILL). This is the first code phasesweep controls after exec, so the
    # mask is cleaned here.
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_SETMASK, set())

    child_pid = os.fork()
    if child_pid == 0:
        os._exit(_run_trainer(ready_fd, ack_fd))

    # The guardian is deliberately outside the trainer's new session and
    # process group. Group-directed SIGKILL can therefore never release its
    # inherited GPU leases before every trainer descendant is confirmed gone.
    os.close(ack_fd)
    try:
        _, status = os.waitpid(child_pid, 0)
        with contextlib.suppress(OSError):
            os.write(ready_fd, _TRAINER_ROOT_EXITED)
        if _reap_remaining_group(child_pid):
            with contextlib.suppress(OSError):
                os.write(ready_fd, _DESCENDANTS_REAPED)
            # A closed pipe is expected after an orchestrator hard exit: there
            # is no parent left to consume the diagnostic, but cleanup continues.
        return _return_child_status(status)
    finally:
        os.close(ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
