"""Process-group probes, termination, and stale trial cleanup.

Reads ``/proc`` to decide whether a recorded PID or process group is still the
one phasesweep launched, terminates groups SIGTERM -> grace -> SIGKILL, and
cleans a stale trial's group only when the boot- and process-bound identity in
``process_identity.json`` proves that signalling it cannot hit a reused PID.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from phasesweep.errors import ForeignPidNamespaceError
from phasesweep.runtime.json import strict_json_loads

# Process supervision logs on one channel, whichever module does the work.
log = logging.getLogger("phasesweep.runtime.process")

_KILL_GRACE_SECONDS = 10.0
PROCESS_IDENTITY_FILE = "process_identity.json"
PROCESS_IDENTITY_SCHEMA_VERSION = 2

#: The kernel constant ``PROC_PID_INIT_INO``: the ``ns/pid`` link text of the
#: initial PID namespace, fixed since Linux 3.8. That namespace is never freed.
INITIAL_PID_NAMESPACE_LINK = "pid:[4026531836]"
#: Namespace identity: ``ns/pid`` inode text plus ``@`` and its PID 1 start
#: time. The pair distinguishes namespace-inode reuse within one boot.
_PID_NAMESPACE_PATTERN = re.compile(r"pid:\[[1-9][0-9]*\]@(?:0|[1-9][0-9]*)")
_PROC = Path("/proc")


@dataclass(frozen=True)
class StaleProcessIdentity:
    """Durable identity of one launched trial process group."""

    schema_version: int
    attempt_id: str
    pid: int
    pgid: int
    proc_starttime: int | None
    boot_id: str | None
    pid_namespace: str


def read_boot_id() -> str | None:
    """Return the current Linux boot identity, or ``None`` when unavailable."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def identity_from_earlier_boot(boot_id: str | None, current_boot_id: str | None) -> bool:
    """Return whether known, different boot IDs prove the recorded process gone.

    Unknown IDs return ``False``. Cleanup callers must separately refuse signaling
    when either boot identity is unknown.

    :param str | None boot_id: Boot identity recorded when the process launched.
    :param str | None current_boot_id: This boot's identity, from :func:`read_boot_id`.
    :return bool: Whether both boot ids are known and differ.
    """
    return boot_id is not None and current_boot_id is not None and boot_id != current_boot_id


def is_pid_namespace(value: object) -> TypeGuard[str]:
    """Return whether ``value`` is a PID namespace as phasesweep records it.

    :param object value: Candidate namespace identity.
    :return bool: Whether it has the form ``pid:[<inode>]@<PID 1 start time>``.
    """
    return isinstance(value, str) and _PID_NAMESPACE_PATTERN.fullmatch(value) is not None


def _pid_namespace_link(pid_namespace: str) -> str:
    """Return the ``ns/pid`` link text that names a recorded namespace's inode.

    :param str pid_namespace: Namespace as :func:`read_pid_namespace` records it.
    :return str: Its ``pid:[<inode>]`` part.
    """
    return pid_namespace.partition("@")[0]


def read_pid_namespace() -> str | None:
    """Return the PID namespace this process's PID numbers belong to.

    PID plus start time identifies a process only inside one PID namespace: the
    same number can name an unrelated process elsewhere. ``/proc`` is
    trustworthy for that only when it is this namespace's own procfs, which
    ``/proc/self`` proves by naming this process's own PID; its ``/proc/1`` is
    then this namespace's PID 1, whose start time keeps a reused inode from
    naming a namespace that has ended.

    :return str | None: ``pid:[<inode>]@<PID 1 start time>``, or ``None`` when
        ``/proc`` is missing, unreadable, mounted from another PID namespace,
        or hides this namespace's PID 1.
    """
    try:
        if os.readlink(_PROC / "self") != str(os.getpid()):
            return None
        link = os.readlink(_PROC / "self" / "ns" / "pid")
    except OSError:
        return None
    init = _read_proc_stat(_PROC / "1")
    if init is None:
        return None
    value = f"{link}@{init.starttime}"
    return value if is_pid_namespace(value) else None


def in_initial_pid_namespace() -> bool:
    """Return whether this process runs in the host's initial PID namespace.

    :return bool: Whether this process's namespace is the host's initial one;
        ``False`` when the namespace is unreadable.
    """
    current = read_pid_namespace()
    return current is not None and _pid_namespace_link(current) == INITIAL_PID_NAMESPACE_LINK


def same_pid_namespace(pid_namespace: str) -> bool:
    """Return whether PIDs recorded in ``pid_namespace`` can be probed from this process.

    Both the inode and the start time of the namespace's PID 1 must match, so
    a namespace that reuses an ended one's inode does not pass for it. An
    unreadable namespace on this side never matches.

    :param str pid_namespace: Namespace recorded with the process identity.
    :return bool: Whether this process is in that namespace.
    """
    current = read_pid_namespace()
    return current is not None and current == pid_namespace


def recorded_process_blocker(
    pid_namespace: str, *, pids: Collection[int], pgids: Collection[int]
) -> str | None:
    """Explain why recorded processes in another PID namespace cannot be proven gone.

    Only the host's initial PID namespace can make this proof. Nested PID, group and
    session IDs identify candidates; a matching or unreadable namespace blocks it.
    Outer-level group matches also block it. Start times are not compared across time
    namespaces. Processes that escaped their recorded group remain a blind spot.

    :param str pid_namespace: Namespace the processes were recorded in.
    :param Collection[int] pids: Recorded PIDs, numbered in that namespace.
    :param Collection[int] pgids: Recorded process groups, numbered in that namespace.
    :return str | None: ``None`` when no candidate remains; otherwise a clause
        naming what blocked the proof, including every blocking host PID.
    """
    if not in_initial_pid_namespace():
        return (
            "only the host's initial PID namespace can see another namespace's processes, "
            "so run this from there"
        )
    link = _pid_namespace_link(pid_namespace)
    if link == INITIAL_PID_NAMESPACE_LINK:
        # That namespace is never freed, so a differing PID 1 start time means
        # the identity was recorded under another clock (a time namespace).
        return "it names the host's initial PID namespace under another clock"
    try:
        entries = tuple(_PROC.iterdir())
    except OSError:
        return "/proc cannot be listed"
    blockers = [
        entry.name
        for entry in entries
        if entry.name.isdigit() and _may_be_recorded_process(entry, link, pids, pgids)
    ]
    if not blockers:
        return None
    named = ",".join(sorted(blockers, key=int))
    return (
        f"host process(es) {named} may be among its recorded processes; inspect them with "
        f"`ps -o pid,pgid,sid,cmd -p {named}` and retry once they have exited"
    )


def _may_be_recorded_process(
    entry: Path, link: str, pids: Collection[int], pgids: Collection[int]
) -> bool:
    """Return whether one ``/proc`` entry may be a recorded process of another namespace.

    :param Path entry: ``/proc/<pid>`` directory of the initial namespace's procfs.
    :param str link: ``ns/pid`` link text of the recorded namespace.
    :param Collection[int] pids: Recorded PIDs, numbered in that namespace.
    :param Collection[int] pgids: Recorded process groups, numbered in that namespace.
    :return bool: Whether the entry is a candidate that its namespace link
        cannot rule out. A process that exits mid-scan is not.
    """
    try:
        status = (entry / "status").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return True
    levels: dict[str, tuple[int, ...]] = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key in {"NSpid", "NSpgid", "NSsid"}:
            try:
                levels[key] = tuple(int(number) for number in value.split())
            except ValueError:
                return True
    nspid, nspgid, nssid = (levels.get(key, ()) for key in ("NSpid", "NSpgid", "NSsid"))
    if not nspid or not len(nspid) == len(nspgid) == len(nssid):
        # Without these lines (Linux before 4.1) nesting cannot be read.
        return True
    leads_recorded_group = [
        nspgid[level] in pgids and nssid[level] == nspgid[level] for level in range(len(nspid))
    ]
    if any(leads_recorded_group[1:-1]):
        return True
    if len(nspid) < 2 or not (nspid[-1] in pids or leads_recorded_group[-1]):
        return False
    try:
        return os.readlink(entry / "ns" / "pid") == link
    except (FileNotFoundError, ProcessLookupError):
        return entry.exists()
    except OSError:
        return True


def foreign_pid_namespace_reason(pid_namespace: str, blocker: str) -> str:
    """Explain why processes recorded in another PID namespace cannot be acted on here.

    :param str pid_namespace: Namespace recorded with the process identity.
    :param str blocker: What stopped the proof, from :func:`recorded_process_blocker`.
    :return str: A clause naming both namespaces and what blocked the proof.
    """
    current = read_pid_namespace() or "unreadable"
    return (
        f"it was recorded in PID namespace {pid_namespace}, but this process is in {current}, "
        f"so its PID numbers name unrelated processes here, and {blocker}"
    )


@dataclass(frozen=True)
class _ProcStat:
    """Parsed fields from one Linux ``/proc/<pid>/stat`` record."""

    state: str
    pgrp: int
    starttime: int


@dataclass(frozen=True)
class _GroupMemberScan:
    """One procfs group scan with the evidence needed for a safe death verdict."""

    pids: tuple[int, ...]
    complete: bool
    all_members_terminal: bool


def _read_proc_stat_result(proc_entry: Path) -> tuple[_ProcStat | None, bool]:
    """Read one proc stat and distinguish disappearance from unreadability.

    :param Path proc_entry: ``/proc/<pid>`` directory to inspect.
    :return tuple: Parsed stat plus ``True`` when the read was conclusive.
        A vanished PID is conclusively absent; permission, I/O, or parse
        failures are incomplete evidence.
    """
    try:
        data = (proc_entry / "stat").read_bytes()
    except FileNotFoundError:
        return None, True
    except OSError:
        return None, False
    rparen = data.rfind(b")")
    if rparen < 0:
        return None, False
    rest = data[rparen + 1 :].strip().split()
    if len(rest) < 20:
        return None, False
    try:
        stat = _ProcStat(
            state=rest[0].decode("ascii"),
            pgrp=int(rest[2]),
            starttime=int(rest[19]),
        )
    except (UnicodeDecodeError, ValueError):
        return None, False
    return stat, True


def _read_proc_stat(proc_entry: Path) -> _ProcStat | None:
    """Parse the proc stat fields phasesweep uses for liveness checks.

    :param Path proc_entry: ``/proc/<pid>`` directory to inspect.
    :return _ProcStat | None: Parsed state, process group, and starttime, or ``None`` when
        unreadable.
    """
    stat, _complete = _read_proc_stat_result(proc_entry)
    return stat


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


def is_same_live_process(pid: int | None, saved_starttime: int | None) -> bool:
    """Return whether a PID identifies the same live, non-zombie process.

    :param int | None pid: Process identifier, if one was recorded.
    :param int | None saved_starttime: Recorded Linux process start time.
    :return bool: Whether the identity still names a live non-zombie process.
    """
    if pid is None or not is_pid_alive(pid):
        return False
    stat = _read_proc_stat(Path("/proc") / str(pid))
    if saved_starttime is not None and (stat is None or stat.starttime != saved_starttime):
        return False
    return stat is None or stat.state != "Z"


def reap_child(pid: int) -> bool:
    """Best-effort non-blocking reap of one exited child.

    Running processes, non-children and already-reaped children are unchanged.
    A child exiting after this call waits for a later reap.

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
        "pid_namespace",
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
            field: Name of the top-level identity field to read from the
                payload and validate; it also names the field in the error
                message.

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
    pid_namespace = payload.get("pid_namespace")
    if not is_pid_namespace(pid_namespace):
        raise ValueError(f"Trial process identity field 'pid_namespace' is invalid at {path}.")
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id=attempt_id,
        pid=positive_int("pid"),
        pgid=positive_int("pgid"),
        proc_starttime=proc_starttime,
        boot_id=boot_id,
        pid_namespace=pid_namespace,
    )


def cleanup_stale_trial_process(
    identity: StaleProcessIdentity,
    *,
    grace_seconds: float = _KILL_GRACE_SECONDS,
) -> bool:
    """Clean a stale trial group using boot, namespace and process identity.

    A different known boot proves cleanup complete. Foreign-namespace processes must
    be proven absent without signaling. Same-namespace cleanup requires known boot
    IDs and saved start time before delegating to :func:`kill_stale_group`.

    :param StaleProcessIdentity identity: Durable process identity read from
        ``process_identity.json``.
    :param float grace_seconds: Seconds to wait after SIGTERM before escalating
        to SIGKILL; forwarded to :func:`kill_stale_group`.
    :raises ForeignPidNamespaceError: The identity was recorded in another PID
        namespace and its processes cannot be proven gone from here.
    :return bool: ``True`` when it is safe to mark the trial ``FAIL`` (nothing
        to clean, the host rebooted, its processes in another namespace are
        provably gone, or cleanup was confirmed). ``False`` when identity
        cannot be verified or :func:`kill_stale_group` reports cleanup is
        uncertain; callers must not advance state in that case.
    """
    current_boot_id = read_boot_id()
    if identity_from_earlier_boot(identity.boot_id, current_boot_id):
        log.warning(
            "Attempt %s belongs to an earlier host boot; no process from that boot remains.",
            identity.attempt_id,
        )
        return True
    if not same_pid_namespace(identity.pid_namespace):
        blocker = recorded_process_blocker(
            identity.pid_namespace, pids=(identity.pid,), pgids=(identity.pgid,)
        )
        if blocker is None:
            log.warning(
                "Attempt %s was recorded in PID namespace %s, where none of its processes remains.",
                identity.attempt_id,
                identity.pid_namespace,
            )
            return True
        raise ForeignPidNamespaceError(
            f"Attempt {identity.attempt_id} (pid={identity.pid}) cannot be cleaned from here: "
            f"{foreign_pid_namespace_reason(identity.pid_namespace, blocker)}. "
            "Nothing was signalled."
        )
    if identity.boot_id is None or identity.proc_starttime is None or current_boot_id is None:
        log.warning(
            "Refusing automatic cleanup for attempt %s because robust process-birth identity "
            "is unavailable on this platform.",
            identity.attempt_id,
        )
        return False
    return kill_stale_group(
        identity.pid,
        identity.proc_starttime,
        pgid=identity.pgid,
        grace_seconds=grace_seconds,
    )


def _terminate_process_group(pgid: int, *, grace_seconds: float) -> bool:
    """Send SIGTERM, wait, then SIGKILL, confirming the group is gone.

    Args:
        pgid: Target process-group ID.
        grace_seconds: Seconds to wait after SIGTERM before escalating to SIGKILL.

    Returns:
        ``True`` if the group is confirmed dead (already gone, or died within
        the SIGTERM grace, or died within 2s after SIGKILL). ``False`` if
        signal delivery failed for non-``ProcessLookupError`` reasons, or the
        group is still alive 2s after SIGKILL.

    """
    return _terminate_process_groups((pgid,), grace_seconds=grace_seconds)[pgid]


def _terminate_process_groups(pgids: tuple[int, ...], *, grace_seconds: float) -> dict[int, bool]:
    """Escalate SIGTERM to SIGKILL with shared wait windows across all groups.

    Per-group verdicts match :func:`_terminate_process_group`; all groups share one
    SIGTERM grace and one post-SIGKILL confirmation window.

    Args:
        pgids: Target process-group IDs.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL, shared across all groups.

    Returns:
        Confirmation verdict per requested pgid — ``True`` only when that
        group is confirmed dead.

    """
    confirmed: dict[int, bool] = {}
    members: dict[int, set[int]] = {}
    pending: list[int] = []
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            confirmed[pgid] = True
            continue
        except (PermissionError, OSError) as exc:
            log.error("Failed to send SIGTERM to process group %d: %s", pgid, exc)
            confirmed[pgid] = False
            continue
        members[pgid] = set(_group_member_pids(pgid).pids)
        pending.append(pgid)

    deadline = time.monotonic() + grace_seconds
    while pending and time.monotonic() < deadline:
        still_alive = []
        for pgid in pending:
            if _process_group_alive_with_members(pgid, members[pgid]):
                still_alive.append(pgid)
            else:
                confirmed[pgid] = True
        pending = still_alive
        if pending:
            time.sleep(0.1)

    survivors: list[int] = []
    for pgid in pending:
        log.warning("Stale process group %d survived SIGTERM — sending SIGKILL", pgid)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            confirmed[pgid] = True
            continue
        except (PermissionError, OSError) as exc:
            log.error("Failed to send SIGKILL to process group %d: %s", pgid, exc)
            confirmed[pgid] = False
            continue
        survivors.append(pgid)

    # Wait briefly for the kernel to actually reap descendants so the reaper's
    # "marked FAIL" state means cleanup completed, not requested.
    kill_deadline = time.monotonic() + 2.0
    while survivors and time.monotonic() < kill_deadline:
        still_alive = []
        for pgid in survivors:
            if _process_group_alive_with_members(pgid, members[pgid]):
                still_alive.append(pgid)
            else:
                confirmed[pgid] = True
        survivors = still_alive
        if survivors:
            time.sleep(0.1)

    for pgid in survivors:
        log.error("Process group %d still appears alive after SIGKILL", pgid)
        confirmed[pgid] = False
    return confirmed


def _process_group_alive(pgid: int) -> bool:
    """Check whether any non-zombie member of ``pgid`` is alive.

    On non-Linux systems without procfs, fall back to ``killpg(0)`` existence checks.

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
    if member_pids is not None and _member_pids_alive(pgid, member_pids):
        return True
    scan = _group_member_pids(pgid)
    if member_pids is not None:
        member_pids.clear()
        member_pids.update(scan.pids)
    if _member_pids_alive(pgid, scan.pids if member_pids is None else member_pids):
        return True
    if not scan.complete:
        return True
    if scan.all_members_terminal:
        return False
    # A kernel-visible group without inspectable members remains uncertain.
    return _process_group_exists(pgid)


def _group_member_pids(pgid: int) -> _GroupMemberScan:
    """Return current ``/proc`` members plus scan completeness.

    :param int pgid: Process-group ID to find under ``/proc``.
    :return _GroupMemberScan: PIDs reporting membership, whether every proc
        entry was inspectable, and whether all observed members were zombies
        or exited.
    """
    proc_root = Path("/proc")
    if not proc_root.exists():
        return _GroupMemberScan(pids=(), complete=False, all_members_terminal=False)
    member_pids: list[int] = []
    all_members_terminal = True
    complete = True
    try:
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            stat, conclusive = _read_proc_stat_result(entry)
            pid = int(entry.name)
            if not conclusive:
                # An unreadable entry blocks proof only when the kernel cannot exclude it
                # from the target group.
                try:
                    unreadable_pgid = os.getpgid(pid)
                except ProcessLookupError:
                    continue
                except OSError:
                    complete = False
                    continue
                if unreadable_pgid != pgid:
                    continue
                member_pids.append(pid)
                complete = False
                all_members_terminal = False
                continue
            if stat is None or stat.pgrp != pgid:
                continue
            member_pids.append(pid)
            if stat.state not in {"Z", "X"}:
                all_members_terminal = False
    except OSError:
        complete = False
    return _GroupMemberScan(
        pids=tuple(member_pids),
        complete=complete,
        all_members_terminal=bool(member_pids) and all_members_terminal,
    )


def _member_pids_alive(pgid: int, member_pids: Collection[int]) -> bool:
    """Return whether any known member PID is still live and in ``pgid``.

    :param int pgid: Process-group ID each PID must still belong to.
    :param Collection[int] member_pids: Candidate member PIDs to inspect.
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
    """Terminate a stale trial group only when its process identity is safe to signal.

    The caller must establish the recorded PID namespace or direct-child ownership.
    Use a matching live PID; after PID reuse or root exit, use the saved PGID only
    if its leader has not been reused. Missing PID and PGID
    means no cleanup is needed; uncertain cleanup must block state advancement.

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
        elif pid_alive and pgid is None:
            log.warning(
                "PID %d is alive but starttime does not match saved value; "
                "PID was reused and no stored PGID is available. Cleanup "
                "status is uncertain.",
                pid,
            )
            return False

    if target_pgid is None and pgid is not None:
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
