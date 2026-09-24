"""Process-group probes, termination, and stale trial cleanup.

Reads ``/proc`` to decide whether a recorded PID or process group is still the
one phasesweep launched, terminates groups SIGTERM -> grace -> SIGKILL, and
cleans a stale trial's group only when the boot- and process-bound identity in
``process_identity.json`` proves that signalling it cannot hit a reused PID.
"""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from phasesweep.runtime.json import strict_json_loads

# Process supervision logs on one channel, whichever module does the work.
log = logging.getLogger("phasesweep.runtime.process")

_KILL_GRACE_SECONDS = 10.0
PROCESS_IDENTITY_FILE = "process_identity.json"
PROCESS_IDENTITY_SCHEMA_VERSION = 1


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


def identity_from_earlier_boot(boot_id: str | None, current_boot_id: str | None) -> bool:
    """Return whether a recorded boot identity proves its process cannot exist.

    PID plus ``/proc`` start time is unique only within one boot: after a
    reboot the kernel restarts both counters, so a saved pair can match an
    unrelated process. A recorded boot id that differs from the current one
    settles the question in the safe direction - nothing launched under the
    earlier boot survived it, so the process and every descendant it ever had
    are conclusively gone and cleanup needs no signal. An unknown boot id on
    either side (a handle written before boot ids were recorded, or a host
    without ``/proc/sys/kernel/random/boot_id``) yields ``False``. Callers
    performing cleanup must separately refuse to signal when either boot id
    is unknown, using the same ``current_boot_id`` read they pass here.

    :param str | None boot_id: Boot identity recorded when the process launched.
    :param str | None current_boot_id: This boot's identity, from :func:`read_boot_id`.
    :return bool: Whether both boot ids are known and differ.
    """
    return boot_id is not None and current_boot_id is not None and boot_id != current_boot_id


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
    if identity_from_earlier_boot(identity.boot_id, current_boot_id):
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
    return _terminate_process_groups((pgid,), grace_seconds=grace_seconds)[pgid]


def _terminate_process_groups(pgids: tuple[int, ...], *, grace_seconds: float) -> dict[int, bool]:
    """Escalate SIGTERM → SIGKILL across process groups with shared wait windows.

    Per-group semantics match :func:`_terminate_process_group`; the SIGTERM
    grace and the post-SIGKILL confirmation window are shared across all
    groups, so total worst-case latency stays roughly ``grace_seconds + 2``
    seconds instead of scaling with the number of live groups. The shutdown
    handler kills every active trial group of an ``n_jobs > 1`` phase through
    this path — trainers that ignore SIGTERM fail correlated, not
    independently, so a serial escalation would multiply the documented
    worst case by the trial parallelism.

    Args:
        pgids: Target process-group IDs.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL, shared across all groups.

    Returns:
        Confirmation verdict per requested pgid — ``True`` only when that
        group is confirmed dead.

    """
    import time

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
        scan = _group_member_pids(pgid)
        if _member_pids_alive(pgid, scan.pids):
            return True
        if not scan.complete:
            return True
        if scan.all_members_terminal:
            return False
        # killpg proved the group existed, but a complete scan found no stable
        # inspectable member. Re-check the kernel verdict; continued existence
        # is uncertainty, not proof of death.
        return _process_group_exists(pgid)
    if _member_pids_alive(pgid, member_pids):
        return True
    scan = _group_member_pids(pgid)
    refreshed = set(scan.pids)
    member_pids.clear()
    member_pids.update(refreshed)
    if _member_pids_alive(pgid, member_pids):
        return True
    if not scan.complete:
        return True
    if scan.all_members_terminal:
        return False
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
                # Hardened procfs mounts commonly hide processes owned by
                # other users. An unreadable host-global entry is irrelevant
                # when the kernel can still prove it belongs to another group;
                # only a possibly-matching entry makes this scan incomplete.
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
