"""PID-namespace binding of process identities: reading, the absence proof, and stale cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import phasesweep.mcp.recovery as mcp_recovery
import phasesweep.mcp.registry as mcp_registry
import phasesweep.mcp.runner as mcp_runner
import phasesweep.mcp.runs as mcp_runs
import phasesweep.mcp.tools as mcp_tools
import phasesweep.runtime.reaper as reaper
from phasesweep.errors import ForeignPidNamespaceError, OperatorAction, PhaseSweepError
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import CatalogError, ExperimentBusyError, RunNamespaceMismatchError
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.registry import Registry, require_linux_mcp_host
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.tool_names import TOOL_CANCEL_RUN
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime import process as runtime_process
from phasesweep.runtime.process import (
    _trial_process_identity,
    _write_process_identity,
    run_supervised,
)
from phasesweep.runtime.reaper import (
    INITIAL_PID_NAMESPACE_LINK,
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    cleanup_stale_trial_process,
    foreign_pid_namespace_reason,
    read_pid_namespace,
    read_proc_starttime,
    read_stale_process_identity,
    recorded_process_blocker,
    same_pid_namespace,
)
from phasesweep.runtime.time import utc_now_iso
from tests.ledger_fixtures import tree_snapshot
from tests.mcp_helpers import (
    claim_runner_handle,
    live_runs,
    patch_popen_capture,
    side_effect_app,
    stage_dead_run,
    stage_live_run,
    write_run_status,
)
from tests.recovery_helpers import write_trial_identity

RECORDED = "pid:[4026539001]@100"
OTHER = "pid:[4026539002]@100"
NESTED_READER = "pid:[4026539333]@100"
#: The reader in the host's initial PID namespace, whose PID 1 started at tick 5.
INITIAL_READER = f"{INITIAL_PID_NAMESPACE_LINK}@5"


def _link(namespace: str) -> str:
    """Return the ``ns/pid`` link text a recorded namespace names."""
    return namespace.partition("@")[0]


def _write_process(
    proc: Path,
    pid: int,
    *,
    nspid: list[int] | None,
    namespace: str | None,
    nspgid: list[int] | None = None,
    nssid: list[int] | None = None,
    starttime: int = 5,
) -> None:
    """Add one fake ``/proc/<pid>`` entry with a stat record and optional nesting lines and namespace link.

    ``nspgid`` defaults to ``nspid`` and ``nssid`` to ``nspgid``: a process that
    leads its own group and session at every level, as runners and trials do.
    """
    entry = proc / str(pid)
    (entry / "ns").mkdir(parents=True)
    lines = ["Name:\tfake", "State:\tS (sleeping)"]
    if nspid is not None:
        groups = nspid if nspgid is None else nspgid
        for key, values in (
            ("NSpid", nspid),
            ("NSpgid", groups),
            ("NSsid", groups if nssid is None else nssid),
        ):
            lines.append(f"{key}:\t" + "\t".join(str(value) for value in values))
    (entry / "status").write_text("\n".join(lines) + "\n")
    (entry / "stat").write_text(f"{pid} (fake) S 0 {pid} {pid} " + "0 " * 15 + f"{starttime}\n")
    if namespace is not None:
        (entry / "ns" / "pid").symlink_to(_link(namespace))


def _hide_namespace(proc: Path, pid: int) -> None:
    """Make one fake entry's namespace link unreadable, as a non-dumpable process's is."""
    (proc / str(pid) / "ns" / "pid").write_text("not a link")


def _fake_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reader: str | None = INITIAL_READER
) -> Path:
    """Replace ``/proc`` with one holding only the host's init, and the reader's namespace with ``reader``."""
    proc = tmp_path / "proc"
    _write_process(proc, 1, nspid=[1], namespace=INITIAL_READER)
    monkeypatch.setattr(reaper, "_PROC", proc)
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: reader)
    return proc


def _identity(
    *,
    pid_namespace: str,
    boot_id: str | None = "boot",
    proc_starttime: int | None = 111,
) -> StaleProcessIdentity:
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id="ns-attempt",
        pid=4242,
        pgid=4242,
        proc_starttime=proc_starttime,
        boot_id=boot_id,
        pid_namespace=pid_namespace,
    )


def test_read_pid_namespace_names_this_process_namespace() -> None:
    namespace = read_pid_namespace()
    if namespace is None:
        pytest.skip("/proc is not this process's own procfs")

    assert namespace == f"{os.readlink('/proc/self/ns/pid')}@{read_proc_starttime(1)}"
    assert reaper.is_pid_namespace(namespace)


def test_read_pid_namespace_refuses_procfs_of_another_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A procfs whose ``self`` is not this PID cannot answer for this namespace's numbers."""
    proc = tmp_path / "proc"
    _write_process(proc, 1, nspid=None, namespace=None, starttime=100)
    _write_process(proc, os.getpid(), nspid=None, namespace=RECORDED)
    monkeypatch.setattr(reaper, "_PROC", proc)

    (proc / "self").symlink_to(str(os.getpid() + 1))
    assert read_pid_namespace() is None

    (proc / "self").unlink()
    (proc / "self").symlink_to(str(os.getpid()))
    assert read_pid_namespace() == RECORDED


def test_read_pid_namespace_needs_its_pid_1_start_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without its PID 1, a namespace's inode could be one an ended namespace used."""
    proc = tmp_path / "proc"
    _write_process(proc, os.getpid(), nspid=None, namespace=RECORDED)
    (proc / "self").symlink_to(str(os.getpid()))
    monkeypatch.setattr(reaper, "_PROC", proc)

    assert read_pid_namespace() is None


def test_read_pid_namespace_rejects_a_malformed_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    _write_process(proc, 1, nspid=None, namespace=None)
    _write_process(proc, os.getpid(), nspid=None, namespace="net:[4026539001]")
    (proc / "self").symlink_to(str(os.getpid()))
    monkeypatch.setattr(reaper, "_PROC", proc)

    assert read_pid_namespace() is None


def test_same_pid_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)

    assert same_pid_namespace(NESTED_READER)
    assert not same_pid_namespace(RECORDED)

    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: None)
    assert not same_pid_namespace(RECORDED)


def test_a_reused_inode_with_another_pid_1_start_time_is_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel reuses an ended namespace's inode within seconds, so the inode alone is not the namespace."""
    reused = f"{_link(RECORDED)}@200"
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: reused)
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    _forbid_run_signals(monkeypatch, reaper)

    assert same_pid_namespace(reused)
    assert not same_pid_namespace(RECORDED)
    with pytest.raises(ForeignPidNamespaceError, match="Nothing was signalled"):
        cleanup_stale_trial_process(_identity(pid_namespace=RECORDED))

    _app, _registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch, reader=reused)
    _forbid_reaping(monkeypatch)
    assert store.runner_view(handle).visibility == "unobservable"
    assert store.observe(handle).state == "running"


# The absence proof: from the host's initial PID namespace only, a recorded
# process of another namespace is gone when no process could still be it.


def test_absence_proof_needs_the_initial_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_proc(tmp_path, monkeypatch, reader=NESTED_READER)

    # A reader inside a nested namespace sees only its own subtree of /proc, so
    # an absent entry proves nothing there.
    blocker = recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,))
    assert blocker is not None
    assert "initial PID namespace" in blocker

    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: INITIAL_READER)
    assert recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,)) is None
    # The initial namespace is never freed: a differing PID 1 start time is a
    # different clock, and its processes may still run.
    other_clock = f"{INITIAL_PID_NAMESPACE_LINK}@9"
    assert recorded_process_blocker(other_clock, pids=(7,), pgids=(7,)) is not None


@pytest.mark.parametrize(
    ("pid", "nspid", "nspgid"),
    [
        pytest.param(500, [500, 7], None, id="recorded-process"),
        pytest.param(501, [501, 9], [480, 7], id="group-leader-exited"),
    ],
)
def test_absence_proof_finds_a_recorded_process_or_surviving_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid: int, nspid, nspgid
) -> None:
    """The recorded PID or a remaining member of its group and session blocks absence."""
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, pid, nspid=nspid, nspgid=nspgid, namespace=RECORDED)

    blocker = recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,))

    assert blocker is not None
    assert str(pid) in blocker


def test_a_readable_link_to_another_namespace_rules_a_candidate_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 502, nspid=[502, 7], namespace=OTHER)

    assert recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,)) is None


def test_an_unreadable_candidate_blocks_only_for_its_exact_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-dumpable process in another sandbox that collides on a number blocks that number alone."""
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 503, nspid=[503, 7], namespace=None)
    _hide_namespace(proc, 503)

    for pids, pgids in (((7,), (7,)), ((7,), (8,)), ((8,), (7,))):
        blocker = recorded_process_blocker(RECORDED, pids=pids, pgids=pgids)
        assert blocker is not None
        assert "503" in blocker
    assert recorded_process_blocker(RECORDED, pids=(8,), pgids=(8,)) is None


def test_a_group_member_of_another_session_is_not_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded group leads its own session, so a group number in another session is a reused one."""
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 504, nspid=[504, 9], nspgid=[480, 7], nssid=[470, 3], namespace=None)
    _hide_namespace(proc, 504)

    assert recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,)) is None


def test_a_group_match_at_an_outer_level_blocks_despite_a_readable_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trial's descendant in a deeper namespace is still in the trial's group one level up."""
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 505, nspid=[505, 40, 2], nspgid=[505, 7, 2], namespace=OTHER)

    blocker = recorded_process_blocker(RECORDED, pids=(99,), pgids=(7,))

    assert blocker is not None
    assert "505" in blocker


def test_a_host_process_is_never_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 7, nspid=[7], namespace=INITIAL_READER)

    assert recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,)) is None


def test_absence_proof_ignores_a_process_that_exits_during_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    (proc / "77").mkdir()

    assert recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,)) is None


@pytest.mark.parametrize("case", ["unreadable_status", "missing_link", "missing_ns_lines"])
def test_absence_proof_treats_an_uninspectable_candidate_as_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    if case == "missing_ns_lines":
        # A kernel older than 4.1 has no NSpid line, so nesting cannot be read.
        _write_process(proc, 60, nspid=None, namespace=OTHER)
    else:
        _write_process(proc, 60, nspid=[60, 7], namespace=None)
    if case == "unreadable_status":
        (proc / "60" / "status").unlink()
        (proc / "60" / "status").mkdir()

    blocker = recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,))

    assert blocker is not None
    assert "60" in blocker


def test_absence_proof_names_every_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator learns every blocking process at once, not one per attempt."""
    proc = _fake_proc(tmp_path, monkeypatch)
    for pid in (601, 600):
        _write_process(proc, pid, nspid=[pid, 7], namespace=None)
        _hide_namespace(proc, pid)

    blocker = recorded_process_blocker(RECORDED, pids=(7,), pgids=(7,))

    assert blocker is not None
    assert "600,601" in blocker


def test_foreign_reason_names_both_namespaces_and_what_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)

    reason = foreign_pid_namespace_reason(RECORDED, "host process(es) 60 may be among them")

    assert RECORDED in reason
    assert NESTED_READER in reason
    assert "host process(es) 60" in reason


def test_trial_process_identity_records_the_launching_namespace(tmp_path: Path) -> None:
    namespace = read_pid_namespace()
    if namespace is None:
        pytest.skip("/proc is not this process's own procfs")

    identity = _trial_process_identity(
        attempt_id="ns-attempt", pid=os.getpid(), pgid=os.getpid(), pid_namespace=namespace
    )
    _write_process_identity(tmp_path / PROCESS_IDENTITY_FILE, identity)

    assert identity.pid_namespace == namespace
    assert json.loads((tmp_path / PROCESS_IDENTITY_FILE).read_text())["pid_namespace"] == namespace
    assert read_stale_process_identity(tmp_path, expected_attempt_id="ns-attempt") == identity


def test_trial_launch_refuses_an_unreadable_pid_namespace_before_anything_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_process, "read_pid_namespace", lambda: None)
    monkeypatch.setattr(
        runtime_process,
        "_spawn_blocked_supervisor",
        lambda **_kwargs: pytest.fail("no trial may start without a recorded namespace"),
    )
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    with (tmp_path / "out.log").open("w") as out, pytest.raises(PhaseSweepError) as caught:
        run_supervised(
            "true",
            env={},
            stdout=out,
            stderr=out,
            timeout=None,
            trial_dir=trial_dir,
            attempt_id="ns-attempt",
        )

    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert "PID namespace" in str(caught.value)
    assert list(trial_dir.iterdir()) == []


def test_read_stale_process_identity_requires_pid_namespace(tmp_path: Path) -> None:
    write_trial_identity(
        tmp_path, attempt_id="ns-attempt", pid=4242, starttime=111, pid_namespace=RECORDED
    )
    path = tmp_path / PROCESS_IDENTITY_FILE
    record = json.loads(path.read_text())
    assert record["pid_namespace"] == RECORDED
    assert read_stale_process_identity(
        tmp_path, expected_attempt_id="ns-attempt"
    ).pid_namespace == (RECORDED)

    record.pop("pid_namespace")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="partial"):
        read_stale_process_identity(tmp_path, expected_attempt_id="ns-attempt")

    for invalid in (None, _link(RECORDED), "pid:[0]@100", f"{_link(RECORDED)}@"):
        record["pid_namespace"] = invalid
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="pid_namespace"):
            read_stale_process_identity(tmp_path, expected_attempt_id="ns-attempt")


def test_cleanup_stale_trial_process_refuses_foreign_namespace_without_signalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)
    _forbid_run_signals(monkeypatch, reaper)

    with pytest.raises(ForeignPidNamespaceError) as caught:
        cleanup_stale_trial_process(_identity(pid_namespace=RECORDED))

    message = str(caught.value)
    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert RECORDED in message
    assert NESTED_READER in message
    assert "Nothing was signalled" in message
    assert "kill -9" not in message


def test_cleanup_stale_trial_process_settles_an_absent_trial_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    proc = _fake_proc(tmp_path, monkeypatch)
    # A process in another sandbox reusing the trial's numbers is ruled out by its link.
    _write_process(proc, 700, nspid=[700, 4242], namespace=OTHER)
    _forbid_run_signals(monkeypatch, reaper)

    assert cleanup_stale_trial_process(_identity(pid_namespace=RECORDED)) is True


def test_cleanup_stale_trial_process_names_what_blocks_the_absence_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 701, nspid=[701, 4242], namespace=RECORDED)
    _forbid_run_signals(monkeypatch, reaper)

    with pytest.raises(ForeignPidNamespaceError) as caught:
        cleanup_stale_trial_process(_identity(pid_namespace=RECORDED))

    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert "701" in str(caught.value)
    assert "kill -9" not in str(caught.value)


def test_cleanup_stale_trial_process_signals_only_within_the_same_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: RECORDED)
    signalled: list[tuple[int | None, int | None]] = []

    def record(pid: int | None, starttime: int | None, **_kwargs: object) -> bool:
        signalled.append((pid, starttime))
        return True

    monkeypatch.setattr(reaper, "kill_stale_group", record)

    assert cleanup_stale_trial_process(_identity(pid_namespace=RECORDED)) is True
    assert signalled == [(4242, 111)]


def test_cleanup_stale_trial_process_refuses_a_foreign_identity_before_checking_its_birth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An identity without boot id or start time is refused as foreign, never with kill advice for its numbers."""
    monkeypatch.setattr(reaper, "read_boot_id", lambda: None)
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)
    _forbid_run_signals(monkeypatch, reaper)

    with pytest.raises(ForeignPidNamespaceError, match="Nothing was signalled"):
        cleanup_stale_trial_process(
            _identity(pid_namespace=RECORDED, boot_id=None, proc_starttime=None)
        )


#: Prints the namespace's link, its PID 1's start time, and the holder's own
#: PID there, then leads a session of its own, as runners and trials do.
_HOLD_NAMESPACE = (
    'readlink /proc/self/ns/pid; cut -d" " -f22 /proc/1/stat; echo $$; exec setsid sleep 120'
)

#: Ways to start a process in a fresh PID namespace with its own procfs. The first
#: needs unprivileged user namespaces; the second works where only bubblewrap may
#: create them. Both die with the launcher process.
_NAMESPACE_LAUNCHERS = (
    ("unshare", "-Urpf", "--kill-child", "--mount-proc"),
    ("bwrap", "--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/", "--proc", "/proc"),
)


def _start_namespace_holder() -> tuple[subprocess.Popen[str], str, int]:
    """Start a process in a new PID namespace; return it, that namespace, and its PID there."""
    refusals: list[str] = []
    for launcher in _NAMESPACE_LAUNCHERS:
        executable = shutil.which(launcher[0])
        if executable is None:
            continue
        child = subprocess.Popen(
            [executable, *launcher[1:], "sh", "-c", _HOLD_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        assert child.stderr is not None
        link = child.stdout.readline().strip()
        namespace = f"{link}@{child.stdout.readline().strip()}"
        if reaper.is_pid_namespace(namespace):
            return child, namespace, int(child.stdout.readline())
        child.kill()
        refusals.append(f"{launcher[0]}: {child.stderr.read().strip()}")
        child.wait(timeout=10)
    pytest.skip("cannot create a PID namespace here: " + "; ".join(refusals))


@pytest.mark.integration
def test_absence_proof_tracks_a_real_child_pid_namespace() -> None:
    """A live recorded process blocks the proof; once its namespace is gone, nothing does."""
    current = read_pid_namespace()
    if current is None or _link(current) != INITIAL_PID_NAMESPACE_LINK:
        pytest.skip("the absence proof needs the initial PID namespace")
    child, namespace, pid = _start_namespace_holder()
    try:
        assert same_pid_namespace(current)
        assert not same_pid_namespace(namespace)
        assert recorded_process_blocker(namespace, pids=(pid,), pgids=(pid,)) is not None

        child.kill()
        child.wait(timeout=10)
        # Inode 1 is never a namespace, so whatever blocks it is an unrelated
        # process that collides on these numbers and hides its namespace from
        # this user (a browser sandbox is not dumpable).
        unrelated = recorded_process_blocker("pid:[1]@0", pids=(pid,), pgids=(pid,))
        if unrelated is not None:
            pytest.skip(f"an unrelated process collides on PID {pid} here: {unrelated}")
        deadline = time.monotonic() + 10
        while (
            time.monotonic() < deadline
            and recorded_process_blocker(namespace, pids=(pid,), pgids=(pid,)) is not None
        ):
            time.sleep(0.05)
        assert recorded_process_blocker(namespace, pids=(pid,), pgids=(pid,)) is None
    finally:
        child.kill()
        child.wait(timeout=10)


# A run whose runner was recorded in another PID namespace is neither alive nor
# dead from here: the MCP server reads it on durable evidence alone and never
# probes, reaps, or signals the numbers it recorded.


def _stage_foreign_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_cancel: bool = False,
    cleanup_uncertain: bool = False,
    reader: str = NESTED_READER,
) -> tuple[PhaseSweepMCP, Registry, RunStore, RunHandle]:
    """Serve the standard experiment from ``reader``'s PID namespace with one run recorded in another."""
    config, app, registry, store = side_effect_app(tmp_path)
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: reader)
    handle = stage_dead_run(
        store,
        "srv-foreign",
        config,
        registry.get("srv").id,
        cleanup_uncertain=cleanup_uncertain,
        allow_cancel=allow_cancel,
        pid_namespace=RECORDED,
    )
    return app, registry, store, handle


def _stage_absent_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_cancel: bool = False,
    cleanup_uncertain: bool = False,
) -> tuple[PhaseSweepMCP, Registry, RunStore, RunHandle, Path]:
    """Serve a foreign run from the host's initial namespace, where none of its processes remains.

    :return tuple: Server, registry, store, handle, and the fake ``/proc`` to add blockers to.
    """
    app, registry, store, handle = _stage_foreign_run(
        tmp_path,
        monkeypatch,
        allow_cancel=allow_cancel,
        cleanup_uncertain=cleanup_uncertain,
        reader=INITIAL_READER,
    )
    proc = _fake_proc(tmp_path / "fake", monkeypatch)
    return app, registry, store, handle, proc


def _add_runner_blocker(proc: Path, handle: RunHandle) -> None:
    """Add a non-dumpable process in another sandbox that collides on the runner's numbers."""
    assert handle.pid is not None
    _write_process(proc, 900, nspid=[900, handle.pid], namespace=None)
    _hide_namespace(proc, 900)


def _write_finished_status(store: RunStore, run_id: str) -> None:
    """Record a clean exit whose result snapshot capture failed, the simplest terminal status."""
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="failed",
        result_snapshot_error="InterruptedFinalization",
    )


def _forbid_run_signals(monkeypatch: pytest.MonkeyPatch, *modules: object) -> None:
    """Fail the test if any of ``modules`` signals a process group."""

    def refuse(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("a process outside this PID namespace must never be signalled")

    for module in modules:
        monkeypatch.setattr(module, "kill_stale_group", refuse)


def _forbid_reaping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_runs,
        "reap_child",
        lambda _pid: pytest.fail("a PID from another namespace must never be waited on"),
    )


def test_foreign_namespace_runner_reads_running_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch)
    before = tree_snapshot(registry.state_dir)
    _forbid_reaping(monkeypatch)

    view = store.runner_view(handle)
    assert view.visibility == "unobservable"
    assert view.reason is not None
    assert RECORDED in view.reason
    assert store.runner_may_be_alive(handle)
    observation = store.observe(handle)
    assert observation.state == "running"
    assert not observation.recovery_required
    assert not observation.snapshot_recovery_required
    assert live_runs(store) == [handle]
    assert tree_snapshot(registry.state_dir) == before


def test_foreign_namespace_terminal_status_still_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable evidence needs no probe, so a finished run reads terminal from anywhere."""
    _app, _registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch)
    _write_finished_status(store, handle.run_id)

    assert store.observe(handle).state == "succeeded"


def test_foreign_namespace_runner_holds_its_concurrency_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store, handle = _stage_foreign_run(tmp_path, monkeypatch)

    with pytest.raises(ExperimentBusyError) as caught:
        app.launch("srv")

    assert handle.run_id in str(caught.value)


def test_absent_runner_reads_as_a_dead_runner_that_needs_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its trials may outlive it, so no read concludes the run, and no read writes."""
    app, registry, store, handle, _proc = _stage_absent_run(tmp_path, monkeypatch)
    before = tree_snapshot(registry.state_dir)
    _forbid_reaping(monkeypatch)

    assert store.runner_view(handle).visibility == "absent"
    assert not store.runner_may_be_alive(handle)
    observation = store.observe(handle)
    assert observation.state == "running"
    assert observation.recovery_required
    assert live_runs(store) == [handle]
    run = app.status(run_id=handle.run_id)["run"]
    assert run["state"] == "running"
    assert run["recovery_required"] is True
    assert run["runner_observable"] is True
    assert tree_snapshot(registry.state_dir) == before


def test_a_blocker_between_reads_never_reads_failed_and_never_unsettles_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absence proof is not monotonic, so only recover-run settles, and its status outlives any later proof."""
    app, registry, store, handle, proc = _stage_absent_run(tmp_path, monkeypatch)
    _forbid_run_signals(monkeypatch, mcp_recovery, mcp_tools, reaper)

    first = app.status(run_id=handle.run_id)["run"]
    _add_runner_blocker(proc, handle)
    second = app.status(run_id=handle.run_id)["run"]

    assert (first["state"], first["recovery_required"], first["runner_observable"]) == (
        "running",
        True,
        True,
    )
    assert (second["state"], second["runner_observable"]) == ("running", False)
    assert live_runs(store) == [handle]

    shutil.rmtree(proc / "900")
    recover_run(registry.state_dir, handle.run_id, confirm=True, emit=lambda _m: None)
    assert store.observe(handle).state == "failed"

    _add_runner_blocker(proc, handle)
    settled = app.status(run_id=handle.run_id)["run"]
    assert settled["state"] == "failed"
    assert settled["recovery_required"] is False


@pytest.mark.parametrize("tool", ["status", "await_run", "winners", "latest_run"])
def test_foreign_namespace_read_tools_leave_state_dir_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    app, registry, _store, handle = _stage_foreign_run(tmp_path, monkeypatch)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)
    before = tree_snapshot(registry.state_dir)

    if tool == "await_run":
        payload = asyncio.run(app.await_run(handle.run_id, timeout_seconds=0))
    elif tool == "winners":
        payload = app.winners(run_id=handle.run_id)
    elif tool == "latest_run":
        payload = app.latest_run("srv")
    else:
        payload = app.status(run_id=handle.run_id)

    assert tree_snapshot(registry.state_dir) == before
    if tool != "winners":
        run = payload["run"]
        assert run["state"] == "running"
        assert run["recovery_required"] is False
        assert run["runner_observable"] is False


def test_a_read_derives_the_runner_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every field of one status read rests on the same derivation and one /proc scan."""
    app, _registry, store, handle, _proc = _stage_absent_run(tmp_path, monkeypatch)
    derivations: list[str] = []
    original = store.runner_view

    def count(saved: RunHandle) -> mcp_runs.RunnerView:
        view = original(saved)
        derivations.append(view.visibility)
        return view

    monkeypatch.setattr(store, "runner_view", count)

    app.status(run_id=handle.run_id)

    assert derivations == ["absent"]


def test_observable_run_reports_runner_observable(tmp_path: Path) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    handle = stage_live_run(store, "srv-here", config, registry.get("srv").id)

    run = app.status(run_id=handle.run_id)["run"]

    assert run["state"] == "running"
    assert run["runner_observable"] is True


def test_finished_foreign_run_reports_runner_observable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a running run can be unobservable: a finished one needs no runner."""
    app, _registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch)
    _write_finished_status(store, handle.run_id)

    run = app.status(run_id=handle.run_id)["run"]

    assert run["state"] == "succeeded"
    assert run["runner_observable"] is True


def test_cancel_refuses_foreign_namespace_without_signalling_or_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch, allow_cancel=True)
    audit_path = registry.state_dir / "audit.jsonl"
    app = PhaseSweepMCP(registry, store, audit=AuditLogger(audit_path))
    _forbid_run_signals(monkeypatch, mcp_tools)

    with pytest.raises(RunNamespaceMismatchError, match="PID namespace") as caught:
        app.cancel(handle.run_id)

    assert "nothing was signalled" in str(caught.value)
    assert "await_run" in str(caught.value)
    assert str(registry.state_dir) not in str(caught.value)
    assert not store.cleanup_uncertain_path(handle.run_id).exists()
    assert store.observe(handle).state == "running"
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [(event["tool"], event["outcome"], event["error_type"]) for event in events] == [
        (TOOL_CANCEL_RUN, "error", "RunNamespaceMismatchError")
    ]


def test_cancel_of_an_absent_runner_signals_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its group is proven gone; the runner's own status confirms its trials' cleanup."""
    app, _registry, store, handle, _proc = _stage_absent_run(
        tmp_path, monkeypatch, allow_cancel=True
    )
    write_run_status(store, handle.run_id, returncode=0, result_snapshot_state="pending")
    _forbid_run_signals(monkeypatch, mcp_tools)

    result = app.cancel(handle.run_id)

    assert result["cleanup_confirmed"] is True
    assert not store.cleanup_uncertain_path(handle.run_id).exists()


@pytest.mark.parametrize("reader", [NESTED_READER, INITIAL_READER])
def test_recover_run_refuses_a_runner_it_cannot_prove_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    _app, registry, _store, handle = _stage_foreign_run(
        tmp_path, monkeypatch, cleanup_uncertain=True, reader=reader
    )
    if reader == INITIAL_READER:
        _add_runner_blocker(_fake_proc(tmp_path / "fake", monkeypatch), handle)
    _forbid_run_signals(monkeypatch, mcp_recovery, reaper)
    before = tree_snapshot(registry.state_dir)

    for confirm in (False, True):
        with pytest.raises(RunRecoveryError) as caught:
            recover_run(registry.state_dir, handle.run_id, confirm=confirm, emit=lambda _m: None)
        message = str(caught.value)
        assert caught.value.action is OperatorAction.FIX_CONFIG
        assert RECORDED in message
        assert reader in message
        assert "kill -9" not in message
        if reader == INITIAL_READER:
            assert "host process(es) 900" in message

    assert tree_snapshot(registry.state_dir) == before


def test_recover_run_settles_an_absent_runner_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, registry, store, handle, _proc = _stage_absent_run(
        tmp_path, monkeypatch, cleanup_uncertain=True
    )
    _forbid_run_signals(monkeypatch, mcp_recovery, mcp_tools, reaper)
    monkeypatch.setattr(
        mcp_recovery,
        "is_same_live_process",
        lambda *_args: pytest.fail("a PID from another namespace must never be probed"),
    )
    before = tree_snapshot(registry.state_dir)
    messages: list[str] = []

    recover_run(registry.state_dir, handle.run_id, confirm=False, emit=messages.append)

    assert messages
    assert tree_snapshot(registry.state_dir) == before

    recover_run(registry.state_dir, handle.run_id, confirm=True, emit=messages.append)

    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["error_class"] == "RunnerExitedWithoutStatus"
    assert terminal["cleanup_confirmed"] is True
    assert not store.cleanup_uncertain_path(handle.run_id).exists()
    observation = store.observe(handle)
    assert observation.state == "failed"
    assert not observation.recovery_required


def test_spawn_refuses_unreadable_pid_namespace_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, app, _registry, store = side_effect_app(tmp_path)
    monkeypatch.setattr(mcp_tools, "read_pid_namespace", lambda: None)
    monkeypatch.setattr(
        "phasesweep.mcp.tools.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("no child may exist without a recorded namespace"),
    )

    with pytest.raises(RuntimeError, match="PID namespace"):
        app.launch("srv")

    assert [h for h in store.list_handles() if h.launch_state == "spawned"] == []


def test_launch_records_the_servers_pid_namespace_in_the_spawned_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, app, _registry, store = side_effect_app(tmp_path)
    patch_popen_capture(monkeypatch)

    run_id = app.launch("srv")["run_id"]

    handle = store.get(run_id)
    assert handle is not None
    assert handle.pid_namespace == read_pid_namespace()


def test_runner_receipt_from_another_namespace_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server compares the runner's receipt with its own view of the child, namespace included."""
    _config, app, _registry, store = side_effect_app(tmp_path)
    captured = patch_popen_capture(monkeypatch)
    monkeypatch.setattr("tests.mcp_helpers.read_pid_namespace", lambda: RECORDED)
    monkeypatch.setattr(mcp_tools, "kill_stale_group", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="did not match"):
        app.launch("srv")

    run_id = captured["cmd"][captured["cmd"].index("--run-id") + 1]
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["returncode"] == 1


def test_runner_refuses_to_persist_a_receipt_without_a_pid_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _app, registry, store = side_effect_app(tmp_path)
    claim_runner_handle(
        store,
        run_id="srv-no-namespace",
        config_sha256="0" * 64,
        config_dir=tmp_path,
        started_at=utc_now_iso(),
    )
    monkeypatch.setattr(mcp_runner, "read_pid_namespace", lambda: None)

    with pytest.raises(RuntimeError, match="PID namespace"):
        mcp_runner._persist_spawned_handle(
            state_dir=registry.state_dir,
            run_id="srv-no-namespace",
            experiment_id="srv",
            config_sha256="0" * 64,
            started_at=utc_now_iso(),
            allow_cancel=False,
        )

    pending = store.get("srv-no-namespace")
    assert pending is not None
    assert pending.launch_state == "launching"


def test_server_startup_requires_a_readable_pid_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_registry, "read_pid_namespace", lambda: None)

    with pytest.raises(CatalogError, match="PID namespace"):
        require_linux_mcp_host()
