"""PID-namespace binding of process identities: reading, the verdict, and stale cleanup."""

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
import phasesweep.mcp.run_control as mcp_run_control
import phasesweep.mcp.runner as mcp_runner
import phasesweep.mcp.runs as mcp_runs
import phasesweep.runtime.reaper as reaper
from phasesweep.errors import ForeignPidNamespaceError, OperatorAction
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import CatalogError, ExperimentBusyError, RunNamespaceMismatchError
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.registry import Registry, require_linux_mcp_host
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.tool_names import TOOL_CANCEL_RUN
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.process import _trial_process_identity, _write_process_identity
from phasesweep.runtime.reaper import (
    INITIAL_PID_NAMESPACE,
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    cleanup_stale_trial_process,
    foreign_pid_namespace_reason,
    pid_namespace_verdict,
    read_pid_namespace,
    read_stale_process_identity,
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

RECORDED = "pid:[4026539001]"
OTHER = "pid:[4026539002]"
NESTED_READER = "pid:[4026539333]"


def _write_process(proc: Path, pid: int, *, nspid: list[int] | None, namespace: str | None) -> None:
    """Add one fake ``/proc/<pid>`` entry with an optional ``NSpid`` line and PID namespace link."""
    entry = proc / str(pid)
    (entry / "ns").mkdir(parents=True)
    lines = ["Name:\tfake", "State:\tS (sleeping)"]
    if nspid is not None:
        lines.append("NSpid:\t" + "\t".join(str(value) for value in nspid))
    (entry / "status").write_text("\n".join(lines) + "\n")
    if namespace is not None:
        (entry / "ns" / "pid").symlink_to(namespace)


def _fake_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reader: str | None = INITIAL_PID_NAMESPACE
) -> Path:
    """Replace ``/proc`` with a readable init entry, and the reader's namespace with ``reader``."""
    proc = tmp_path / "proc"
    _write_process(proc, 1, nspid=[1], namespace=None)
    monkeypatch.setattr(reaper, "_PROC", proc)
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: reader)
    return proc


def _identity(*, pid_namespace: str | None, boot_id: str | None = "boot") -> StaleProcessIdentity:
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id="ns-attempt",
        pid=4242,
        pgid=4242,
        proc_starttime=111,
        boot_id=boot_id,
        pid_namespace=pid_namespace,
    )


def _forbid_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reaper,
        "kill_stale_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a process outside this PID namespace must never be signalled")
        ),
    )


def test_read_pid_namespace_names_this_process_namespace() -> None:
    namespace = read_pid_namespace()
    if namespace is None:
        pytest.skip("/proc is not this process's own procfs")

    assert namespace == os.readlink("/proc/self/ns/pid")
    assert reaper.is_pid_namespace(namespace)


def test_read_pid_namespace_refuses_procfs_of_another_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A procfs whose ``self`` is not this PID cannot answer for this namespace's numbers."""
    proc = tmp_path / "proc"
    _write_process(proc, os.getpid(), nspid=None, namespace=RECORDED)
    monkeypatch.setattr(reaper, "_PROC", proc)

    (proc / "self").symlink_to(str(os.getpid() + 1))
    assert read_pid_namespace() is None

    (proc / "self").unlink()
    (proc / "self").symlink_to(str(os.getpid()))
    assert read_pid_namespace() == RECORDED


def test_read_pid_namespace_rejects_a_malformed_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    _write_process(proc, os.getpid(), nspid=None, namespace="net:[4026539001]")
    (proc / "self").symlink_to(str(os.getpid()))
    monkeypatch.setattr(reaper, "_PROC", proc)

    assert read_pid_namespace() is None


def test_verdict_same_and_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)

    assert pid_namespace_verdict(NESTED_READER) == "same"
    assert pid_namespace_verdict(RECORDED) == "foreign"
    assert pid_namespace_verdict(None) == "foreign"
    assert pid_namespace_verdict("not-a-namespace") == "foreign"

    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: None)
    assert pid_namespace_verdict(RECORDED) == "foreign"


def test_verdict_gone_only_from_initial_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 50, nspid=[50, 1], namespace=OTHER)
    _write_process(proc, 51, nspid=[51], namespace=INITIAL_PID_NAMESPACE)

    assert pid_namespace_verdict(RECORDED) == "gone"
    assert pid_namespace_verdict(OTHER) == "foreign"

    # A reader inside a nested namespace sees only its own subtree of /proc, so
    # an absent entry proves nothing there.
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)
    assert pid_namespace_verdict(RECORDED) == "foreign"


def test_gone_scan_ignores_a_process_that_exits_during_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    (proc / "77").mkdir()
    _write_process(proc, 78, nspid=[78, 2], namespace=OTHER)

    assert pid_namespace_verdict(RECORDED) == "gone"


@pytest.mark.parametrize(
    "case",
    ["unreadable_link", "missing_link", "missing_nspid"],
)
def test_gone_scan_treats_an_uninspectable_entry_as_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    if case == "missing_nspid":
        # A kernel older than 4.1 has no NSpid line, so nesting cannot be read.
        _write_process(proc, 60, nspid=None, namespace=OTHER)
    else:
        _write_process(proc, 60, nspid=[60, 3], namespace=None)
        if case == "unreadable_link":
            # A regular file where the namespace link belongs is unreadable as a link.
            (proc / "60" / "ns" / "pid").write_text("not a link")

    assert pid_namespace_verdict(RECORDED) == "foreign"


def test_gone_scan_refuses_a_procfs_that_hides_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a readable init entry, other users' processes may be hidden from the scan."""
    proc = _fake_proc(tmp_path, monkeypatch)
    shutil.rmtree(proc / "1")

    assert pid_namespace_verdict(RECORDED) == "foreign"


def test_foreign_reason_names_both_namespaces_and_the_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)

    reason = foreign_pid_namespace_reason(RECORDED)

    assert RECORDED in reason
    assert NESTED_READER in reason
    assert "initial PID namespace" in reason


def test_foreign_reason_from_the_initial_namespace_names_what_blocks_the_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path, monkeypatch)
    _write_process(proc, 60, nspid=[60, 3], namespace=None)
    (proc / "60" / "ns" / "pid").write_text("not a link")

    assert "host process 60" in foreign_pid_namespace_reason(RECORDED)

    shutil.rmtree(proc / "1")
    assert "hides other users' processes" in foreign_pid_namespace_reason(RECORDED)


def test_trial_process_identity_records_the_launching_namespace(tmp_path: Path) -> None:
    namespace = read_pid_namespace()
    if namespace is None:
        pytest.skip("/proc is not this process's own procfs")

    identity = _trial_process_identity(attempt_id="ns-attempt", pid=os.getpid(), pgid=os.getpid())
    _write_process_identity(tmp_path / PROCESS_IDENTITY_FILE, identity)

    assert identity.pid_namespace == namespace
    assert json.loads((tmp_path / PROCESS_IDENTITY_FILE).read_text())["pid_namespace"] == namespace
    assert read_stale_process_identity(tmp_path, expected_attempt_id="ns-attempt") == identity


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

    record["pid_namespace"] = "pid:[0]"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="pid_namespace"):
        read_stale_process_identity(tmp_path, expected_attempt_id="ns-attempt")

    record["pid_namespace"] = None
    path.write_text(json.dumps(record))
    unrecorded = read_stale_process_identity(tmp_path, expected_attempt_id="ns-attempt")
    assert unrecorded.pid_namespace is None


def test_cleanup_stale_trial_process_refuses_foreign_namespace_without_signalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)
    _forbid_signals(monkeypatch)

    with pytest.raises(ForeignPidNamespaceError) as caught:
        cleanup_stale_trial_process(_identity(pid_namespace=RECORDED))

    message = str(caught.value)
    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert RECORDED in message
    assert NESTED_READER in message
    assert "Nothing was signalled" in message
    assert "kill -9" not in message


def test_cleanup_stale_trial_process_settles_gone_namespace_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    _fake_proc(tmp_path, monkeypatch)
    _forbid_signals(monkeypatch)

    assert cleanup_stale_trial_process(_identity(pid_namespace=RECORDED)) is True


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


def test_cleanup_stale_trial_process_refuses_an_identity_without_a_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reaper, "read_boot_id", lambda: "boot")
    _forbid_signals(monkeypatch)

    assert cleanup_stale_trial_process(_identity(pid_namespace=None)) is False


_HOLD_NAMESPACE = "readlink /proc/self/ns/pid; exec sleep 120"

#: Ways to start a process in a fresh PID namespace with its own procfs. The first
#: needs unprivileged user namespaces; the second works where only bubblewrap may
#: create them. Both die with the launcher process.
_NAMESPACE_LAUNCHERS = (
    ("unshare", "-Urpf", "--kill-child", "--mount-proc"),
    ("bwrap", "--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/", "--proc", "/proc"),
)


def _blocker_note() -> str:
    """Name the host process that stops a scan from proving an absent namespace empty."""
    return f"host process {reaper._pid_namespace_blocker('pid:[1]')}"


def _start_namespace_holder() -> tuple[subprocess.Popen[str], str]:
    """Start a process in a new PID namespace and return it with that namespace's identity."""
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
        namespace = child.stdout.readline().strip()
        if reaper.is_pid_namespace(namespace):
            return child, namespace
        child.kill()
        refusals.append(f"{launcher[0]}: {child.stderr.read().strip()}")
        child.wait(timeout=10)
    pytest.skip("cannot create a PID namespace here: " + "; ".join(refusals))


@pytest.mark.integration
def test_verdict_tracks_a_real_child_pid_namespace() -> None:
    """A namespace with a live member reads foreign; once its init dies it reads gone."""
    if read_pid_namespace() != INITIAL_PID_NAMESPACE:
        pytest.skip("the gone verdict needs the initial PID namespace")
    child, namespace = _start_namespace_holder()
    try:
        assert pid_namespace_verdict(read_pid_namespace()) == "same"
        assert pid_namespace_verdict(namespace) == "foreign"

        child.kill()
        child.wait(timeout=10)
        # Inode 1 is never a namespace. If even that is unprovable, an unrelated
        # process (a browser sandbox is not dumpable) hides its namespace from
        # this user, and the scan correctly refuses to call anything empty.
        if pid_namespace_verdict("pid:[1]") != "gone":
            pytest.skip(f"this host cannot prove any namespace empty: {_blocker_note()}")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and pid_namespace_verdict(namespace) != "gone":
            time.sleep(0.05)
        assert pid_namespace_verdict(namespace) == "gone"
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
) -> tuple[PhaseSweepMCP, Registry, RunStore, RunHandle]:
    """Serve the standard experiment from a nested PID namespace with one run recorded in another."""
    config, app, registry, store = side_effect_app(tmp_path)
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)
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


def test_foreign_namespace_runner_reads_running_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch)
    before = tree_snapshot(registry.state_dir)
    monkeypatch.setattr(
        mcp_runs,
        "reap_child",
        lambda _pid: pytest.fail("a PID from another namespace must never be waited on"),
    )

    assert store.runner_visibility(handle) == "unobservable"
    assert store.runner_may_be_alive(handle)
    assert store.state(handle) == "running"
    assert not store.recovery_required(handle)
    assert not store.snapshot_recovery_required(handle)
    assert live_runs(store) == [handle]
    assert tree_snapshot(registry.state_dir) == before


def test_foreign_namespace_terminal_status_still_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable evidence needs no probe, so a finished run reads terminal from anywhere."""
    _app, _registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch)
    _write_finished_status(store, handle.run_id)

    assert store.state(handle) == "succeeded"


def test_foreign_namespace_runner_holds_its_concurrency_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store, handle = _stage_foreign_run(tmp_path, monkeypatch)

    with pytest.raises(ExperimentBusyError) as caught:
        app.launch("srv")

    assert handle.run_id in str(caught.value)


def test_gone_namespace_settles_like_an_earlier_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, registry, store, handle = _stage_foreign_run(
        tmp_path, monkeypatch, cleanup_uncertain=True
    )
    _fake_proc(tmp_path / "fake", monkeypatch)
    before = tree_snapshot(registry.state_dir)

    assert store.runner_visibility(handle) == "gone"
    assert not store.runner_may_be_alive(handle)
    assert store.state(handle) == "failed"
    assert not store.recovery_required(handle)
    assert live_runs(store) == []
    assert tree_snapshot(registry.state_dir) == before


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
    _forbid_run_signals(monkeypatch, mcp_run_control)

    with pytest.raises(RunNamespaceMismatchError, match="PID namespace") as caught:
        app.cancel(handle.run_id)

    assert "nothing was signalled" in str(caught.value)
    assert str(registry.state_dir) not in str(caught.value)
    assert not store.cleanup_uncertain_path(handle.run_id).exists()
    assert store.state(handle) == "running"
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [(event["tool"], event["outcome"], event["error_type"]) for event in events] == [
        (TOOL_CANCEL_RUN, "error", "RunNamespaceMismatchError")
    ]


def test_cancel_settles_gone_namespace_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reached, as the earlier-boot twin is, through an unfinalized terminal snapshot."""
    app, _registry, store, handle = _stage_foreign_run(tmp_path, monkeypatch, allow_cancel=True)
    write_run_status(store, handle.run_id, returncode=0, result_snapshot_state="pending")
    _fake_proc(tmp_path / "fake", monkeypatch)
    _forbid_run_signals(monkeypatch, mcp_run_control)

    result = app.cancel(handle.run_id)

    assert result["cleanup_confirmed"] is True


def test_recover_run_refuses_foreign_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, registry, _store, handle = _stage_foreign_run(
        tmp_path, monkeypatch, cleanup_uncertain=True
    )
    _forbid_run_signals(monkeypatch, mcp_recovery)
    before = tree_snapshot(registry.state_dir)

    for confirm in (False, True):
        with pytest.raises(RunRecoveryError) as caught:
            recover_run(registry.state_dir, handle.run_id, confirm=confirm, emit=lambda _m: None)
        assert caught.value.action is OperatorAction.FIX_CONFIG
        assert RECORDED in str(caught.value)
        assert NESTED_READER in str(caught.value)
        assert "kill -9" not in str(caught.value)

    assert tree_snapshot(registry.state_dir) == before


def test_recover_run_settles_gone_namespace_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, registry, store, handle = _stage_foreign_run(
        tmp_path, monkeypatch, cleanup_uncertain=True
    )
    _fake_proc(tmp_path / "fake", monkeypatch)
    _forbid_run_signals(monkeypatch, mcp_recovery)
    messages: list[str] = []

    recover_run(registry.state_dir, handle.run_id, confirm=False, emit=messages.append)

    assert messages
    assert "still appears live" not in "\n".join(messages)
    assert store.cleanup_uncertain_path(handle.run_id).is_file()


def test_spawn_refuses_unreadable_pid_namespace_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, app, _registry, store = side_effect_app(tmp_path)
    monkeypatch.setattr(mcp_run_control, "read_pid_namespace", lambda: None)
    monkeypatch.setattr(
        "phasesweep.mcp.run_control.subprocess.Popen",
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
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", lambda *_args, **_kwargs: True)

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
