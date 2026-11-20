"""MCP server cancel behavior. Logic that does not need a real detached runner."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import phasesweep.mcp.errors as mcp_errors
import phasesweep.mcp.runs as mcp_runs
import phasesweep.mcp.tools as mcp_tools
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.mcp.errors import (
    ExperimentBusyError,
    McpToolError,
    PermissionDeniedError,
    RunLaunchUnsettledError,
)
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runner import _cleanup_failure_payload, _cleanup_uncertain_payload
from phasesweep.mcp.runs import RunHandle, RunnerView, RunObservation, RunStore
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime import reaper
from phasesweep.runtime.reaper import (
    read_boot_id,
    read_pid_namespace,
)
from tests.conftest import (
    reaped_pid,
)
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    call_mcp_tool,
    earlier_boot_id,
    make_dead_run_handle,
    make_mcp_app,
    make_run_handle,
    side_effect_app,
    stage_dead_run,
    write_mcp_catalog,
    write_run_status,
    write_srv_catalog,
    write_srv_config,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import (
    recover_run_cli,
    write_uncertain_failed_trial,
)


def test_cancel_refuses_unsettled_launch_without_runner_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    pending = make_run_handle(
        run_id="srv-launch-gap",
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
        allow_cancel=True,
    )
    store.create(pending)

    with pytest.raises(RunLaunchUnsettledError, match="verified runner identity"):
        app.cancel(pending.run_id)

    assert not store.cleanup_uncertain_path(pending.run_id).exists()
    assert store.observe(pending).recovery_required

    runner_pid = reaped_pid()
    spawned = replace(
        pending,
        launch_state="spawned",
        pid=runner_pid,
        pgid=runner_pid,
        pid_starttime=111,
        boot_id=read_boot_id(),
        pid_namespace=read_pid_namespace(),
    )
    original_observe = store.observe
    first_read = True

    def complete_launch_after_observation(
        saved: RunHandle, *, view: RunnerView | None = None
    ) -> RunObservation:
        nonlocal first_read
        observation = original_observe(saved, view=view)
        if first_read:
            first_read = False
            store.update(spawned)
        return observation

    signalled: list[tuple[int | None, int | None, int | None]] = []

    def kill_runner(
        pid: int | None, starttime: int | None, *, pgid: int | None, grace_seconds: float
    ) -> bool:
        signalled.append((pid, starttime, pgid))
        return True

    monkeypatch.setattr(store, "observe", complete_launch_after_observation)
    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", kill_runner)

    with pytest.raises(McpToolError, match="recover-run"):
        app.cancel(pending.run_id)

    assert store.get(pending.run_id) == spawned
    assert signalled == []
    assert store.observe(spawned).recovery_required


def test_cancel_on_an_earlier_boot_confirms_cleanup_without_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reboot settles cleanup; the saved PGID now belongs to someone else.

    Reached through a run left ``running`` by an unfinalized terminal snapshot,
    which is the state a reboot mid-finalization leaves behind.
    """
    config, app, _registry, store = side_effect_app(tmp_path)
    handle = make_dead_run_handle(
        run_id="srv-boot",
        experiment_id="srv",
        allow_cancel=True,
        boot_id=earlier_boot_id(),
    )
    store.create(handle)
    write_run_status(store, "srv-boot", returncode=0, result_snapshot_state="pending")
    signalled: list[tuple[object, ...]] = []

    def record_kill(*args: object, **kwargs: object) -> bool:
        signalled.append(args)
        return True

    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", record_kill)
    assert store.observe(handle).state == "running"

    result = app.cancel("srv-boot")

    assert signalled == []
    assert result["cleanup_confirmed"] is True
    # The orphaned pending snapshot is a separate operator concern from process
    # cleanup, so it still asks for recovery.
    assert result["recovery_required"] is True


def test_cancel_refuses_to_signal_when_boot_identity_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host whose own boot id is unreadable cannot rule out PID reuse."""
    current_boot = read_boot_id()
    if current_boot is None:
        pytest.skip("boot id unavailable on this platform")
    config, app, _registry, store = side_effect_app(tmp_path)
    handle = make_dead_run_handle(
        run_id="srv-unknown-boot",
        experiment_id="srv",
        allow_cancel=True,
        boot_id=current_boot,
    )
    store.create(handle)
    monkeypatch.setattr(mcp_runs, "read_boot_id", lambda: None)
    monkeypatch.setattr(mcp_tools, "read_boot_id", lambda: None)
    signalled: list[object] = []

    def record_signal(*args: object, **kwargs: object) -> bool:
        signalled.append((args, kwargs))
        return True

    monkeypatch.setattr(mcp_tools, "kill_stale_group", record_signal)

    with pytest.raises(McpToolError, match="recover-run|PID namespace"):
        app.cancel(handle.run_id)

    assert signalled == []
    assert store.observe(handle).recovery_required


@pytest.mark.parametrize(
    ("catalog_allows_cancel", "run_allows_cancel"),
    [
        pytest.param(False, True, id="catalog-denies-cancel"),
        pytest.param(True, False, id="launch-snapshot-denies-cancel"),
    ],
)
def test_cancel_requires_catalog_and_launch_time_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    catalog_allows_cancel: bool,
    run_allows_cancel: bool,
) -> None:
    """Cancellation remains denied when either authority source forbids it."""
    monkeypatch.setattr(mcp_errors, "in_initial_pid_namespace", lambda: True)
    config = write_srv_config(tmp_path)
    allowed = ALLOW_SIDE_EFFECTS
    if not catalog_allows_cancel:
        allowed = {"launch": True, "cancel": False, "from_phase": True}
    app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=allowed),
    )
    reg = registry.get("srv")
    run_id = "srv-denied"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=run_allows_cancel,
    )
    store.create(handle)

    with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted") as denied:
        app.cancel(run_id)
    assert (
        f"To stop this run, the operator can run `kill -TERM -- -{handle.pgid}` from a host "
        f"shell, which signals runner PID {handle.pid} and its process group {handle.pgid}; "
        "the runner handles SIGTERM as cancellation. Do not retry."
    ) in str(denied.value)


def test_cancel_refusal_for_a_dead_runner_names_recovery_not_a_signal(tmp_path: Path) -> None:
    """A run whose runner is gone has no group to signal, so the refusal gives its recover-run step."""
    config, app, registry, store = side_effect_app(tmp_path)
    run_id = "srv-denied-dead"
    stage_dead_run(store, run_id, config, registry.get("srv").id, cleanup_uncertain=True)

    with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted") as denied:
        app.cancel(run_id)

    message = str(denied.value)
    assert "kill -TERM" not in message
    assert f"phasesweep mcp recover-run --state-dir STATE_DIR --run-id {run_id}" in message
    assert app.status(run_id=run_id)["run"]["recovery_required"] is True


def test_cancel_refusal_gives_no_pid_number_from_a_container_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server in a nested PID namespace finds the runner by run id, since its numbers name other processes on the host."""
    monkeypatch.setattr(mcp_errors, "in_initial_pid_namespace", lambda: False)
    config = write_srv_config(tmp_path)
    app, registry, store = make_mcp_app(
        write_srv_catalog(
            tmp_path,
            config,
            allow={"launch": True, "cancel": False, "from_phase": True},
        )
    )
    reg = registry.get("srv")
    handle = make_run_handle(
        run_id="srv-denied-nested",
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.create(handle)

    with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted") as denied:
        app.cancel(handle.run_id)

    message = str(denied.value)
    assert f"run `pgrep -af {handle.run_id}` from a host shell" in message
    assert str(handle.pid) not in message
    assert str(handle.pgid) not in message


@pytest.mark.parametrize("runner", ["foreign_namespace", "finished"])
def test_cancel_refusal_names_a_stop_route_only_it_can_give(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: str
) -> None:
    """A runner whose PID cannot be read here is found with pgrep; a finished run gets no stop route."""
    config = write_srv_config(tmp_path)
    app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    reg = registry.get("srv")
    run_id = "srv-denied"
    if runner == "foreign_namespace":
        monkeypatch.setattr(reaper, "read_pid_namespace", lambda: "pid:[4026539333]@100")
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            pid_namespace="pid:[4026539001]@100" if runner == "foreign_namespace" else None,
        )
    )
    if runner == "finished":
        write_run_status(store, run_id, returncode=0, result_snapshot_state="failed")

    with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted") as denied:
        app.cancel(run_id)

    message = str(denied.value)
    if runner == "finished":
        assert "To stop this run" not in message
    else:
        assert (
            f"To stop this run, the operator can run `pgrep -af {run_id}` from a host shell, take "
            f"the PID of its `runner.py --run-id {run_id}` line and run `kill -TERM -- -<that "
            "PID>`, since the runner leads its own process group; the runner handles SIGTERM as "
            "cancellation. Do not retry."
        ) in message


def test_cancel_decataloged_run_uses_launch_time_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A removed catalog entry must not strand a live detached runner.

    Status reads still resolve the run from its snapshot and report it running,
    and ``phasesweep mcp recover-run`` refuses while the runner is alive, so a
    hard deny here would leave no supported way to stop the sweep. There is no
    current policy to intersect with, and cancelling is risk-reducing.
    """
    old_config = write_srv_config(tmp_path, name="old")
    old_snapshot = old_config.read_bytes()
    other_config = write_srv_config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, allow=ALLOW_SIDE_EFFECTS)
    )
    run_id = "old-running"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id="old",
        config_sha256=hashlib.sha256(old_snapshot).hexdigest(),
        allow_cancel=True,
    )
    store.create(handle)
    store.mark_cleanup_uncertain(handle)
    assert store.observe(handle).state == "running"
    assert store.cleanup_uncertain_path(run_id).is_file()

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        write_run_status(
            store,
            run_id,
            returncode=143,
            error_class="cancelled",
            cleanup_confirmed=True,
        )
        return True

    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    assert result == {
        "run_id": run_id,
        "state": "cancelled",
        "cleanup_confirmed": True,
        "recovery_required": False,
    }
    assert not store.cleanup_uncertain_path(run_id).exists()


def test_cancel_decataloged_run_is_denied_without_launch_permission(tmp_path: Path) -> None:
    """Removing the entry cannot grant authority the launch never had."""
    old_config = write_srv_config(tmp_path, name="old")
    old_snapshot = old_config.read_bytes()
    other_config = write_srv_config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, allow=ALLOW_SIDE_EFFECTS)
    )
    run_id = "old-running-no-permission"
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id="old",
            config_sha256=hashlib.sha256(old_snapshot).hexdigest(),
            allow_cancel=False,
        )
    )

    with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted"):
        app.cancel(run_id)


@pytest.mark.parametrize("declared_cancel", [False, True])
def test_cancel_of_an_entry_that_failed_to_load_follows_its_declared_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared_cancel: bool
) -> None:
    """A failed entry keeps the ``allow.cancel`` its catalog declares, so a revocation holds and a grant stays usable."""
    config = write_srv_config(tmp_path, name="broken")
    snapshot = config.read_bytes()
    catalog = write_mcp_catalog(
        tmp_path,
        {"broken": config},
        allow={"launch": True, "cancel": declared_cancel, "from_phase": True},
        cwd={"broken": tmp_path / "missing-cwd"},
    )
    registry = Registry.load(catalog, collect_errors=True)
    assert [failure.experiment_id for failure in registry.failures] == ["broken"]
    store = RunStore(registry.state_dir)
    app = PhaseSweepMCP(registry, store)
    run_id = "broken-running"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id="broken",
        config_sha256=hashlib.sha256(snapshot).hexdigest(),
        allow_cancel=True,
    )
    store.create(handle)
    store.mark_cleanup_uncertain(handle)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        write_run_status(
            store, run_id, returncode=143, error_class="cancelled", cleanup_confirmed=True
        )
        return True

    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", fake_kill_stale_group)

    if declared_cancel:
        assert app.cancel(run_id)["state"] == "cancelled"
    else:
        with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted"):
            app.cancel(run_id)
        assert store.observe(handle).state == "running"


@pytest.mark.parametrize(
    (
        "runner_group_gone",
        "status_cleanup_confirmed",
        "preexisting_marker",
        "check_launch_gate",
        "initial_namespace",
    ),
    [
        pytest.param(False, None, False, True, True, id="runner-group-still-live"),
        pytest.param(False, None, False, True, False, id="runner-group-still-live-in-container"),
        pytest.param(True, None, False, True, True, id="forced-runner-kill-without-status"),
        pytest.param(True, False, False, False, True, id="runner-status-cleanup-unconfirmed"),
        pytest.param(True, True, True, False, True, id="runner-status-cleanup-confirmed"),
    ],
)
def test_cancel_cleanup_confirmation_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_group_gone: bool,
    status_cleanup_confirmed: bool | None,
    preexisting_marker: bool,
    check_launch_gate: bool,
    initial_namespace: bool,
) -> None:
    """Cancellation confirms cleanup or reserves it with a static remedy every reader retains."""
    monkeypatch.setattr(mcp_errors, "in_initial_pid_namespace", lambda: initial_namespace)
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-cancel-policy"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        config_dir=str(config.parent),
        allow_cancel=True,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    if preexisting_marker:
        store.mark_cleanup_uncertain(handle)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        assert store.cleanup_uncertain_path(run_id).is_file()
        if status_cleanup_confirmed is not None:
            write_run_status(
                store,
                run_id,
                returncode=143,
                error_class="cancelled",
                cleanup_confirmed=status_cleanup_confirmed,
            )
        return runner_group_gone

    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", fake_kill_stale_group)

    cleanup_confirmed = runner_group_gone and status_cleanup_confirmed is True
    if cleanup_confirmed:
        result = app.cancel(run_id)
        assert result == {
            "run_id": run_id,
            "state": "cancelled",
            "cleanup_confirmed": True,
            "recovery_required": False,
        }
    else:
        with pytest.raises(McpToolError, match="recover-run"):
            app.cancel(run_id)
    assert store.cleanup_uncertain_path(run_id).exists() is not cleanup_confirmed
    if not runner_group_gone:
        failure = json.loads(store.cleanup_uncertain_path(run_id).read_text())["failure"]
        remedy = failure["remediation"]
        if initial_namespace:
            assert f"runner PID {handle.pid} (process group {handle.pgid})" in remedy
        else:
            assert f"`pgrep -af {run_id}` from a host shell" in remedy
            assert str(handle.pid) not in remedy
            assert str(handle.pgid) not in remedy
        assert "is still running on the host, clear the cause there" in remedy
        assert "uninterruptible I/O ignores SIGKILL until the I/O ends" in remedy
        assert "cancel_run" not in remedy
        assert len(remedy.splitlines()) == 1
        for visibility in ("observable", "absent", "unobservable"):
            view = RunnerView(visibility)
            monkeypatch.setattr(store, "runner_view", lambda _handle, view=view: view)
            for tool, args in (
                ("get_run_status", {"run_id": run_id}),
                ("await_run", {"run_id": run_id, "timeout_seconds": 5}),
                ("get_run_results", {"run_id": run_id}),
                ("get_latest_run", {"experiment_id": reg.id}),
            ):
                result = call_mcp_tool(app, tool, args)
                assert not result.isError, result
                payload = result.structuredContent
                recorded = payload["run"]["failure"] if "run" in payload else payload["failure"]
                assert recorded["remediation"] == remedy
    if status_cleanup_confirmed is None:
        assert not store.status_path(run_id).exists()
    if check_launch_gate:
        with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
            app.launch("srv")


def test_dead_runner_cancel_retains_the_status_failure_for_other_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation fails with the status remedy and persists it for an unobservable reader."""
    config, app, _registry, store = side_effect_app(tmp_path)
    handle = stage_dead_run(
        store, "srv-dead-cancel", config, "srv", cleanup_uncertain=False, allow_cancel=True
    )
    failure = app.status(handle.run_id)["run"]["failure"]

    result = call_mcp_tool(app, "cancel_run", {"run_id": handle.run_id})

    assert result.isError
    assert "recover-run" in result.content[0].text
    assert failure["remediation"] in result.content[0].text
    marker = json.loads(store.cleanup_uncertain_path(handle.run_id).read_text())
    assert marker["failure"] == failure
    monkeypatch.setattr(store, "runner_view", lambda _handle: RunnerView("unobservable"))
    assert app.status(handle.run_id)["run"]["failure"] == failure
    again = call_mcp_tool(app, "cancel_run", {"run_id": handle.run_id})
    assert again.isError
    assert failure["remediation"] in again.content[0].text


@pytest.mark.parametrize("unobservable", [False, True], ids=["host", "sandbox"])
def test_terminal_cleanup_failure_precedes_the_cancellation_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unobservable: bool
) -> None:
    """Terminal storage remedies survive an older marker in both cancellation and reads."""
    config, app, _registry, store = side_effect_app(tmp_path)
    handle = stage_dead_run(
        store, "srv-terminal-failure", config, "srv", cleanup_uncertain=False, allow_cancel=True
    )
    specific = _cleanup_failure_payload(
        StudyStorageUnavailableError("ledger unavailable"),
        run_id=handle.run_id,
        cause_stage="execution",
    )
    store.mark_cleanup_uncertain(
        handle, failure=_cleanup_uncertain_payload([], run_id=handle.run_id)
    )
    write_unsafe_cleanup_status(store, handle.run_id, failure=specific)
    if unobservable:
        monkeypatch.setattr(store, "runner_view", lambda _handle: RunnerView("unobservable"))

    cancelled = call_mcp_tool(app, "cancel_run", {"run_id": handle.run_id})

    assert cancelled.isError
    assert specific["remediation"] in cancelled.content[0].text
    assert app.status(handle.run_id)["run"]["failure"] == specific


def test_live_cancellation_reservation_preserves_a_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live cancellation reserves cleanup without masking the runner's eventual diagnosis."""
    config, app, _registry, store = side_effect_app(tmp_path)
    snapshot = config.read_bytes()
    handle = make_run_handle(
        run_id="srv-live-failure",
        experiment_id="srv",
        allow_cancel=True,
        config_sha256=hashlib.sha256(snapshot).hexdigest(),
        config_dir=str(config.parent),
    )
    store.create(handle)
    store.config_snapshot_path(handle.run_id).write_bytes(snapshot)
    specific = _cleanup_failure_payload(
        StudyStorageUnavailableError("ledger unavailable"),
        run_id=handle.run_id,
        cause_stage="execution",
    )

    def record_terminal_failure(*_args: object, **_kwargs: object) -> bool:
        assert store.cleanup_failure(handle) is None
        write_unsafe_cleanup_status(store, handle.run_id, failure=specific)
        return True

    monkeypatch.setattr(mcp_tools, "kill_stale_group", record_terminal_failure)

    cancelled = call_mcp_tool(app, "cancel_run", {"run_id": handle.run_id})

    assert cancelled.isError
    assert specific["remediation"] in cancelled.content[0].text
    assert store.cleanup_failure(handle) is None
    assert app.status(handle.run_id)["run"]["failure"] == specific


def test_concurrent_cancel_calls_converge_on_the_same_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-concurrent-cancel"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=True,
    )
    store.create(handle)
    barrier = threading.Barrier(2)
    status_lock = threading.Lock()

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        barrier.wait(timeout=2.0)
        with status_lock:
            if not store.status_path(run_id).exists():
                write_run_status(
                    store,
                    run_id,
                    returncode=143,
                    error_class="cancelled",
                    cleanup_confirmed=True,
                )
        return True

    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", fake_kill_stale_group)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: app.cancel(run_id), range(2)))

    assert results == [
        {
            "run_id": run_id,
            "state": "cancelled",
            "cleanup_confirmed": True,
            "recovery_required": False,
        },
        {
            "run_id": run_id,
            "state": "cancelled",
            "cleanup_confirmed": True,
            "recovery_required": False,
        },
    ]
    assert not store.cleanup_uncertain_path(run_id).exists()


def test_cancel_does_not_resurrect_marker_after_operator_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_srv_config(tmp_path)
    run_id = "srv-cancel-after-recovery"
    write_uncertain_failed_trial(config, generation_id=run_id)
    app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    reg = registry.get("srv")
    handle = stage_dead_run(
        store, run_id, config, reg.id, cleanup_uncertain=True, allow_cancel=True
    )
    write_unsafe_cleanup_status(store, run_id)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process", lambda _identity, **_kwargs: True
    )

    read_running = threading.Event()
    recovery_done = threading.Event()
    original_observe = store.observe
    first_cancel_read = True

    def pause_cancel_observation(
        saved: RunHandle, *, view: RunnerView | None = None
    ) -> RunObservation:
        nonlocal first_cancel_read
        observation = original_observe(saved, view=view)
        if threading.current_thread() is not threading.main_thread() and first_cancel_read:
            first_cancel_read = False
            read_running.set()
            assert recovery_done.wait(timeout=10)
        return observation

    monkeypatch.setattr(store, "observe", pause_cancel_observation)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        cancelling = pool.submit(app.cancel, run_id)
        assert read_running.wait(timeout=10)
        confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)
        recovery_done.set()
        assert confirmed.exit_code == 0, confirmed.output
        result = cancelling.result(timeout=10)

    assert result["state"] == "failed"
    assert result["recovery_required"] is False
    assert not store.cleanup_uncertain_path(run_id).exists()
    assert store.observe(handle).state == "failed"
