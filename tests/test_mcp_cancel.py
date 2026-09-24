"""MCP server cancel behavior. Logic that does not need a real detached runner."""

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import phasesweep.mcp.run_control as mcp_run_control
import phasesweep.mcp.runs as mcp_runs
from phasesweep.mcp.errors import (
    ExperimentBusyError,
    PermissionDeniedError,
    RunLaunchUnsettledError,
)
from phasesweep.mcp.runs import RunHandle, RunState
from phasesweep.runtime.reaper import (
    read_boot_id,
)
from tests.conftest import (
    reaped_pid,
)
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    _catalog,
    _config,
    make_mcp_app,
    make_run_handle,
    stage_dead_run,
    write_mcp_catalog,
    write_run_status,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import (
    recover_run_cli,
    write_uncertain_failed_trial,
)


def test_cancel_refuses_unsettled_launch_without_runner_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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
    assert store.recovery_required(pending)

    runner_pid = reaped_pid()
    spawned = replace(
        pending,
        launch_state="spawned",
        pid=runner_pid,
        pgid=runner_pid,
        pid_starttime=111,
        boot_id=read_boot_id(),
    )
    original_state = store.state
    first_read = True

    def complete_launch_after_state_read(
        saved: RunHandle, *, transition_locked: bool = False
    ) -> RunState:
        nonlocal first_read
        state = original_state(saved, transition_locked=transition_locked)
        if first_read:
            first_read = False
            store.update(spawned)
        return state

    signalled: list[tuple[int | None, int | None, int | None]] = []

    def kill_runner(
        pid: int | None, starttime: int | None, *, pgid: int | None, grace_seconds: float
    ) -> bool:
        signalled.append((pid, starttime, pgid))
        return True

    monkeypatch.setattr(store, "state", complete_launch_after_state_read)
    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", kill_runner)

    cancelled = app.cancel(pending.run_id)

    assert store.get(pending.run_id) == spawned
    assert signalled == [(runner_pid, 111, runner_pid)]
    assert cancelled["state"] == "running"
    assert cancelled["recovery_required"] is True


def test_cancel_on_an_earlier_boot_confirms_cleanup_without_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reboot settles cleanup; the saved PGID now belongs to someone else.

    Reached through a run left ``running`` by an unfinalized terminal snapshot,
    which is the state a reboot mid-finalization leaves behind.
    """
    current_boot = read_boot_id()
    if current_boot is None:
        pytest.skip("boot id unavailable on this platform")
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    handle = replace(
        make_run_handle(
            run_id="srv-boot",
            experiment_id="srv",
            pid=reaped_pid(),
            starttime=111,
            allow_cancel=True,
        ),
        boot_id=(
            "00000000-0000-0000-0000-000000000000"
            if current_boot != "00000000-0000-0000-0000-000000000000"
            else "11111111-1111-1111-1111-111111111111"
        ),
    )
    store.create(handle)
    write_run_status(store, "srv-boot", returncode=0, result_snapshot_state="pending")
    signalled: list[tuple[object, ...]] = []

    def record_kill(*args: object, **kwargs: object) -> bool:
        signalled.append(args)
        return True

    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", record_kill)
    assert store.state(handle) == "running"

    result = app.cancel("srv-boot")

    assert signalled == []
    assert result["cleanup_confirmed"] is True
    # The orphaned pending snapshot is a separate operator concern from process
    # cleanup, so it still asks for recovery.
    assert result["recovery_required"] is True


@pytest.mark.parametrize("unknown_side", ["saved", "current"])
def test_cancel_refuses_to_signal_when_boot_identity_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown_side: str,
) -> None:
    current_boot = read_boot_id()
    if current_boot is None:
        pytest.skip("boot id unavailable on this platform")
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    handle = replace(
        make_run_handle(
            run_id="srv-unknown-boot",
            experiment_id="srv",
            pid=reaped_pid(),
            starttime=111,
            allow_cancel=True,
        ),
        boot_id=None if unknown_side == "saved" else current_boot,
    )
    store.create(handle)
    if unknown_side == "current":
        monkeypatch.setattr(mcp_runs, "read_boot_id", lambda: None)
        monkeypatch.setattr(mcp_run_control, "read_boot_id", lambda: None)
    signalled: list[object] = []

    def record_signal(*args: object, **kwargs: object) -> bool:
        signalled.append((args, kwargs))
        return True

    monkeypatch.setattr(mcp_run_control, "kill_stale_group", record_signal)

    result = app.cancel(handle.run_id)

    assert signalled == []
    assert result["state"] == "running"
    assert result["cleanup_confirmed"] is False
    assert result["recovery_required"] is True


@pytest.mark.parametrize(
    ("catalog_allows_cancel", "run_allows_cancel"),
    [
        pytest.param(False, True, id="catalog-denies-cancel"),
        pytest.param(True, False, id="launch-snapshot-denies-cancel"),
    ],
)
def test_cancel_requires_catalog_and_launch_time_permission(
    tmp_path: Path,
    catalog_allows_cancel: bool,
    run_allows_cancel: bool,
) -> None:
    """Cancellation remains denied when either authority source forbids it."""
    config = _config(tmp_path)
    allowed = ALLOW_SIDE_EFFECTS
    if not catalog_allows_cancel:
        allowed = {"launch": True, "cancel": False, "from_phase": True}
    app, registry, store = make_mcp_app(
        _catalog(tmp_path, config, allow=allowed),
    )
    reg = registry.get("srv")
    run_id = "srv-denied"
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            allow_cancel=run_allows_cancel,
        )
    )

    with pytest.raises(PermissionDeniedError, match="action 'cancel' is not permitted"):
        app.cancel(run_id)


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
    old_config = _config(tmp_path, name="old")
    old_snapshot = old_config.read_bytes()
    other_config = _config(tmp_path, name="other")
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
    assert store.state(handle) == "running"
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

    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", fake_kill_stale_group)

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
    old_config = _config(tmp_path, name="old")
    old_snapshot = old_config.read_bytes()
    other_config = _config(tmp_path, name="other")
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


@pytest.mark.parametrize(
    ("runner_group_gone", "status_cleanup_confirmed", "preexisting_marker", "check_launch_gate"),
    [
        pytest.param(False, None, False, True, id="runner-group-still-live"),
        pytest.param(True, None, False, True, id="forced-runner-kill-without-status"),
        pytest.param(True, False, False, False, id="runner-status-cleanup-unconfirmed"),
        pytest.param(True, True, True, False, id="runner-status-cleanup-confirmed"),
    ],
)
def test_cancel_cleanup_confirmation_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_group_gone: bool,
    status_cleanup_confirmed: bool | None,
    preexisting_marker: bool,
    check_launch_gate: bool,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-cancel-policy"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=True,
    )
    store.create(handle)
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

    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    cleanup_confirmed = runner_group_gone and status_cleanup_confirmed is True
    assert result == {
        "run_id": run_id,
        "state": "cancelled" if cleanup_confirmed else "running",
        "cleanup_confirmed": cleanup_confirmed,
        "recovery_required": not cleanup_confirmed,
    }
    assert store.cleanup_uncertain_path(run_id).exists() is not cleanup_confirmed
    if status_cleanup_confirmed is None:
        assert not store.status_path(run_id).exists()
    if check_launch_gate:
        with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
            app.launch("srv")


def test_concurrent_cancel_calls_converge_on_the_same_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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

    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", fake_kill_stale_group)

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
    config = _config(tmp_path)
    run_id = "srv-cancel-after-recovery"
    write_uncertain_failed_trial(config, generation_id=run_id)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = stage_dead_run(
        store, run_id, config, reg.id, cleanup_uncertain=True, allow_cancel=True
    )
    write_unsafe_cleanup_status(store, run_id)
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process", lambda _identity: True
    )

    read_running = threading.Event()
    recovery_done = threading.Event()
    original_state = store.state
    first_cancel_read = True

    def pause_cancel_state(saved: RunHandle, *, transition_locked: bool = False) -> RunState:
        nonlocal first_cancel_read
        state = original_state(saved, transition_locked=transition_locked)
        if threading.current_thread() is not threading.main_thread() and first_cancel_read:
            first_cancel_read = False
            read_running.set()
            assert recovery_done.wait(timeout=10)
        return state

    monkeypatch.setattr(store, "state", pause_cancel_state)
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
    assert store.state(handle) == "failed"
