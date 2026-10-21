"""MCP server recover-run and cleanup-uncertainty behavior. Logic that does not need a real detached runner."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple
from unittest.mock import DEFAULT, Mock

import optuna
import pytest
import yaml
from click.testing import Result

import phasesweep.engine.attempts as attempts_module
import phasesweep.mcp.recovery as mcp_recovery
import phasesweep.mcp.runs as mcp_runs
from phasesweep.config.io import load_experiment
from phasesweep.config.models import Experiment
from phasesweep.engine.artifact_roots import (
    _write_artifact_root_binding,
)
from phasesweep.engine.attempts import (
    _AttemptRecord,
    _preflight_active_attempts,
    _PreflightCleanupReport,
    _register_active_attempt,
    _register_tracking_operation,
    _retire_active_attempt,
    _tracking_cleanup_receipts,
)
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.guards import _reconcile_existing_studies
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _attempts_dir,
    _experiment_dir,
    _generation_record_path,
    _trial_dir_for,
)
from phasesweep.engine.run import run_experiment
from phasesweep.engine.state import (
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
)
from phasesweep.errors import OPERATOR_STEPS, OperatorAction
from phasesweep.mcp.errors import (
    ExperimentBusyError,
    RunLaunchUnsettledError,
)
from phasesweep.mcp.runs import MCP_STATE_FORMAT_VERSION, RunHandle, RunStore
from phasesweep.mcp.snapshots import capture_result_snapshot, finalize_result_snapshot
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.reaper import (
    StaleProcessIdentity,
    read_boot_id,
)
from phasesweep.tracking.publication import sync_tracking
from phasesweep.tracking.session import tracking_root
from tests.conftest import (
    mark_current_format,
    reaped_pid,
)
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    _catalog,
    _config,
    earlier_boot_id,
    live_runs,
    make_dead_run_handle,
    make_mcp_app,
    make_run_handle,
    patch_popen_capture,
    record_runner_cleanup,
    side_effect_app,
    stage_dead_run,
    write_run_status,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import (
    _new_phase_study,
    recover_run_cli,
    stamp_artifact_root,
    write_launched_stale_trial,
    write_trial_identity,
    write_uncertain_failed_trial,
)


def test_recovery_cli_renders_state_directory_remedy(tmp_path: Path) -> None:
    """An unsupported MCP store calls for a fresh state directory."""
    _config, _app, registry, store = side_effect_app(tmp_path)
    marker = store._format_marker_path
    marker.write_text(json.dumps({"schema_version": MCP_STATE_FORMAT_VERSION + 1}))
    before = marker.read_bytes()

    result = recover_run_cli(registry.state_dir, "refused-run", confirm=True)

    assert result.exit_code == 1
    assert "use a fresh MCP state directory" in result.output
    assert f"fix: {OPERATOR_STEPS[OperatorAction.FRESH_NAMESPACE]}." in result.output
    assert "artifact root" not in result.output
    assert marker.read_bytes() == before


def _interrupt_first_cleanup_clear() -> Callable[[RunStore, RunHandle], None]:
    """Return a cleanup-marker clear that fails once, then delegates normally."""
    real_clear = RunStore.clear_cleanup_uncertain
    clear_calls = 0

    def interrupt_first_clear(candidate_store: RunStore, candidate: RunHandle) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            # The marker's removal is filesystem work, so an interruption there
            # is an OSError recovery reports; a defect would propagate instead.
            raise OSError("interrupted before clearing cleanup marker")
        real_clear(candidate_store, candidate)

    return interrupt_first_clear


def _load_first_phase_study(config: Path) -> optuna.Study:
    exp = load_experiment(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    return optuna.load_study(
        study_name=f"{exp.experiment}::{phase.name}",
        storage=_resolve_storage(exp.resolved_storage),
    )


def _load_phase_trial(config: Path, trial_number: int) -> optuna.trial.FrozenTrial:
    study = _load_first_phase_study(config)
    return next(trial for trial in study.get_trials(deepcopy=False) if trial.number == trial_number)


def _stage_stale_running_recovery_scaffold(
    tmp_path: Path,
    *,
    run_id: str,
    error_class: str,
    include_generation_record: bool,
    mark_cleanup_uncertain: bool,
    snapshot_bound_to_generation: bool,
    kill_stale_group_stub: Callable[..., bool],
    cleanup_trial_stub: Callable[..., bool],
    monkeypatch: pytest.MonkeyPatch,
    allow_cancel: bool = False,
    earlier_boot: bool = False,
) -> tuple[PhaseSweepMCP, RunStore, RunHandle, str, Callable[[], Result]]:
    """Shared scaffold for the interrupted-recovery --confirm retry tests: a
    stale RUNNING trial, a run handle, and a terminal status with a captured
    ``result_snapshot``, with ``kill_stale_group``/``cleanup_stale_trial_process``
    stubbed to succeed. Callers monkeypatch their own fail-once target and
    call the returned ``recover`` twice. ``allow_cancel`` permits an intervening
    cancellation in a retry test. ``earlier_boot`` records both runner and
    trial identities under a different boot ID. Returns ``(app, store, handle,
    attempt_id, recover)``.
    """
    config = _config(tmp_path)
    recorded_boot_id = earlier_boot_id() if earlier_boot else None
    trial_number = write_launched_stale_trial(
        config,
        cleanup_confirmed=False,
        generation_id=run_id,
        boot_id=recorded_boot_id,
    )
    attempt_id = f"stale-attempt-{trial_number}"
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = make_dead_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=allow_cancel,
        boot_id=recorded_boot_id,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    if include_generation_record:
        generation_path = _generation_record_path(experiment, run_id)
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(yaml.safe_dump({"generation_id": run_id}))
    if mark_cleanup_uncertain:
        store.mark_cleanup_uncertain(handle)
    snapshot = (
        capture_result_snapshot(experiment, generation_id=run_id)
        if snapshot_bound_to_generation
        else capture_result_snapshot(experiment)
    )
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class=error_class,
        cleanup_confirmed=False,
        result_snapshot_state="complete",
        result_snapshot=snapshot,
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", kill_stale_group_stub)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        cleanup_trial_stub,
    )
    recover = partial(recover_run_cli, registry.state_dir, run_id, confirm=True)
    return app, store, handle, attempt_id, recover


def _stage_terminal_uncertain_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    mark_uncertain: bool,
) -> tuple[RunStore, RunHandle, int, Path, Path, Callable[[], Result]]:
    """Stage a run whose only cleanup evidence is one of its own
    cleanup-uncertain terminal trials (generation id == run id, matching the
    detached-runner contract). Returns ``(store, handle, trial_number,
    config, state_dir, recover)``."""
    config = _config(tmp_path)
    trial_number = write_uncertain_failed_trial(config, generation_id=run_id)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=mark_uncertain)
    write_unsafe_cleanup_status(store, run_id)

    def fake_cleanup(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", fake_cleanup)
    recover = partial(recover_run_cli, registry.state_dir, run_id, confirm=True)
    return store, handle, trial_number, config, registry.state_dir, recover


@pytest.mark.parametrize("interrupted_recovery", [False, True])
def test_operator_recovery_clears_abandoned_transactional_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted_recovery: bool
) -> None:
    """A free launch lease makes a persisted launch recoverable without losing its log."""
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-abandoned-preparation"
    pending = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
        allow_cancel=True,
    )
    preparation = store.prepare_launch(pending, config.read_bytes())
    with pytest.raises(RunLaunchUnsettledError, match="verified runner identity"):
        app.cancel(run_id)
    log_path = store.log_path(run_id)
    recovered_log = log_path.with_suffix(".log.recovered")
    log_bytes = b"runner failed before persisting its process identity\n"
    log_path.write_bytes(log_bytes)
    artifacts = (
        registry.state_dir / "runs" / f"{run_id}.json",
        store.config_snapshot_path(run_id),
        store.launch_lease_path(run_id),
        store._logs_dir / f"{run_id}.transition.lock",
        log_path,
    )
    original = {path: path.read_bytes() for path in artifacts}

    held = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert held.exit_code != 0
    assert "launch outcome is unresolved" in held.output
    assert {path: path.read_bytes() for path in artifacts} == original

    preparation.close()

    if interrupted_recovery:

        def interrupt_unlink(_path: Path) -> None:
            raise OSError("interrupted after archiving runner log")

        with monkeypatch.context() as patch:
            patch.setattr(mcp_runs, "_strict_unlink", interrupt_unlink)
            with pytest.raises(OSError, match="interrupted after archiving"):
                store.clear_pre_spawn_orphan(run_id)
        assert recovered_log.read_bytes() == log_bytes
        assert not log_path.exists()
        original.pop(log_path)
        original[recovered_log] = log_bytes

    preflight = recover_run_cli(registry.state_dir, run_id)

    assert preflight.exit_code == 0, preflight.output
    assert "no runner can still start a trainer under this identity" in preflight.output
    if interrupted_recovery:
        assert f"Runner log already preserved at {recovered_log}." in preflight.output
    else:
        assert f" runner log {log_path} as {recovered_log}." in preflight.output
        assert not recovered_log.exists()
    assert {path: path.read_bytes() for path in original} == original

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert "no runner can still start a trainer under this identity" in confirmed.output
    assert str(recovered_log) in confirmed.output
    assert all(not path.exists() for path in artifacts)
    assert recovered_log.read_bytes() == log_bytes
    assert store.launch_inventory() == ([], set())


@pytest.mark.parametrize(
    "terminal_evidence",
    ["status", "cleanup_uncertain", "cleanup_recovery"],
)
def test_operator_recovery_refuses_leased_preparation_with_terminal_evidence(
    tmp_path: Path,
    terminal_evidence: str,
) -> None:
    """Terminal evidence prevents a launch lease from proving a pre-spawn orphan."""
    config, app, registry, store = side_effect_app(tmp_path)
    del app
    reg = registry.get("srv")
    run_id = f"srv-leased-{terminal_evidence.replace('_', '-')}"
    pending = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    preparation = store.prepare_launch(pending, config.read_bytes())
    preparation.close()
    evidence_path = {
        "status": store.status_path(run_id),
        "cleanup_uncertain": store.cleanup_uncertain_path(run_id),
        "cleanup_recovery": store.cleanup_recovery_path(run_id),
    }[terminal_evidence]
    evidence_path.write_text(json.dumps({"run_id": run_id}) + "\n")
    artifacts = (
        registry.state_dir / "runs" / f"{run_id}.json",
        store.config_snapshot_path(run_id),
        store.launch_lease_path(run_id),
        evidence_path,
    )
    original = {path: path.read_bytes() for path in artifacts}

    result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code != 0
    assert "launch outcome is unresolved" in result.output
    assert {path: path.read_bytes() for path in artifacts} == original


def test_terminal_cleanup_uncertainty_blocks_relaunch(tmp_path: Path) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-terminal-uncertain"
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)
    write_unsafe_cleanup_status(store, run_id)

    assert store.observe(handle).state == "running"
    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")


def test_operator_recovery_clears_no_status_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-operator-recover"
    stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=True)

    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")

    dry = recover_run_cli(registry.state_dir, run_id)

    assert dry.exit_code == 0, dry.output
    assert "Re-run with --confirm" in dry.output
    assert "Recovery preflight" in dry.output
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.status_path(run_id).exists()

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert "Cleared cleanup uncertainty" in confirmed.output
    assert not store.cleanup_uncertain_path(run_id).exists()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["run_id"] == run_id
    assert recovery["config_sha256"] == reg.config_sha256
    assert recovery["cleanup_confirmed"] is True
    terminal_status = json.loads(store.status_path(run_id).read_text())
    assert terminal_status["error_class"] == "RunnerExitedWithoutStatus"
    assert terminal_status["cleanup_confirmed"] is True
    assert terminal_status["result_snapshot_state"] == "failed"
    assert terminal_status["result_snapshot_error"] == "HistoricalSnapshotUnavailable"
    unavailable = app.status(run_id=run_id)
    assert unavailable["result_source"] == "terminal_snapshot_unavailable"
    assert unavailable["run"]["failure"]["code"] == "result_snapshot_unavailable"
    assert unavailable["phases"][0]["trial_data_available"] is False

    captured = patch_popen_capture(monkeypatch)
    launched = app.launch("srv")
    assert launched["state"] == "running"
    assert captured["cmd"]


def test_recovery_preserves_status_written_after_initial_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-status-during-recovery"
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=True)
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    snapshot = capture_result_snapshot(experiment, generation_id=run_id)

    def runner_finishes_during_cleanup(*_args: object, **_kwargs: object) -> None:
        write_run_status(
            store,
            run_id,
            returncode=0,
            error_class=None,
            cleanup_confirmed=True,
            result_snapshot_state="complete",
            result_snapshot=snapshot,
        )

    monkeypatch.setattr(mcp_recovery, "_cleanup_runner", runner_finishes_during_cleanup)
    result = recover_run_cli(registry.state_dir, run_id, confirm=True)
    assert result.exit_code == 0, result.output
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["returncode"] == 0
    assert terminal["result_snapshot_state"] == "complete"


def test_operator_recovery_refuses_unknown_boot_process_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spawned handle always records its own boot id; only the current lookup can fail.

    A recovery-time boot id read failure still must not signal, since it
    cannot rule out PID reuse after a reboot.
    """
    current_boot = read_boot_id()
    if current_boot is None:
        pytest.skip("boot id unavailable on this platform")
    config, _app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-recovery-unknown-boot"
    stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=True)
    monkeypatch.setattr(mcp_runs, "read_boot_id", lambda: None)
    monkeypatch.setattr(mcp_recovery, "read_boot_id", lambda: None, raising=False)
    signal_group = Mock(return_value=True)
    monkeypatch.setattr(mcp_recovery, "kill_stale_group", signal_group)

    for confirmed in (False, True):
        result = recover_run_cli(registry.state_dir, run_id, confirm=confirmed)
        assert result.exit_code != 0
        assert "boot id" in f"{result.output} {result.exception}".lower()

    signal_group.assert_not_called()
    assert store.cleanup_uncertain_path(run_id).is_file()


def test_operator_recovery_skips_liveness_and_signalling_for_earlier_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PID recycled after a reboot must not fake runner liveness or absorb
    the recovery group signal: a recorded boot id from an earlier boot proves
    the runner and its descendants are gone.

    The handle deliberately records this test process's own live PID and start
    time — the strongest possible "looks live" collision — under a foreign
    boot id.
    """
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-earlier-boot"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        boot_id="00000000-0000-0000-0000-000000000000",
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.mark_cleanup_uncertain(handle)

    signal_group = Mock(return_value=True)
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", signal_group)

    dry = recover_run_cli(registry.state_dir, run_id)
    assert dry.exit_code == 0, dry.output
    assert "still appears live" not in dry.output

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)
    assert confirmed.exit_code == 0, confirmed.output
    assert "Cleared cleanup uncertainty" in confirmed.output
    signal_group.assert_not_called()
    assert not store.cleanup_uncertain_path(run_id).exists()


def test_earlier_boot_runner_without_status_never_reads_later_shared_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebooted no-status run frees capacity without borrowing later results."""
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-earlier-boot-no-status"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        boot_id=earlier_boot_id(),
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    later_trial = write_launched_stale_trial(config, generation_id="srv-later-run")

    assert _load_phase_trial(config, later_trial).state == optuna.trial.TrialState.RUNNING
    observation = store.observe(handle)
    assert observation.state == "failed"
    assert not observation.recovery_required
    assert live_runs(store) == []

    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)
    status = app.status(run_id=run_id)
    winners = app.winners(run_id=run_id)
    awaited = asyncio.run(app.await_run(run_id, timeout_seconds=0))

    failure = {
        "code": "result_snapshot_unavailable",
        "stage": "cleanup",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            "Report that this run's historical results are unavailable and ask "
            "the operator to inspect the PhaseSweep run or server diagnostics; "
            "do not substitute mutable experiment-level results."
        ),
    }
    for payload in (status, awaited):
        assert payload["result_source"] == "terminal_snapshot_unavailable"
        assert payload["represented_generation_id"] is None
        assert payload["run"]["state"] == "failed"
        assert payload["run"]["recovery_required"] is False
        assert payload["run"]["failure"] == failure
        assert payload["phases"][0]["running_trials_total"] is None
        assert payload["phases"][0]["trial_data_available"] is False
    assert winners["result_source"] == "terminal_snapshot_unavailable"
    assert winners["represented_generation_id"] is None
    assert winners["failure"] == failure
    assert app.latest_run("srv")["run"]["failure"] == failure
    assert awaited["reason"] == "terminal"


def test_operator_recovery_refuses_engine_lock_contention_before_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-recovery-lock-contention"
    stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=True)
    cleanup = Mock(return_value=True)
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", cleanup)
    with _experiment_lock(reg.experiment):
        result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code != 0
    assert "Another phasesweep process" in result.output
    assert "retry the same operation" in result.output
    assert "start a new run" not in result.output
    cleanup.assert_not_called()
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.cleanup_recovery_path(run_id).exists()
    assert not store.status_path(run_id).exists()


def test_operator_recovery_finalizes_orphaned_pending_snapshot(tmp_path: Path) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-pending-finalization"
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="pending",
    )

    observation = store.observe(handle)
    assert observation.state == "running"
    assert observation.recovery_required
    status = app.status(run_id=run_id)
    winners = app.winners(run_id=run_id)
    awaited = asyncio.run(app.await_run(run_id))

    for payload in (status, awaited):
        assert payload["result_source"] == "terminal_snapshot_unavailable"
        assert payload["publication_integrity"] == "unknown"
        assert payload["represented_generation_id"] is None
        assert payload["run"]["state"] == "running"
        assert payload["run"]["recovery_required"] is True
        assert payload["run"]["failure"]["code"] == "result_snapshot_unavailable"
    assert winners["result_source"] == "terminal_snapshot_unavailable"
    assert winners["publication_integrity"] == "unknown"
    assert winners["winner_count"] == 0
    assert awaited["reason"] == "recovery_required"
    preflight = recover_run_cli(registry.state_dir, run_id)
    assert preflight.exit_code == 0, preflight.output
    assert "historical terminal snapshot is unavailable" in preflight.output

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "failed"
    assert terminal["result_snapshot_error"] == "InterruptedFinalization"
    # The engine exit status stays authoritative: an unavailable frozen
    # snapshot degrades result reads (below), never the run outcome itself.
    observation = store.observe(handle)
    assert observation.state == "succeeded"
    assert not observation.recovery_required
    assert live_runs(store) == []
    unavailable = app.status(run_id=run_id)
    assert unavailable["result_source"] == "terminal_snapshot_unavailable"
    assert unavailable["run"]["state"] == "succeeded"
    assert unavailable["run"]["failure"]["code"] == "result_snapshot_unavailable"


def test_operator_recovery_keeps_unresolved_launch_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-launch-failed"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())

    def unexpected_cleanup(*args: object, **kwargs: object) -> bool:
        raise AssertionError("pre-spawn launch failure must not run process cleanup")

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", unexpected_cleanup)

    result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code != 0
    assert "launch outcome is unresolved" in result.output
    assert "remains reserved" in result.output
    assert "automated recovery cannot safely" in result.output
    assert "retry the same operation" in result.output
    assert "start a new run" not in result.output
    assert not store.cleanup_recovery_path(run_id).exists()
    assert not store.status_path(run_id).exists()
    observation = store.observe(handle)
    assert observation.state == "running"
    assert observation.recovery_required
    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")


def test_operator_recovery_reconciles_registry_attempt_when_storage_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial supervisor can outlive its runner in a separate process group.

    With no loadable Optuna study, the experiment-level attempt registry is
    the only remaining authority. Recovery must inspect it in dry-run mode,
    refuse to clear the run while its process cleanup is uncertain, and only
    finalize after the registered supervisor is confirmed gone.
    """
    config = _config(tmp_path)
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-registry-only-recovery"
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)

    mark_current_format(experiment)
    attempt_id = "registry-only-attempt"
    trial_dir = _experiment_dir(experiment) / "p" / "trial_registry_only"
    trial_dir.mkdir(parents=True)
    write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    write_trial_identity(
        trial_dir,
        attempt_id=attempt_id,
        pid=reaped_pid(),
        starttime=222,
    )
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name="removed-phase",
        study_name="srv::removed-phase",
        trial_number=0,
        trial_dir=trial_dir,
        generation_id=run_id,
    )
    entry_path = _attempts_dir(experiment) / f"{attempt_id}.json"

    runner_cleanup = Mock(return_value=True)
    trial_cleanup = Mock(return_value=False)
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", runner_cleanup)
    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", trial_cleanup)

    dry = recover_run_cli(registry.state_dir, run_id)

    assert dry.exit_code == 0, dry.output
    assert "reconcile 1 registered attempt" in dry.output
    runner_cleanup.assert_not_called()
    trial_cleanup.assert_not_called()

    refused = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert refused.exit_code != 0
    assert "process cleanup could not be confirmed" in refused.output
    runner_cleanup.assert_called_once()
    trial_cleanup.assert_called_once()
    assert entry_path.is_file()
    assert not store.status_path(run_id).exists()
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.observe(handle).state == "running"
    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")

    trial_cleanup.return_value = True
    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert "reconciled 1 registered attempt" in confirmed.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["registered_attempts_reconciled"] == 1
    assert not entry_path.exists()
    assert store.observe(handle).state == "failed"


class _TrackingRun(NamedTuple):
    """A dead run whose cleanup uncertainty rests on one registered tracking operation."""

    experiment: Experiment
    state_dir: Path
    store: RunStore
    handle: RunHandle
    study: optuna.Study
    operation_id: str

    @property
    def entry_path(self) -> Path:
        """Return the operation's active-attempt registry entry."""
        return _attempts_dir(self.experiment) / f"{self.operation_id}.json"

    def receipt_path(self, generation_id: str) -> Path:
        """Return the operation's cleanup receipt under its producing generation."""
        root = tracking_root(self.experiment)
        return root / generation_id / "cleanup-receipts" / f"{self.operation_id}.json"


def _stage_tracking_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation_generation: str | None = None,
    uncertain_attempt_ids: list[str] | None = None,
) -> _TrackingRun:
    """Stage a dead cleanup-uncertain run plus a registered tracking operation.

    :param Path tmp_path: Test directory holding the config, catalog, and operation directory.
    :param pytest.MonkeyPatch monkeypatch: Patches the runner process-group cleanup to succeed.
    :param str | None operation_generation: Generation producing the operation; the run's own by default.
    :param list[str] | None uncertain_attempt_ids: Attempts the runner's terminal status reported.
    :return _TrackingRun: The staged run, its study, and the operation id.
    """
    config = _config(tmp_path)
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    _, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    run_id = "srv-tracking-worker-recovery"
    mark_current_format(experiment)
    study = _new_phase_study(experiment, experiment.phases[0].name)
    stamp_artifact_root(study, experiment)
    handle = stage_dead_run(store, run_id, config, registry.get("srv").id, cleanup_uncertain=True)
    write_unsafe_cleanup_status(
        store,
        run_id,
        uncertain_attempt_ids=uncertain_attempt_ids or [],
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment, generation_id=run_id),
    )
    operation_id = "a" * 32
    directory = tmp_path / "tracking-operation"
    directory.mkdir()
    write_attempt_lifecycle(directory, attempt_id=operation_id, state="allocated")
    _register_tracking_operation(
        experiment,
        operation_id=operation_id,
        generation_id=operation_generation or run_id,
        directory=directory,
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *a, **k: True)
    return _TrackingRun(experiment, registry.state_dir, store, handle, study, operation_id)


def _recovered_tracking_run(run: _TrackingRun) -> None:
    """Assert that recover-run clears the run on tracking evidence alone, inspecting first."""
    trials_before = run.study.get_trials()

    inspected = recover_run_cli(run.state_dir, run.handle.run_id)

    assert inspected.exit_code == 0, inspected.output
    assert not run.store.cleanup_recovery_path(run.handle.run_id).exists()
    confirmed = recover_run_cli(run.state_dir, run.handle.run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    recovery = json.loads(run.store.cleanup_recovery_path(run.handle.run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [run.operation_id]
    assert recovery["reaped_running_trials"] == 0
    assert not run.entry_path.exists()
    observation = run.store.observe(run.handle)
    assert observation.state == "failed"
    assert not observation.recovery_required
    assert run.study.get_trials() == trials_before


@pytest.mark.parametrize("consumer", ["nobody", "sync_preflight", "run_preflight"])
def test_operator_recovery_accepts_a_resolved_tracking_operation_as_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    """A tracking worker whose cleanup was uncertain stays the run's evidence after any pass."""
    run = _stage_tracking_run(tmp_path, monkeypatch)
    if consumer == "sync_preflight":
        _preflight_active_attempts(
            run.experiment, _PreflightCleanupReport(), confirm=True, tracking_only=True
        )
    elif consumer == "run_preflight":
        _reconcile_existing_studies(run.experiment, cleanup_report=_PreflightCleanupReport())
    assert run.entry_path.exists() == (consumer == "nobody")
    assert run.receipt_path(run.handle.run_id).is_file() == (consumer != "nobody")

    _recovered_tracking_run(run)


@pytest.mark.parametrize("failing_pass", ["sync_preflight", "recover_run"])
def test_tracking_entry_outlives_a_failed_receipt_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_pass: str
) -> None:
    """The entry is the operation's only proof until some pass records its receipt."""
    run = _stage_tracking_run(tmp_path, monkeypatch)
    run_id = run.handle.run_id

    def read_only_receipts(path: Path, text: str, **kwargs: bool) -> None:
        if path.parent.name == "cleanup-receipts":
            raise PermissionError("tracking tree is read-only")
        private_atomic_write_text(path, text, **kwargs)

    report = _PreflightCleanupReport()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(attempts_module, "private_atomic_write_text", read_only_receipts)
        if failing_pass == "sync_preflight":
            _preflight_active_attempts(run.experiment, report, confirm=True, tracking_only=True)
        else:
            # The run's own recovery record is proof enough for the run, but
            # the entry must still wait for a receipt that any other run could use.
            recovered = recover_run_cli(run.state_dir, run_id, confirm=True)
            assert recovered.exit_code == 0, recovered.output
            record = json.loads(run.store.cleanup_recovery_path(run_id).read_text())
            assert record["reaped_attempt_ids"] == [run.operation_id]
            assert not run.store.observe(run.handle).recovery_required

    assert run.entry_path.is_file()
    assert not run.receipt_path(run_id).exists()
    if failing_pass == "sync_preflight":
        assert report.recovered_attempt_ids == {run.operation_id}
        assert report.unreceipted_attempt_ids == {run.operation_id}

    _preflight_active_attempts(
        run.experiment, _PreflightCleanupReport(), confirm=True, tracking_only=True
    )

    assert not run.entry_path.exists()
    assert run.receipt_path(run_id).is_file()
    if failing_pass == "sync_preflight":
        _recovered_tracking_run(run)


@pytest.mark.parametrize("causally_reported", [False, True])
def test_tracking_receipt_of_another_generation_is_evidence_only_when_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, causally_reported: bool
) -> None:
    """A receipt names its own generation, so a run needs its runner's report to claim it."""
    run = _stage_tracking_run(
        tmp_path,
        monkeypatch,
        operation_generation="other-generation",
        uncertain_attempt_ids=["a" * 32] if causally_reported else [],
    )
    _preflight_active_attempts(
        run.experiment, _PreflightCleanupReport(), confirm=True, tracking_only=True
    )
    assert run.receipt_path("other-generation").is_file()

    if causally_reported:
        _recovered_tracking_run(run)
        return
    refused = recover_run_cli(run.state_dir, run.handle.run_id, confirm=True)

    assert refused.exit_code != 0
    assert "could not confirm any trial-level cleanup evidence" in refused.output
    assert run.store.observe(run.handle).recovery_required


@pytest.mark.parametrize(
    "damage",
    [
        "schema_version",
        "boolean_schema_version",
        "operation_id",
        "generation_id",
        "unconfirmed",
        "truthy_not_true",
        "extra_field",
        "missing_field",
        "not_json",
        "shared_mode",
        "symlink",
        "misfiled",
        "renamed",
    ],
)
def test_damaged_tracking_receipt_is_not_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    damage: str,
) -> None:
    """Only an intact receipt filed under its own generation and operation proves cleanup."""
    run = _stage_tracking_run(tmp_path, monkeypatch)
    run_id = run.handle.run_id
    _preflight_active_attempts(
        run.experiment, _PreflightCleanupReport(), confirm=True, tracking_only=True
    )
    path = run.receipt_path(run_id)
    receipt = json.loads(path.read_text())
    assert _tracking_cleanup_receipts(run.experiment) == {run.operation_id: run_id}
    field_damage: dict[str, dict[str, object]] = {
        "schema_version": {"schema_version": 2},
        "boolean_schema_version": {"schema_version": True},
        "operation_id": {"operation_id": "b" * 32},
        "generation_id": {"generation_id": "other-generation"},
        "unconfirmed": {"cleanup_confirmed": False},
        "truthy_not_true": {"cleanup_confirmed": 1},
        "extra_field": {"note": "unexpected"},
    }
    if damage in field_damage:
        text = json.dumps({**receipt, **field_damage[damage]}) + "\n"
        private_atomic_write_text(path, text, require_private_dir=False)
    elif damage == "missing_field":
        del receipt["cleanup_confirmed"]
        private_atomic_write_text(path, json.dumps(receipt) + "\n", require_private_dir=False)
    elif damage == "not_json":
        private_atomic_write_text(path, "{", require_private_dir=False)
    elif damage == "shared_mode":
        path.chmod(0o644)
    elif damage == "symlink":
        forged = tmp_path / "forged.json"
        path.rename(forged)
        path.symlink_to(forged)
    elif damage == "misfiled":
        elsewhere = run.receipt_path("other-generation")
        elsewhere.parent.mkdir(parents=True)
        path.rename(elsewhere)
    else:
        path.rename(path.with_name(f"{'b' * 32}.json"))

    with caplog.at_level("WARNING", logger=attempts_module.log.name):
        assert _tracking_cleanup_receipts(run.experiment) == {}
    refused = recover_run_cli(run.state_dir, run_id, confirm=True)

    assert "Ignoring" in caplog.text
    assert refused.exit_code != 0
    assert "could not confirm any trial-level cleanup evidence" in refused.output
    assert run.store.observe(run.handle).recovery_required


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["own_generation", "other_generation"])
def test_sync_tracking_leaves_the_originating_runs_recovery_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    """The real sync retires a tracking entry in any generation scope, with its proof kept."""
    pytest.importorskip("mlflow")
    from tests.tracking_helpers import _remote_example

    config, _uploads = _remote_example(tmp_path, monkeypatch)
    config["storage"] = "journal:///" + str(tmp_path / "study.journal")
    config["phases"] = config["phases"][:1]
    experiment = Experiment.model_validate(config)
    run_id = "tracking-recovery-run"
    run_experiment(experiment, generation_id=run_id)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, registry, store = make_mcp_app(_catalog(tmp_path, config_path, allow=ALLOW_SIDE_EFFECTS))
    handle = stage_dead_run(
        store, run_id, config_path, registry.get("srv").id, cleanup_uncertain=True
    )
    write_unsafe_cleanup_status(
        store,
        run_id,
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment, generation_id=run_id),
    )
    operation_id = "a" * 32
    directory = tracking_root(experiment) / run_id / "operations" / operation_id
    directory.mkdir(parents=True)
    write_attempt_lifecycle(directory, attempt_id=operation_id, state="allocated")
    _register_tracking_operation(
        experiment, operation_id=operation_id, generation_id=run_id, directory=directory
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *a, **k: True)

    if scope == "own_generation":
        assert sync_tracking(experiment, run_id)["pending"] == 0
    else:
        with pytest.raises(ValueError, match="No tracking deliveries exist"):
            sync_tracking(experiment, "other-generation")

    study = _load_first_phase_study(config_path)
    _recovered_tracking_run(
        _TrackingRun(experiment, registry.state_dir, store, handle, study, operation_id)
    )


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["registered", "retired"])
def test_sync_tracking_finishes_an_attempt_whose_cleanup_recover_run_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """recover-run consumes a failed attempt's uncertainty in the study ledger, and sync honors that.

    A registered attempt is confirmed through the registry pass. Once its entry is gone, only the
    terminal trial's own record remains, which the terminal-trial pass confirms instead. Neither
    rewrites the trial, so the ledger is the only place sync can see the confirmation.
    """
    pytest.importorskip("mlflow")
    from tests.tracking_helpers import (
        _leave_cleanup_uncertain_attempt,
        _remote_example,
    )

    config, uploads = _remote_example(tmp_path, monkeypatch)
    config["storage"] = "journal:///" + str(tmp_path / "study.journal")
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, sampler={"type": "random", "seed": 123})
    experiment = Experiment.model_validate(config)
    run_id = "mcp-uncertain-attempt"
    attempt = _leave_cleanup_uncertain_attempt(experiment, monkeypatch, generation_id=run_id)
    if entry == "retired":
        _retire_active_attempt(experiment, attempt)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, registry, store = make_mcp_app(_catalog(tmp_path, config_path, allow=ALLOW_SIDE_EFFECTS))
    stage_dead_run(store, run_id, config_path, registry.get("srv").id, cleanup_uncertain=True)
    write_unsafe_cleanup_status(store, run_id)
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *a, **k: True)
    uploads.clear()
    with pytest.raises(ValueError, match="No tracking deliveries exist"):
        sync_tracking(experiment, run_id)
    assert uploads == []

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert _load_phase_trial(config_path, 0).user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    study = _load_first_phase_study(config_path)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [0]
    journal = (tmp_path / "study.journal").read_bytes()
    assert sync_tracking(experiment, run_id)["pending"] == 0
    [upload] = [r for r in uploads if r["binding"]["attempt_id"] == attempt]
    assert upload["finish"] == "FAILED"
    assert (tmp_path / "study.journal").read_bytes() == journal


@pytest.mark.parametrize(
    (
        "causally_reported",
        "interrupt_recovery_write",
        "anonymous_snapshot",
        "changed_storage",
    ),
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, True, False),
        (True, True, False, True),
    ],
)
def test_operator_recovery_scopes_cleanup_evidence_to_its_reported_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    causally_reported: bool,
    interrupt_recovery_write: bool,
    anonymous_snapshot: bool,
    changed_storage: bool,
) -> None:
    """Cross-generation evidence is causal and survives an interrupted write."""
    config = _config(tmp_path)
    attempt_config = config
    if changed_storage:
        attempt_config = tmp_path / "srv-later-storage.yaml"
        attempt_config.write_text(
            config.read_text().replace("/srv.db", "/srv-later.db"),
        )
    earlier_run_id = "srv-earlier-uncertain"
    later_generation_id = "later-cli-generation"
    trial_number = write_launched_stale_trial(
        attempt_config,
        generation_id=later_generation_id,
        persist_trial_attrs=not anonymous_snapshot,
    )
    experiment = load_experiment(attempt_config)
    assert isinstance(experiment, Experiment)
    phase = experiment.phases[0]
    study = _load_first_phase_study(attempt_config)
    attempt_id = f"stale-attempt-{trial_number}"
    trial_dir = _trial_dir_for(
        experiment,
        phase.name,
        trial_number,
        generation_id=later_generation_id,
        attempt_id=attempt_id,
    )
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase.name,
        study_name=study.study_name,
        trial_number=trial_number,
        trial_dir=trial_dir,
        generation_id=later_generation_id,
    )
    if changed_storage:
        recovery_experiment = load_experiment(config)
        assert isinstance(recovery_experiment, Experiment)
        _write_artifact_root_binding(recovery_experiment)

    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = stage_dead_run(store, earlier_run_id, config, reg.id, cleanup_uncertain=False)
    status_kwargs: dict[str, object] = {}
    if anonymous_snapshot:
        snapshot = capture_result_snapshot(
            experiment,
            generation_id=earlier_run_id,
        )
        frozen_attempt = snapshot["status"]["phases"][0]["running_attempts"][0]
        assert frozen_attempt["attempt_id"] is None
        assert frozen_attempt["generation_id"] is None
        status_kwargs = {
            "result_snapshot_state": "complete",
            "result_snapshot": snapshot,
        }
    write_unsafe_cleanup_status(
        store,
        earlier_run_id,
        uncertain_attempt_ids=[attempt_id] if causally_reported else [],
        **status_kwargs,
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        lambda _identity: True,
    )
    recovery_path = store.cleanup_recovery_path(earlier_run_id)
    entry_path = _attempts_dir(experiment) / f"{attempt_id}.json"
    if interrupt_recovery_write:
        recovery_path.mkdir()

    result = recover_run_cli(registry.state_dir, earlier_run_id, confirm=True)

    assert study.get_trials(deepcopy=False)[trial_number].state == optuna.trial.TrialState.FAIL
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]
    if interrupt_recovery_write:
        assert result.exit_code != 0
        assert entry_path.exists()
        recovery_path.rmdir()
        result = recover_run_cli(registry.state_dir, earlier_run_id, confirm=True)
    if causally_reported:
        assert result.exit_code == 0, result.output
        assert recovery_path.is_file()
        recovery = json.loads(recovery_path.read_text())
        assert recovery["reaped_attempt_ids"] == [attempt_id]
        assert not entry_path.exists()
        if anonymous_snapshot:
            assert recovery["reaped_attempt_locations"] == {
                attempt_id: {
                    "phase": phase.name,
                    "trial_number": trial_number,
                    "generation_id": later_generation_id,
                }
            }
            terminal = json.loads(store.status_path(earlier_run_id).read_text())
            frozen_phase = terminal["result_snapshot"]["status"]["phases"][0]
            assert frozen_phase["trials"]["RUNNING"] == 0
            assert frozen_phase["trials"]["INTERRUPTED"] == 1
            assert frozen_phase["running_attempts"] == []
        assert store.observe(handle).state == "failed"
    else:
        assert result.exit_code != 0
        assert "could not confirm any trial-level cleanup evidence" in result.output
        assert not recovery_path.exists()
        assert store.observe(handle).state == "running"
        with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
            app.launch("srv")


def test_operator_recovery_refuses_to_rebuild_missing_historical_snapshot(tmp_path: Path) -> None:
    config, app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-result-repair"
    stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="failed",
        result_snapshot_error="RuntimeError",
    )

    unavailable = app.status(run_id=run_id)
    assert unavailable["result_source"] == "terminal_snapshot_unavailable"
    assert unavailable["run"]["failure"]["code"] == "result_snapshot_unavailable"
    assert app.winners(run_id=run_id)["failure"]["code"] == "result_snapshot_unavailable"

    preflight = recover_run_cli(registry.state_dir, run_id)
    assert preflight.exit_code != 0
    assert "cannot be rebuilt from the current shared study" in preflight.output
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "failed"

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)
    assert confirmed.exit_code != 0
    assert "cannot be rebuilt from the current shared study" in confirmed.output
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "failed"
    assert terminal["result_snapshot_error"] == "RuntimeError"


@pytest.mark.parametrize("terminal", [False, True], ids=["running", "cleanup-uncertain-terminal"])
def test_registered_trial_recovery_counts_match_preflight_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    """Registry reconciliation contributes the same trial count preflight promised."""
    config = _config(tmp_path)
    run_id = "srv-registered-trial-count"
    setup = write_uncertain_failed_trial if terminal else write_launched_stale_trial
    trial_number = setup(config, generation_id=run_id)
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    phase = experiment.phases[0]
    attempt_id = f"stale-attempt-{trial_number}"
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase.name,
        study_name=f"{experiment.experiment}::{phase.name}",
        trial_number=trial_number,
        trial_dir=_trial_dir_for(
            experiment, phase.name, trial_number, generation_id=run_id, attempt_id=attempt_id
        ),
        generation_id=run_id,
    )
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    stage_dead_run(store, run_id, config, "srv", cleanup_uncertain=False)
    write_unsafe_cleanup_status(store, run_id)
    record_runner_cleanup(monkeypatch, module="phasesweep.mcp.recovery")
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process", Mock(return_value=True)
    )

    preflight = recover_run_cli(registry.state_dir, run_id)

    assert preflight.exit_code == 0, preflight.output
    planned = "recover 1 cleanup-uncertain terminal trial" if terminal else "reap 1 stale trial"
    assert planned in preflight.output
    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["registered_attempts_reconciled"] == 1
    assert recovery["reaped_running_trials"] == (0 if terminal else 1)
    assert recovery["cleanup_uncertain_terminal_trials"] == (1 if terminal else 0)
    actual = "confirmed 1 cleanup-uncertain trial" if terminal else "reaped 1 stale trial"
    assert actual in confirmed.output
    trial = _load_phase_trial(config, trial_number)
    assert trial.state == optuna.trial.TrialState.FAIL
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]


@pytest.mark.parametrize(
    (
        "trial_setup_fn",
        "trial_starttime",
        "dry_run_substring",
        "confirm_output_substring",
        "expected_reaped_running",
        "expected_cleanup_uncertain_terminal",
        "expect_running_before_confirm",
    ),
    [
        pytest.param(
            write_uncertain_failed_trial,
            111,
            "recover 1 cleanup-uncertain terminal trial",
            "Cleared cleanup uncertainty",
            0,
            1,
            False,
            id="terminal-cleanup-uncertain-trial",
        ),
        pytest.param(
            partial(write_launched_stale_trial, cleanup_confirmed=False),
            222,
            "reap 1 stale trial",
            "reaped 1 stale trial",
            1,
            0,
            True,
            id="stale-running-trial",
        ),
    ],
)
@pytest.mark.integration
def test_operator_recovery_clears_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_setup_fn,
    trial_starttime: int,
    dry_run_substring: str,
    confirm_output_substring: str,
    expected_reaped_running: int,
    expected_cleanup_uncertain_terminal: int,
    expect_running_before_confirm: bool,
) -> None:
    """Recovery clears cleanup uncertainty from a terminal cleanup-uncertain
    trial and a stale RUNNING trial reaped by recovery.
    """
    config = _config(tmp_path)
    run_id = "srv-cleanup-uncertainty-recover"
    trial_pid = reaped_pid()
    trial_number = trial_setup_fn(config, generation_id=run_id, pid=trial_pid)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)
    status_kwargs: dict[str, object] = {}
    if expect_running_before_confirm:
        exp = load_experiment(config)
        assert isinstance(exp, Experiment)
        status_kwargs["result_snapshot"] = capture_result_snapshot(exp)
    write_unsafe_cleanup_status(store, run_id, **status_kwargs)
    if expect_running_before_confirm:
        assert app.status(run_id=run_id)["phases"][0]["running_trials_total"] == 1

    trial_cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def fake_trial_cleanup(identity: StaleProcessIdentity) -> bool:
        trial_cleanup_calls.append((identity.pid, identity.proc_starttime, identity.pgid))
        return True

    runner_cleanup_calls = record_runner_cleanup(monkeypatch, module="phasesweep.mcp.recovery")
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process", fake_trial_cleanup
    )

    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")

    dry = recover_run_cli(registry.state_dir, run_id)

    assert dry.exit_code == 0, dry.output
    assert "Re-run with --confirm" in dry.output
    assert dry_run_substring in dry.output
    assert runner_cleanup_calls == []
    assert trial_cleanup_calls == []
    assert not store.cleanup_uncertain_path(run_id).exists()
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.observe(handle).state == "running"
    if expect_running_before_confirm:
        study = _load_first_phase_study(config)
        trial = next(t for t in study.get_trials(deepcopy=False) if t.number == trial_number)
        assert trial.state == optuna.trial.TrialState.RUNNING
        assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    else:
        trial = _load_phase_trial(config, trial_number)
        assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
        study = _load_first_phase_study(config)
        assert CLEANUP_RECOVERED_TRIALS_ATTR not in study.user_attrs

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert confirm_output_substring in confirmed.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["run_id"] == run_id
    assert recovery["config_sha256"] == reg.config_sha256
    assert recovery["cleanup_confirmed"] is True
    assert recovery["reaped_running_trials"] == expected_reaped_running
    assert recovery["cleanup_uncertain_terminal_trials"] == expected_cleanup_uncertain_terminal
    assert store.observe(handle).state == "failed"
    assert runner_cleanup_calls == [(handle.pid, 111, handle.pid)]
    assert trial_cleanup_calls == [(trial_pid, trial_starttime, trial_pid)]
    trial = _load_phase_trial(config, trial_number)
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    assert trial.state == optuna.trial.TrialState.FAIL
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]
    if expect_running_before_confirm:
        recovered_status = app.status(run_id=run_id)
        assert recovered_status["phases"][0]["trials"] == {
            "WAITING": 0,
            "RUNNING": 0,
            "COMPLETE": 0,
            "FAIL": 0,
            "INTERRUPTED": 1,
        }
        assert recovered_status["phases"][0]["running_trials_total"] == 0

    final_status = store.status_path(run_id).read_bytes()
    final_recovery = store.cleanup_recovery_path(run_id).read_bytes()
    repeat = recover_run_cli(registry.state_dir, run_id, confirm=True)
    if expect_running_before_confirm:
        assert repeat.exit_code == 0, repeat.output
        assert "No cleanup uncertainty or terminal result repair" in repeat.output
    else:
        # Cleanup is settled, but an absent historical snapshot remains an
        # explicit refusal; retry must not repeat process cleanup to repair it.
        assert repeat.exit_code == 1, repeat.output
        assert "no immutable terminal result snapshot" in repeat.output
    assert store.status_path(run_id).read_bytes() == final_status
    assert store.cleanup_recovery_path(run_id).read_bytes() == final_recovery
    assert runner_cleanup_calls == [(handle.pid, 111, handle.pid)]
    assert trial_cleanup_calls == [(trial_pid, trial_starttime, trial_pid)]

    captured = patch_popen_capture(monkeypatch)
    launched = app.launch("srv")
    assert launched["state"] == "running"
    assert captured["cmd"]


@pytest.mark.parametrize("earlier_boot", [False, True])
@pytest.mark.integration
def test_operator_snapshot_repair_retry_reuses_cleanup_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    earlier_boot: bool,
) -> None:
    """A --confirm retry after an interrupted
    ``finalize_result_snapshot`` call reuses the cleanup-recovery evidence
    persisted by the first (failed) attempt instead of re-running runner/trial
    cleanup or re-reaping the stale trial.
    """
    run_id = "srv-retry-result-repair"
    runner_cleanup = Mock(return_value=True)
    trial_cleanup = Mock(return_value=True)

    app, store, handle, attempt_id, recover = _stage_stale_running_recovery_scaffold(
        tmp_path,
        run_id=run_id,
        error_class="UnsafeProcessCleanupError",
        include_generation_record=True,
        mark_cleanup_uncertain=False,
        snapshot_bound_to_generation=True,
        kill_stale_group_stub=runner_cleanup,
        cleanup_trial_stub=trial_cleanup,
        monkeypatch=monkeypatch,
        earlier_boot=earlier_boot,
    )
    observation = store.observe(handle)
    assert observation.state == ("failed" if earlier_boot else "running")
    assert observation.recovery_required

    snapshot = Mock(
        wraps=finalize_result_snapshot,
        side_effect=[OSError("snapshot finalization failed"), DEFAULT],
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.finalize_result_snapshot", snapshot)

    first = recover()

    assert first.exit_code != 0
    assert "failed to finalize terminal result snapshot" in first.output
    assert store.cleanup_recovery_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert store.observe(handle).recovery_required
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "complete"
    frozen_status = app.status(run_id=run_id)
    assert frozen_status["result_source"] == "frozen_run_snapshot"
    assert frozen_status["phases"][0]["trials"]["RUNNING"] == 1
    assert runner_cleanup.call_count == (0 if earlier_boot else 1)
    trial_cleanup.assert_called_once()

    retry = recover()

    assert retry.exit_code == 0, retry.output
    assert "Finalized stored terminal result snapshot" in retry.output
    assert runner_cleanup.call_count == (0 if earlier_boot else 1)
    trial_cleanup.assert_called_once()
    assert snapshot.call_count == 2
    assert [set(call.kwargs["confirmed_attempt_ids"]) for call in snapshot.call_args_list] == [
        {attempt_id},
        {attempt_id},
    ]
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    phase = terminal["result_snapshot"]["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["INTERRUPTED"] == 1
    assert phase["generation_trials"]["RUNNING"] == 0
    assert phase["generation_trials"]["INTERRUPTED"] == 1
    assert not store.observe(handle).recovery_required
    recovered_status = app.status(run_id=run_id)["phases"][0]
    # An interrupted trial is not terminal for the budget: it reruns.
    assert recovered_status["terminal_trials_this_run"] == 0
    assert recovered_status["terminal_trials_before_run"] == 0
    assert recovered_status["target_already_satisfied"] is False

    final_status = store.status_path(run_id).read_bytes()

    def forbid_redundant_finalization(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("completed recovery must not finalize again")

    with monkeypatch.context() as patch:
        patch.setattr(
            "phasesweep.mcp.recovery.finalize_result_snapshot", forbid_redundant_finalization
        )
        repeat = recover()

    assert repeat.exit_code == 0, repeat.output
    assert "No cleanup uncertainty or terminal result repair" in repeat.output
    assert store.status_path(run_id).read_bytes() == final_status
    assert not store.observe(handle).recovery_required


@pytest.mark.integration
def test_operator_recovery_keeps_frozen_snapshot_when_final_status_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed final status persistence keeps the prior frozen result and reservation."""
    run_id = "srv-pending-write-failure"
    app, store, handle, attempt_id, recover = _stage_stale_running_recovery_scaffold(
        tmp_path,
        run_id=run_id,
        error_class="UnsafeProcessCleanupError",
        include_generation_record=True,
        mark_cleanup_uncertain=False,
        snapshot_bound_to_generation=True,
        kill_stale_group_stub=lambda *_args, **_kwargs: True,
        cleanup_trial_stub=lambda *_args, **_kwargs: True,
        monkeypatch=monkeypatch,
        allow_cancel=True,
    )

    def refuse_status_write(_path: Path, _payload: dict) -> None:
        raise OSError("status file unavailable")

    with monkeypatch.context() as patch:
        patch.setattr("phasesweep.mcp.recovery.write_status_file", refuse_status_write)
        failed = recover()

    assert failed.exit_code != 0
    assert "failed to finalize terminal result snapshot" in failed.output
    assert store.observe(handle).recovery_required
    frozen_status = app.status(run_id=run_id)
    assert frozen_status["run"]["recovery_required"] is True
    assert frozen_status["result_source"] == "frozen_run_snapshot"
    assert frozen_status["phases"][0]["trials"]["RUNNING"] == 1
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)
    awaited = asyncio.run(app.await_run(run_id, timeout_seconds=0))
    assert awaited["result_source"] == "frozen_run_snapshot"
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "complete"
    assert store.cleanup_recovery_path(run_id).is_file()
    assert store.cleanup_uncertain_path(run_id).is_file()
    monkeypatch.setattr("phasesweep.mcp.tools.kill_stale_group", lambda *_args, **_kwargs: True)
    cancelled = app.cancel(run_id)
    assert cancelled["state"] == "running"
    assert cancelled["recovery_required"] is True
    assert store.cleanup_uncertain_path(run_id).is_file()
    with pytest.raises(ExperimentBusyError):
        app.launch("srv")

    retried = recover()
    assert retried.exit_code == 0, retried.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert not store.cleanup_uncertain_path(run_id).exists()
    assert not store.observe(handle).recovery_required
    phase = app.status(run_id=run_id)["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["INTERRUPTED"] == 1


def test_operator_recovery_uses_runner_reconciliation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run_id = "srv-runner-reconciled"
    trial_number = write_launched_stale_trial(
        config,
        cleanup_confirmed=False,
        generation_id=run_id,
    )
    attempt_id = f"stale-attempt-{trial_number}"
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    study = _load_first_phase_study(config)
    reconciled_attempts: dict[str, _AttemptRecord] = {}
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        lambda _identity: True,
    )
    assert (
        _reap_stale_trials(
            study,
            experiment,
            experiment.phases[0].name,
            confirm=True,
            recovered_attempts=reconciled_attempts,
        )
        == 1
    )
    reconciled_attempt_ids = set(reconciled_attempts)
    reconciled_attempt_generations = {
        aid: record.generation_id for aid, record in reconciled_attempts.items()
    }
    assert reconciled_attempt_ids == {attempt_id}
    assert reconciled_attempt_generations == {attempt_id: run_id}

    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="cancelled",
        cleanup_confirmed=False,
        recovered_attempt_ids=sorted(reconciled_attempt_ids),
        recovered_attempt_generations=reconciled_attempt_generations,
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment),
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *args, **kwargs: True)

    result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code == 0, result.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert recovery["reaped_running_trials"] == 0
    assert recovery["cleanup_uncertain_terminal_trials"] == 0
    assert store.observe(handle).state == "cancelled"
    assert live_runs(store) == []


@pytest.mark.integration
def test_operator_cleanup_recovery_retry_counts_persisted_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A --confirm retry after an interrupted
    ``RunStore.clear_cleanup_uncertain`` call still counts the reaped-attempt
    evidence persisted by the first attempt, rather than treating the retry as
    having found no cleanup evidence.
    """
    run_id = "srv-cleanup-recovery-retry"
    app, store, handle, attempt_id, recover = _stage_stale_running_recovery_scaffold(
        tmp_path,
        run_id=run_id,
        error_class="cancelled",
        include_generation_record=False,
        mark_cleanup_uncertain=True,
        snapshot_bound_to_generation=False,
        kill_stale_group_stub=lambda *args, **kwargs: True,
        cleanup_trial_stub=lambda _identity: True,
        monkeypatch=monkeypatch,
    )

    monkeypatch.setattr(
        RunStore,
        "clear_cleanup_uncertain",
        _interrupt_first_cleanup_clear(),
    )

    first = recover()

    assert first.exit_code != 0
    assert "interrupted before clearing cleanup marker" in first.output
    assert store.cleanup_uncertain_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    frozen_status = app.status(run_id=run_id)
    assert frozen_status["result_source"] == "frozen_run_snapshot"
    assert app.winners(run_id=run_id)["result_source"] == "frozen_run_snapshot"

    retry = recover()

    assert retry.exit_code == 0, retry.output
    assert not store.cleanup_uncertain_path(run_id).exists()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert recovery["reaped_running_trials"] == 0
    terminal = json.loads(store.status_path(run_id).read_text())
    phase = terminal["result_snapshot"]["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["INTERRUPTED"] == 1
    assert store.observe(handle).state == "cancelled"
    assert live_runs(store) == []


def test_operator_recovery_consumes_terminal_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_run = "srv-terminal-first"
    store, handle, trial_number, config, state_dir, recover = _stage_terminal_uncertain_run(
        tmp_path, monkeypatch, run_id=first_run, mark_uncertain=False
    )
    first = recover()

    assert first.exit_code == 0, first.output
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]

    second_run = "srv-terminal-second"
    second_handle = stage_dead_run(
        store, second_run, config, handle.experiment_id, cleanup_uncertain=False
    )
    write_unsafe_cleanup_status(store, second_run)

    replay = recover_run_cli(state_dir, second_run, confirm=True)

    assert replay.exit_code != 0
    assert "could not confirm any trial-level cleanup evidence" in replay.output
    assert not store.cleanup_recovery_path(second_run).exists()
    assert store.observe(second_handle).state == "running"


def test_operator_recovery_retry_counts_ledger_evidence_after_lost_recovery_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the study-ledger write but before the run-level recovery
    record must not wedge the run forever: the --confirm retry counts the
    durable ledger evidence for this run's own generation instead of refusing
    with "no trial-level cleanup evidence"."""
    from phasesweep.runtime.files import private_atomic_write_text as real_write

    run_id = "srv-ledger-retry"
    store, handle, trial_number, config, _state_dir, recover = _stage_terminal_uncertain_run(
        tmp_path, monkeypatch, run_id=run_id, mark_uncertain=False
    )
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    write_unsafe_cleanup_status(
        store,
        run_id,
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment),
    )
    recovery_record = store.cleanup_recovery_path(run_id)

    def crash_on_recovery_record(path: Path, text: str) -> None:
        if path == recovery_record:
            raise RuntimeError("simulated crash before the recovery record")
        real_write(path, text)

    monkeypatch.setattr(
        "phasesweep.mcp.recovery.private_atomic_write_text", crash_on_recovery_record
    )

    first = recover()

    assert first.exit_code != 0
    # The durable study ledger consumed the trial before the crash; the
    # run-level record never landed.
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]
    assert not recovery_record.exists()
    assert store.observe(handle).recovery_required
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "complete"
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    assert app.status(run_id=run_id)["result_source"] == "frozen_run_snapshot"
    assert app.winners(run_id=run_id)["result_source"] == "frozen_run_snapshot"

    monkeypatch.setattr("phasesweep.mcp.recovery.private_atomic_write_text", real_write)

    retry = recover()

    assert retry.exit_code == 0, retry.output
    recovery = json.loads(recovery_record.read_text())
    assert recovery["cleanup_confirmed"] is True
    assert live_runs(store) == []


def test_operator_recovery_retry_clears_marker_after_terminal_only_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupting ``clear_cleanup_uncertain`` after a recovery whose only
    evidence was a cleanup-uncertain terminal trial must not wedge the run:
    the retry finds nothing fresh to recover but must not refuse while the
    recovery record is already on disk. The study ledger supplies the
    evidence."""
    run_id = "srv-terminal-marker-retry"
    store, handle, _trial, _config, _state_dir, recover = _stage_terminal_uncertain_run(
        tmp_path, monkeypatch, run_id=run_id, mark_uncertain=True
    )

    monkeypatch.setattr(
        RunStore,
        "clear_cleanup_uncertain",
        _interrupt_first_cleanup_clear(),
    )

    first = recover()

    assert first.exit_code != 0
    assert store.cleanup_uncertain_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["cleanup_uncertain_terminal_trials"] == 1

    retry = recover()

    assert retry.exit_code == 0, retry.output
    assert not store.cleanup_uncertain_path(run_id).exists()
    assert live_runs(store) == []


@pytest.mark.parametrize(
    ("snapshot_suffix", "terminal_without_evidence", "expected_message"),
    [
        pytest.param(
            b"",
            True,
            "could not confirm any trial-level cleanup evidence",
            id="terminal-uncertainty-without-trial-evidence",
        ),
        pytest.param(
            b"\n# drifted\n",
            False,
            "run snapshot hash mismatch",
            id="snapshot-hash-mismatch",
        ),
        pytest.param(None, False, "run config snapshot is missing", id="snapshot-missing"),
    ],
)
def test_operator_recovery_refuses_unverifiable_run(
    tmp_path: Path,
    snapshot_suffix: bytes | None,
    terminal_without_evidence: bool,
    expected_message: str,
) -> None:
    """Recovery refuses missing cleanup evidence and altered launch snapshots."""
    config, _app, registry, store = side_effect_app(tmp_path)
    reg = registry.get("srv")
    run_id = "srv-unverifiable-recovery"
    handle = make_dead_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.create(handle)
    if snapshot_suffix is not None:
        store.config_snapshot_path(run_id).write_bytes(config.read_bytes() + snapshot_suffix)
    if terminal_without_evidence:
        write_unsafe_cleanup_status(store, run_id)
    else:
        store.mark_cleanup_uncertain(handle)

    result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code != 0
    assert expected_message in result.output
    if not terminal_without_evidence:
        assert "restore the files and permissions identified in the error" in result.output
        assert "experiment tree" not in result.output
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.observe(handle).state == "running"
    if not terminal_without_evidence:
        assert store.cleanup_uncertain_path(run_id).is_file()
