"""MCP server recover-run and cleanup-uncertainty behavior. Logic that does not need a real detached runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import optuna
import pytest
import yaml
from click.testing import Result

import phasesweep.mcp.recovery as mcp_recovery
import phasesweep.mcp.runs as mcp_runs
from phasesweep.config import (
    Experiment,
    load_config,
)
from phasesweep.engine import (
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    TerminalReport,
)
from phasesweep.engine.artifact_roots import (
    _bind_study_artifact_root,
    _write_artifact_root_binding,
)
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _attempts_dir,
    _experiment_dir,
    _generation_record_path,
    _trial_dir_for,
)
from phasesweep.engine.state import (
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.engine.trial import UnsafeProcessCleanupError
from phasesweep.mcp.errors import (
    ExperimentBusyError,
    RunLaunchUnsettledError,
)
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.snapshots import capture_result_snapshot, finalize_result_snapshot
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.reaper import (
    StaleProcessIdentity,
    read_boot_id,
)
from tests.conftest import (
    mark_current_format,
    reaped_pid,
)
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    _catalog,
    _config,
    claim_runner_handle,
    make_mcp_app,
    make_run_handle,
    patch_popen_capture,
    runner_argv,
    runner_main,
    stage_dead_run,
    write_run_status,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import (
    recover_run_cli,
    write_launched_stale_trial,
    write_trial_identity,
    write_uncertain_failed_trial,
)


def _counting_success_callback(calls: list[None]) -> Callable[..., bool]:
    """Return a permissive cleanup stub that records each invocation."""

    def callback(*args: object, **kwargs: object) -> bool:
        calls.append(None)
        return True

    return callback


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
    exp = load_config(config)
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
    recorded_boot_id = None
    if earlier_boot:
        current_boot_id = read_boot_id()
        if current_boot_id is None:
            pytest.skip("boot id unavailable on this platform")
        recorded_boot_id = "00000000-0000-0000-0000-000000000000"
        if recorded_boot_id == current_boot_id:
            recorded_boot_id = "11111111-1111-1111-1111-111111111111"
    trial_number = write_launched_stale_trial(
        config,
        cleanup_confirmed=False,
        generation_id=run_id,
        boot_id=recorded_boot_id,
    )
    attempt_id = f"stale-attempt-{trial_number}"
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=reaped_pid(),
        starttime=111,
        allow_cancel=allow_cancel,
    )
    if recorded_boot_id is not None:
        handle = replace(handle, boot_id=recorded_boot_id)
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
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
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
) -> tuple[RunStore, RunHandle, int, Path, Callable[[], Result]]:
    """Stage a run whose only cleanup evidence is one of its own
    cleanup-uncertain terminal trials (generation id == run id, matching the
    detached-runner contract). Returns ``(store, handle, trial_number,
    config, recover)``."""
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
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", fake_cleanup)
    recover = partial(recover_run_cli, registry.state_dir, run_id, confirm=True)
    return store, handle, trial_number, config, recover


@pytest.mark.parametrize("interrupted_recovery", [False, True])
def test_operator_recovery_clears_abandoned_transactional_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted_recovery: bool
) -> None:
    """A free launch lease makes a persisted launch recoverable without losing its log."""
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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


@pytest.mark.integration
def test_runner_records_cleanup_uncertainty_for_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )

    def raise_cleanup_uncertain(*args: object, **kwargs: object) -> None:
        raise UnsafeProcessCleanupError("trial cleanup uncertain")

    monkeypatch.setattr("phasesweep.mcp.runner.run_experiment", raise_cleanup_uncertain)

    with pytest.raises(UnsafeProcessCleanupError, match="trial cleanup uncertain"):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "UnsafeProcessCleanupError"
    assert status["cleanup_confirmed"] is False


@pytest.mark.integration
def test_runner_makes_cleanup_uncertainty_actionable_and_preserves_primary_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r-secondary-cleanup"
    status_path = store.status_path(run_id)
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )
    primary = NoFeasibleTrialError("trainer failed")

    def fail_with_uncertain_cleanup(
        config_obj: Experiment,
        *,
        from_phase: str | None,
        dry_run: bool,
        terminal_callback,
        generation_id: str,
        publication_hook: object,
    ) -> None:
        del config_obj, from_phase, dry_run, publication_hook
        terminal_callback(
            TerminalReport(
                generation_id=generation_id,
                primary_error=primary,
                cleanup_confirmed=False,
                recovered_attempt_ids=frozenset({"attempt-reconciled"}),
                uncertain_attempt_ids=frozenset({"attempt-1"}),
                recovered_attempt_generations={"attempt-reconciled": generation_id},
                cleanup_error=ProcessCleanupUncertainError("cleanup uncertain"),
            )
        )
        raise ProcessCleanupUncertainError("cleanup uncertain") from primary

    monkeypatch.setattr(
        "phasesweep.mcp.runner.run_experiment",
        fail_with_uncertain_cleanup,
    )

    with pytest.raises(ProcessCleanupUncertainError):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "ProcessCleanupUncertainError"
    assert status["cleanup_confirmed"] is False
    assert status["recovered_attempt_ids"] == ["attempt-reconciled"]
    assert status["recovered_attempt_generations"] == {
        "attempt-reconciled": run_id,
    }
    assert status["uncertain_attempt_ids"] == ["attempt-1"]
    assert status["failure"]["code"] == "cleanup_uncertain"
    assert status["failure"]["stage"] == "cleanup"
    assert status["failure"]["retryable"] is False
    assert status["failure"]["actor"] == "operator"
    assert status["failure"]["cause"]["code"] == "trainer_failed"
    assert status["failure"]["cause"]["stage"] == "execution"


@pytest.mark.integration
def test_runner_persists_registered_terminal_identity_uncertainty(tmp_path: Path) -> None:
    """A terminal trial missing its attempt id remains attributable to recover-run."""
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r-terminal-identity"
    status_path = store.status_path(run_id)
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )

    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    phase = experiment.phases[0]
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase.name}",
        storage=_resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )
    mark_current_format(experiment, study)
    _bind_study_artifact_root(study, experiment)
    trial = study.ask()
    attempt_id = "terminal-identity-attempt"
    generation_id = "earlier-generation"
    trial_dir = _trial_dir_for(
        experiment,
        phase.name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase.name,
        study_name=study.study_name,
        trial_number=trial.number,
        trial_dir=trial_dir,
        generation_id=generation_id,
    )

    with pytest.raises(ProcessCleanupUncertainError, match="identity is missing"):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
        )

    status = json.loads(status_path.read_text())
    assert status["cleanup_confirmed"] is False
    assert status["uncertain_attempt_ids"] == [attempt_id]
    # The refusal names the ledger as the first repair, so the agent hears the same.
    assert status["failure"]["code"] == "cleanup_uncertain"
    assert status["failure"]["remediation"] == (
        "Ask the operator to restore or repair the storage ledger and access to it, then "
        "run phasesweep mcp recover-run before another launch. The error in the "
        "PhaseSweep run log gives the details."
    )


def test_terminal_cleanup_uncertainty_blocks_relaunch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-terminal-uncertain"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=reaped_pid(),
        starttime=111,
    )
    store.create(handle)
    write_unsafe_cleanup_status(store, run_id)

    assert store.state(handle) == "running"
    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")


def test_operator_recovery_clears_no_status_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-status-during-recovery"
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=True)
    experiment = load_config(config)
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


@pytest.mark.parametrize("unknown_side", ["saved", "current"])
def test_operator_recovery_refuses_unknown_boot_process_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown_side: str,
) -> None:
    current_boot = read_boot_id()
    if current_boot is None:
        pytest.skip("boot id unavailable on this platform")
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-recovery-unknown-boot"
    handle = replace(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            pid=reaped_pid(),
            starttime=111,
        ),
        boot_id=None if unknown_side == "saved" else current_boot,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.mark_cleanup_uncertain(handle)
    if unknown_side == "current":
        monkeypatch.setattr(mcp_runs, "read_boot_id", lambda: None)
        monkeypatch.setattr(mcp_recovery, "read_boot_id", lambda: None, raising=False)
    signalled: list[object] = []

    def record_signal(*args: object, **kwargs: object) -> bool:
        signalled.append((args, kwargs))
        return True

    monkeypatch.setattr(mcp_recovery, "kill_stale_group", record_signal)

    for confirmed in (False, True):
        result = recover_run_cli(registry.state_dir, run_id, confirm=confirmed)
        assert result.exit_code != 0
        assert "boot id" in f"{result.output} {result.exception}".lower()

    assert signalled == []
    assert store.cleanup_uncertain_path(run_id).is_file()


def test_operator_recovery_skips_liveness_and_signalling_for_earlier_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PID recycled after a reboot must not fake runner liveness or absorb
    the recovery group signal: a recorded boot id from an earlier boot proves
    the runner and its descendants are gone (review v0.5.17 / finding E).

    The handle deliberately records this test process's own live PID and start
    time — the strongest possible "looks live" collision — under a foreign
    boot id.
    """
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-earlier-boot"
    handle = replace(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
        ),
        boot_id="00000000-0000-0000-0000-000000000000",
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.mark_cleanup_uncertain(handle)

    signalled: list[object] = []

    def spy_kill_stale_group(*args: object, **kwargs: object) -> bool:
        signalled.append((args, kwargs))
        return True

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", spy_kill_stale_group)

    dry = recover_run_cli(registry.state_dir, run_id)
    assert dry.exit_code == 0, dry.output
    assert "still appears live" not in dry.output

    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)
    assert confirmed.exit_code == 0, confirmed.output
    assert "Cleared cleanup uncertainty" in confirmed.output
    assert signalled == []
    assert not store.cleanup_uncertain_path(run_id).exists()


def test_earlier_boot_runner_without_status_never_reads_later_shared_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebooted no-status run frees capacity without borrowing later results."""
    current_boot = read_boot_id()
    if current_boot is None:
        pytest.skip("boot id unavailable on this platform")
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-earlier-boot-no-status"
    earlier_boot = "00000000-0000-0000-0000-000000000000"
    if earlier_boot == current_boot:
        earlier_boot = "11111111-1111-1111-1111-111111111111"
    handle = replace(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
        ),
        boot_id=earlier_boot,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    later_trial = write_launched_stale_trial(config, generation_id="srv-later-run")

    assert _load_phase_trial(config, later_trial).state == optuna.trial.TrialState.RUNNING
    assert store.state(handle) == "failed"
    assert not store.recovery_required(handle)
    assert store.live_runs() == []

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
        assert payload["phases"][0]["running_trials_total"] == 0
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
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-recovery-lock-contention"
    stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=True)
    cleanup_calls = 0

    def unexpected_cleanup(*args: object, **kwargs: object) -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return True

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", unexpected_cleanup)
    with _experiment_lock(reg.experiment):
        result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code != 0
    assert "Another phasesweep process" in result.output
    assert cleanup_calls == 0
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.cleanup_recovery_path(run_id).exists()
    assert not store.status_path(run_id).exists()


def test_operator_recovery_finalizes_orphaned_pending_snapshot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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

    assert store.state(handle) == "running"
    assert store.recovery_required(handle)
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
    # snapshot degrades result reads (below), never the run outcome itself
    # (review v0.5.16 / blocker 2).
    assert store.state(handle) == "succeeded"
    assert not store.recovery_required(handle)
    assert store.live_runs() == []
    unavailable = app.status(run_id=run_id)
    assert unavailable["result_source"] == "terminal_snapshot_unavailable"
    assert unavailable["run"]["state"] == "succeeded"
    assert unavailable["run"]["failure"]["code"] == "result_snapshot_unavailable"


def test_operator_recovery_keeps_unresolved_launch_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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
    assert not store.cleanup_recovery_path(run_id).exists()
    assert not store.status_path(run_id).exists()
    assert store.state(handle) == "running"
    assert store.recovery_required(handle)
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
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-registry-only-recovery"
    handle = stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)

    # A real run binds the tree in preflight, long before it registers an
    # attempt into it. Fabricating the registry entry without the binding would
    # leave a tree this release correctly reads as unmarked pre-cutover state.
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

    runner_cleanup_calls: list[None] = []
    trial_cleanup_allowed = False
    trial_cleanup_calls = 0

    def trial_cleanup(_identity: StaleProcessIdentity) -> bool:
        nonlocal trial_cleanup_calls
        trial_cleanup_calls += 1
        return trial_cleanup_allowed

    monkeypatch.setattr(
        "phasesweep.mcp.recovery.kill_stale_group",
        _counting_success_callback(runner_cleanup_calls),
    )
    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", trial_cleanup)
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", trial_cleanup)

    dry = recover_run_cli(registry.state_dir, run_id)

    assert dry.exit_code == 0, dry.output
    assert "reconcile 1 registered attempt" in dry.output
    assert len(runner_cleanup_calls) == 0
    assert trial_cleanup_calls == 0

    refused = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert refused.exit_code != 0
    assert "may still have a live process group" in refused.output
    assert len(runner_cleanup_calls) == 1
    assert trial_cleanup_calls == 1
    assert entry_path.is_file()
    assert not store.status_path(run_id).exists()
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.state(handle) == "running"
    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        app.launch("srv")

    trial_cleanup_allowed = True
    confirmed = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert "reconciled 1 registered attempt" in confirmed.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["registered_attempts_reconciled"] == 1
    assert not entry_path.exists()
    assert store.state(handle) == "failed"


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
    experiment = load_config(attempt_config)
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
        # The stale trial was written against the later ledger, which also
        # bound the tree to it. Recovery loads the original config, and this
        # release refuses to operate a tree bound to a different ledger. The
        # realistic shape is therefore a tree owned by the config being
        # recovered, with only the registry entry still naming the old
        # locator -- which is exactly the reconciliation under test.
        recovery_experiment = load_config(config)
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
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
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
            assert frozen_phase["trials"]["FAIL"] == 1
            assert frozen_phase["running_attempts"] == []
        assert store.state(handle) == "failed"
    else:
        assert result.exit_code != 0
        assert "could not confirm any trial-level cleanup evidence" in result.output
        assert not recovery_path.exists()
        assert store.state(handle) == "running"
        with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
            app.launch("srv")


def test_operator_recovery_refuses_to_rebuild_missing_historical_snapshot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
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
    """Recovery clears cleanup uncertainty from two distinct evidence branches that
    share the same launch-refusal -> dry-run -> --confirm -> relaunch scaffold: a
    terminal trial already recorded as cleanup-uncertain
    (``write_uncertain_failed_trial``, pinned via
    ``cleanup_uncertain_terminal_trials``) and a stale RUNNING trial reaped by the
    recovery pass itself (``write_launched_stale_trial``, pinned via
    ``reaped_running_trials``).
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
        exp = load_config(config)
        assert isinstance(exp, Experiment)
        status_kwargs["result_snapshot"] = capture_result_snapshot(exp)
    write_unsafe_cleanup_status(store, run_id, **status_kwargs)
    if expect_running_before_confirm:
        assert app.status(run_id=run_id)["phases"][0]["running_trials_total"] == 1

    runner_cleanup_calls: list[tuple[int | None, int | None, int | None]] = []
    trial_cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def fake_runner_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
        grace_seconds: float = 30.0,
    ) -> bool:
        runner_cleanup_calls.append((pid, saved_starttime, pgid))
        return True

    def fake_trial_cleanup(identity: StaleProcessIdentity) -> bool:
        trial_cleanup_calls.append((identity.pid, identity.proc_starttime, identity.pgid))
        return True

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", fake_runner_cleanup)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process", fake_trial_cleanup
    )
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", fake_trial_cleanup)

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
    assert store.state(handle) == "running"
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
    assert store.state(handle) == "failed"
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
            "PRUNED": 0,
            "FAIL": 1,
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
    """Pins that a --confirm retry after an interrupted
    ``finalize_result_snapshot`` call reuses the cleanup-recovery evidence
    persisted by the first (failed) attempt instead of re-running runner/trial
    cleanup or re-reaping the stale trial.
    """
    run_id = "srv-retry-result-repair"
    runner_cleanup_calls: list[None] = []
    trial_cleanup_calls: list[None] = []

    app, store, handle, attempt_id, recover = _stage_stale_running_recovery_scaffold(
        tmp_path,
        run_id=run_id,
        error_class="UnsafeProcessCleanupError",
        include_generation_record=True,
        mark_cleanup_uncertain=False,
        snapshot_bound_to_generation=True,
        kill_stale_group_stub=_counting_success_callback(runner_cleanup_calls),
        cleanup_trial_stub=_counting_success_callback(trial_cleanup_calls),
        monkeypatch=monkeypatch,
        earlier_boot=earlier_boot,
    )
    assert store.state(handle) == ("failed" if earlier_boot else "running")
    assert store.recovery_required(handle)

    snapshot_calls = 0
    snapshot_attempt_ids: list[set[str]] = []

    def flaky_snapshot(
        snapshot: dict,
        *,
        confirmed_attempt_ids=(),
        confirmed_attempt_locations=None,
    ) -> dict:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshot_attempt_ids.append(set(confirmed_attempt_ids))
        if snapshot_calls == 1:
            # An operator-repairable failure: recovery reports it as a refusal,
            # whereas a defect would reach the internal-error boundary instead.
            raise OSError("snapshot finalization failed")
        return finalize_result_snapshot(
            snapshot,
            confirmed_attempt_ids=confirmed_attempt_ids,
            confirmed_attempt_locations=confirmed_attempt_locations,
        )

    monkeypatch.setattr("phasesweep.mcp.recovery.finalize_result_snapshot", flaky_snapshot)

    first = recover()

    assert first.exit_code != 0
    assert "failed to finalize terminal result snapshot" in first.output
    assert store.cleanup_recovery_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert store.recovery_required(handle)
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "complete"
    frozen_status = app.status(run_id=run_id)
    assert frozen_status["result_source"] == "frozen_run_snapshot"
    assert frozen_status["phases"][0]["trials"]["RUNNING"] == 1
    assert len(runner_cleanup_calls) == (0 if earlier_boot else 1)
    assert len(trial_cleanup_calls) == 1

    retry = recover()

    assert retry.exit_code == 0, retry.output
    assert "Finalized stored terminal result snapshot" in retry.output
    assert len(runner_cleanup_calls) == (0 if earlier_boot else 1)
    assert len(trial_cleanup_calls) == 1
    assert snapshot_calls == 2
    assert snapshot_attempt_ids == [{attempt_id}, {attempt_id}]
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    phase = terminal["result_snapshot"]["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["FAIL"] == 1
    assert phase["generation_trials"]["RUNNING"] == 0
    assert phase["generation_trials"]["FAIL"] == 1
    assert not store.recovery_required(handle)
    recovered_status = app.status(run_id=run_id)["phases"][0]
    assert recovered_status["terminal_trials_this_run"] == 1
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
    assert not store.recovery_required(handle)


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
    assert store.recovery_required(handle)
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
    monkeypatch.setattr(
        "phasesweep.mcp.run_control.kill_stale_group", lambda *_args, **_kwargs: True
    )
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
    assert not store.recovery_required(handle)
    phase = app.status(run_id=run_id)["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["FAIL"] == 1


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
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    study = _load_first_phase_study(config)
    reconciled_attempt_ids: set[str] = set()
    reconciled_attempt_generations: dict[str, str] = {}
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        lambda _identity: True,
    )
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
        lambda _identity: True,
    )
    assert (
        _reap_stale_trials(
            study,
            experiment,
            experiment.phases[0].name,
            recovered_attempt_ids=reconciled_attempt_ids,
            recovered_attempt_generations=reconciled_attempt_generations,
        )
        == 1
    )
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
    assert store.state(handle) == "cancelled"
    assert store.live_runs() == []


@pytest.mark.integration
def test_operator_cleanup_recovery_retry_counts_persisted_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins that a --confirm retry after an interrupted
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
    assert phase["trials"]["FAIL"] == 1
    assert store.state(handle) == "cancelled"
    assert store.live_runs() == []


def test_operator_recovery_consumes_terminal_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first_run = "srv-terminal-first"
    trial_number = write_uncertain_failed_trial(
        config,
        generation_id=first_run,
    )
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")

    def fake_cleanup(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", fake_cleanup)

    stage_dead_run(store, first_run, config, reg.id, cleanup_uncertain=False)
    write_unsafe_cleanup_status(store, first_run)

    first = recover_run_cli(registry.state_dir, first_run, confirm=True)

    assert first.exit_code == 0, first.output
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]

    second_run = "srv-terminal-second"
    second_handle = stage_dead_run(store, second_run, config, reg.id, cleanup_uncertain=False)
    write_unsafe_cleanup_status(store, second_run)

    replay = recover_run_cli(registry.state_dir, second_run, confirm=True)

    assert replay.exit_code != 0
    assert "could not confirm any trial-level cleanup evidence" in replay.output
    assert not store.cleanup_recovery_path(second_run).exists()
    assert store.state(second_handle) == "running"


def test_operator_recovery_retry_counts_ledger_evidence_after_lost_recovery_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the study-ledger write but before the run-level recovery
    record must not wedge the run forever: the --confirm retry counts the
    durable ledger evidence for this run's own generation instead of refusing
    with "no trial-level cleanup evidence" (review v0.5.17 gap hunt)."""
    from phasesweep.runtime.files import private_atomic_write_text as real_write

    run_id = "srv-ledger-retry"
    store, handle, trial_number, config, recover = _stage_terminal_uncertain_run(
        tmp_path, monkeypatch, run_id=run_id, mark_uncertain=False
    )
    experiment = load_config(config)
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
    assert store.recovery_required(handle)
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "complete"
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    assert app.status(run_id=run_id)["result_source"] == "frozen_run_snapshot"
    assert app.winners(run_id=run_id)["result_source"] == "frozen_run_snapshot"

    monkeypatch.setattr("phasesweep.mcp.recovery.private_atomic_write_text", real_write)

    retry = recover()

    assert retry.exit_code == 0, retry.output
    recovery = json.loads(recovery_record.read_text())
    assert recovery["cleanup_confirmed"] is True
    assert store.live_runs() == []


def test_operator_recovery_retry_clears_marker_after_terminal_only_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupting ``clear_cleanup_uncertain`` after a recovery whose only
    evidence was a cleanup-uncertain terminal trial used to wedge the run:
    the retry found nothing fresh to recover and refused even though the
    recovery record was already on disk. The study ledger now supplies the
    evidence (review v0.5.17 gap hunt; the existing retry pin covered only
    the reaped-RUNNING-trial variant, whose attempt ids persist in status)."""
    run_id = "srv-terminal-marker-retry"
    store, handle, _trial_number, _config_path, recover = _stage_terminal_uncertain_run(
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
    assert store.live_runs() == []


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
    ],
)
def test_operator_recovery_refuses_unverifiable_run(
    tmp_path: Path,
    snapshot_suffix: bytes,
    terminal_without_evidence: bool,
    expected_message: str,
) -> None:
    """Recovery refuses missing cleanup evidence and altered launch snapshots."""
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-unverifiable-recovery"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=reaped_pid(),
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes() + snapshot_suffix)
    if terminal_without_evidence:
        write_unsafe_cleanup_status(store, run_id)
    else:
        store.mark_cleanup_uncertain(handle)

    result = recover_run_cli(registry.state_dir, run_id, confirm=True)

    assert result.exit_code != 0
    assert expected_message in result.output
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.state(handle) == "running"
    if not terminal_without_evidence:
        assert store.cleanup_uncertain_path(run_id).is_file()
