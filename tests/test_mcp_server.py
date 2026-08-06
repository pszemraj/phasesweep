"""MCP server logic that does not need a real detached runner."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml
from click.testing import CliRunner

from phasesweep.cli import cli as cli_main
from phasesweep.config import Experiment, Phase, Sampler, load_config
from phasesweep.engine import (
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    TerminalReport,
    run_experiment,
)
from phasesweep.engine.errors import StudyFingerprintMismatchError, StudySchemaMismatchError
from phasesweep.engine.guards import _experiment_lock, _phase_fingerprint, _reap_stale_trials
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _last_successful_generation_path,
    _trial_dir_for,
    _winner_path,
)
from phasesweep.engine.trial import UnsafeProcessCleanupError
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import (
    ConcurrencyLimitError,
    RunLaunchUnsettledError,
    UnknownExperimentError,
)
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.server import (
    TOOL_AWAIT_RUN,
    TOOL_GET_RUN_RESULTS,
    TOOL_GET_RUN_STATUS,
    TOOL_LAUNCH_RUN,
    GetRunStatusResult,
    PhaseSweepMCP,
    _safe_tool,
    _status_next_action,
)
from phasesweep.mcp.snapshots import capture_result_snapshot, finalize_result_snapshot
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    _write_process_identity,
    read_boot_id,
    read_proc_starttime,
)
from tests.conftest import make_experiment, write_constant_trainer
from tests.mcp_helpers import (
    claim_runner_handle,
    make_mcp_app,
    make_run_handle,
    mcp_experiment_config_text,
    patch_popen_capture,
    runner_main,
    write_mcp_catalog,
    write_run_status,
)

ALLOW_SIDE_EFFECTS = {"launch": True, "cancel": True, "from_phase": True}


def _config(tmp_path: Path, *, name: str = "srv", phases: str | None = None) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(mcp_experiment_config_text(tmp_path, name=name, phases=phases))
    return path


def _catalog(
    tmp_path: Path,
    config: Path,
    allow: dict[str, bool] | None = None,
    *,
    visible_params: object | None = None,
) -> Path:
    return write_mcp_catalog(
        tmp_path,
        {"srv": config},
        allow=allow,
        visible_params=None if visible_params is None else {"srv": visible_params},
        filename="srv.catalog.yaml",
    )


def _write_trial_process_identity(
    trial_dir: Path,
    *,
    attempt_id: str,
    pid: int,
    starttime: int,
) -> None:
    _write_process_identity(
        trial_dir / PROCESS_IDENTITY_FILE,
        StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id=attempt_id,
            pid=pid,
            pgid=pid,
            proc_starttime=starttime,
            boot_id=read_boot_id(),
        ),
    )


def _write_cleanup_uncertain_failed_trial(
    config: Path, *, generation_id: str = "stale-generation"
) -> int:
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = optuna.create_study(
        study_name=f"{exp.experiment}::{phase.name}",
        storage=exp.storage,
        direction="minimize",
    )
    trial = study.ask()
    attempt_id = f"stale-attempt-{trial.number}"
    trial_dir = _trial_dir_for(
        exp,
        phase.name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    _write_trial_process_identity(
        trial_dir,
        attempt_id=attempt_id,
        pid=4242,
        starttime=111,
    )
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return trial.number


def _write_stale_running_trial(
    config: Path,
    *,
    cleanup_confirmed: bool | None = None,
    generation_id: str = "stale-generation",
) -> int:
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = optuna.create_study(
        study_name=f"{exp.experiment}::{phase.name}",
        storage=exp.storage,
        direction="minimize",
    )
    trial = study.ask()
    attempt_id = f"stale-attempt-{trial.number}"
    trial_dir = _trial_dir_for(
        exp,
        phase.name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    _write_trial_process_identity(
        trial_dir,
        attempt_id=attempt_id,
        pid=4343,
        starttime=222,
    )
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    if cleanup_confirmed is not None:
        trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, cleanup_confirmed)
    return trial.number


def _load_first_phase_study(config: Path) -> optuna.Study:
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    return optuna.load_study(
        study_name=f"{exp.experiment}::{phase.name}",
        storage=exp.storage,
    )


def _load_phase_trial(config: Path, trial_number: int) -> optuna.trial.FrozenTrial:
    study = _load_first_phase_study(config)
    return next(trial for trial in study.get_trials(deepcopy=False) if trial.number == trial_number)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _interrupt_first_cleanup_clear() -> Callable[[RunStore, RunHandle], None]:
    """Return a cleanup-marker clear that fails once, then delegates normally."""
    real_clear = RunStore.clear_cleanup_uncertain
    clear_calls = 0

    def interrupt_first_clear(candidate_store: RunStore, candidate: RunHandle) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            raise RuntimeError("interrupted before clearing cleanup marker")
        real_clear(candidate_store, candidate)

    return interrupt_first_clear


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
) -> tuple[PhaseSweepMCP, RunStore, RunHandle, str, list[str]]:
    """Shared scaffold for the interrupted-recovery --confirm retry tests: a
    stale RUNNING trial, a run handle, and a terminal status with a captured
    ``result_snapshot``, with ``kill_stale_group``/``cleanup_stale_trial_process``
    stubbed to succeed. Callers monkeypatch their own fail-once target and
    invoke the returned command twice. Returns ``(app, store, handle,
    attempt_id, command)``.
    """
    config = _config(tmp_path)
    trial_number = _write_stale_running_trial(
        config,
        cleanup_confirmed=False,
        generation_id=run_id,
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
        pid=999999,
        starttime=111,
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
    monkeypatch.setattr("phasesweep.cli.kill_stale_group", kill_stale_group_stub)
    monkeypatch.setattr(
        "phasesweep.engine.guards.cleanup_stale_trial_process",
        cleanup_trial_stub,
    )
    command = [
        "mcp",
        "recover-run",
        "--state-dir",
        str(registry.state_dir),
        "--run-id",
        run_id,
        "--confirm",
    ]
    return app, store, handle, attempt_id, command


def test_safe_tool_returns_safe_mcp_error() -> None:
    @_safe_tool
    def boom() -> None:
        raise UnknownExperimentError("srv")

    with pytest.raises(ValueError, match="unknown experiment id 'srv'"):
        boom()


def test_concurrency_limit_error_bounds_actionable_run_ids() -> None:
    run_ids = [
        "block-alpha",
        "block-beta",
        "block-gamma",
        "block-delta",
        "block-epsilon",
        "block-zeta",
        "block-eta",
    ]

    message = str(ConcurrencyLimitError(7, 3, run_ids))

    for run_id in run_ids[:5]:
        assert repr(run_id) in message
    for run_id in run_ids[5:]:
        assert repr(run_id) not in message
    assert "2 more active" in message
    assert "await_run" in message
    assert "only after that run is terminal" in message


def test_safe_tool_redacts_unexpected_exception() -> None:
    @_safe_tool
    def boom() -> None:
        raise OSError("/tmp/SECRET_PATH/config.yaml")

    with pytest.raises(ValueError) as excinfo:
        boom()
    assert str(excinfo.value) == (
        "internal server error in boom; report it to the operator and do not retry immediately"
    )
    assert "SECRET_PATH" not in str(excinfo.value)


def test_launch_permission_denied_before_spawn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))

    with pytest.raises(Exception, match="action 'launch' is not permitted"):
        app.launch("srv")


def test_from_phase_permission_denied_before_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(
        _catalog(tmp_path, config, allow={"launch": True, "cancel": True, "from_phase": False}),
    )

    with pytest.raises(Exception, match="action 'from_phase' is not permitted"):
        app.launch("srv", from_phase="p")


def test_invalid_from_phase_rejected_before_spawn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with pytest.raises(Exception, match="phase 'missing' is not a phase"):
        app.launch("srv", from_phase="missing")


RESUMABLE_PHASES = """\
  - name: p
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  - name: q
    inherits: [p]
    n_trials: 1
    sampler: { type: random, seed: 1 }
    search_space:
      wd: { type: float, low: 0.0, high: 0.1 }
"""


def test_resume_requires_prior_winner(tmp_path: Path) -> None:
    config = _config(tmp_path, phases=RESUMABLE_PHASES)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with pytest.raises(Exception, match="earlier phase 'p' has no winner yet"):
        app.launch("srv", from_phase="q")


def _write_winner_yaml(
    experiment: Experiment,
    phase_name: str,
    *,
    phase_fingerprint: str,
    incomplete: bool = False,
    generation_id: str | None = None,
) -> None:
    path = (
        _winner_path(experiment, phase_name)
        if generation_id is None
        else _generation_winner_path(experiment, generation_id, phase_name)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "phase": phase_name,
                "trial_number": 0,
                "metric": {experiment.metric.name: 0.123, "goal": experiment.metric.goal},
                "params": {"lr": 0.001},
                "effective_overrides": {"lr": 0.001},
                "completion": {"incomplete": incomplete},
                "phase_fingerprint": phase_fingerprint,
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": phase_name,
                    "trial_number": 0,
                    "generation_id": generation_id,
                    "attempt_id": None,
                    "study": None,
                },
            }
        )
    )


@pytest.mark.parametrize(
    ("fingerprint_override", "incomplete"),
    [
        pytest.param("0" * 64, False, id="stale"),
        pytest.param(None, True, id="incomplete"),
    ],
)
def test_resume_rejects_incompatible_winner_before_spawn(
    tmp_path: Path,
    fingerprint_override: str | None,
    incomplete: bool,
) -> None:
    config = _config(tmp_path, phases=RESUMABLE_PHASES)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase_fingerprint = fingerprint_override or _phase_fingerprint(exp, exp.phases[0], {})
    _write_winner_yaml(
        exp,
        "p",
        phase_fingerprint=phase_fingerprint,
        incomplete=incomplete,
    )

    with pytest.raises(Exception, match="compatible winner"):
        app.launch("srv", from_phase="q")

    assert store.list_handles() == []


def test_launch_refuses_config_changed_after_registry_load(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    config.write_text(config.read_text().replace("python train.py", "python changed.py"))

    with pytest.raises(Exception, match="changed since server startup"):
        app.launch("srv")

    assert store.list_handles() == []


def test_launch_passes_config_snapshot_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(
        _catalog(
            tmp_path,
            config,
            allow=ALLOW_SIDE_EFFECTS,
            visible_params=["lr"],
        )
    )
    captured = patch_popen_capture(monkeypatch)
    original_create = store.create
    snapshots_before_publish: list[bytes] = []

    def create_after_snapshot(handle: RunHandle) -> None:
        snapshots_before_publish.append(store.config_snapshot_path(handle.run_id).read_bytes())
        original_create(handle)

    monkeypatch.setattr(store, "create", create_after_snapshot)

    result = app.launch("srv")

    cmd = captured["cmd"]
    config_arg = Path(cmd[cmd.index("--config") + 1])
    sha_arg = cmd[cmd.index("--config-sha256") + 1]
    assert config_arg != config.resolve()
    assert config_arg.read_bytes() == config.read_bytes()
    assert snapshots_before_publish == [config.read_bytes()]
    assert sha_arg == registry.get("srv").config_sha256
    assert config_arg == registry.state_dir / "logs" / f"{result['run_id']}.config.yaml"
    assert Path(cmd[cmd.index("--state-dir") + 1]) == registry.state_dir
    assert cmd[cmd.index("--experiment-id") + 1] == "srv"
    assert list(config_arg.parent.glob("*.tmp")) == []
    assert list(config_arg.parent.glob(".*.tmp")) == []
    handle = store.get(result["run_id"])
    assert handle is not None
    assert handle.launch_state == "spawned"
    assert handle.allow_cancel is True
    assert handle.visible_params_at_launch == ["lr"]
    assert cmd[cmd.index("--started-at") + 1] == handle.started_at


def test_launch_retries_run_id_collision_without_touching_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    collision_id = "srv-collision"
    fresh_id = "srv-fresh"
    existing = make_run_handle(
        run_id=collision_id,
        experiment_id="srv",
        config_sha256=registry.get("srv").config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(existing)
    store.config_snapshot_path(collision_id).write_bytes(b"existing config\n")
    store.log_path(collision_id).write_bytes(b"existing log\n")
    write_run_status(store, collision_id, returncode=0, cleanup_confirmed=True)
    before = {
        path: path.read_bytes()
        for path in (
            registry.state_dir / "runs" / f"{collision_id}.json",
            store.config_snapshot_path(collision_id),
            store.log_path(collision_id),
            store.status_path(collision_id),
        )
    }
    minted = iter((collision_id, fresh_id))
    monkeypatch.setattr(store, "new_run_id", lambda _experiment_id: next(minted))

    result = app.launch("srv")

    assert result["run_id"] == fresh_id
    assert store.get(collision_id) == existing
    assert {path: path.read_bytes() for path in before} == before


def test_runner_persists_spawned_handle_for_restart_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "srv-recover"
    started_at = "2026-06-24T00:00:00Z"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )
    calls: list[tuple[str, str | None, bool]] = []

    def fake_run_experiment(
        config_obj: Experiment,
        *,
        from_phase: str | None,
        dry_run: bool,
        terminal_callback,
        generation_id: str,
    ) -> None:
        assert generation_id == run_id
        calls.append((config_obj.experiment, from_phase, dry_run))
        generation_path = _generation_record_path(config_obj, generation_id)
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(f"generation_id: {generation_id}\n")
        terminal_callback(
            TerminalReport(
                generation_id=generation_id,
                primary_error=None,
                cleanup_confirmed=True,
                recovered_attempt_ids=frozenset(),
                uncertain_attempt_ids=frozenset(),
            )
        )

    monkeypatch.setattr("phasesweep.mcp.runner.run_experiment", fake_run_experiment)

    def capture_without_fake_trial_storage(
        experiment: Experiment,
        *,
        generation_id: str,
        engine_winners: object = None,
    ) -> dict[str, object]:
        return capture_result_snapshot(
            experiment,
            generation_id=generation_id,
        )

    monkeypatch.setattr(
        "phasesweep.mcp.runner.capture_result_snapshot",
        capture_without_fake_trial_storage,
    )

    assert (
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(store.status_path(run_id)),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "srv",
                "--started-at",
                started_at,
            ]
        )
        == 0
    )

    handle = store.get(run_id)
    assert handle is not None
    assert handle.launch_state == "spawned"
    assert handle.experiment_id == "srv"
    assert handle.config_sha256 == config_sha256
    assert handle.pid == os.getpid()
    assert handle.pgid == (os.getpgrp() if hasattr(os, "getpgrp") else os.getpid())
    assert handle.pid_starttime == read_proc_starttime(os.getpid())
    assert handle.started_at == started_at
    assert store.state(handle) == "succeeded"
    assert calls == [("srv", None, False)]
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["result_snapshot"]["status"]["phases"][0]["phase"] == "p"
    assert terminal["result_snapshot"]["winners"] == []


def test_launch_does_not_spawn_when_pending_handle_create_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    def fail_create(handle: RunHandle) -> None:
        raise OSError("runs directory is not writable")

    monkeypatch.setattr(store, "create", fail_create)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert "cmd" not in captured
    assert store.list_handles() == []
    assert list((tmp_path / "state" / "logs").glob("*.config.yaml")) == []


def test_launch_finalizes_pending_handle_when_popen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise OSError("runner executable is unavailable")

    monkeypatch.setattr("phasesweep.mcp.server.subprocess.Popen", fail_popen)

    with pytest.raises(OSError, match="runner executable"):
        app.launch("srv")

    (handle,) = store.list_handles()
    assert handle.launch_state == "launching"
    assert store.state(handle) == "failed"
    assert not store.recovery_required(handle)
    assert store.live_runs() == []
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["error_class"] == "OSError"
    assert terminal["cleanup_confirmed"] is True
    assert terminal["result_snapshot_state"] == "failed"
    assert terminal["failure"] == {
        "code": "internal_error",
        "stage": "preflight",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            "Ask the operator to inspect the PhaseSweep server diagnostics before retrying."
        ),
    }
    latest = app.latest_run("srv")
    failure = latest["run"]["failure"]
    assert failure["code"] == "result_snapshot_unavailable"
    assert failure["cause"] == terminal["failure"]
    assert app.status(run_id=handle.run_id)["run"]["failure"] == failure


def test_restarted_server_reserves_unresolved_launching_handle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _first, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    pending = make_run_handle(
        run_id="srv-launch-gap",
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    store.create(pending)

    restarted = PhaseSweepMCP(registry, RunStore(registry.state_dir))
    with pytest.raises(Exception, match="already has a running sweep"):
        restarted.launch("srv")

    assert store.recovery_required(pending)
    assert store.live_runs() == [pending]


def test_cancel_refuses_unsettled_launch_without_runner_identity(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("cleanup_confirmed", "expected_state"),
    [(True, "failed"), (False, "running")],
)
def test_launch_refuses_runner_without_linux_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_confirmed: bool,
    expected_state: str,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, saved_starttime, pgid))
        return cleanup_confirmed

    monkeypatch.setattr("phasesweep.mcp.server.read_proc_starttime", lambda _pid: None)
    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_cleanup)

    with pytest.raises(RuntimeError, match="has no Linux /proc start time"):
        app.launch("srv")

    handles = store.list_handles()
    assert len(handles) == 1
    assert handles[0].launch_state == "launching"
    assert store.state(handles[0]) == expected_state
    assert store.recovery_required(handles[0]) is (not cleanup_confirmed)
    assert bool(store.cleanup_uncertain(handles[0])) is (not cleanup_confirmed)
    assert cleanup_calls and cleanup_calls[0][1] is None


def test_launch_terminates_spawned_runner_when_handle_update_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    updated: list[RunHandle] = []
    terminated: list[tuple[int | None, int | None, int | None]] = []

    def fail_update(handle: RunHandle) -> None:
        updated.append(handle)
        raise OSError("runs directory is not writable")

    def fake_kill_stale_group(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        terminated.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(store, "update", fail_update)
    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert [handle.launch_state for handle in updated] == ["spawned"]
    spawned = updated[0]
    assert terminated == [(spawned.pid, spawned.pid_starttime, spawned.pgid)]
    pending = store.get(spawned.run_id)
    assert pending is not None
    assert pending.launch_state == "launching"
    assert store.state(pending) == "failed"


def test_launch_logs_when_cleanup_marker_write_fails_after_update_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    updated: list[RunHandle] = []

    def fail_update(handle: RunHandle) -> None:
        updated.append(handle)
        raise OSError("runs directory is not writable")

    def fail_marker(_handle: RunHandle) -> None:
        raise OSError("logs directory is not writable")

    monkeypatch.setattr(store, "update", fail_update)
    monkeypatch.setattr(store, "mark_cleanup_uncertain", fail_marker)
    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", lambda *args, **kwargs: False)
    caplog.set_level(logging.ERROR, logger="phasesweep.mcp.server")

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert [handle.launch_state for handle in updated] == ["spawned"]
    assert "failed to persist cleanup uncertainty marker" in caplog.text
    assert "original error" in caplog.text
    assert "cleanup uncertain after failed runner launch bookkeeping" in caplog.text


def _launch_with_poison_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    """Launch against a project dir seeded with import-time code and capture the spawn.

    The project directory holds every hook that runs before the runner's first
    statement: a ``phasesweep`` shadow package (found when the cwd is on
    ``sys.path``), a ``sitecustomize`` (imported by ``site`` when it is
    importable at all), and a ``PYTHONSTARTUP``/``PYTHONPATH`` pair exported
    into the server's own environment. Each writes a marker file naming itself.

    The patches are undone before returning so callers can spawn real children:
    patching ``phasesweep.mcp.server.subprocess.Popen`` patches the one shared
    ``subprocess`` module.

    :param Path tmp_path: Test-scoped directory.
    :param pytest.MonkeyPatch monkeypatch: Patcher for env and ``Popen``.
    :return tuple[dict[str, Any], Path, Path, dict[str, str]]: Captured spawn,
        project dir, marker dir, and the poisoned parent environment.
    """
    config = _config(tmp_path)
    project = tmp_path / "project"
    markers = tmp_path / "markers"
    markers.mkdir()
    shadow = project / "phasesweep" / "mcp"
    shadow.mkdir(parents=True)

    def poison(path: Path, name: str) -> None:
        path.write_text(
            "import pathlib\n"
            f"pathlib.Path({str(markers)!r}).joinpath({name!r}).write_text('executed')\n"
        )

    poison(project / "phasesweep" / "__init__.py", "shadow_package")
    poison(shadow / "__init__.py", "shadow_mcp")
    poison(shadow / "runner.py", "shadow_runner")
    poison(project / "sitecustomize.py", "sitecustomize")
    poison(project / "startup.py", "pythonstartup")

    monkeypatch.setenv("PYTHONPATH", str(project))
    monkeypatch.setenv("PYTHONSTARTUP", str(project / "startup.py"))
    monkeypatch.setenv("PYTHONHOME", str(project))
    monkeypatch.setenv("PYTHONEXECUTABLE", str(project / "python"))
    app, _registry, _store = make_mcp_app(
        write_mcp_catalog(
            tmp_path,
            {"srv": config},
            allow=ALLOW_SIDE_EFFECTS,
            cwd={"srv": project},
        )
    )
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    parent_env = dict(os.environ)
    monkeypatch.undo()
    return captured, project, markers, parent_env


def test_launch_spawns_runner_in_neutral_dir_and_passes_project_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child must not start inside the project it is about to run.

    Interpreter startup from the experiment's directory executes project-local
    code before the server can name the process it created (review v0.5.17 /
    blocker 9), so ``Popen`` uses the server-owned state directory and the
    project directory travels as an explicit argument the runner applies later.
    """
    config = _config(tmp_path)
    runner_cwd = tmp_path / "runner-cwd"
    runner_cwd.mkdir()
    app, registry, _store = make_mcp_app(
        write_mcp_catalog(
            tmp_path,
            {"srv": config},
            allow=ALLOW_SIDE_EFFECTS,
            cwd={"srv": runner_cwd},
        )
    )
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    assert captured["cwd"] == str(registry.state_dir)
    assert captured["cwd"] != str(runner_cwd.resolve())
    cmd = captured["cmd"]
    assert cmd[cmd.index("--cwd") + 1] == str(runner_cwd.resolve())


def test_launch_hardens_runner_interpreter_flags_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "attacker"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "attacker"))
    monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "attacker" / "startup.py"))
    monkeypatch.setenv("PYTHONEXECUTABLE", str(tmp_path / "attacker" / "python"))
    monkeypatch.setenv("PHASESWEEP_KEEP_ME", "kept")
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    # The flags must precede -m, or the interpreter treats them as module args.
    assert cmd[1:3] == ["-P", "-s"]
    assert cmd[3:5] == ["-m", "phasesweep.mcp.runner"]
    env = captured["env"]
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    assert "PYTHONEXECUTABLE" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PHASESWEEP_KEEP_ME"] == "kept"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX spawn contract")
def test_spawned_runner_cannot_execute_project_local_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawn a real child under the server's contract; no project code may run.

    The runner module itself is not started (its import graph is irrelevant to
    the question); the child instead reports which ``phasesweep`` the hardened
    interpreter resolves and whether any poisoned hook left a marker.
    """
    captured, project, markers, _parent_env = _launch_with_poison_project(tmp_path, monkeypatch)
    resolved = tmp_path / "resolved.txt"
    probe = (
        "import pathlib, sys\n"
        "import phasesweep\n"
        "pathlib.Path(sys.argv[1]).write_text(phasesweep.__file__ + '\\n' + repr(sys.path))\n"
    )
    cmd = captured["cmd"]
    flags = cmd[1 : cmd.index("-m")]

    completed = subprocess.run(
        [cmd[0], *flags, "-c", probe, str(resolved)],
        cwd=captured["cwd"],
        env=captured["env"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in markers.iterdir()) == []
    # Neither the resolved module nor any sys.path entry comes from the project.
    assert str(project) not in resolved.read_text()


def test_launch_records_the_current_boot_id_in_the_spawned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)

    run_id = app.launch("srv")["run_id"]

    handle = store.get(run_id)
    assert handle is not None
    assert handle.boot_id == read_boot_id()


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
            pid=999999,
            starttime=111,
            allow_cancel=True,
        ),
        boot_id="0" * len(current_boot) if current_boot != "0" * len(current_boot) else "1",
    )
    store.create(handle)
    write_run_status(store, "srv-boot", returncode=0, result_snapshot_state="pending")
    signalled: list[tuple[object, ...]] = []

    def record_kill(*args: object, **kwargs: object) -> bool:
        signalled.append(args)
        return True

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", record_kill)
    assert store.state(handle) == "running"

    result = app.cancel("srv-boot")

    assert signalled == []
    assert result["cleanup_confirmed"] is True
    # The orphaned pending snapshot is a separate operator concern from process
    # cleanup, so it still asks for recovery.
    assert result["recovery_required"] is True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX spawn contract")
def test_project_local_shadow_package_would_run_under_the_old_spawn_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the test above: the poison is genuinely executable.

    Reproducing the pre-fix contract - project cwd, no ``-P``/``-s``, inherited
    environment - must import the shadow package, so a passing hardened case is
    evidence about the fix rather than about an inert fixture.
    """
    _captured, project, markers, parent_env = _launch_with_poison_project(tmp_path, monkeypatch)
    # PYTHONHOME alone would break the interpreter before it could import
    # anything; drop it so the control isolates the cwd and PYTHONPATH hooks.
    env = {name: value for name, value in parent_env.items() if name != "PYTHONHOME"}

    completed = subprocess.run(
        [sys.executable, "-c", "import phasesweep.mcp.runner"],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in markers.iterdir()) == [
        "shadow_mcp",
        "shadow_package",
        "shadow_runner",
        "sitecustomize",
    ]


@pytest.mark.parametrize("method_name", ["status", "winners"])
def test_run_tools_read_launched_config_snapshot_after_catalog_edit(
    tmp_path: Path,
    method_name: str,
) -> None:
    config = _config(tmp_path)
    catalog = _catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    _app, registry, store = make_mcp_app(catalog)
    reg = registry.get("srv")
    exp = reg.experiment
    run_id = "srv-launched"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
        )
    )
    if method_name == "winners":
        _write_winner_yaml(
            exp,
            "p",
            phase_fingerprint="0" * 64,
            generation_id=run_id,
        )
    config.write_text(config.read_text().replace("- name: p", "- name: edited"))

    restarted_registry = Registry.load(catalog)
    restarted_app = PhaseSweepMCP(restarted_registry, store)

    by_run = getattr(restarted_app, method_name)(run_id=run_id)
    assert [phase["phase"] for phase in by_run["phases"]] == ["p"]
    if method_name == "status":
        assert by_run["run"]["state"] == "running"
    else:
        assert by_run["phases"][0]["metric"] == 0.123

    by_experiment = getattr(restarted_app, method_name)(experiment_id="srv")
    assert [phase["phase"] for phase in by_experiment["phases"]] == ["p"]
    if method_name == "status":
        assert by_experiment["run"]["run_id"] == run_id


def test_winners_by_run_id_defaults_to_redacted_params_after_decatalog(
    tmp_path: Path,
) -> None:
    old_config = _config(tmp_path, name="old")
    old_exp = load_config(old_config)
    assert isinstance(old_exp, Experiment)
    run_id = "old-launched"
    _write_winner_yaml(
        old_exp,
        "p",
        phase_fingerprint=_phase_fingerprint(old_exp, old_exp.phases[0], {}),
        generation_id=run_id,
    )
    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, visible_params={"other": "all"})
    )
    snapshot = old_config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id="old",
            config_sha256=hashlib.sha256(snapshot).hexdigest(),
        )
    )

    result = app.winners(run_id=run_id)

    assert result["experiment_id"] == "old"
    assert result["phases"][0]["params"] == {"lr": "<redacted>"}


@pytest.mark.parametrize("method_name", ["status", "winners"])
def test_run_tools_reject_config_snapshot_hash_mismatch(
    tmp_path: Path,
    method_name: str,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    run_id = "srv-mismatch"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=registry.get("srv").id,
            config_sha256=hashlib.sha256(snapshot + b"\n").hexdigest(),
        )
    )

    with pytest.raises(Exception, match="saved config snapshot"):
        getattr(app, method_name)(run_id=run_id)


@pytest.mark.parametrize("method_name", ["status", "winners"])
def test_read_tools_require_exactly_one_identifier(tmp_path: Path, method_name: str) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))
    method = getattr(app, method_name)

    with pytest.raises(Exception, match="exactly one of experiment_id or run_id"):
        method()
    with pytest.raises(Exception, match="exactly one of experiment_id or run_id"):
        method(experiment_id="srv", run_id="nope-123")
    with pytest.raises(Exception, match="unknown run id"):
        method(run_id="nope-123")


def test_experiment_status_next_action_steers_a_finished_experiment_to_results(
    tmp_path: Path,
) -> None:
    """A completed experiment must not report "nothing left to do".

    ``live_run_for`` matches only running handles, so an experiment-scoped
    status read of a finished sweep carries ``run: null``. Deriving
    ``next_action`` from the run alone would answer null - the agent's documented
    stop signal - while the winners sit on disk unread.
    """
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-finished"
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.create(
        make_run_handle(run_id=run_id, experiment_id=reg.id, config_sha256=reg.config_sha256)
    )
    write_run_status(store, run_id, returncode=0, cleanup_confirmed=True)

    before = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))

    assert before.run is None
    assert not any(phase.winner_present for phase in before.phases)
    assert _status_next_action(before) is None  # nothing has produced results yet

    _write_winner_yaml(
        reg.experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(reg.experiment, reg.experiment.phases[0], {}),
    )

    after = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))

    assert after.run is None
    assert _status_next_action(after) == TOOL_GET_RUN_RESULTS


def test_status_and_winners_carry_the_publication_integrity_verdict(tmp_path: Path) -> None:
    """Review v0.5.18 / finding F4: an agent must distinguish corrupt from fresh.

    Both read surfaces reported a corrupt publication exactly like a workdir
    that had never published, so an agent would cheerfully propose the one
    action that destroys the evidence.
    """
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = make_experiment(
        experiment="srv",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=[
            Phase(
                name="p",
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={},
            )
        ],
    )
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))

    fresh = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))
    assert fresh.publication_integrity == "absent"
    assert app.winners(experiment_id="srv")["publication_integrity"] == "absent"

    run_experiment(experiment)

    healthy = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))
    assert healthy.publication_integrity == "ok"
    assert healthy.is_published is True
    assert app.winners(experiment_id="srv")["publication_integrity"] == "ok"

    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    payload = app.status(experiment_id="srv")
    corrupt = GetRunStatusResult.model_validate(payload)
    assert corrupt.publication_integrity == "failed"
    assert corrupt.published_generation_id is None
    assert corrupt.is_published is False
    winners = app.winners(experiment_id="srv")
    assert winners["publication_integrity"] == "failed"
    # The verdict is a closed enum, so nothing path-shaped rides along with it.
    serialized = json.dumps([payload, winners], default=str)
    assert str(tmp_path) not in serialized


def test_experiment_status_next_action_awaits_a_live_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-live"
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.create(
        make_run_handle(run_id=run_id, experiment_id=reg.id, config_sha256=reg.config_sha256)
    )

    result = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))

    assert result.run is not None
    assert result.run.state == "running"
    assert _status_next_action(result) == TOOL_AWAIT_RUN


def test_get_run_status_tool_chains_a_finished_experiment_to_results(tmp_path: Path) -> None:
    """The registered tool, not only the helper, must emit the follow-up."""
    pytest.importorskip("mcp")

    from phasesweep.mcp.server import build_server

    config = _config(tmp_path)
    app, registry, _store = make_mcp_app(_catalog(tmp_path, config))
    reg = registry.get("srv")
    _write_winner_yaml(
        reg.experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(reg.experiment, reg.experiment.phases[0], {}),
    )
    server = build_server(app)

    result = asyncio.run(server._tool_manager.get_tool(TOOL_GET_RUN_STATUS).fn(experiment_id="srv"))

    assert result.run is None
    assert result.next_action == TOOL_GET_RUN_RESULTS


def test_winners_apply_catalog_visible_params_policy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    default_app, default_registry, _store = make_mcp_app(_catalog(tmp_path, config))
    reg = default_registry.get("srv")
    _write_winner_yaml(
        reg.experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(reg.experiment, reg.experiment.phases[0], {}),
    )

    assert default_app.winners(experiment_id="srv")["phases"][0]["params"] == {"lr": "<redacted>"}

    visible_app, _visible_registry, _visible_store = make_mcp_app(
        write_mcp_catalog(
            tmp_path,
            {"srv": config},
            visible_params={"srv": ["lr"]},
        )
    )

    assert visible_app.winners(experiment_id="srv")["phases"][0]["params"] == {"lr": 0.001}


@pytest.mark.parametrize(
    ("launch_policy", "current_policy", "expected"),
    [
        pytest.param("none", "all", "<redacted>", id="later-loosening-cannot-reveal"),
        pytest.param("all", "none", "<redacted>", id="later-tightening-redacts"),
        pytest.param(["lr"], "all", 0.001, id="launch-allowlist-remains-visible"),
        pytest.param("all", ["lr"], 0.001, id="current-allowlist-restricts"),
    ],
)
def test_run_winner_visibility_intersects_launch_and_restarted_catalog_policy(
    tmp_path: Path,
    launch_policy: object,
    current_policy: object,
    expected: object,
) -> None:
    config = _config(tmp_path)
    catalog = _catalog(tmp_path, config, visible_params=launch_policy)
    launch_registry = Registry.load(catalog)
    store = RunStore(launch_registry.state_dir)
    reg = launch_registry.get("srv")
    run_id = "srv-historical"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            visible_params_at_launch=reg.visible_params,
        )
    )
    _write_winner_yaml(
        reg.experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(reg.experiment, reg.experiment.phases[0], {}),
        generation_id=run_id,
    )

    _catalog(tmp_path, config, visible_params=current_policy)
    restarted = PhaseSweepMCP(Registry.load(catalog), store)

    assert restarted.winners(run_id=run_id)["phases"][0]["params"]["lr"] == expected


def test_legacy_run_without_visibility_never_falls_back_to_current_catalog(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, visible_params="all"))
    reg = registry.get("srv")
    run_id = "srv-legacy"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            visible_params_at_launch=None,
        )
    )
    _write_winner_yaml(
        reg.experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(reg.experiment, reg.experiment.phases[0], {}),
        generation_id=run_id,
    )

    assert app.winners(run_id=run_id)["phases"][0]["params"] == {"lr": "<redacted>"}


def test_corrupt_run_handle_fails_closed_for_experiment_scoped_winners(tmp_path: Path) -> None:
    """A published generation whose run handle no longer decodes must not be
    rendered under the current catalog policy: the frozen launch authority is
    unreadable, and the current policy may be wider than the launch grant."""
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"srv": config}, visible_params={"srv": "all"})
    )
    reg = registry.get("srv")
    run_id = "srv-corrupt-handle"
    _write_winner_yaml(
        reg.experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(reg.experiment, reg.experiment.phases[0], {}),
        generation_id=run_id,
    )
    # Legacy (pre-manifest) publication: the pointer and summary pass the
    # identity-only gate, so the experiment-scoped read represents run_id.
    summary_path = _generation_summary_path(reg.experiment, run_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump({"experiment": reg.experiment.experiment, "generation_id": run_id})
    )
    pointer = _last_successful_generation_path(reg.experiment)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        yaml.safe_dump({"experiment": reg.experiment.experiment, "generation_id": run_id})
    )
    # The handle file exists but no longer decodes.
    handle_path = store._runs_dir / f"{run_id}.json"
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle_path.write_text("{ not json")

    winners = app.winners(experiment_id="srv")

    assert winners["phases"][0]["params"] == {"lr": "<redacted>"}


def test_await_run_never_starts_a_status_read_past_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow status read must not launch after the deadline: that overshoot is
    what pushes the default await past the client-side request timeout the
    default was chosen to stay under. The just-collected snapshot is returned."""
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-await-bound"
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.create(
        make_run_handle(run_id=run_id, experiment_id=reg.id, config_sha256=reg.config_sha256)
    )

    clock = {"now": 0.0}
    read_starts: list[float] = []
    real_read = app._read_status_target

    def slow_read(**kwargs: Any) -> Any:
        read_starts.append(clock["now"])
        result = real_read(**kwargs)
        clock["now"] += 4.9  # one disk read burns nearly the whole 5s budget
        return result

    async def virtual_sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(app, "_read_status_target", slow_read)
    monkeypatch.setattr(
        "phasesweep.mcp.server.time", types.SimpleNamespace(monotonic=lambda: clock["now"])
    )
    monkeypatch.setattr(
        "phasesweep.mcp.server.asyncio",
        types.SimpleNamespace(sleep=virtual_sleep, to_thread=asyncio.to_thread),
    )

    awaited = asyncio.run(app.await_run(run_id, timeout_seconds=5))

    assert awaited["reason"] == "timeout"
    assert awaited["run"]["state"] == "running"
    # A second equally slow read would have returned at 9.8 virtual seconds,
    # far past the 5s deadline; instead the await returns the first read's
    # snapshot as soon as the cost estimate rules another read out.
    assert read_starts == [0.0]
    assert clock["now"] == pytest.approx(4.9)


def test_list_experiments_pages_catalog(tmp_path: Path) -> None:
    configs = {f"srv{i}": _config(tmp_path, name=f"srv{i}") for i in range(3)}
    registry = Registry.load(write_mcp_catalog(tmp_path, configs))
    store = RunStore(registry.state_dir)
    app = PhaseSweepMCP(registry, store)

    first = app.list_experiments(limit=2)
    assert [item["id"] for item in first["experiments"]] == ["srv0", "srv1"]
    assert first["total_count"] == 3
    assert first["next_cursor"] == "2"

    second = app.list_experiments(limit=2, cursor=first["next_cursor"])
    assert [item["id"] for item in second["experiments"]] == ["srv2"]
    assert second["total_count"] == 3
    assert second["next_cursor"] is None

    with pytest.raises(Exception, match="invalid cursor"):
        app.list_experiments(cursor="not-a-cursor")
    with pytest.raises(Exception, match="limit must be between"):
        app.list_experiments(limit=0)


def test_validate_rejects_config_changed_after_startup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))
    config.write_text(config.read_text() + "\n# changed after server startup\n")

    with pytest.raises(Exception, match="restart the MCP server"):
        app.validate("srv")


def test_latest_run_returns_one_computed_reattachment_handle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config))

    assert app.latest_run("srv") == {
        "experiment_id": "srv",
        "found": False,
        "run": None,
    }

    handle = make_run_handle(
        run_id="srv-current",
        experiment_id=registry.get("srv").id,
        config_sha256=registry.get("srv").config_sha256,
    )
    store.create(handle)

    result = app.latest_run("srv")
    assert result["found"] is True
    assert result["run"] == {
        "run_id": handle.run_id,
        "state": "running",
        "started_at": handle.started_at,
        "recovery_required": False,
        "failure": None,
    }


def test_audit_log_records_side_effects_without_sensitive_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    registry = Registry.load(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    store = RunStore(registry.state_dir)
    audit_path = registry.state_dir / "audit.jsonl"
    app = PhaseSweepMCP(registry, store, audit=AuditLogger(audit_path))
    captured = patch_popen_capture(monkeypatch)

    app.validate("srv")
    launched = app.launch("srv")
    with pytest.raises(Exception, match="already has a running sweep") as exc_info:
        app.launch("srv")
    busy_message = str(exc_info.value)
    assert launched["run_id"] in busy_message
    assert "lost the response" in busy_message
    assert "await_run" in busy_message
    assert "Cancel it only if the user wants" in busy_message

    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [record["tool"] for record in records] == [
        TOOL_LAUNCH_RUN,
        TOOL_LAUNCH_RUN,
    ]
    assert {record["session_id"] for record in records}
    assert all(record["actor"] == "local-stdio" for record in records)
    assert all(record["transport"] == "stdio" for record in records)

    launch_record = records[0]
    assert launch_record["outcome"] == "success"
    assert launch_record["args"] == {"experiment_id": "srv"}
    assert launch_record["resolved"] == {"experiment_id": "srv", "run_id": launched["run_id"]}
    assert launch_record["state_before"] == {"live_runs": 0}
    assert launch_record["state_after"] == {"live_runs": 1, "run_state": "running"}
    assert launch_record["result_counts"] == {"runs": 1}

    busy_record = records[1]
    assert busy_record["outcome"] == "error"
    assert busy_record["args"] == {"experiment_id": "srv"}
    assert busy_record["resolved"] == {"experiment_id": "srv"}
    assert busy_record["state_before"] == {"live_runs": 1}
    assert busy_record["error_type"] == "ExperimentBusyError"
    assert "already has a running sweep" in busy_record["error"]

    blob = audit_path.read_text()
    for needle in ("train.py", "sqlite", str(config), str(tmp_path / "runs")):
        assert needle not in blob
    assert captured["cmd"]  # sanity: the launch path really reached Popen


def test_audit_log_caps_agent_supplied_string_values(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLogger(audit_path)

    audit.record(tool=TOOL_LAUNCH_RUN, args={"cursor": "x" * 500}, outcome="success")

    record = json.loads(audit_path.read_text())
    assert record["args"]["cursor"] == ("x" * 253) + "..."


def test_launch_artifacts_and_audit_are_private_under_permissive_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    catalog = _catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    registry = Registry.load(catalog)
    store = RunStore(registry.state_dir)
    audit_path = registry.state_dir / "audit.jsonl"
    app = PhaseSweepMCP(registry, store, audit=AuditLogger(audit_path))
    captured = patch_popen_capture(monkeypatch)

    old_umask = os.umask(0)
    try:
        result = app.launch("srv")
    finally:
        os.umask(old_umask)

    run_id = result["run_id"]
    assert result["state"] == "running"
    assert captured["cmd"]
    assert _mode(registry.state_dir) == 0o700
    assert _mode(registry.state_dir / "runs") == 0o700
    assert _mode(registry.state_dir / "logs") == 0o700
    assert _mode(registry.state_dir / "runs" / f"{run_id}.json") == 0o600
    assert _mode(store.log_path(run_id)) == 0o600
    assert _mode(store.config_snapshot_path(run_id)) == 0o600
    assert _mode(audit_path) == 0o600


def test_runner_rejects_config_snapshot_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256="0" * 64,
        started_at=started_at,
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config),
                "--config-sha256",
                "0" * 64,
                "--status-path",
                str(status_path),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "srv",
                "--started-at",
                started_at,
            ]
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "RuntimeError"
    assert status["cleanup_confirmed"] is True


def test_preflight_failure_is_actionable_through_run_reads(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = make_experiment(
        experiment="srv",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=[
            Phase(
                name="p",
                n_trials=1,
                fixed_overrides={"k": 1},
                sampler=Sampler(type="random", seed=0),
                search_space={},
            )
        ],
    )
    run_experiment(experiment)
    changed = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"fixed_overrides": {"k": 2}})]}
    )
    config.write_text(yaml.safe_dump(changed.model_dump(mode="json"), sort_keys=False))
    registry = Registry.load(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    store = RunStore(registry.state_dir)
    run_id = "srv-preflight-failure"
    snapshot_path = store.config_snapshot_path(run_id)
    snapshot_path.write_bytes(config.read_bytes())
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )

    with pytest.raises(StudyFingerprintMismatchError, match="different phase config"):
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(snapshot_path),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(store.status_path(run_id)),
                "--state-dir",
                str(registry.state_dir),
                "--experiment-id",
                "srv",
                "--started-at",
                started_at,
            ]
        )

    handle = store.get(run_id)
    assert handle is not None
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["failure"]["code"] == "fingerprint_mismatch"
    app = PhaseSweepMCP(registry, store)
    latest = app.latest_run("srv")
    status = app.status(run_id=run_id)
    awaited = asyncio.run(app.await_run(run_id))
    winners = app.winners(run_id=run_id)

    for payload in (latest["run"], status["run"], awaited["run"]):
        assert payload["failure"]["code"] == "fingerprint_mismatch"
        assert payload["failure"]["retryable"] is False
        assert payload["failure"]["actor"] == "operator"
    assert status["result_source"] == "frozen_run_snapshot"
    assert status["summary_present"] is False
    assert status["phases"][0]["attempts_launched_this_run"] == 0
    assert awaited["reason"] == "terminal"
    assert winners["winner_count"] == 0
    assert winners["failure"]["code"] == "fingerprint_mismatch"


def test_aggregated_schema_preflight_preserves_actionable_failure_category(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        phases="""\
  - name: a
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space: {}
  - name: b
    n_trials: 1
    sampler: { type: random, seed: 1 }
    search_space: {}
""",
    )
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    for phase in experiment.phases:
        study = optuna.create_study(
            study_name=f"{experiment.experiment}::{phase.name}",
            storage=experiment.storage,
            direction="minimize",
        )
        study.add_trial(
            optuna.trial.create_trial(
                value=0.5,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    store = RunStore(tmp_path / "state")
    run_id = "srv-schema-mismatch"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )

    with pytest.raises(StudySchemaMismatchError, match="multiple unsafe studies"):
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(store.status_path(run_id)),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "srv",
                "--started-at",
                started_at,
            ]
        )

    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["failure"]["code"] == "study_schema_mismatch"
    assert terminal["failure"]["retryable"] is False
    assert "unsupported persistent study" in terminal["failure"]["remediation"]


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
            [
                "--run-id",
                run_id,
                "--config",
                str(config),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(status_path),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "srv",
                "--started-at",
                started_at,
            ]
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "UnsafeProcessCleanupError"
    assert status["cleanup_confirmed"] is False


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
    ) -> None:
        del config_obj, from_phase, dry_run
        terminal_callback(
            TerminalReport(
                generation_id=generation_id,
                primary_error=primary,
                cleanup_confirmed=False,
                recovered_attempt_ids=frozenset({"attempt-reconciled"}),
                uncertain_attempt_ids=frozenset({"attempt-1"}),
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
            [
                "--run-id",
                run_id,
                "--config",
                str(config),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(status_path),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "srv",
                "--started-at",
                started_at,
            ]
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "ProcessCleanupUncertainError"
    assert status["cleanup_confirmed"] is False
    assert status["recovered_attempt_ids"] == ["attempt-reconciled"]
    assert status["failure"]["code"] == "cleanup_uncertain"
    assert status["failure"]["stage"] == "cleanup"
    assert status["failure"]["retryable"] is False
    assert status["failure"]["actor"] == "operator"
    assert status["failure"]["cause"]["code"] == "trainer_failed"
    assert status["failure"]["cause"]["stage"] == "execution"


def test_terminal_cleanup_uncertainty_blocks_relaunch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-terminal-uncertain"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    assert store.state(handle) == "running"
    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")


def test_cancel_permission_denied_before_signalling(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(
        _catalog(tmp_path, config, allow={"launch": True, "cancel": False, "from_phase": True}),
    )
    reg = registry.get("srv")
    run_id = "srv-denied"
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            allow_cancel=True,
        )
    )

    with pytest.raises(Exception, match="action 'cancel' is not permitted"):
        app.cancel(run_id)


def test_cancel_cannot_be_enabled_after_launch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-launch-denied-cancel"
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            allow_cancel=False,
        )
    )

    with pytest.raises(Exception, match="action 'cancel' is not permitted"):
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
    assert store.state(handle) == "running"

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        write_run_status(
            store,
            run_id,
            returncode=143,
            error_class="cancelled",
            cleanup_confirmed=True,
        )
        return True

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    assert result == {
        "run_id": run_id,
        "state": "cancelled",
        "cleanup_confirmed": True,
        "recovery_required": False,
    }


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

    with pytest.raises(Exception, match="action 'cancel' is not permitted"):
        app.cancel(run_id)


def test_cancel_uncertain_cleanup_keeps_run_live_for_launch_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-uncertain"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=True,
        pid=999999,
        starttime=111,
    )
    store.create(handle)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        assert store.cleanup_uncertain_path(run_id).is_file()
        return False

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    assert result == {
        "run_id": run_id,
        "state": "running",
        "cleanup_confirmed": False,
        "recovery_required": True,
    }
    assert store.cleanup_uncertain_path(run_id).is_file()

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")


def test_cancel_forced_runner_kill_without_status_keeps_cleanup_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-force-kill"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=True,
    )
    store.create(handle)

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", lambda *args, **kwargs: True)

    result = app.cancel(run_id)

    assert result == {
        "run_id": run_id,
        "state": "running",
        "cleanup_confirmed": False,
        "recovery_required": True,
    }
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.status_path(run_id).exists()

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")


def test_cancel_requires_runner_status_cleanup_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-status-uncertain"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=True,
    )
    store.create(handle)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        write_run_status(
            store,
            run_id,
            returncode=143,
            error_class="cancelled",
            cleanup_confirmed=False,
        )
        return True

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    assert result == {
        "run_id": run_id,
        "state": "running",
        "cleanup_confirmed": False,
        "recovery_required": True,
    }
    assert store.cleanup_uncertain_path(run_id).is_file()


def test_cancel_clears_uncertainty_only_with_runner_cleanup_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-status-confirmed"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        allow_cancel=True,
    )
    store.create(handle)
    store.mark_cleanup_uncertain(handle)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        write_run_status(
            store,
            run_id,
            returncode=143,
            error_class="cancelled",
            cleanup_confirmed=True,
        )
        return True

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    assert result == {
        "run_id": run_id,
        "state": "cancelled",
        "cleanup_confirmed": True,
        "recovery_required": False,
    }
    assert not store.cleanup_uncertain_path(run_id).exists()


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

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

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


def test_operator_recovery_clears_no_status_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-operator-recover"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.mark_cleanup_uncertain(handle)

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    runner = CliRunner()
    dry = runner.invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )

    assert dry.exit_code == 0, dry.output
    assert "Re-run with --confirm" in dry.output
    assert "Recovery preflight" in dry.output
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.status_path(run_id).exists()

    confirmed = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

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

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", spy_kill_stale_group)

    runner = CliRunner()
    dry = runner.invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )
    assert dry.exit_code == 0, dry.output
    assert "still appears live" not in dry.output

    confirmed = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )
    assert confirmed.exit_code == 0, confirmed.output
    assert "Cleared cleanup uncertainty" in confirmed.output
    assert signalled == []
    assert not store.cleanup_uncertain_path(run_id).exists()


def test_operator_recovery_refuses_engine_lock_contention_before_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-recovery-lock-contention"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.mark_cleanup_uncertain(handle)
    cleanup_calls = 0

    def unexpected_cleanup(*args: object, **kwargs: object) -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return True

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", unexpected_cleanup)
    with _experiment_lock(reg.experiment):
        result = CliRunner().invoke(
            cli_main,
            [
                "mcp",
                "recover-run",
                "--state-dir",
                str(registry.state_dir),
                "--run-id",
                run_id,
                "--confirm",
            ],
        )

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
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
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
        assert payload["result_source"] == "current_shared_study"
        assert payload["run"]["state"] == "running"
        assert payload["run"]["recovery_required"] is True
    assert winners["result_source"] == "current_shared_study"
    assert awaited["reason"] == "recovery_required"
    preflight = CliRunner().invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )
    assert preflight.exit_code == 0, preflight.output
    assert "historical terminal snapshot is unavailable" in preflight.output

    confirmed = CliRunner().invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

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


def test_read_tools_use_live_view_while_result_snapshot_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-live-pending-finalization"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="pending",
    )

    status = app.status(run_id=run_id)
    winners = app.winners(run_id=run_id)
    monkeypatch.setattr("phasesweep.mcp.server.AWAIT_MIN_TIMEOUT_SECONDS", 0)
    awaited = asyncio.run(app.await_run(run_id, timeout_seconds=0))

    for payload in (status, awaited):
        assert payload["result_source"] == "current_shared_study"
        assert payload["run"]["state"] == "running"
        assert payload["run"]["recovery_required"] is False
    assert winners["result_source"] == "current_shared_study"
    assert awaited["reason"] == "timeout"


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

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", unexpected_cleanup)

    result = CliRunner().invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert result.exit_code != 0
    assert "launch outcome is unresolved" in result.output
    assert "remains reserved" in result.output
    assert "automated recovery cannot safely" in result.output
    assert not store.cleanup_recovery_path(run_id).exists()
    assert not store.status_path(run_id).exists()
    assert store.state(handle) == "running"
    assert store.recovery_required(handle)
    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")


def test_operator_recovery_refuses_to_rebuild_missing_historical_snapshot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-result-repair"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
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

    runner = CliRunner()
    preflight = runner.invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )
    assert preflight.exit_code != 0
    assert "cannot be rebuilt from the current shared study" in preflight.output
    assert json.loads(store.status_path(run_id).read_text())["result_snapshot_state"] == "failed"

    confirmed = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )
    assert confirmed.exit_code != 0
    assert "cannot be rebuilt from the current shared study" in confirmed.output
    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "failed"
    assert terminal["result_snapshot_error"] == "RuntimeError"


@pytest.mark.parametrize(
    (
        "trial_setup_fn",
        "expected_identity",
        "dry_run_substring",
        "confirm_output_substring",
        "expected_reaped_running",
        "expected_cleanup_uncertain_terminal",
        "expect_running_before_confirm",
    ),
    [
        pytest.param(
            _write_cleanup_uncertain_failed_trial,
            (4242, 111, 4242),
            "recover 1 cleanup-uncertain terminal trial",
            "Cleared cleanup uncertainty",
            0,
            1,
            False,
            id="terminal-cleanup-uncertain-trial",
        ),
        pytest.param(
            lambda config: _write_stale_running_trial(config, cleanup_confirmed=False),
            (4343, 222, 4343),
            "reap 1 stale trial",
            "reaped 1 stale trial",
            1,
            0,
            True,
            id="stale-running-trial",
        ),
    ],
)
def test_operator_recovery_clears_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_setup_fn,
    expected_identity: tuple[int, int, int],
    dry_run_substring: str,
    confirm_output_substring: str,
    expected_reaped_running: int,
    expected_cleanup_uncertain_terminal: int,
    expect_running_before_confirm: bool,
) -> None:
    """Recovery clears cleanup uncertainty from two distinct evidence branches that
    share the same launch-refusal -> dry-run -> --confirm -> relaunch scaffold: a
    terminal trial already recorded as cleanup-uncertain
    (``_write_cleanup_uncertain_failed_trial``, pinned via
    ``cleanup_uncertain_terminal_trials``) and a stale RUNNING trial reaped by the
    recovery pass itself (``_write_stale_running_trial``, pinned via
    ``reaped_running_trials``).
    """
    config = _config(tmp_path)
    trial_number = trial_setup_fn(config)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-cleanup-uncertainty-recover"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    status_kwargs: dict[str, object] = {}
    if expect_running_before_confirm:
        exp = load_config(config)
        assert isinstance(exp, Experiment)
        status_kwargs["result_snapshot"] = capture_result_snapshot(exp)
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
        **status_kwargs,
    )
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

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", fake_runner_cleanup)
    monkeypatch.setattr(
        "phasesweep.engine.guards.cleanup_stale_trial_process",
        fake_trial_cleanup,
    )

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    runner = CliRunner()
    dry = runner.invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )

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

    confirmed = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert confirmed.exit_code == 0, confirmed.output
    assert confirm_output_substring in confirmed.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["run_id"] == run_id
    assert recovery["config_sha256"] == reg.config_sha256
    assert recovery["cleanup_confirmed"] is True
    assert recovery["reaped_running_trials"] == expected_reaped_running
    assert recovery["cleanup_uncertain_terminal_trials"] == expected_cleanup_uncertain_terminal
    assert store.state(handle) == "failed"
    assert runner_cleanup_calls == [(999999, 111, 999999)]
    assert trial_cleanup_calls == [expected_identity]
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

    captured = patch_popen_capture(monkeypatch)
    launched = app.launch("srv")
    assert launched["state"] == "running"
    assert captured["cmd"]


def test_operator_snapshot_repair_retry_reuses_cleanup_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins that a --confirm retry after an interrupted
    ``finalize_result_snapshot`` call reuses the cleanup-recovery evidence
    persisted by the first (failed) attempt instead of re-running runner/trial
    cleanup or re-reaping the stale trial.
    """
    run_id = "srv-retry-result-repair"
    runner_cleanup_calls = 0
    trial_cleanup_calls = 0

    def fake_runner_cleanup(*args: object, **kwargs: object) -> bool:
        nonlocal runner_cleanup_calls
        runner_cleanup_calls += 1
        return True

    def fake_trial_cleanup(*args: object, **kwargs: object) -> bool:
        nonlocal trial_cleanup_calls
        trial_cleanup_calls += 1
        return True

    app, store, handle, attempt_id, command = _stage_stale_running_recovery_scaffold(
        tmp_path,
        run_id=run_id,
        error_class="UnsafeProcessCleanupError",
        include_generation_record=True,
        mark_cleanup_uncertain=False,
        snapshot_bound_to_generation=True,
        kill_stale_group_stub=fake_runner_cleanup,
        cleanup_trial_stub=fake_trial_cleanup,
        monkeypatch=monkeypatch,
    )

    snapshot_calls = 0
    snapshot_attempt_ids: list[set[str]] = []

    def flaky_snapshot(
        snapshot: dict,
        *,
        confirmed_attempt_ids=(),
    ) -> dict:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshot_attempt_ids.append(set(confirmed_attempt_ids))
        if snapshot_calls == 1:
            raise RuntimeError("snapshot finalization failed")
        return finalize_result_snapshot(
            snapshot,
            confirmed_attempt_ids=confirmed_attempt_ids,
        )

    monkeypatch.setattr("phasesweep.cli.finalize_result_snapshot", flaky_snapshot)

    runner = CliRunner()
    first = runner.invoke(cli_main, command)

    assert first.exit_code != 0
    assert "failed to finalize terminal result snapshot" in first.output
    assert store.cleanup_recovery_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert store.recovery_required(handle)
    assert runner_cleanup_calls == 1
    assert trial_cleanup_calls == 1

    retry = runner.invoke(cli_main, command)

    assert retry.exit_code == 0, retry.output
    assert "Finalized stored terminal result snapshot" in retry.output
    assert runner_cleanup_calls == 1
    assert trial_cleanup_calls == 1
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


def test_operator_recovery_uses_runner_reconciliation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run_id = "srv-runner-reconciled"
    trial_number = _write_stale_running_trial(
        config,
        cleanup_confirmed=False,
        generation_id=run_id,
    )
    attempt_id = f"stale-attempt-{trial_number}"
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    study = _load_first_phase_study(config)
    reconciled_attempt_ids: set[str] = set()
    monkeypatch.setattr(
        "phasesweep.engine.guards.cleanup_stale_trial_process",
        lambda _identity: True,
    )
    assert (
        _reap_stale_trials(
            study,
            experiment,
            experiment.phases[0].name,
            recovered_attempt_ids=reconciled_attempt_ids,
        )
        == 1
    )
    assert reconciled_attempt_ids == {attempt_id}

    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="cancelled",
        cleanup_confirmed=False,
        recovered_attempt_ids=sorted(reconciled_attempt_ids),
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment),
    )
    monkeypatch.setattr("phasesweep.cli.kill_stale_group", lambda *args, **kwargs: True)

    result = CliRunner().invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert result.exit_code == 0, result.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert recovery["reaped_running_trials"] == 0
    assert recovery["cleanup_uncertain_terminal_trials"] == 0
    assert store.state(handle) == "cancelled"
    assert store.live_runs() == []


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
    _app, store, handle, attempt_id, command = _stage_stale_running_recovery_scaffold(
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
    runner = CliRunner()

    first = runner.invoke(cli_main, command)

    assert first.exit_code != 0
    assert "interrupted before clearing cleanup marker" in first.output
    assert store.cleanup_uncertain_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]

    retry = runner.invoke(cli_main, command)

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
    trial_number = _write_cleanup_uncertain_failed_trial(config)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")

    def fake_cleanup(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.guards.cleanup_stale_trial_process", fake_cleanup)

    first_run = "srv-terminal-first"
    first_handle = make_run_handle(
        run_id=first_run,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(first_handle)
    store.config_snapshot_path(first_run).write_bytes(config.read_bytes())
    write_run_status(
        store,
        first_run,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    runner = CliRunner()
    first = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            first_run,
            "--confirm",
        ],
    )

    assert first.exit_code == 0, first.output
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]

    second_run = "srv-terminal-second"
    second_handle = make_run_handle(
        run_id=second_run,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999998,
        starttime=112,
    )
    store.create(second_handle)
    store.config_snapshot_path(second_run).write_bytes(config.read_bytes())
    write_run_status(
        store,
        second_run,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    replay = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            second_run,
            "--confirm",
        ],
    )

    assert replay.exit_code != 0
    assert "could not confirm any trial-level cleanup evidence" in replay.output
    assert not store.cleanup_recovery_path(second_run).exists()
    assert store.state(second_handle) == "running"


def _stage_terminal_uncertain_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    mark_uncertain: bool,
) -> tuple[RunStore, RunHandle, int, Path, list[str]]:
    """Stage a run whose only cleanup evidence is one of its own
    cleanup-uncertain terminal trials (generation id == run id, matching the
    detached-runner contract). Returns ``(store, handle, trial_number,
    config, command)``."""
    config = _config(tmp_path)
    trial_number = _write_cleanup_uncertain_failed_trial(config, generation_id=run_id)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    if mark_uncertain:
        store.mark_cleanup_uncertain(handle)
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    def fake_cleanup(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.guards.cleanup_stale_trial_process", fake_cleanup)
    command = [
        "mcp",
        "recover-run",
        "--state-dir",
        str(registry.state_dir),
        "--run-id",
        run_id,
        "--confirm",
    ]
    return store, handle, trial_number, config, command


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
    store, handle, trial_number, config, command = _stage_terminal_uncertain_run(
        tmp_path, monkeypatch, run_id=run_id, mark_uncertain=False
    )
    recovery_record = store.cleanup_recovery_path(run_id)

    def crash_on_recovery_record(path: Path, text: str) -> None:
        if path == recovery_record:
            raise RuntimeError("simulated crash before the recovery record")
        real_write(path, text)

    monkeypatch.setattr("phasesweep.cli.private_atomic_write_text", crash_on_recovery_record)
    runner = CliRunner()

    first = runner.invoke(cli_main, command)

    assert first.exit_code != 0
    # The durable study ledger consumed the trial before the crash; the
    # run-level record never landed.
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]
    assert not recovery_record.exists()

    monkeypatch.setattr("phasesweep.cli.private_atomic_write_text", real_write)

    retry = runner.invoke(cli_main, command)

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
    store, handle, _trial_number, _config_path, command = _stage_terminal_uncertain_run(
        tmp_path, monkeypatch, run_id=run_id, mark_uncertain=True
    )

    monkeypatch.setattr(
        RunStore,
        "clear_cleanup_uncertain",
        _interrupt_first_cleanup_clear(),
    )
    runner = CliRunner()

    first = runner.invoke(cli_main, command)

    assert first.exit_code != 0
    assert store.cleanup_uncertain_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["cleanup_uncertain_terminal_trials"] == 1

    retry = runner.invoke(cli_main, command)

    assert retry.exit_code == 0, retry.output
    assert not store.cleanup_uncertain_path(run_id).exists()
    assert store.live_runs() == []


def test_operator_recovery_refuses_terminal_uncertainty_without_trial_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-terminal-no-evidence"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    result = CliRunner().invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert result.exit_code != 0
    assert "could not confirm any trial-level cleanup evidence" in result.output
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.state(handle) == "running"


def test_operator_recovery_refuses_snapshot_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-bad-snapshot"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes() + b"\n# drifted\n")
    store.mark_cleanup_uncertain(handle)

    result = CliRunner().invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert result.exit_code != 0
    assert "run snapshot hash mismatch" in result.output
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.cleanup_recovery_path(run_id).exists()
