"""MCP server logic that does not need a real detached runner."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml
from click.testing import CliRunner

import phasesweep.mcp.runs as mcp_runs
import phasesweep.mcp.server as mcp_server
from phasesweep.cli import cli as cli_main
from phasesweep.config import (
    ExecutionContext,
    Experiment,
    IntParam,
    JsonEnvelopeExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    load_config,
)
from phasesweep.engine import (
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    TerminalReport,
    run_experiment,
)
from phasesweep.engine.artifact_roots import _validate_artifact_root_binding
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.errors import StudyFingerprintMismatchError, StudySchemaMismatchError
from phasesweep.engine.fingerprints import (
    _experiment_semantic_fingerprint,
    _phase_fingerprint,
)
from phasesweep.engine.generation import _write_generation_state
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _attempts_dir,
    _experiment_dir,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _trial_dir_for,
    _winner_path,
)
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    GENERATION_ID_ATTR,
    PUBLICATION_POINTER_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
)
from phasesweep.engine.trial import UnsafeProcessCleanupError
from phasesweep.evidence.models import objective_evidence_assurance
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import (
    ConcurrencyLimitError,
    RunCapacityUnknownError,
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
    AwaitRunResult,
    GetRunResultsResult,
    GetRunStatusResult,
    PhaseSweepMCP,
    _safe_tool,
    _status_next_action,
)
from phasesweep.mcp.snapshots import capture_result_snapshot, finalize_result_snapshot
from phasesweep.runtime.files import open_private_text
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    _write_process_identity,
    read_boot_id,
    read_proc_starttime,
    write_attempt_lifecycle,
)
from tests.conftest import file_mode, make_experiment, write_constant_trainer
from tests.mcp_helpers import (
    claim_runner_handle,
    make_mcp_app,
    make_run_handle,
    mcp_experiment_config_text,
    patch_popen_capture,
    runner_argv,
    runner_main,
    write_mcp_catalog,
    write_run_status,
)

ALLOW_SIDE_EFFECTS = {"launch": True, "cancel": True, "from_phase": True}


def _counting_success_callback(calls: list[None]) -> Callable[..., bool]:
    """Return a permissive cleanup stub that records each invocation."""

    def callback(*args: object, **kwargs: object) -> bool:
        calls.append(None)
        return True

    return callback


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
    _validate_artifact_root_binding(exp, claim_fresh=True)
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
    persist_trial_attrs: bool = True,
) -> int:
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = optuna.create_study(
        study_name=f"{exp.experiment}::{phase.name}",
        storage=exp.storage,
        direction="minimize",
    )
    _validate_artifact_root_binding(exp, claim_fresh=True)
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
    if persist_trial_attrs:
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
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", kill_stale_group_stub)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        cleanup_trial_stub,
    )
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
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
    _validate_artifact_root_binding(experiment, claim_fresh=True)
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


@pytest.mark.parametrize("record_kind", ["malformed_handle", "orphan_config"])
def test_launch_refuses_when_persisted_run_capacity_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)
    if record_kind == "malformed_handle":
        (tmp_path / "state" / "runs" / "srv-broken.json").write_text("{not valid json")
    else:
        with open_private_text(store.config_snapshot_path("srv-orphan"), "x") as output:
            output.write("experiment: srv\n")

    with pytest.raises(
        RunCapacityUnknownError, match="cannot prove available launch capacity"
    ) as exc:
        app.launch("srv")

    if record_kind == "orphan_config":
        assert "srv-orphan" in str(exc.value)
        assert "recover-run" in str(exc.value)

    assert "cmd" not in captured
    assert store.list_handles() == []


def test_operator_recovery_clears_pre_spawn_orphan_snapshot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    del app
    run_id = "srv-pre-spawn-orphan"
    snapshot = store.config_snapshot_path(run_id)
    with open_private_text(snapshot, "x") as output:
        output.write(config.read_text())

    preflight = CliRunner().invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )
    assert preflight.exit_code == 0, preflight.output
    assert "no runner can still start a trainer under this identity" in preflight.output
    assert snapshot.is_file()

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
    assert "no runner can still start a trainer under this identity" in confirmed.output
    assert not snapshot.exists()


def test_operator_recovery_clears_abandoned_transactional_preparation(tmp_path: Path) -> None:
    """A free launch lease makes a persisted launching handle recoverable."""
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    del app
    reg = registry.get("srv")
    run_id = "srv-abandoned-preparation"
    pending = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    preparation = store.prepare_launch(pending, config.read_bytes())
    artifacts = (
        registry.state_dir / "runs" / f"{run_id}.json",
        store.config_snapshot_path(run_id),
        store.launch_lease_path(run_id),
    )
    original = {path: path.read_bytes() for path in artifacts}

    held = CliRunner().invoke(
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

    assert held.exit_code != 0
    assert "launch outcome is unresolved" in held.output
    assert {path: path.read_bytes() for path in artifacts} == original

    preparation.close()

    preflight = CliRunner().invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )

    assert preflight.exit_code == 0, preflight.output
    assert "no runner can still start a trainer under this identity" in preflight.output
    assert {path: path.read_bytes() for path in artifacts} == original

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
    assert "no runner can still start a trainer under this identity" in confirmed.output
    assert all(not path.exists() for path in artifacts)


@pytest.mark.parametrize("evidence", ["log", "dangling_handle"])
def test_operator_recovery_refuses_ambiguous_pre_spawn_orphan(
    tmp_path: Path,
    evidence: str,
) -> None:
    """Only a snapshot with no other run evidence is safe to remove."""
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    del app
    run_id = "srv-ambiguous-orphan"
    snapshot = store.config_snapshot_path(run_id)
    with open_private_text(snapshot, "x") as output:
        output.write(config.read_text())
    if evidence == "log":
        with open_private_text(store.log_path(run_id), "x") as output:
            output.write("runner may have started\n")
    else:
        handle_path = registry.state_dir / "runs" / f"{run_id}.json"
        handle_path.symlink_to("missing-handle.json")

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
    assert "unknown run id" in result.output
    assert snapshot.is_file()


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
        publication_hook: object,
    ) -> None:
        del publication_hook
        assert generation_id == run_id
        calls.append((config_obj.experiment, from_phase, dry_run))
        _validate_artifact_root_binding(config_obj, claim_fresh=True)
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
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
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
    assert not store.launch_lease_path(handle.run_id).exists()
    patch_popen_capture(monkeypatch)
    assert app.launch("srv")["state"] == "running"


def test_launch_terminates_real_runner_when_log_context_exit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child created before log close fails is owned and terminated."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    real_popen = subprocess.Popen
    real_open_private_text = mcp_server.open_private_text
    spawned: list[subprocess.Popen[Any]] = []

    @contextlib.contextmanager
    def fail_log_close(path: Path, mode: str) -> Iterator[Any]:
        with real_open_private_text(path, mode) as handle:
            yield handle
        if path.suffix == ".log":
            raise OSError("injected log context exit failure")

    def sleeping_popen(cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        proc = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        run_id = cmd[cmd.index("--run-id") + 1]
        pending = store.get(run_id)
        assert pending is not None
        store.update(
            replace(
                pending,
                pid=proc.pid,
                pgid=proc.pid,
                pid_starttime=read_proc_starttime(proc.pid),
                boot_id=read_boot_id(),
                launch_state="spawned",
            )
        )
        os.write(int(cmd[cmd.index("--launch-ready-fd") + 1]), b"R")
        spawned.append(proc)
        return proc

    monkeypatch.setattr(mcp_server, "open_private_text", fail_log_close)
    monkeypatch.setattr(mcp_server.subprocess, "Popen", sleeping_popen)
    try:
        with pytest.raises(OSError, match="log context exit"):
            app.launch("srv")
        assert len(spawned) == 1
        spawned[0].wait(timeout=5)

        (pending,) = store.list_handles()
        terminal = store.recorded_terminal_status(pending)
        assert terminal is not None
        assert terminal["cleanup_confirmed"] is True
        assert terminal["error_class"] == "OSError"
        assert store.state(pending) == "failed"
        assert not store.recovery_required(pending)
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize(
    ("failing_reader", "cleanup_confirmed", "expected_state"),
    [
        ("starttime", True, "failed"),
        ("boot_id", True, "failed"),
        ("starttime", False, "running"),
    ],
)
def test_launch_records_real_cleanup_result_when_process_identity_read_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_reader: str,
    cleanup_confirmed: bool,
    expected_state: str,
) -> None:
    """A fallible identity read cannot turn a created child into a pre-spawn failure."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def fail_identity_read(*_args: object) -> None:
        raise OSError(f"injected {failing_reader} read failure")

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, saved_starttime, pgid))
        return cleanup_confirmed

    monkeypatch.setattr(mcp_server, "kill_stale_group", fake_cleanup)
    monkeypatch.setattr(
        mcp_server,
        "read_proc_starttime" if failing_reader == "starttime" else "read_boot_id",
        fail_identity_read,
    )

    with pytest.raises(OSError, match=failing_reader):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is cleanup_confirmed
    assert store.state(pending) == expected_state
    assert store.recovery_required(pending) is (not cleanup_confirmed)
    assert cleanup_calls
    if failing_reader == "starttime":
        assert cleanup_calls[0][1] is None
    else:
        assert cleanup_calls[0][1] is not None


def test_launch_cleans_up_when_identity_bookkeeping_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyboardInterrupt after Popen is cleanup work, not a pre-spawn failure."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def interrupt_boot_id() -> None:
        raise KeyboardInterrupt

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(mcp_server, "read_boot_id", interrupt_boot_id)
    monkeypatch.setattr(mcp_server, "kill_stale_group", fake_cleanup)

    with pytest.raises(KeyboardInterrupt):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is True
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert store.state(pending) == "failed"
    assert cleanup_calls and cleanup_calls[0][1] is not None


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


def test_restarted_server_reaps_abandoned_transaction_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server crash before Popen leaves a provably removable preparation."""
    config = _config(tmp_path)
    _first, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    abandoned_id = "srv-abandoned-preparation"
    pending = make_run_handle(
        run_id=abandoned_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    preparation = store.prepare_launch(pending, config.read_bytes())
    preparation.close()  # Simulate the launching server disappearing before Popen.

    restarted_store = RunStore(registry.state_dir)
    restarted = PhaseSweepMCP(registry, restarted_store)
    monkeypatch.setattr(restarted_store, "new_run_id", lambda _experiment_id: "srv-retry")
    patch_popen_capture(monkeypatch)

    result = restarted.launch("srv")

    assert result["run_id"] == "srv-retry"
    assert restarted_store.get(abandoned_id) is None
    assert not restarted_store.config_snapshot_path(abandoned_id).exists()
    assert not restarted_store.launch_lease_path(abandoned_id).exists()


def test_missing_runner_receipt_fails_without_reserving_retry_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that never publishes a receipt never receives permission to work."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    class UnreadyProc:
        pid = os.getppid()

    monkeypatch.setattr(mcp_server.subprocess, "Popen", lambda *_args, **_kwargs: UnreadyProc())
    monkeypatch.setattr(mcp_server, "kill_stale_group", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="did not persist its launch receipt"):
        app.launch("srv")

    (failed,) = store.list_handles()
    assert store.state(failed) == "failed"
    assert not store.launch_lease_path(failed.run_id).exists()
    patch_popen_capture(monkeypatch)
    assert app.launch("srv")["state"] == "running"


def test_acknowledgement_write_failure_terminates_receipted_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable receipt alone cannot let the child cross the work boundary."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []
    real_write = os.write

    def fail_ack(fd: int, data: bytes) -> int:
        if data == b"A":
            raise OSError("injected acknowledgement failure")
        return real_write(fd, data)

    def confirm_cleanup(
        pid: int | None,
        starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, starttime, pgid))
        return True

    monkeypatch.setattr(mcp_server.os, "write", fail_ack)
    monkeypatch.setattr(mcp_server, "kill_stale_group", confirm_cleanup)

    with pytest.raises(OSError, match="acknowledgement"):
        app.launch("srv")

    (handle,) = store.list_handles()
    assert handle.launch_state == "spawned"
    assert cleanup_calls == [(handle.pid, handle.pid_starttime, handle.pgid)]
    assert store.state(handle) == "failed"
    assert not store.launch_lease_path(handle.run_id).exists()


def test_completed_lease_cleanup_failure_cannot_replace_launch_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lease sidecar cleanup is non-authoritative after an acknowledged receipt."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    real_unlink = mcp_runs._strict_unlink

    def fail_lease_unlink(path: Path) -> None:
        if path.suffix == ".lock":
            raise OSError("injected lease cleanup failure")
        real_unlink(path)

    monkeypatch.setattr(mcp_runs, "_strict_unlink", fail_lease_unlink)
    caplog.set_level(logging.WARNING, logger="phasesweep.mcp.runs")

    result = app.launch("srv")

    assert result["state"] == "running"
    assert "could not remove completed launch lease" in caplog.text
    assert store.launch_lease_path(result["run_id"]).is_file()


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
    assert handles[0].launch_state == "spawned"
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
    assert pending.launch_state == "spawned"
    assert store.state(pending) == "failed"


def test_launch_interrupt_during_handle_update_terminates_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown interrupt cannot escape after spawn but before durable identity."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    terminated: list[tuple[int | None, int | None, int | None]] = []

    def interrupt_update(_handle: RunHandle) -> None:
        raise KeyboardInterrupt

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        terminated.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(store, "update", interrupt_update)
    monkeypatch.setattr(mcp_server, "kill_stale_group", fake_cleanup)

    with pytest.raises(KeyboardInterrupt):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is True
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert store.state(pending) == "failed"
    assert terminated and terminated[0][1] is not None


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


def test_launch_retains_recovery_reservation_when_cleanup_marker_cannot_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed process death is not enough when its uncertainty marker cannot clear."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)

    def fail_update(_handle: RunHandle) -> None:
        raise OSError("runs directory is not writable")

    def fail_clear(_handle: RunHandle) -> None:
        raise OSError("cleanup marker cannot be removed")

    monkeypatch.setattr(store, "update", fail_update)
    monkeypatch.setattr(store, "clear_cleanup_uncertain", fail_clear)
    monkeypatch.setattr(mcp_server, "kill_stale_group", lambda *args, **kwargs: True)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is False
    assert store.cleanup_uncertain(pending)
    assert store.state(pending) == "running"
    assert store.recovery_required(pending)


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


def test_decataloged_live_run_leaves_config_drift_unknown(tmp_path: Path) -> None:
    old_config = _config(tmp_path, name="old")
    old_exp = load_config(old_config)
    assert isinstance(old_exp, Experiment)
    run_id = "old-live-published"
    _write_winner_yaml(
        old_exp,
        "p",
        phase_fingerprint=_phase_fingerprint(old_exp, old_exp.phases[0], {}),
        generation_id=run_id,
    )
    _generation_summary_path(old_exp, run_id).write_text(
        yaml.safe_dump(
            {
                "config_fingerprint": _experiment_semantic_fingerprint(old_exp),
                "metric": {"name": "loss", "goal": "minimize"},
                "phase_plan": [{"name": "p"}],
            },
            sort_keys=False,
        )
    )

    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(write_mcp_catalog(tmp_path, {"other": other_config}))
    snapshot = old_config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id="old",
            config_sha256=hashlib.sha256(snapshot).hexdigest(),
        )
    )

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.result_source == "current_shared_study"
    assert results.result_source == "current_shared_study"
    assert status.represented_generation_id == run_id
    assert results.represented_generation_id == run_id
    assert status.published_config_matches_current is None
    assert results.published_config_matches_current is None


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
        override_format="argparse",
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
    if os.geteuid() != 0:
        snapshot = (
            _generation_summary_path(experiment, generation_id).parent / "config.snapshot.yaml"
        )
        original_mode = snapshot.stat().st_mode & 0o777
        snapshot.chmod(0o000)
        try:
            denied_status_payload = app.status(experiment_id="srv")
            denied_status = GetRunStatusResult.model_validate(denied_status_payload)
            denied_winners = app.winners(experiment_id="srv")
        finally:
            snapshot.chmod(original_mode)
        assert denied_status.publication_integrity == "permission_denied"
        assert denied_status.published_generation_id is None
        assert denied_status.is_published is False
        assert denied_winners["publication_integrity"] == "permission_denied"
        assert denied_winners["winner_count"] == 0
        # MCP deliberately forwards the enum, not the local permission detail.
        assert "publication_error" not in json.dumps(
            [denied_status_payload, denied_winners], default=str
        )

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


# --------------------------------------------------------------------------
# A published result is historical evidence. Editing the catalog config never
# relabels it, and the agent is told when the two have diverged (review
# v0.5.16 / blocker 4).
# --------------------------------------------------------------------------


def _drift_experiment(
    tmp_path: Path,
    trainer: Path,
    *,
    metric_name: str = "x",
    goal: str = "minimize",
    extractor: object | None = None,
    phase_name: str = "p",
) -> Experiment:
    """Build the one-phase experiment the catalog-drift tests publish and edit."""
    return make_experiment(
        experiment="srv",
        storage=f"sqlite:///{tmp_path / 'drift.db'}",
        workdir=str(tmp_path / "runs"),
        execution=ExecutionContext(cwd=str(tmp_path)),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            name=metric_name,
            goal=goal,
            extractor=extractor
            or LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(
                name=phase_name,
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={"lr": IntParam(type="int", low=1, high=2)},
            )
        ],
    )


def _write_experiment_config(config: Path, experiment: Experiment) -> None:
    """Rewrite a cataloged config file in place, as an operator edit would."""
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))


def _publish_drift_experiment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Publish one real generation and return its trainer, config, and catalog."""
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    published = _drift_experiment(tmp_path, trainer)
    _write_experiment_config(config, published)
    catalog = _catalog(tmp_path, config)
    run_experiment(published)
    return trainer, config, catalog


def _record_published_run_snapshot(
    tmp_path: Path,
) -> tuple[str, Path, Path, Path]:
    """Publish one generation whose id also has a completed MCP run snapshot."""
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = _drift_experiment(tmp_path, trainer)
    _write_experiment_config(config, experiment)
    catalog = _catalog(tmp_path, config)
    _app, registry, store = make_mcp_app(catalog)
    reg = registry.get("srv")
    run_id = "srv-frozen-result"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    run_experiment(experiment, generation_id=run_id)
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment, generation_id=run_id),
    )
    return run_id, trainer, config, catalog


def test_published_results_keep_their_own_metric_after_a_catalog_metric_edit(
    tmp_path: Path,
) -> None:
    """A result published as x/minimize is never reported as y/maximize.

    Both read surfaces must agree: status already resolved the historical
    metric, while results built its label from the current catalog, so the
    same number was reported under two different metrics by two tools of the
    same server.
    """
    trainer, config, catalog = _publish_drift_experiment(tmp_path)

    app, _registry, _store = make_mcp_app(catalog)
    baseline_status = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))
    baseline_results = GetRunResultsResult.model_validate(app.winners(experiment_id="srv"))

    # Unchanged config: the historical labels are also the current ones, and
    # the drift flag says so rather than staying silent.
    assert baseline_results.metric.name == "x"
    assert baseline_results.metric.goal == "minimize"
    assert baseline_results.result_context == "represented_generation"
    assert baseline_results.published_config_matches_current is True
    assert baseline_status.published_config_matches_current is True
    assert baseline_status.result_phase_plan == ["p"]
    assert baseline_results.winner_count == 1

    _write_experiment_config(
        config,
        _drift_experiment(tmp_path, trainer, metric_name="y", goal="maximize"),
    )
    restarted, _restarted_registry, _restarted_store = make_mcp_app(catalog)

    status = GetRunStatusResult.model_validate(restarted.status(experiment_id="srv"))
    results = GetRunResultsResult.model_validate(restarted.winners(experiment_id="srv"))

    assert (results.metric.name, results.metric.goal) == ("x", "minimize")
    assert (status.metric.name, status.metric.goal) == (results.metric.name, results.metric.goal)
    assert results.result_context == "represented_generation"
    assert results.published_config_matches_current is False
    assert status.published_config_matches_current is False
    # The winner itself is unchanged: only its provenance disclosure improved.
    assert results.winner_count == 1
    assert results.phases[0].metric == baseline_results.phases[0].metric


def test_run_scoped_snapshot_recomputes_config_drift_against_current_catalog(
    tmp_path: Path,
) -> None:
    run_id, trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    _write_experiment_config(
        config,
        _drift_experiment(tmp_path, trainer, metric_name="y", goal="maximize"),
    )
    app, _registry, _store = make_mcp_app(catalog)

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.result_source == "frozen_run_snapshot"
    assert results.result_source == "frozen_run_snapshot"
    assert status.published_config_matches_current is False
    assert results.published_config_matches_current is False
    assert (results.metric.name, results.metric.goal) == ("x", "minimize")


def test_run_scoped_snapshot_survives_later_artifact_corruption(tmp_path: Path) -> None:
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, _store = make_mcp_app(catalog)
    experiment = app._registry.get("srv").experiment
    winner_path = _generation_winner_path(experiment, run_id, "p")
    winner_path.write_text("broken: true\n")

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.publication_integrity == "ok"
    assert status.is_published is True
    assert status.phases[0].winner_present is True
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1


def test_run_scoped_snapshot_keeps_captured_generation_pointers(tmp_path: Path) -> None:
    run_id, _trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    later_failed_id = "srv-later-failed"
    _write_generation_state(
        experiment,
        generation_id=later_failed_id,
        state="failed",
        from_phase=None,
        publish_current=True,
        error_class="RuntimeError",
    )
    app, _registry, _store = make_mcp_app(catalog)

    after_failure = GetRunStatusResult.model_validate(app.status(run_id=run_id))

    assert after_failure.current_generation_id == run_id
    assert after_failure.published_generation_id == run_id
    assert after_failure.represented_generation_id == run_id
    assert after_failure.is_published is True

    later_published_id = "srv-later-published"
    run_experiment(experiment, generation_id=later_published_id)

    after_publication = GetRunStatusResult.model_validate(app.status(run_id=run_id))

    assert after_publication.current_generation_id == run_id
    assert after_publication.published_generation_id == run_id
    assert after_publication.represented_generation_id == run_id
    assert after_publication.is_published is True


def test_run_scoped_snapshot_survives_publication_pointer_removal(
    tmp_path: Path,
) -> None:
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, _store = make_mcp_app(catalog)
    experiment = app._registry.get("srv").experiment
    _last_successful_generation_path(experiment).unlink()

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.publication_integrity == "ok"
    assert status.is_published is True
    assert status.phases[0].winner_present is True
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1


def test_run_scoped_snapshot_survives_artifact_tree_relocation(tmp_path: Path) -> None:
    """Completed run reads use frozen evidence even when the live tree moved."""
    run_id, _trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, _store = make_mcp_app(catalog)
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    source = _experiment_dir(experiment)
    relocated = tmp_path / "relocated" / source.name
    relocated.parent.mkdir()
    shutil.move(source, relocated)

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))
    awaited = AwaitRunResult.model_validate(asyncio.run(app.await_run(run_id)))

    for payload in (status, awaited):
        assert payload.result_source == "frozen_run_snapshot"
        assert payload.publication_integrity == "ok"
        assert payload.phases[0].winner_present is True
    assert results.result_source == "frozen_run_snapshot"
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1


def test_published_results_keep_their_objective_evidence_after_an_extractor_swap(
    tmp_path: Path,
) -> None:
    """A log-scraped number must not inherit a structured extractor's guarantees.

    Reporting the current extractor's assurance beside a historical winner
    claims evidence properties that run never had - the one field an agent is
    told to use when deciding how far to trust a metric.
    """
    trainer, config, catalog = _publish_drift_experiment(tmp_path)
    published_assurance = objective_evidence_assurance(
        _drift_experiment(tmp_path, trainer).metric.extractor
    )

    _write_experiment_config(
        config,
        _drift_experiment(
            tmp_path,
            trainer,
            extractor=JsonEnvelopeExtractor(
                type="json_envelope",
                path="r.json",
                objective_name="x",
                split="test",
                policy="test",
            ),
        ),
    )
    app, _registry, _store = make_mcp_app(catalog)

    results = GetRunResultsResult.model_validate(app.winners(experiment_id="srv"))
    evidence = results.metric.objective_evidence.model_dump()

    assert evidence == published_assurance
    assert evidence["kind"] == "log_regex"
    # The four guarantees the swapped-in extractor would have asserted.
    assert evidence["objective_name_bound"] is False
    assert evidence["split_bound"] is False
    assert evidence["evaluation_policy_bound"] is False
    assert evidence["source_identity_keyed"] is False
    assert results.published_config_matches_current is False


def test_drifted_recorded_objective_evidence_falls_back_without_internal_error(
    tmp_path: Path,
) -> None:
    trainer, _config, catalog = _publish_drift_experiment(tmp_path)
    experiment = _drift_experiment(tmp_path, trainer)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["metric"]["objective_evidence"]["future_flag"] = True
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    app, _registry, _store = make_mcp_app(catalog)

    results = GetRunResultsResult.model_validate(app.winners(experiment_id="srv"))
    status = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))

    expected = objective_evidence_assurance(experiment.metric.extractor)
    assert results.metric.objective_evidence.model_dump() == expected
    assert status.metric.objective_evidence.model_dump() == expected


def test_published_winner_survives_a_catalog_phase_rename(tmp_path: Path) -> None:
    """Renaming a phase must not delete the published result from the payload.

    Enumerating winners under the *current* phase names returned an "ok"
    publication with zero winners and the new name listed as missing: an
    agent's cue to launch a run over evidence that was there all along.
    """
    trainer, config, catalog = _publish_drift_experiment(tmp_path)
    _write_experiment_config(config, _drift_experiment(tmp_path, trainer, phase_name="q"))
    app, _registry, _store = make_mcp_app(catalog)

    results = GetRunResultsResult.model_validate(app.winners(experiment_id="srv"))

    assert results.publication_integrity == "ok"
    assert [phase.phase for phase in results.phases] == ["p"]
    assert results.winner_count == 1
    assert results.declared_phase_count == 1
    assert results.missing_phases == []
    assert results.all_phases_have_winners is True
    assert results.published_config_matches_current is False

    status = GetRunStatusResult.model_validate(app.status(experiment_id="srv"))

    # Status keeps reporting the *current* plan's progress - a run would have
    # to produce a winner for "q" - but no longer implies the publication is
    # empty: it names the plan that publication used and flags the drift.
    assert status.is_published is True
    assert [phase.phase for phase in status.phases] == ["q"]
    assert status.phases[0].winner_present is False
    assert status.result_phase_plan == ["p"]
    assert status.published_config_matches_current is False
    # ... and it still points at the result instead of answering "stop": the
    # winner the results tool returns is right there.
    assert _status_next_action(status) == TOOL_GET_RUN_RESULTS


def test_current_visibility_policy_redacts_historical_values_without_relabeling_them(
    tmp_path: Path,
) -> None:
    """Redaction is the current operator's call; labeling is the result's own.

    The two must not be conflated in either direction: tightening the catalog
    policy still hides a historical param value, and it never licenses
    reporting that value under today's metric or phase name.
    """
    trainer, config, catalog = _publish_drift_experiment(tmp_path)
    _write_experiment_config(
        config,
        _drift_experiment(tmp_path, trainer, metric_name="y", goal="maximize", phase_name="q"),
    )

    redacted_app, _registry, _store = make_mcp_app(catalog)
    redacted = GetRunResultsResult.model_validate(redacted_app.winners(experiment_id="srv"))

    assert redacted.phases[0].params == {"lr": "<redacted>"}
    assert redacted.phases[0].params_redacted is True
    assert redacted.phases[0].phase == "p"
    assert (redacted.metric.name, redacted.metric.goal) == ("x", "minimize")

    visible_app, _visible_registry, _visible_store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"srv": config}, visible_params={"srv": ["lr"]})
    )
    visible = GetRunResultsResult.model_validate(visible_app.winners(experiment_id="srv"))

    assert visible.phases[0].params["lr"] in (1, 2)
    assert visible.phases[0].params_redacted is False
    assert visible.phases[0].phase == "p"
    assert (visible.metric.name, visible.metric.goal) == ("x", "minimize")


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


def _represent_legacy_generation(experiment: Experiment, generation_id: str) -> None:
    """Fabricate an identity-only publication that represents ``generation_id``.

    Legacy (pre-manifest) publication: the pointer and summary pass the
    identity-only gate, so the experiment-scoped read represents the given
    generation without a real run.
    """
    _write_winner_yaml(
        experiment,
        "p",
        phase_fingerprint=_phase_fingerprint(experiment, experiment.phases[0], {}),
        generation_id=generation_id,
    )
    summary_path = _generation_summary_path(experiment, generation_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_bytes = yaml.safe_dump(
        {"experiment": experiment.experiment, "generation_id": generation_id}
    ).encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    pointer = _last_successful_generation_path(experiment)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        yaml.safe_dump(
            {
                "schema_version": PUBLICATION_POINTER_SCHEMA_VERSION,
                "experiment": experiment.experiment,
                "generation_id": generation_id,
                "summary_size_bytes": len(summary_bytes),
                "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            }
        )
    )


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
    _represent_legacy_generation(reg.experiment, run_id)
    # The handle file exists but no longer decodes.
    handle_path = store._runs_dir / f"{run_id}.json"
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle_path.write_text("{ not json")

    winners = app.winners(experiment_id="srv")

    assert winners["phases"][0]["params"] == {"lr": "<redacted>"}


@pytest.mark.parametrize(
    "surviving",
    [
        "config_snapshot_path",
        "status_path",
        "log_path",
        "cleanup_uncertain_path",
        "cleanup_recovery_path",
    ],
)
def test_deleted_run_handle_with_surviving_run_evidence_fails_closed(
    tmp_path: Path, surviving: str
) -> None:
    """A deleted handle whose sibling per-run files survive is a launched MCP
    run with unreadable frozen authority, not a never-MCP generation: the
    current catalog policy may be wider than the lost launch grant, so the
    narrowest policy applies (PR #5 review / P2 missing-handle authority)."""
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"srv": config}, visible_params={"srv": "all"})
    )
    reg = registry.get("srv")
    run_id = "srv-deleted-handle"
    _represent_legacy_generation(reg.experiment, run_id)
    # No handle file at all -- deleted after the run -- but one sibling
    # per-run file under the same state dir survives it.
    evidence = getattr(store, surviving)(run_id)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("orphaned\n")

    winners = app.winners(experiment_id="srv")

    assert winners["phases"][0]["params"] == {"lr": "<redacted>"}


def test_generation_with_no_run_evidence_and_no_id_source_keeps_current_policy(
    tmp_path: Path,
) -> None:
    """A legacy tree must not be over-closed: a represented generation with no
    handle, no surviving per-run file, and no recorded id source predates the
    provenance marker, and the current catalog policy legitimately applies."""
    config = _config(tmp_path)
    app, registry, _store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"srv": config}, visible_params={"srv": "all"})
    )
    reg = registry.get("srv")
    _represent_legacy_generation(reg.experiment, "legacy-cli-generation")

    winners = app.winners(experiment_id="srv")

    assert winners["phases"][0]["params"] == {"lr": 0.001}


def _real_run_app(tmp_path: Path) -> tuple[PhaseSweepMCP, Experiment]:
    """Build an app over a real runnable experiment cataloged with ``visible_params: all``."""
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = make_experiment(
        experiment="srv",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=[
            Phase(
                name="p",
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            )
        ],
    )
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, visible_params="all"))
    return app, experiment


def test_replaced_state_dir_fails_closed_for_a_caller_identified_generation(
    tmp_path: Path,
) -> None:
    """Losing the MCP state dir must not widen visibility: the generation's own
    reproducibility record proves its id -- and so its launch authority -- was
    caller-granted, and with no handle answering for that frozen grant the
    narrowest policy applies (PR #5 review / P2 missing-handle authority)."""
    app, experiment = _real_run_app(tmp_path)
    # What the detached runner does: publish under the launcher-granted run id.
    # The app's store holds nothing for it, as after a state-dir replacement.
    run_experiment(experiment, generation_id="srv-detached-1")

    winners = app.winners(experiment_id="srv")

    assert winners["phases"][0]["params"] == {"x": "<redacted>"}


def test_engine_minted_generation_without_a_handle_uses_current_catalog_policy(
    tmp_path: Path,
) -> None:
    """The durable id-source marker must not over-close: a CLI-launched
    generation records ``engine``, so with no MCP evidence anywhere the current
    catalog policy legitimately renders its winner values."""
    app, experiment = _real_run_app(tmp_path)
    run_experiment(experiment)

    winners = app.winners(experiment_id="srv")

    params = winners["phases"][0]["params"]
    assert isinstance(params["x"], int)


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
    assert file_mode(registry.state_dir) == 0o700
    assert file_mode(registry.state_dir / "runs") == 0o700
    assert file_mode(registry.state_dir / "logs") == 0o700
    assert file_mode(registry.state_dir / "runs" / f"{run_id}.json") == 0o600
    assert file_mode(store.log_path(run_id)) == 0o600
    assert file_mode(store.config_snapshot_path(run_id)) == 0o600
    assert file_mode(audit_path) == 0o600


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
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256="0" * 64,
                experiment_id="srv",
                started_at=started_at,
            )
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
        override_format="argparse",
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
            runner_argv(
                store,
                run_id=run_id,
                config=snapshot_path,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
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
        # A study a prior run left behind carries this workdir's artifact-root
        # binding; without it the pre-binding migration refusal preempts the
        # schema aggregation this test is about (re-review v0.5.19 / blocker B1).
        study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment)))
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
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
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

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

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

    with pytest.raises(Exception, match="action 'cancel' is not permitted"):
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

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

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
        with pytest.raises(Exception, match="already has a running sweep"):
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

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", spy_kill_stale_group)

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

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", unexpected_cleanup)
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
    assert winners["represented_generation_id"] == run_id
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

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", unexpected_cleanup)

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
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())

    attempt_id = "registry-only-attempt"
    trial_dir = _experiment_dir(experiment) / "p" / "trial_registry_only"
    trial_dir.mkdir(parents=True)
    write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    _write_trial_process_identity(
        trial_dir,
        attempt_id=attempt_id,
        pid=4242,
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
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        trial_cleanup,
    )
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
        trial_cleanup,
    )
    runner = CliRunner()
    command = ["mcp", "recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id]

    dry = runner.invoke(cli_main, command)

    assert dry.exit_code == 0, dry.output
    assert "reconcile 1 registered attempt" in dry.output
    assert len(runner_cleanup_calls) == 0
    assert trial_cleanup_calls == 0

    refused = runner.invoke(cli_main, [*command, "--confirm"])

    assert refused.exit_code != 0
    assert "may still have a live process group" in refused.output
    assert len(runner_cleanup_calls) == 1
    assert trial_cleanup_calls == 1
    assert entry_path.is_file()
    assert not store.status_path(run_id).exists()
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.state(handle) == "running"
    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    trial_cleanup_allowed = True
    confirmed = runner.invoke(cli_main, [*command, "--confirm"])

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
    trial_number = _write_stale_running_trial(
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

    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    handle = make_run_handle(
        run_id=earlier_run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(earlier_run_id).write_bytes(config.read_bytes())
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
    write_run_status(
        store,
        earlier_run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
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

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "mcp",
            "recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            earlier_run_id,
            "--confirm",
        ],
    )

    assert study.get_trials(deepcopy=False)[trial_number].state == optuna.trial.TrialState.FAIL
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]
    if interrupt_recovery_write:
        assert result.exit_code != 0
        assert entry_path.exists()
        recovery_path.rmdir()
        result = runner.invoke(
            cli_main,
            [
                "mcp",
                "recover-run",
                "--state-dir",
                str(registry.state_dir),
                "--run-id",
                earlier_run_id,
                "--confirm",
            ],
        )
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
            lambda config, *, generation_id: _write_stale_running_trial(
                config,
                cleanup_confirmed=False,
                generation_id=generation_id,
            ),
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
    run_id = "srv-cleanup-uncertainty-recover"
    trial_number = trial_setup_fn(config, generation_id=run_id)
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

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", fake_runner_cleanup)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        fake_trial_cleanup,
    )
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
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
    runner_cleanup_calls: list[None] = []
    trial_cleanup_calls: list[None] = []

    app, store, handle, attempt_id, command = _stage_stale_running_recovery_scaffold(
        tmp_path,
        run_id=run_id,
        error_class="UnsafeProcessCleanupError",
        include_generation_record=True,
        mark_cleanup_uncertain=False,
        snapshot_bound_to_generation=True,
        kill_stale_group_stub=_counting_success_callback(runner_cleanup_calls),
        cleanup_trial_stub=_counting_success_callback(trial_cleanup_calls),
        monkeypatch=monkeypatch,
    )

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
            raise RuntimeError("snapshot finalization failed")
        return finalize_result_snapshot(
            snapshot,
            confirmed_attempt_ids=confirmed_attempt_ids,
            confirmed_attempt_locations=confirmed_attempt_locations,
        )

    monkeypatch.setattr("phasesweep.mcp.recovery.finalize_result_snapshot", flaky_snapshot)

    runner = CliRunner()
    first = runner.invoke(cli_main, command)

    assert first.exit_code != 0
    assert "failed to finalize terminal result snapshot" in first.output
    assert store.cleanup_recovery_path(run_id).is_file()
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["reaped_attempt_ids"] == [attempt_id]
    assert store.recovery_required(handle)
    assert len(runner_cleanup_calls) == 1
    assert len(trial_cleanup_calls) == 1

    retry = runner.invoke(cli_main, command)

    assert retry.exit_code == 0, retry.output
    assert "Finalized stored terminal result snapshot" in retry.output
    assert len(runner_cleanup_calls) == 1
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
        recovered_attempt_generations=reconciled_attempt_generations,
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment),
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *args, **kwargs: True)

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
    first_run = "srv-terminal-first"
    trial_number = _write_cleanup_uncertain_failed_trial(
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

    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", fake_cleanup)
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", fake_cleanup)
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

    monkeypatch.setattr(
        "phasesweep.mcp.recovery.private_atomic_write_text", crash_on_recovery_record
    )
    runner = CliRunner()

    first = runner.invoke(cli_main, command)

    assert first.exit_code != 0
    # The durable study ledger consumed the trial before the crash; the
    # run-level record never landed.
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]
    assert not recovery_record.exists()

    monkeypatch.setattr("phasesweep.mcp.recovery.private_atomic_write_text", real_write)

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
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes() + snapshot_suffix)
    if terminal_without_evidence:
        write_run_status(
            store,
            run_id,
            returncode=1,
            error_class="UnsafeProcessCleanupError",
            cleanup_confirmed=False,
        )
    else:
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
    assert expected_message in result.output
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.state(handle) == "running"
    if not terminal_without_evidence:
        assert store.cleanup_uncertain_path(run_id).is_file()
