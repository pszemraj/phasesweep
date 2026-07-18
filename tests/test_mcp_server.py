"""MCP server logic that does not need a real detached runner."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from pathlib import Path

import optuna
import pytest
import yaml
from click.testing import CliRunner

from phasesweep.cli import main as cli_main
from phasesweep.config import Experiment, load_config
from phasesweep.engine.guards import _phase_fingerprint
from phasesweep.engine.state import (
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    TRIAL_DIR_ATTR,
    _trial_dir_for,
    _winner_path,
)
from phasesweep.engine.trial import UnsafeProcessCleanupError
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import UnknownExperimentError
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runner import main as runner_main
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.server import (
    TOOL_LAUNCH_SWEEP,
    TOOL_LIST_EXPERIMENTS,
    TOOL_VALIDATE_CONFIG,
    PhaseSweepMCP,
    _safe_tool,
)
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.runtime.process import read_proc_starttime
from tests.mcp_helpers import (
    make_mcp_app,
    make_run_handle,
    mcp_experiment_config_text,
    patch_popen_capture,
    write_mcp_catalog,
    write_run_status,
)

ALLOW_SIDE_EFFECTS = {"launch": True, "cancel": True, "from_phase": True}


def _config(tmp_path: Path, *, name: str = "srv", phases: str | None = None) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(mcp_experiment_config_text(tmp_path, name=name, phases=phases))
    return path


def _catalog(tmp_path: Path, config: Path, allow: dict[str, bool] | None = None) -> Path:
    return write_mcp_catalog(
        tmp_path,
        {"srv": config},
        allow=allow,
        filename="srv.catalog.yaml",
    )


def _write_cleanup_uncertain_failed_trial(config: Path) -> int:
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = optuna.create_study(
        study_name=f"{exp.experiment}::{phase.name}",
        storage=exp.storage,
        direction="minimize",
    )
    trial = study.ask()
    trial_dir = _trial_dir_for(exp, phase.name, trial.number)
    trial_dir.mkdir(parents=True)
    (trial_dir / "pid").write_text("4242\n")
    (trial_dir / "pgid").write_text("4242\n")
    (trial_dir / "pid_starttime").write_text("111\n")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return trial.number


def _write_stale_running_trial(
    config: Path,
    *,
    cleanup_confirmed: bool | None = None,
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
    trial_dir = _trial_dir_for(exp, phase.name, trial.number)
    trial_dir.mkdir(parents=True)
    (trial_dir / "pid").write_text("4343\n")
    (trial_dir / "pgid").write_text("4343\n")
    (trial_dir / "pid_starttime").write_text("222\n")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
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


def test_safe_tool_returns_safe_mcp_error() -> None:
    @_safe_tool
    def boom() -> None:
        raise UnknownExperimentError("srv")

    with pytest.raises(ValueError, match="unknown experiment id 'srv'"):
        boom()


def test_safe_tool_redacts_unexpected_exception() -> None:
    @_safe_tool
    def boom() -> None:
        raise OSError("/tmp/SECRET_PATH/config.yaml")

    with pytest.raises(ValueError) as excinfo:
        boom()
    assert str(excinfo.value) == "internal error"
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


def test_resume_requires_prior_winner(tmp_path: Path) -> None:
    phases = """\
  - name: p
    n_trials: 1
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  - name: q
    inherits: [p]
    n_trials: 1
    search_space:
      wd: { type: float, low: 0.0, high: 0.1 }
"""
    config = _config(tmp_path, phases=phases)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with pytest.raises(Exception, match="earlier phase 'p' has no winner yet"):
        app.launch("srv", from_phase="q")


def _write_winner_yaml(
    experiment: Experiment,
    phase_name: str,
    *,
    phase_fingerprint: str,
    incomplete: bool = False,
) -> None:
    path = _winner_path(experiment, phase_name)
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
            }
        )
    )


def test_resume_rejects_stale_winner_before_spawn(tmp_path: Path) -> None:
    phases = """\
  - name: p
    n_trials: 1
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  - name: q
    inherits: [p]
    n_trials: 1
    search_space:
      wd: { type: float, low: 0.0, high: 0.1 }
"""
    config = _config(tmp_path, phases=phases)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    _write_winner_yaml(exp, "p", phase_fingerprint="0" * 64)

    with pytest.raises(Exception, match="compatible winner"):
        app.launch("srv", from_phase="q")

    assert store.list_handles() == []


def test_resume_rejects_incomplete_winner_before_spawn(tmp_path: Path) -> None:
    phases = """\
  - name: p
    n_trials: 1
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  - name: q
    inherits: [p]
    n_trials: 1
    search_space:
      wd: { type: float, low: 0.0, high: 0.1 }
"""
    config = _config(tmp_path, phases=phases)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    fp = _phase_fingerprint(exp, exp.phases[0], {})
    _write_winner_yaml(exp, "p", phase_fingerprint=fp, incomplete=True)

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
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    result = app.launch("srv")

    cmd = captured["cmd"]
    config_arg = Path(cmd[cmd.index("--config") + 1])
    sha_arg = cmd[cmd.index("--config-sha256") + 1]
    assert config_arg != config.resolve()
    assert config_arg.read_bytes() == config.read_bytes()
    assert sha_arg == registry.get("srv").config_sha256
    assert config_arg == registry.state_dir / "logs" / f"{result['run_id']}.config.yaml"
    assert Path(cmd[cmd.index("--state-dir") + 1]) == registry.state_dir
    assert cmd[cmd.index("--experiment-id") + 1] == "srv"
    assert list(config_arg.parent.glob("*.tmp")) == []
    assert list(config_arg.parent.glob(".*.tmp")) == []
    handle = store.get(result["run_id"])
    assert handle is not None
    assert handle.launch_state == "spawned"
    assert cmd[cmd.index("--started-at") + 1] == handle.started_at


def test_runner_persists_spawned_handle_for_restart_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "srv-recover"
    started_at = "2026-06-24T00:00:00Z"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    calls: list[tuple[str, str | None, bool]] = []

    def fake_run_config(config_obj: Experiment, *, from_phase: str | None, dry_run: bool) -> None:
        calls.append((config_obj.experiment, from_phase, dry_run))

    monkeypatch.setattr("phasesweep.engine.run_config", fake_run_config)

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
    assert terminal["result_snapshot"]["status"]["phases"][0]["phase"] == "p"
    assert terminal["result_snapshot"]["winners"] == []


def test_launch_does_not_spawn_when_pending_handle_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    def fail_save(handle: RunHandle) -> None:
        raise OSError("runs directory is not writable")

    monkeypatch.setattr(store, "save", fail_save)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert "cmd" not in captured
    assert store.list_handles() == []


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
    assert bool(store.cleanup_uncertain(handles[0])) is (not cleanup_confirmed)
    assert cleanup_calls and cleanup_calls[0][1] is None


def test_launch_terminates_spawned_runner_when_final_handle_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    original_save = store.save
    saved: list[RunHandle] = []
    terminated: list[tuple[int | None, int | None, int | None]] = []

    def fail_second_save(handle: RunHandle) -> None:
        saved.append(handle)
        if len(saved) == 1:
            original_save(handle)
            return
        raise OSError("runs directory is not writable")

    def fake_kill_stale_group(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        terminated.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(store, "save", fail_second_save)
    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert [handle.launch_state for handle in saved] == ["launching", "spawned"]
    assert terminated == [(saved[1].pid, saved[1].pid_starttime, saved[1].pgid)]
    pending = store.get(saved[0].run_id)
    assert pending is not None
    assert pending.launch_state == "launching"
    assert store.state(pending) == "failed"


def test_launch_logs_when_cleanup_marker_write_fails_after_final_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    original_save = store.save
    saved: list[RunHandle] = []

    def fail_second_save(handle: RunHandle) -> None:
        saved.append(handle)
        if len(saved) == 1:
            original_save(handle)
            return
        raise OSError("runs directory is not writable")

    def fail_marker(_handle: RunHandle) -> None:
        raise OSError("logs directory is not writable")

    monkeypatch.setattr(store, "save", fail_second_save)
    monkeypatch.setattr(store, "mark_cleanup_uncertain", fail_marker)
    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", lambda *args, **kwargs: False)
    caplog.set_level(logging.ERROR, logger="phasesweep.mcp.server")

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert [handle.launch_state for handle in saved] == ["launching", "spawned"]
    assert "failed to persist cleanup uncertainty marker" in caplog.text
    assert "original save error" in caplog.text
    assert "cleanup uncertain after failed handle save" in caplog.text


def test_launch_spawns_runner_with_registered_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    runner_cwd = tmp_path / "runner-cwd"
    runner_cwd.mkdir()
    app, _registry, _store = make_mcp_app(
        write_mcp_catalog(
            tmp_path,
            {"srv": config},
            allow=ALLOW_SIDE_EFFECTS,
            cwd={"srv": runner_cwd},
        )
    )
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    assert captured["cwd"] == str(runner_cwd.resolve())


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
    store.save(
        make_run_handle(
            store,
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
        )
    )
    if method_name == "winners":
        _write_winner_yaml(exp, "p", phase_fingerprint="0" * 64)
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
    _write_winner_yaml(
        old_exp,
        "p",
        phase_fingerprint=_phase_fingerprint(old_exp, old_exp.phases[0], {}),
    )
    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, visible_params={"other": "all"})
    )
    snapshot = old_config.read_bytes()
    run_id = "old-launched"
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.save(
        make_run_handle(
            store,
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
    store.save(
        make_run_handle(
            store,
            run_id=run_id,
            experiment_id=registry.get("srv").id,
            config_sha256=hashlib.sha256(snapshot + b"\n").hexdigest(),
        )
    )

    with pytest.raises(Exception, match="saved config snapshot"):
        getattr(app, method_name)(run_id=run_id)


def test_winners_requires_exactly_one_identifier(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))

    with pytest.raises(Exception, match="exactly one of experiment_id or run_id"):
        app.winners()
    with pytest.raises(Exception, match="exactly one of experiment_id or run_id"):
        app.winners(experiment_id="srv", run_id="nope-123")
    with pytest.raises(Exception, match="unknown run id"):
        app.winners(run_id="nope-123")


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


def test_list_experiments_pages_catalog_and_audits(tmp_path: Path) -> None:
    configs = {f"srv{i}": _config(tmp_path, name=f"srv{i}") for i in range(3)}
    registry = Registry.load(write_mcp_catalog(tmp_path, configs))
    store = RunStore(registry.state_dir)
    audit_path = registry.state_dir / "audit.jsonl"
    app = PhaseSweepMCP(registry, store, audit=AuditLogger(audit_path))

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

    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert records[0]["tool"] == TOOL_LIST_EXPERIMENTS
    assert records[0]["args"] == {"limit": 2}
    assert records[0]["result_counts"] == {"experiments": 2, "total_count": 3}
    assert records[1]["args"] == {"cursor": "2", "limit": 2}
    assert records[1]["result_counts"] == {"experiments": 1, "total_count": 3}
    assert records[2]["outcome"] == "error"
    assert records[2]["error_type"] == "McpToolError"


def test_audit_log_records_success_and_error_without_sensitive_fields(
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
    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [record["tool"] for record in records] == [
        TOOL_VALIDATE_CONFIG,
        TOOL_LAUNCH_SWEEP,
        TOOL_LAUNCH_SWEEP,
    ]
    assert {record["session_id"] for record in records}
    assert all(record["actor"] == "local-stdio" for record in records)
    assert all(record["transport"] == "stdio" for record in records)

    validate_record = records[0]
    assert validate_record["outcome"] == "success"
    assert validate_record["args"] == {"experiment_id": "srv"}
    assert validate_record["resolved"] == {"experiment_id": "srv"}
    assert validate_record["result_counts"] == {"phases": 1, "search_space_keys": 1}

    launch_record = records[1]
    assert launch_record["outcome"] == "success"
    assert launch_record["args"] == {"experiment_id": "srv"}
    assert launch_record["resolved"] == {"experiment_id": "srv", "run_id": launched["run_id"]}
    assert launch_record["state_before"] == {"live_runs": 0}
    assert launch_record["state_after"] == {"live_runs": 1, "run_state": "running"}
    assert launch_record["result_counts"] == {"runs": 1}

    busy_record = records[2]
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

    audit.record(tool=TOOL_LIST_EXPERIMENTS, args={"cursor": "x" * 500}, outcome="success")

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
                "2026-06-24T00:00:00Z",
            ]
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "RuntimeError"
    assert status["cleanup_confirmed"] is True


def test_runner_records_cleanup_uncertainty_for_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phasesweep.engine as engine_module

    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)

    def raise_cleanup_uncertain(*args: object, **kwargs: object) -> None:
        raise UnsafeProcessCleanupError("trial cleanup uncertain")

    monkeypatch.setattr(engine_module, "run_config", raise_cleanup_uncertain)

    with pytest.raises(UnsafeProcessCleanupError, match="trial cleanup uncertain"):
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config),
                "--config-sha256",
                hashlib.sha256(config.read_bytes()).hexdigest(),
                "--status-path",
                str(status_path),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "srv",
                "--started-at",
                "2026-06-24T00:00:00Z",
            ]
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "UnsafeProcessCleanupError"
    assert status["cleanup_confirmed"] is False


def test_terminal_cleanup_uncertainty_blocks_relaunch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-terminal-uncertain"
    handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
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
    store.save(
        make_run_handle(
            store,
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
        )
    )

    with pytest.raises(Exception, match="action 'cancel' is not permitted"):
        app.cancel(run_id)


def test_cancel_decataloged_run_uses_launch_time_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_config = _config(tmp_path, name="old")
    old_snapshot = old_config.read_bytes()
    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, allow=ALLOW_SIDE_EFFECTS)
    )
    run_id = "old-running"
    handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id="old",
        config_sha256=hashlib.sha256(old_snapshot).hexdigest(),
        allow_cancel=True,
    )
    store.save(handle)

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

    assert result == {"run_id": run_id, "state": "cancelled", "cleanup_confirmed": True}
    assert not store.cleanup_uncertain_path(run_id).exists()


def test_cancel_decataloged_run_without_launch_time_permission_denied(tmp_path: Path) -> None:
    old_config = _config(tmp_path, name="old")
    old_snapshot = old_config.read_bytes()
    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, allow=ALLOW_SIDE_EFFECTS)
    )
    run_id = "old-no-cancel"
    store.save(
        make_run_handle(
            store,
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
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.save(handle)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        assert store.cleanup_uncertain_path(run_id).is_file()
        return False

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", fake_kill_stale_group)

    result = app.cancel(run_id)

    assert result == {"run_id": run_id, "state": "running", "cleanup_confirmed": False}
    assert store.cleanup_uncertain_path(run_id).is_file()

    stale_handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(stale_handle)

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
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.save(handle)

    monkeypatch.setattr("phasesweep.mcp.server.kill_stale_group", lambda *args, **kwargs: True)

    result = app.cancel(run_id)

    assert result == {"run_id": run_id, "state": "running", "cleanup_confirmed": False}
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
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.save(handle)

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

    assert result == {"run_id": run_id, "state": "running", "cleanup_confirmed": False}
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
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.save(handle)
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

    assert result == {"run_id": run_id, "state": "cancelled", "cleanup_confirmed": True}
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
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    store.mark_cleanup_uncertain(handle)

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    runner = CliRunner()
    dry = runner.invoke(
        cli_main,
        ["mcp-recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )

    assert dry.exit_code == 0, dry.output
    assert "Re-run with --confirm" in dry.output
    assert "Recovery preflight" in dry.output
    assert store.cleanup_uncertain_path(run_id).is_file()
    assert not store.status_path(run_id).exists()

    confirmed = runner.invoke(
        cli_main,
        [
            "mcp-recover-run",
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
    assert terminal_status["result_snapshot"]["status"]["phases"][0]["running"] == 0

    captured = patch_popen_capture(monkeypatch)
    launched = app.launch("srv")
    assert launched["state"] == "running"
    assert captured["cmd"]


def test_operator_recovery_does_not_create_mistyped_state_directory(tmp_path: Path) -> None:
    missing = tmp_path / "mistyped-state"

    result = CliRunner().invoke(
        cli_main,
        ["mcp-recover-run", "--state-dir", str(missing), "--run-id", "missing"],
    )

    assert result.exit_code != 0
    assert "Directory" in result.output
    assert not missing.exists()


def test_operator_recovery_clears_terminal_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    trial_number = _write_cleanup_uncertain_failed_trial(config)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-terminal-recover"
    handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

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

    def fake_trial_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
        grace_seconds: float = 30.0,
    ) -> bool:
        trial_cleanup_calls.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", fake_runner_cleanup)
    monkeypatch.setattr("phasesweep.engine.guards.kill_stale_group", fake_trial_cleanup)

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    runner = CliRunner()
    dry = runner.invoke(
        cli_main,
        ["mcp-recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )

    assert dry.exit_code == 0, dry.output
    assert "Re-run with --confirm" in dry.output
    assert "recover 1 cleanup-uncertain terminal trial" in dry.output
    assert runner_cleanup_calls == []
    assert trial_cleanup_calls == []
    assert not store.cleanup_uncertain_path(run_id).exists()
    assert store.state(handle) == "running"
    trial = _load_phase_trial(config, trial_number)
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    study = _load_first_phase_study(config)
    assert CLEANUP_RECOVERED_TRIALS_ATTR not in study.user_attrs

    confirmed = runner.invoke(
        cli_main,
        [
            "mcp-recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert confirmed.exit_code == 0, confirmed.output
    assert "Cleared cleanup uncertainty" in confirmed.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["run_id"] == run_id
    assert recovery["config_sha256"] == reg.config_sha256
    assert recovery["cleanup_confirmed"] is True
    assert recovery["reaped_running_trials"] == 0
    assert recovery["cleanup_uncertain_terminal_trials"] == 1
    assert store.state(handle) == "failed"
    assert runner_cleanup_calls == [(999999, 111, 999999)]
    assert trial_cleanup_calls == [(4242, 111, 4242)]
    trial = _load_phase_trial(config, trial_number)
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    study = _load_first_phase_study(config)
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]

    captured = patch_popen_capture(monkeypatch)
    launched = app.launch("srv")
    assert launched["state"] == "running"
    assert captured["cmd"]


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
    monkeypatch.setattr("phasesweep.engine.guards.kill_stale_group", fake_cleanup)

    first_run = "srv-terminal-first"
    first_handle = make_run_handle(
        store,
        run_id=first_run,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(first_handle)
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
            "mcp-recover-run",
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
        store,
        run_id=second_run,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999998,
        starttime=112,
    )
    store.save(second_handle)
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
            "mcp-recover-run",
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


def test_operator_recovery_counts_reaped_running_trials_as_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    trial_number = _write_stale_running_trial(config, cleanup_confirmed=False)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-running-recover"
    handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
        result_snapshot=capture_result_snapshot(exp, cleanup_confirmed=False),
    )
    assert app.status(run_id=run_id)["phases"][0]["running"] == 1

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

    def fake_trial_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
        grace_seconds: float = 30.0,
    ) -> bool:
        trial_cleanup_calls.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr("phasesweep.cli.kill_stale_group", fake_runner_cleanup)
    monkeypatch.setattr("phasesweep.engine.guards.kill_stale_group", fake_trial_cleanup)

    with pytest.raises(Exception, match="already has a running sweep"):
        app.launch("srv")

    runner = CliRunner()
    dry = runner.invoke(
        cli_main,
        ["mcp-recover-run", "--state-dir", str(registry.state_dir), "--run-id", run_id],
    )

    assert dry.exit_code == 0, dry.output
    assert "reap 1 stale trial" in dry.output
    assert runner_cleanup_calls == []
    assert trial_cleanup_calls == []
    assert not store.cleanup_recovery_path(run_id).exists()
    assert store.state(handle) == "running"

    study = optuna.load_study(study_name="srv::p", storage=exp.storage)
    trial = study.get_trials(deepcopy=False)[trial_number]
    assert trial.state == optuna.trial.TrialState.RUNNING
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False

    result = runner.invoke(
        cli_main,
        [
            "mcp-recover-run",
            "--state-dir",
            str(registry.state_dir),
            "--run-id",
            run_id,
            "--confirm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "reaped 1 stale trial" in result.output
    recovery = json.loads(store.cleanup_recovery_path(run_id).read_text())
    assert recovery["run_id"] == run_id
    assert recovery["config_sha256"] == reg.config_sha256
    assert recovery["cleanup_confirmed"] is True
    assert recovery["reaped_running_trials"] == 1
    assert recovery["cleanup_uncertain_terminal_trials"] == 0
    assert store.state(handle) == "failed"
    recovered_status = app.status(run_id=run_id)
    assert recovered_status["phases"][0]["trials"] == {"FAIL": 1}
    assert recovered_status["phases"][0]["running"] == 0
    assert runner_cleanup_calls == [(999999, 111, 999999)]
    assert trial_cleanup_calls == [(4343, 222, 4343)]

    study = optuna.load_study(study_name="srv::p", storage=exp.storage)
    trial = study.get_trials(deepcopy=False)[trial_number]
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [trial_number]

    captured = patch_popen_capture(monkeypatch)
    launched = app.launch("srv")
    assert launched["state"] == "running"
    assert captured["cmd"]


def test_operator_recovery_refuses_terminal_uncertainty_without_trial_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-terminal-no-evidence"
    handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
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
            "mcp-recover-run",
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
        store,
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes() + b"\n# drifted\n")
    store.mark_cleanup_uncertain(handle)

    result = CliRunner().invoke(
        cli_main,
        [
            "mcp-recover-run",
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
