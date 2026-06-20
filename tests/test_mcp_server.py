"""MCP server logic that does not need a real detached runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from phasesweep.config import Experiment, load_config
from phasesweep.engine.guards import _phase_fingerprint
from phasesweep.engine.state import _winner_path
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
from tests.mcp_helpers import make_mcp_app, make_run_handle, patch_popen_capture, write_mcp_catalog

ALLOW_SIDE_EFFECTS = {"launch": True, "cancel": True, "from_phase": True}


def _config(tmp_path: Path, *, name: str = "srv", phases: str | None = None) -> Path:
    if phases is None:
        phases = """\
  - name: p
    n_trials: 1
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
"""
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""\
experiment: {name}
storage: sqlite:///{tmp_path}/{name}.db
workdir: {tmp_path}/runs/{name}
trial_command: "python train.py --out {{trial_dir}}/r.json {{overrides}}"
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
phases:
{phases}"""
    )
    return path


def _catalog(tmp_path: Path, config: Path, allow: dict[str, bool] | None = None) -> Path:
    return write_mcp_catalog(
        tmp_path,
        {"srv": config},
        allow=allow,
        filename="srv.catalog.yaml",
    )


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
    app, registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    cmd = captured["cmd"]
    config_arg = Path(cmd[cmd.index("--config") + 1])
    sha_arg = cmd[cmd.index("--config-sha256") + 1]
    assert config_arg != config.resolve()
    assert config_arg.read_bytes() == config.read_bytes()
    assert sha_arg == registry.get("srv").config_sha256


def test_launch_terminates_spawned_runner_when_handle_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    saved: dict[str, RunHandle] = {}
    terminated: list[int] = []

    def fail_save(handle: RunHandle) -> None:
        saved["handle"] = handle
        raise OSError("runs directory is not writable")

    def fake_terminate_group(pgid: int) -> bool:
        terminated.append(pgid)
        return True

    monkeypatch.setattr(store, "save", fail_save)
    monkeypatch.setattr("phasesweep.mcp.server.terminate_group", fake_terminate_group)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    handle = saved["handle"]
    assert terminated == [handle.pgid]
    assert store.list_handles() == []


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


def test_runner_rejects_config_snapshot_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    status_path = tmp_path / "status.json"

    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner_main(
            [
                "--run-id",
                "r1",
                "--config",
                str(config),
                "--config-sha256",
                "0" * 64,
                "--status-path",
                str(status_path),
            ]
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "RuntimeError"


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
