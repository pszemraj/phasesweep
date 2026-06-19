"""MCP server logic that does not need a real detached runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from phasesweep.config import Experiment, load_config
from phasesweep.engine.guards import _phase_fingerprint
from phasesweep.engine.state import _winner_path
from phasesweep.mcp.errors import UnknownExperimentError
from phasesweep.mcp.runner import main as runner_main
from phasesweep.mcp.runs import RunHandle, utc_now_iso
from phasesweep.mcp.server import _safe_tool
from phasesweep.runtime.process import read_proc_starttime
from tests.mcp_helpers import make_mcp_app, write_mcp_catalog

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
    captured: dict[str, list[str]] = {}

    class DummyProc:
        pid = os.getpid()

    def fake_popen(cmd: list[str], **_kwargs: object) -> DummyProc:
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr("phasesweep.mcp.server.subprocess.Popen", fake_popen)

    app.launch("srv")

    cmd = captured["cmd"]
    config_arg = Path(cmd[cmd.index("--config") + 1])
    sha_arg = cmd[cmd.index("--config-sha256") + 1]
    assert config_arg != config.resolve()
    assert config_arg.read_bytes() == config.read_bytes()
    assert sha_arg == registry.get("srv").config_sha256


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
    pid = os.getpid()
    run_id = "srv-denied"
    store.save(
        RunHandle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            pid=pid,
            pgid=pid,
            pid_starttime=read_proc_starttime(pid),
            started_at=utc_now_iso(),
            log_path=str(store.log_path(run_id)),
            status_path=str(store.status_path(run_id)),
        )
    )

    with pytest.raises(Exception, match="action 'cancel' is not permitted"):
        app.cancel(run_id)
