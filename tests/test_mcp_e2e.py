"""End-to-end MCP flow: launch a real detached sweep, monitor it to completion,
read winners, and exercise the launch -> cancel -> relaunch cycle.
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import pytest

from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.server import PhaseSweepMCP
from tests.conftest import REPO

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="detached runner + cancel rely on POSIX process groups + /proc liveness",
)

_TRAINER = REPO / "examples" / "fake_train.py"


def _write_catalog(tmp_path: Path, *, entry_id: str, config_body: str) -> Path:
    config = tmp_path / f"{entry_id}.yaml"
    config.write_text(config_body)
    catalog = tmp_path / f"{entry_id}.catalog.yaml"
    catalog.write_text(
        f"state_dir: {tmp_path}/state\nexperiments:\n  - id: {entry_id}\n    config: {config}\n"
    )
    return catalog


def _write_multi_catalog(
    tmp_path: Path, configs: dict[str, str], *, max_concurrent_runs: int | None = None
) -> Path:
    lines = [f"state_dir: {tmp_path}/state"]
    if max_concurrent_runs is not None:
        lines.append(f"max_concurrent_runs: {max_concurrent_runs}")
    lines.append("experiments:")
    for entry_id, body in configs.items():
        config = tmp_path / f"{entry_id}.yaml"
        config.write_text(body)
        lines += [f"  - id: {entry_id}", f"    config: {config}"]
    catalog = tmp_path / "multi.catalog.yaml"
    catalog.write_text("\n".join(lines) + "\n")
    return catalog


def _chained_config(tmp_path: Path) -> str:
    return f"""\
experiment: e2e_lm
storage: sqlite:///{tmp_path}/phases.db
workdir: {tmp_path}/runs
trial_command: "{sys.executable} {_TRAINER} --out {{trial_dir}}/result.json {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
constraints:
  - name: param_bytes
    extractor: {{ type: json, path: result.json, key: param_bytes }}
    max: 16777216
phases:
  - name: depth
    n_trials: 2
    sampler: {{ type: grid }}
    search_space:
      n_layers: {{ type: categorical, choices: [4, 8] }}
  - name: lr
    inherits: [depth]
    n_trials: 3
    sampler: {{ type: tpe, seed: 0 }}
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
"""


def _slow_config(tmp_path: Path, *, name: str = "slow", sleep: float = 30.0) -> str:
    return f"""\
experiment: {name}
storage: sqlite:///{tmp_path}/{name}.db
workdir: {tmp_path}/runs/{name}
trial_command: "{sys.executable} {_TRAINER} --out {{trial_dir}}/result.json --sleep {sleep} {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
phases:
  - name: p
    n_trials: 1
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
"""


def _app(catalog: Path) -> tuple[PhaseSweepMCP, RunStore]:
    registry = Registry.load(catalog)
    store = RunStore(registry.state_dir)
    return PhaseSweepMCP(registry, store), store


def _wait_state(app: PhaseSweepMCP, run_id: str, *, want: set[str], timeout: float) -> str:
    deadline = time.time() + timeout
    state = "unknown"
    while time.time() < deadline:
        state = app.status(run_id=run_id)["run"]["state"]
        if state in want:
            return state
        time.sleep(0.3)
    return state


def _wait_for_running_trial(app: PhaseSweepMCP, run_id: str, *, timeout: float) -> str:
    # Wait until a trial is actually executing, which guarantees the runner has
    # entered run_experiment and installed its signal handlers - so a cancel
    # hits the graceful SIGTERM->143 path rather than a pre-handler startup kill.
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = app.status(run_id=run_id)
        if status["run"]["state"] in {"succeeded", "failed", "cancelled"}:
            return status["run"]["state"]
        if status["phases"][0]["running"] >= 1:
            return "running"
        time.sleep(0.2)
    return "timeout"


def _cancel_quietly(app: PhaseSweepMCP, run_id: str) -> None:
    with contextlib.suppress(Exception):
        app.cancel(run_id)


def test_list_validate_launch_monitor_winners(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path, entry_id="e2e_lm", config_body=_chained_config(tmp_path))
    app, store = _app(catalog)

    # Catalog metadata is path-free and well-formed.
    summaries = app.list_experiments()
    assert summaries[0]["id"] == "e2e_lm"
    assert summaries[0]["phases"] == ["depth", "lr"]
    structure = app.validate("e2e_lm")
    assert [p["name"] for p in structure["phases"]] == ["depth", "lr"]
    assert structure["phases"][0]["sampler"] == "grid"
    assert structure["phases"][1]["search_space"] == ["lr"]  # keys only

    run_id = app.launch("e2e_lm")["run_id"]
    try:
        state = _wait_state(app, run_id, want={"succeeded", "failed", "cancelled"}, timeout=120)
        log = Path(store.get(run_id).log_path).read_text()
        assert state == "succeeded", f"run ended {state}; log:\n{log}"

        winners = app.winners("e2e_lm")
        phases = winners["phases"]
        assert [p["phase"] for p in phases] == ["depth", "lr"]
        for p in phases:
            assert isinstance(p["metric"], float)
            assert p["metric"] == p["metric"]  # not NaN
            assert "effective_overrides" in p
        # The chained phase carries the inherited depth winner.
        assert "n_layers" in phases[1]["effective_overrides"]
    finally:
        _cancel_quietly(app, run_id)


def test_launch_then_cancel_then_relaunch(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path, entry_id="slow", config_body=_slow_config(tmp_path))
    app, _store = _app(catalog)

    run_id = app.launch("slow")["run_id"]
    second_run_id = None
    try:
        got = _wait_for_running_trial(app, run_id, timeout=30)
        assert got == "running", f"expected a running trial, got {got}"
        # A second launch while one is live is refused.
        with pytest.raises(Exception, match="already has a running sweep"):
            app.launch("slow")

        result = app.cancel(run_id)
        assert result["state"] == "cancelled"
        assert result["cleanup_confirmed"] is True

        # Once the prior run is terminal the lock is released and relaunch works.
        second = app.launch("slow")
        second_run_id = second["run_id"]
        assert second["state"] == "running"
    finally:
        _cancel_quietly(app, run_id)
        if second_run_id is not None:
            _cancel_quietly(app, second_run_id)


def test_global_concurrency_cap_serializes_sweeps(tmp_path: Path) -> None:
    catalog = _write_multi_catalog(
        tmp_path,
        {
            "slowa": _slow_config(tmp_path, name="slowa"),
            "slowb": _slow_config(tmp_path, name="slowb"),
        },
    )  # default max_concurrent_runs == 1
    app, _store = _app(catalog)

    run_a = app.launch("slowa")["run_id"]
    run_b = None
    try:
        assert _wait_for_running_trial(app, run_a, timeout=30) == "running"
        # A *different* experiment cannot start while one sweep is live (single-GPU cap).
        with pytest.raises(Exception, match="limit 1"):
            app.launch("slowb")
        # Freeing the slot lets the other experiment launch.
        assert app.cancel(run_a)["state"] == "cancelled"
        result = app.launch("slowb")
        run_b = result["run_id"]
        assert result["state"] == "running"
    finally:
        _cancel_quietly(app, run_a)
        if run_b is not None:
            _cancel_quietly(app, run_b)


def test_launch_refused_while_launch_lock_held(tmp_path: Path) -> None:
    # Holding the launch lock stands in for a concurrent launch mid-decision.
    # The cap check and spawn are serialized under it, so a second launch is
    # told to retry rather than racing past the cap. The refused path spawns
    # nothing (it raises before _spawn), so no real sweep is needed and the
    # test stays deterministic. White-box: reach for the store's lock directly.
    from phasesweep.runtime.files import try_lock_file, unlock_file

    catalog = _write_catalog(tmp_path, entry_id="slow", config_body=_slow_config(tmp_path))
    app, store = _app(catalog)

    held = try_lock_file(store._launch_lock_path)
    assert held is not None
    try:
        with pytest.raises(Exception, match="launch is in progress"):
            app.launch("slow")
    finally:
        unlock_file(held)

    # The lock is advisory and released on unlock: a fresh acquire succeeds.
    regrab = try_lock_file(store._launch_lock_path)
    assert regrab is not None
    unlock_file(regrab)


def test_status_and_cancel_error_paths(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path, entry_id="e2e_lm", config_body=_chained_config(tmp_path))
    app, _store = _app(catalog)

    with pytest.raises(Exception, match="unknown run id"):
        app.status(run_id="nope-123")
    with pytest.raises(Exception, match="unknown run id"):
        app.cancel("nope-123")
    # A path-shaped run_id never reaches the filesystem: it reads as unknown.
    for traversal in ("../../etc/passwd", "../../../runs/secret"):
        with pytest.raises(Exception, match="unknown run id"):
            app.status(run_id=traversal)
        with pytest.raises(Exception, match="unknown run id"):
            app.cancel(traversal)
    with pytest.raises(Exception, match="either experiment_id or run_id"):
        app.status()
    # No run launched yet: experiment-level status reports no live run.
    assert app.status(experiment_id="e2e_lm")["run"] is None


def test_fastmcp_registers_six_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from phasesweep.mcp.server import build_server

    catalog = _write_catalog(tmp_path, entry_id="e2e_lm", config_body=_chained_config(tmp_path))
    app, _store = _app(catalog)
    server = build_server(app)
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        "list_experiments",
        "validate_config",
        "get_status",
        "get_winners",
        "launch_sweep",
        "cancel_sweep",
    }
    assert all(t.description for t in tools)

    # The _safe_tool wrapper (functools.wraps + *args/**kwargs) must not erase
    # the parameter schema FastMCP derives from each signature, or the agent
    # could not call the tools. Lock the shapes in.
    schemas = {t.name: t.inputSchema for t in tools}
    assert sorted(schemas["launch_sweep"]["properties"]) == ["experiment_id", "from_phase"]
    assert schemas["launch_sweep"]["required"] == ["experiment_id"]
    assert sorted(schemas["get_status"]["properties"]) == ["experiment_id", "run_id"]
    assert schemas["get_status"].get("required") is None  # both optional
    assert schemas["cancel_sweep"]["required"] == ["run_id"]
    assert schemas["validate_config"]["required"] == ["experiment_id"]
    assert not schemas["list_experiments"]["properties"]
