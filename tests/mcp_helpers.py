"""Shared MCP test setup helpers."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle, RunLaunchState, RunStore, write_status_file
from phasesweep.mcp.server import PhaseSweepMCP
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.process import read_proc_starttime


def write_mcp_catalog(
    tmp_path: Path,
    entries: Mapping[str, Path],
    *,
    allow: Mapping[str, bool] | None = None,
    cwd: Mapping[str, Path] | None = None,
    visible_params: Mapping[str, object] | None = None,
    max_concurrent_runs: int | None = None,
    filename: str = "catalog.yaml",
) -> Path:
    lines = [f"state_dir: {tmp_path}/state"]
    if max_concurrent_runs is not None:
        lines.append(f"max_concurrent_runs: {max_concurrent_runs}")
    lines.append("experiments:")
    for entry_id, config in entries.items():
        lines += [f"  - id: {entry_id}", f"    config: {config}"]
        if cwd is not None and entry_id in cwd:
            lines.append(f"    cwd: {cwd[entry_id]}")
        if visible_params is not None and entry_id in visible_params:
            value = visible_params[entry_id]
            if isinstance(value, list):
                lines.append("    visible_params:")
                lines.extend(f"      - {item}" for item in value)
            else:
                lines.append(f"    visible_params: {value}")
        if allow is not None:
            lines.append("    allow:")
            lines.extend(f"      {key}: {str(value).lower()}" for key, value in allow.items())
    catalog = tmp_path / filename
    catalog.write_text("\n".join(lines) + "\n")
    return catalog


def write_mcp_config_catalog(
    tmp_path: Path,
    configs: Mapping[str, str],
    *,
    allow: Mapping[str, bool] | None = None,
    cwd: Mapping[str, Path] | None = None,
    visible_params: Mapping[str, object] | None = None,
    max_concurrent_runs: int | None = None,
    filename: str = "catalog.yaml",
) -> Path:
    entries = {}
    for entry_id, body in configs.items():
        config = tmp_path / f"{entry_id}.yaml"
        config.write_text(body)
        entries[entry_id] = config
    return write_mcp_catalog(
        tmp_path,
        entries,
        allow=allow,
        cwd=cwd,
        visible_params=visible_params,
        max_concurrent_runs=max_concurrent_runs,
        filename=filename,
    )


def mcp_experiment_config_text(
    tmp_path: Path,
    *,
    name: str = "srv",
    phases: str | None = None,
    with_storage: bool = True,
) -> str:
    if phases is None:
        phases = """\
  - name: p
    n_trials: 1
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
"""
    storage = f"storage: sqlite:///{tmp_path}/{name}.db\n" if with_storage else ""
    return f"""\
experiment: {name}
{storage}workdir: {tmp_path}/runs/{name}
trial_command: "python train.py --out {{trial_dir}}/r.json {{overrides}}"
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
phases:
{phases}"""


def slow_mcp_config_text(
    tmp_path: Path,
    *,
    trainer: Path,
    name: str = "slow",
    sleep: float = 30.0,
) -> str:
    return f"""\
experiment: {name}
storage: sqlite:///{tmp_path}/{name}.db
workdir: {tmp_path}/runs/{name}
trial_command: "{sys.executable} {trainer} --out {{trial_dir}}/result.json --sleep {sleep} {{overrides}}"
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


def make_mcp_app(catalog: Path) -> tuple[PhaseSweepMCP, Registry, RunStore]:
    registry = Registry.load(catalog)
    store = RunStore(registry.state_dir)
    return PhaseSweepMCP(registry, store), registry, store


def wait_for_mcp_state(
    app: PhaseSweepMCP,
    run_id: str,
    *,
    want: set[str],
    timeout: float,
) -> str:
    deadline = time.time() + timeout
    state = "unknown"
    while time.time() < deadline:
        state = app.status(run_id=run_id)["run"]["state"]
        if state in want:
            return state
        time.sleep(0.3)
    return state


def wait_for_mcp_running_trial(app: PhaseSweepMCP, run_id: str, *, timeout: float) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = app.status(run_id=run_id)
        if status["run"]["state"] in {"succeeded", "failed", "cancelled"}:
            return status["run"]["state"]
        if status["phases"][0]["running"] >= 1:
            return "running"
        time.sleep(0.2)
    return "timeout"


def cancel_mcp_run_quietly(app: PhaseSweepMCP, run_id: str) -> None:
    with contextlib.suppress(Exception):
        app.cancel(run_id)


def assert_no_sensitive(payload: Any, sensitive: Iterable[str]) -> None:
    """Raise ``AssertionError`` if any string leaf contains a sensitive value."""
    needles = [s for s in sensitive if s]

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for needle in needles:
                assert needle not in node, f"sensitive value leaked into payload: {needle!r}"
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)


def make_run_handle(
    store: RunStore,
    *,
    run_id: str,
    experiment_id: str = "exp",
    config_sha256: str = "0" * 64,
    pid: int | None = None,
    starttime: int | None = None,
    launch_state: RunLaunchState = "spawned",
    allow_cancel: bool = False,
) -> RunHandle:
    if launch_state == "launching":
        return RunHandle(
            run_id=run_id,
            experiment_id=experiment_id,
            config_sha256=config_sha256,
            pid=None,
            pgid=None,
            pid_starttime=None,
            started_at=utc_now_iso(),
            log_path=str(store.log_path(run_id)),
            status_path=str(store.status_path(run_id)),
            launch_state=launch_state,
            allow_cancel=allow_cancel,
        )
    process_id = os.getpid() if pid is None else pid
    return RunHandle(
        run_id=run_id,
        experiment_id=experiment_id,
        config_sha256=config_sha256,
        pid=process_id,
        pgid=process_id,
        pid_starttime=read_proc_starttime(process_id) if starttime is None else starttime,
        started_at=utc_now_iso(),
        log_path=str(store.log_path(run_id)),
        status_path=str(store.status_path(run_id)),
        launch_state=launch_state,
        allow_cancel=allow_cancel,
    )


def write_run_status(store: RunStore, run_id: str, **payload: object) -> None:
    full_payload = {"run_id": run_id, **payload}
    write_status_file(store.status_path(run_id), full_payload)


def patch_popen_capture(monkeypatch: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class DummyProc:
        pid = os.getppid()

    def fake_popen(cmd: list[str], **kwargs: object) -> DummyProc:
        stdout = kwargs.get("stdout")
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("stderr") is subprocess.STDOUT
        assert kwargs.get("start_new_session") is True
        assert stdout is not None and not getattr(stdout, "closed", True)
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return DummyProc()

    monkeypatch.setattr("phasesweep.mcp.server.subprocess.Popen", fake_popen)
    return captured
