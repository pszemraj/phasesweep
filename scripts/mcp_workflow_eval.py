#!/usr/bin/env python3
"""Run small deterministic evaluations for the PhaseSweep MCP workflow."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.server import PhaseSweepMCP

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "examples" / "fake_train.py"


@dataclass
class ScenarioResult:
    """One MCP workflow scenario outcome."""

    scenario: str
    success: bool
    tool_calls: int
    error_count: int
    duration_seconds: float
    terminal_state: str | None = None
    message: str = ""


class EvalClient:
    """Tiny counter around the SDK-free MCP implementation."""

    def __init__(self, app: PhaseSweepMCP) -> None:
        self.app = app
        self.tool_calls = 0
        self.error_count = 0

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a named service method and count tool-level outcomes."""
        self.tool_calls += 1
        try:
            return getattr(self.app, name)(*args, **kwargs)
        except Exception:
            self.error_count += 1
            raise


def _write_config(root: Path, *, name: str = "mcp_eval") -> Path:
    """Write a small persistent experiment config for local evals."""
    root.mkdir(parents=True, exist_ok=True)
    config = root / f"{name}.yaml"
    config.write_text(
        f"""\
experiment: {name}
storage: sqlite:///{root}/{name}.db
workdir: {root}/runs/{name}
trial_command: "{sys.executable} {TRAINER} --out {{trial_dir}}/result.json {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
phases:
  - name: depth
    n_trials: 2
    sampler: {{ type: grid }}
    search_space:
      n_layers: {{ type: categorical, choices: [4, 8] }}
"""
    )
    return config


def _write_catalog(root: Path, config: Path, *, allow_launch: bool) -> Path:
    """Write a one-entry MCP catalog."""
    allow = (
        """\
    allow:
      launch: true
      cancel: true
      from_phase: true
"""
        if allow_launch
        else ""
    )
    catalog = root / "catalog.yaml"
    catalog.write_text(
        f"""\
state_dir: {root}/state
max_concurrent_runs: 1
experiments:
  - id: mcp-eval
    config: {config}
    description: "MCP workflow eval"
{allow}"""
    )
    return catalog


def _make_client(catalog: Path) -> EvalClient:
    """Build an SDK-free MCP client facade from a catalog."""
    registry = Registry.load(catalog)
    return EvalClient(PhaseSweepMCP(registry, RunStore(registry.state_dir)))


def _wait_terminal(client: EvalClient, run_id: str, *, timeout: float) -> str:
    """Poll run status until terminal or timeout."""
    deadline = time.monotonic() + timeout
    state = "unknown"
    while time.monotonic() < deadline:
        status = client.call("status", run_id=run_id)
        state = status["run"]["state"]
        if state in {"succeeded", "failed", "cancelled"}:
            return state
        time.sleep(0.25)
    return state


def _result(
    scenario: str,
    client: EvalClient,
    started: float,
    *,
    success: bool,
    terminal_state: str | None = None,
    message: str = "",
) -> ScenarioResult:
    """Build a scenario result with counters."""
    return ScenarioResult(
        scenario=scenario,
        success=success,
        tool_calls=client.tool_calls,
        error_count=client.error_count,
        duration_seconds=round(time.monotonic() - started, 3),
        terminal_state=terminal_state,
        message=message,
    )


def discovery_scenario(root: Path) -> ScenarioResult:
    """Verify the agent can discover and validate a read-only catalog."""
    client = _make_client(_write_catalog(root, _write_config(root), allow_launch=False))
    started = time.monotonic()
    try:
        listing = client.call("list_experiments")
        structure = client.call("validate", "mcp-eval")
        success = (
            listing["experiments"][0]["id"] == "mcp-eval"
            and structure["phases"][0]["name"] == "depth"
        )
        return _result("discovery", client, started, success=success)
    except Exception as exc:
        return _result("discovery", client, started, success=False, message=str(exc))


def read_only_safety_scenario(root: Path) -> ScenarioResult:
    """Verify side effects are refused when catalog permissions are read-only."""
    client = _make_client(_write_catalog(root, _write_config(root), allow_launch=False))
    started = time.monotonic()
    try:
        client.call("launch", "mcp-eval")
    except Exception as exc:
        return _result(
            "read_only_safety",
            client,
            started,
            success="not permitted" in str(exc),
            message=str(exc),
        )
    return _result(
        "read_only_safety", client, started, success=False, message="launch unexpectedly succeeded"
    )


def happy_path_scenario(root: Path, *, timeout: float) -> ScenarioResult:
    """Launch, monitor, and read winners for a tiny real sweep."""
    client = _make_client(_write_catalog(root, _write_config(root), allow_launch=True))
    started = time.monotonic()
    run_id: str | None = None
    terminal_state: str | None = None
    try:
        run_id = client.call("launch", "mcp-eval")["run_id"]
        terminal_state = _wait_terminal(client, run_id, timeout=timeout)
        winners = client.call("winners", "mcp-eval")
        success = terminal_state == "succeeded" and len(winners["phases"]) == 1
        return _result(
            "happy_path", client, started, success=success, terminal_state=terminal_state
        )
    except Exception as exc:
        return _result(
            "happy_path",
            client,
            started,
            success=False,
            terminal_state=terminal_state,
            message=str(exc),
        )
    finally:
        if run_id is not None and terminal_state not in {"succeeded", "failed", "cancelled"}:
            with contextlib.suppress(Exception):
                client.call("cancel", run_id)


@contextlib.contextmanager
def _isolated_runtime_env(root: Path) -> Iterator[None]:
    """Run eval scenarios with lock and CUDA state rooted in the temp tree."""
    lock_dir = root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    keys = ("PHASESWEEP_LOCK_DIR", "CUDA_VISIBLE_DEVICES")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["PHASESWEEP_LOCK_DIR"] = str(lock_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    """Run all MCP workflow eval scenarios and print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="Seconds to wait for the happy-path run."
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="phasesweep-mcp-eval-") as tmp:
        root = Path(tmp)
        with _isolated_runtime_env(root):
            results = [
                discovery_scenario(root / "discovery"),
                read_only_safety_scenario(root / "readonly"),
                happy_path_scenario(root / "happy", timeout=args.timeout),
            ]

    payload = {
        "success": all(item.success for item in results),
        "scenarios": [asdict(item) for item in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
