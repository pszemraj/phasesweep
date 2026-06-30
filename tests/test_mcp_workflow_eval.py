"""Quality-gate script behavior for the MCP workflow eval."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _load_workflow_eval() -> ModuleType:
    """Load the repo-local workflow eval script as a testable module."""
    spec = importlib.util.spec_from_file_location(
        "phasesweep_mcp_workflow_eval", REPO / "scripts" / "mcp_workflow_eval.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_eval_main_isolates_lock_and_cuda_env(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Scenarios should not inherit host GPU visibility or lock directories."""
    workflow_eval = _load_workflow_eval()
    seen: list[tuple[str, Path, str | None, str | None]] = []

    class FakeTemporaryDirectory:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, *exc_info: object) -> None:
            return None

    def scenario(name: str):
        def run(root: Path, **_kwargs: object) -> object:
            seen.append(
                (
                    name,
                    root,
                    os.environ.get("PHASESWEEP_LOCK_DIR"),
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                )
            )
            return workflow_eval.ScenarioResult(
                scenario=name,
                success=True,
                tool_calls=0,
                error_count=0,
                duration_seconds=0.0,
            )

        return run

    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", "/ambient/locks")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-ambient")
    monkeypatch.setattr(workflow_eval.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(workflow_eval, "discovery_scenario", scenario("discovery"))
    monkeypatch.setattr(workflow_eval, "read_only_safety_scenario", scenario("read_only_safety"))
    monkeypatch.setattr(workflow_eval, "happy_path_scenario", scenario("happy_path"))

    assert workflow_eval.main(["--timeout", "0.1"]) == 0

    lock_dir = tmp_path / "locks"
    assert lock_dir.is_dir()
    assert seen == [
        ("discovery", tmp_path / "discovery", str(lock_dir), "-1"),
        ("read_only_safety", tmp_path / "readonly", str(lock_dir), "-1"),
        ("happy_path", tmp_path / "happy", str(lock_dir), "-1"),
    ]
    assert os.environ["PHASESWEEP_LOCK_DIR"] == "/ambient/locks"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-ambient"
    assert '"success": true' in capsys.readouterr().out
