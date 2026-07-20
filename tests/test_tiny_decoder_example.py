"""CPU-only checks for the tiny-decoder trial wrapper's evidence contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER_PATH = REPO_ROOT / "examples" / "tiny_decoder_enwik8" / "run_trial.py"


def _load_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tiny_decoder_run_trial", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_publishes_attempt_scoped_final_checkpoint_result(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    template_root = tmp_path / "upstream"
    template_root.mkdir()
    base_config = tmp_path / "base.yaml"
    base_config.write_text("num_batches: 1000\nvalidate_every: 100\nval_batches: 5\n")
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text('{"learning_rate": 0.001}\n')
    trial_dir = tmp_path / "trial"
    observed: dict[str, object] = {}

    def fake_run(_template_root: Path, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text())
        observed["config"] = config
        run_dir = Path(config["run_dir"])
        run_dir.mkdir(parents=True)
        (run_dir / "final.pt").write_bytes(b"checkpoint")

    def fake_evaluate(_template_root: Path, run_dir: Path) -> dict[str, object]:
        assert (run_dir / "final.pt").read_bytes() == b"checkpoint"
        return {
            "checkpoint": "final.pt",
            "policy": "final_checkpoint",
            "step": 1000,
            "val_loss": 0.25,
        }

    monkeypatch.setattr(wrapper, "_run_template", fake_run)
    monkeypatch.setattr(wrapper, "_evaluate_final_checkpoint", fake_evaluate)
    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    overrides_sha256 = hashlib.sha256(overrides_path.read_bytes()).hexdigest()
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", overrides_sha256)

    assert (
        wrapper.main(
            [
                "--template-root",
                str(template_root),
                "--base-config",
                str(base_config),
                "--overrides-path",
                str(overrides_path),
                "--trial-dir",
                str(trial_dir),
            ]
        )
        == 0
    )

    assert observed["config"]["num_batches"] == 1000
    assert observed["config"]["validate_every"] == 100
    result = json.loads((trial_dir / "result.json").read_text())
    assert result == {
        "attempt_id": "attempt-test",
        "evaluation": {
            "checkpoint": "final.pt",
            "policy": "final_checkpoint",
            "step": 1000,
        },
        "generation_id": "generation-test",
        "objective": {"name": "val_loss", "split": "validation", "value": 0.25},
        "overrides": {"learning_rate": 0.001},
        "overrides_sha256": overrides_sha256,
        "schema_version": 1,
        "status": "complete",
        "val_loss": 0.25,
    }
    assert list(trial_dir.glob(".result.json.*.tmp")) == []


def test_result_publish_failure_preserves_existing_evidence(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result_path = trial_dir / "result.json"
    result_path.write_text('{"status": "old"}\n')
    original = result_path.read_bytes()
    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    overrides_sha256 = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", overrides_sha256)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(wrapper.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        wrapper._write_result(
            trial_dir,
            {},
            overrides_sha256,
            {
                "checkpoint": "final.pt",
                "policy": "final_checkpoint",
                "step": 1000,
                "val_loss": 0.25,
            },
        )

    assert result_path.read_bytes() == original
    assert list(trial_dir.glob(".result.json.*.tmp")) == []
