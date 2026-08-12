"""CPU-only checks for the tiny-decoder trial wrapper's evidence contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    trial_dir = tmp_path / "trial"
    config_path = trial_dir / "trainer_config.yaml"
    trial_dir.mkdir()
    config_path.write_text(
        f"run_dir: {trial_dir / 'trainer'}\n"
        "num_batches: 1000\n"
        "validate_every: 100\n"
        "val_batches: 5\n"
        "learning_rate: 0.001\n"
    )
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
            "device_type": "cuda",
            "policy": "final_checkpoint",
            "step": 1000,
            "val_loss": 0.25,
        }

    monkeypatch.setattr(wrapper, "_run_template", fake_run)
    monkeypatch.setattr(wrapper, "_evaluate_final_checkpoint", fake_evaluate)
    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", config_sha256)
    monkeypatch.setenv("PHASESWEEP_OBJECTIVE_PATH", str(trial_dir / "result.json"))

    assert (
        wrapper.main(
            [
                "--template-root",
                str(template_root),
                "--config",
                str(config_path),
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
        "overrides_sha256": config_sha256,
        "runtime": {"device_type": "cuda"},
        "schema_version": 1,
        "status": "complete",
    }
    assert list(trial_dir.glob(".result.json.*.tmp")) == []


def test_wrapper_rejects_a_config_from_another_attempt(tmp_path, monkeypatch) -> None:
    wrapper = _load_wrapper()
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", "expected")
    monkeypatch.setenv("PHASESWEEP_OBJECTIVE_PATH", str(trial_dir / "result.json"))

    with pytest.raises(ValueError, match="Trainer config does not match"):
        wrapper._write_result(
            "different",
            {
                "checkpoint": "final.pt",
                "device_type": "cuda",
                "policy": "final_checkpoint",
                "step": 1000,
                "val_loss": 0.25,
            },
        )

    assert not (trial_dir / "result.json").exists()


def test_final_evaluator_applies_zero_seed(tmp_path, monkeypatch) -> None:
    wrapper = _load_wrapper()
    observed: list[int] = []

    class FakeModel:
        def to(self, _device):
            return self

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return None

    fake_torch = SimpleNamespace(
        load=lambda *_args, **_kwargs: {
            "config": {"seed": 0, "data_path": "data.bin"},
            "model": {},
        },
        manual_seed=observed.append,
    )
    fake_trainer = SimpleNamespace(
        torch=fake_torch,
        get_optimal_device=lambda: ("cpu", "cpu", None),
        Llama=lambda **_kwargs: FakeModel(),
        load_data=lambda _path: (_ for _ in ()).throw(RuntimeError("evaluation reached data")),
    )
    monkeypatch.setattr(wrapper, "_load_upstream_trainer", lambda _root: fake_trainer)

    with pytest.raises(RuntimeError, match="evaluation reached data"):
        wrapper._evaluate_final_checkpoint(tmp_path, tmp_path)

    assert observed == [0]


def test_final_evaluator_rejects_empty_validation_budget(tmp_path, monkeypatch) -> None:
    """A zero-batch evaluation cannot publish a data-free validation objective."""
    wrapper = _load_wrapper()

    class FakeModel:
        def to(self, _device):
            return self

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return None

    fake_torch = SimpleNamespace(
        load=lambda *_args, **_kwargs: {
            "config": {
                "batch_size": 1,
                "data_path": "data.bin",
                "seq_len": 8,
                "val_batches": 0,
            },
            "model": {},
        },
    )
    fake_trainer = SimpleNamespace(
        torch=fake_torch,
        get_optimal_device=lambda: ("cpu", "cpu", None),
        Llama=lambda **_kwargs: FakeModel(),
        load_data=lambda _path: (b"train", b"validation"),
        SequenceDataset=lambda *_args: object(),
        DataLoader=lambda *_args, **_kwargs: object(),
        cycle=lambda _loader: iter(()),
    )
    monkeypatch.setattr(wrapper, "_load_upstream_trainer", lambda _root: fake_trainer)

    with pytest.raises(ValueError, match="evaluated no validation tokens"):
        wrapper._evaluate_final_checkpoint(tmp_path, tmp_path)
