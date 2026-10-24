"""CPU-only checks for the tiny-decoder trial wrapper's evidence contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from phasesweep import load_experiment
from phasesweep.evidence.evaluation import evaluate_gates
from tests.conftest import make_trial_context

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER_PATH = REPO_ROOT / "examples" / "tiny_decoder_enwik8" / "run_trial.py"


def _load_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tiny_decoder_run_trial", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trainer_script(tmp_path: Path) -> Path:
    """Provide a checkout marker for workflows that replace training with a stub."""
    script = tmp_path / "upstream" / "train.py"
    script.parent.mkdir()
    script.touch()
    return script


@pytest.mark.parametrize("device_type", ["cuda", "mps", "cpu"])
def test_wrapper_publishes_attempt_scoped_final_checkpoint_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_type: str, trainer_script: Path
) -> None:
    wrapper = _load_wrapper()
    monkeypatch.setitem(sys.modules, "mlflow", None)
    template_root = trainer_script.parent
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

    def fake_run(train_py: Path, config_path: Path) -> None:
        assert train_py == trainer_script
        config = yaml.safe_load(config_path.read_text())
        observed["config"] = config
        run_dir = Path(config["run_dir"])
        run_dir.mkdir(parents=True)
        (run_dir / "final.pt").write_bytes(b"checkpoint")

    def fake_evaluate(train_py: Path, run_dir: Path) -> dict[str, object]:
        assert train_py == trainer_script
        assert (run_dir / "final.pt").read_bytes() == b"checkpoint"
        return {
            "checkpoint": "final.pt",
            "device_type": device_type,
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
        "runtime": {"device_type": device_type, "accelerated": device_type in {"cuda", "mps"}},
        "schema_version": 1,
        "status": "complete",
    }
    assert list(trial_dir.glob(".result.json.*.tmp")) == []
    smoke = load_experiment(WRAPPER_PATH.parent / "gpu_smoke.yaml")
    gates = evaluate_gates(make_trial_context(trial_dir), smoke.phases[0].gates)
    assert len(gates) == 1
    assert gates[0].passed is (device_type != "cpu")


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize(
    "device_type,trainer_fails",
    [("cuda", False), ("cpu", False), ("cuda", True)],
    ids=["cuda-success", "cpu-success", "trainer-failure"],
)
def test_wrapper_logs_native_mlflow_checkpoint_in_both_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed: bool,
    device_type: str,
    trainer_fails: bool,
    trainer_script: Path,
) -> None:
    """The native run covers training and closes on failure without logging a score."""
    wrapper = _load_wrapper()
    config_path = tmp_path / "trainer.yaml"
    config = {"run_dir": str(tmp_path / "trainer"), "learning_rate": 0.001, "num_batches": 10}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    observed: dict[str, object] = {}

    @contextmanager
    def start_run(*, run_id: str | None) -> Iterator[SimpleNamespace]:
        observed["resumed"] = run_id
        observed["active"] = True
        try:
            yield SimpleNamespace(info=SimpleNamespace(run_id="native-run"))
        except RuntimeError:
            observed["status"] = "FAILED"
            raise
        else:
            observed["status"] = "FINISHED"
        finally:
            observed["active"] = False

    def train(train_py: Path, _config: Path) -> None:
        assert train_py == trainer_script
        assert observed["active"] is True
        assert observed["params"] == config
        if trainer_fails:
            raise RuntimeError("trainer failed")

    def log_metrics(values: dict[str, float], *, step: int) -> None:
        observed["metrics"] = values
        observed["step"] = step

    sdk = SimpleNamespace(
        start_run=start_run,
        log_params=lambda params: observed.update(params=params),
        log_metrics=log_metrics,
        set_tags=lambda tags: observed.update(tags=tags),
    )
    monkeypatch.setitem(sys.modules, "mlflow", sdk)
    monkeypatch.setattr(
        "phasesweep.tracking.bind_mlflow_run", lambda run_id: observed.update(bound=run_id)
    )
    if managed:
        monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    else:
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setattr(wrapper, "_run_template", train)
    monkeypatch.setattr(
        wrapper,
        "_evaluate_final_checkpoint",
        lambda *_args: {
            "val_loss": 0.25,
            "step": 10,
            "device_type": device_type,
            "policy": "final_checkpoint",
            "checkpoint": "final.pt",
        },
    )
    monkeypatch.setenv(
        "PHASESWEEP_OVERRIDES_SHA256", hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    # Native mode has no JSON-envelope destination supplied by the controller.
    monkeypatch.delenv("PHASESWEEP_OBJECTIVE_PATH", raising=False)
    args = ["--template-root", str(trainer_script.parent), "--config", str(config_path), "--mlflow"]
    if trainer_fails:
        with pytest.raises(RuntimeError, match="trainer failed"):
            wrapper.main(args)
        assert observed["status"] == "FAILED"
        assert "metrics" not in observed
    else:
        assert wrapper.main(args) == 0
        assert observed["status"] == "FINISHED"
        assert observed["metrics"] == {
            "val_loss": 0.25,
            "runtime/cuda": float(device_type == "cuda"),
        }
        assert observed["step"] == 10
        assert observed["tags"] == {
            "runtime.device_type": device_type,
            "evaluation.policy": "final_checkpoint",
            "evaluation.checkpoint": "final.pt",
            "evaluation.split": "validation",
        }
    assert observed["resumed"] == ("managed-run" if managed else None)
    assert observed.get("bound") == (None if managed else "native-run")
    assert observed["active"] is False


@pytest.mark.parametrize("mlflow_mode", [False, True])
def test_wrapper_rejects_a_config_from_another_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mlflow_mode: bool, trainer_script: Path
) -> None:
    """Neither reporting mode publishes an evaluation of another attempt's config."""
    wrapper = _load_wrapper()
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    config_path = trial_dir / "trainer_config.yaml"
    config_path.write_text(yaml.safe_dump({"run_dir": str(trial_dir / "trainer")}))
    scored: list[str] = []

    @contextmanager
    def start_run(*, run_id: str | None) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    sdk = SimpleNamespace(
        start_run=start_run,
        log_params=lambda _params: None,
        log_metrics=lambda *_args, **_kwargs: scored.append("metrics"),
        set_tags=lambda _tags: scored.append("tags"),
    )
    monkeypatch.setitem(sys.modules, "mlflow", sdk)
    monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    monkeypatch.setattr(wrapper, "_run_template", lambda *_args: None)
    monkeypatch.setattr(
        wrapper,
        "_evaluate_final_checkpoint",
        lambda *_args: {
            "checkpoint": "final.pt",
            "device_type": "cuda",
            "policy": "final_checkpoint",
            "step": 1000,
            "val_loss": 0.25,
        },
    )
    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", "different")
    monkeypatch.setenv("PHASESWEEP_OBJECTIVE_PATH", str(trial_dir / "result.json"))

    args = ["--template-root", str(trainer_script.parent), "--config", str(config_path)]
    with pytest.raises(ValueError, match="Trainer config does not match"):
        wrapper.main([*args, "--mlflow"] if mlflow_mode else args)

    assert scored == []
    assert not (trial_dir / "result.json").exists()


class _FakeEvaluationModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return None


def test_final_evaluator_applies_zero_seed(tmp_path, monkeypatch) -> None:
    wrapper = _load_wrapper()
    observed: list[int] = []

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
        Llama=lambda **_kwargs: _FakeEvaluationModel(),
        load_data=lambda _path: (_ for _ in ()).throw(RuntimeError("evaluation reached data")),
    )
    monkeypatch.setattr(wrapper, "_load_upstream_trainer", lambda _root: fake_trainer)

    with pytest.raises(RuntimeError, match="evaluation reached data"):
        wrapper._evaluate_final_checkpoint(tmp_path / "train.py", tmp_path)

    assert observed == [0]


def test_final_evaluator_rejects_empty_validation_budget(tmp_path, monkeypatch) -> None:
    """A zero-batch evaluation cannot publish a data-free validation objective."""
    wrapper = _load_wrapper()

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
        Llama=lambda **_kwargs: _FakeEvaluationModel(),
        load_data=lambda _path: (b"train", b"validation"),
        SequenceDataset=lambda *_args: object(),
        DataLoader=lambda *_args, **_kwargs: object(),
        cycle=lambda _loader: iter(()),
    )
    monkeypatch.setattr(wrapper, "_load_upstream_trainer", lambda _root: fake_trainer)

    with pytest.raises(ValueError, match="evaluated no validation tokens"):
        wrapper._evaluate_final_checkpoint(tmp_path / "train.py", tmp_path)
