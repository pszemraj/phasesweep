"""Executable guidance for comparing final positions in training-length sweeps."""

from __future__ import annotations

import re
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep import report_objective
from phasesweep.engine.confound import _assess_population, _confound_summary
from phasesweep.evidence.evaluation import run_extractor
from phasesweep.evidence.models import JsonEnvelopeExtractor
from tests.conftest import make_trial_context


def _documented_recipe() -> tuple[JsonEnvelopeExtractor, str]:
    """Read the actual configured extractor and trainer example from the guide."""
    guide = (Path(__file__).resolve().parents[1] / "docs" / "config.md").read_text()
    configurations = re.findall(r"```yaml\n(.*?)```", guide, flags=re.DOTALL)
    config_example = next(
        (block for block in configurations if "evaluation_axis: schedule_fraction" in block), None
    )
    assert config_example is not None, (
        "Document a copyable schedule_fraction extractor configuration"
    )
    extractor = JsonEnvelopeExtractor.model_validate(
        yaml.safe_load(config_example)["metric"]["extractor"]
    )
    snippets = re.findall(r"```python\n(.*?)```", guide, flags=re.DOTALL)
    trainer_example = next(
        (
            block
            for block in snippets
            if "schedule_fraction" in block and "report_objective(" in block
        ),
        None,
    )
    assert trainer_example is not None, (
        "Document report_objective with the actual schedule fraction"
    )
    return extractor, trainer_example


@pytest.mark.parametrize(
    "axis, second_actual_steps, expected_verdict",
    [
        ("step", 200, "heterogeneous"),
        ("schedule_fraction", 200, "ok"),
        ("schedule_fraction", 160, "heterogeneous"),
    ],
    ids=["raw-final-steps-stay-flagged", "both-planned-finishes", "early-stop-stays-flagged"],
)
def test_training_length_recipe_preserves_evaluation_and_mcp_verdicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    second_actual_steps: int,
    expected_verdict: str,
) -> None:
    """Relative completion compares planned finishes while retaining early-stop evidence."""
    if axis == "schedule_fraction":
        extractor, example = _documented_recipe()
    else:
        extractor = JsonEnvelopeExtractor(
            type="json_envelope",
            path="result.json",
            objective_name="eval_loss",
            split="validation",
            policy="final",
        )
        example = None
    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", "a" * 64)
    study = optuna.create_study()
    provenance: dict = {}
    for number, (actual_steps, planned_steps) in enumerate(
        [(100, 100), (second_actual_steps, 200)]
    ):
        trial_dir = tmp_path / str(number)
        monkeypatch.setenv("PHASESWEEP_OBJECTIVE_PATH", str(trial_dir / "result.json"))
        value = 0.5 - 0.1 * number
        if example is None:
            report_objective(
                value,
                name="eval_loss",
                split="validation",
                policy="final",
                checkpoint=f"checkpoint-{actual_steps}",
                step=actual_steps,
            )
        else:
            exec(
                compile(example, "docs/config.md schedule_fraction example", "exec"),
                {
                    "eval_loss": value,
                    "checkpoint_path": f"checkpoint-{actual_steps}",
                    "actual_steps": actual_steps,
                    "planned_steps": planned_steps,
                },
            )
        record: dict = {}
        measured = run_extractor(make_trial_context(trial_dir), extractor, provenance=record)
        assert measured == value
        provenance[number] = record
        study.add_trial(
            optuna.trial.create_trial(
                value=measured,
                params={"max_steps": planned_steps},
                distributions={
                    "max_steps": optuna.distributions.CategoricalDistribution([100, 200])
                },
            )
        )
    trials = study.get_trials(deepcopy=False)
    block = _assess_population(trials, trials, trials[1], provenance=provenance)

    assert block["checks"]["evaluation_point"]["verdict"] == expected_verdict
    summary = _confound_summary(block, winner_trial=1, extractor_kind="json_envelope")
    assert summary["evaluation_point"]["verdict"] == expected_verdict
    assert summary["flagged"] == (
        ["evaluation_point"] if expected_verdict == "heterogeneous" else []
    )
    if axis == "schedule_fraction" and expected_verdict == "heterogeneous":
        assert (
            "already measures actual_steps / planned_steps" in summary["evaluation_point"]["remedy"]
        )
        assert "set metric.extractor.evaluation_axis" not in summary["evaluation_point"]["remedy"]
