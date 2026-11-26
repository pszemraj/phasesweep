"""The fix step when a trainer ran but its objective could not be read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import METRIC_EXTRACTOR_LABEL
from phasesweep.evidence.evaluation import ExtractorError, run_extractor
from phasesweep.evidence.models import JsonEnvelopeExtractor, JsonExtractor
from tests.conftest import (
    addressed_phase_study,
    make_experiment,
    make_trial_context,
    no_winner_remedy,
    write_trainer,
)

_UNREADABLE = "A trainer ran but its objective could not be read"
_KEY_MISSING = f"{METRIC_EXTRACTOR_LABEL}: Key 'lossz' not found in r.json (failed at 'lossz')."


def _remedy(tmp_path: Path, reasons: list[str], *, persistent: bool) -> str:
    """Return the remedy an experiment gives when every trial of phase ``p`` failed with ``reasons``."""
    experiment = make_experiment(persistent=tmp_path if persistent else None, n_trials=len(reasons))
    return no_winner_remedy(experiment, list(reasons))


def test_an_unreadable_objective_says_which_fix_continues_the_study_and_which_starts_one(
    tmp_path: Path,
) -> None:
    """A wrong `metric` is re-read; the retry step is offered only as the alternative to that."""
    remedy = _remedy(tmp_path, [_KEY_MISSING] * 2, persistent=True)

    assert remedy.index(_UNREADABLE) < remedy.index(
        "not retrained. Otherwise, fix the failure, then raise phases[p].n_trials"
    )
    assert (
        f"{_UNREADABLE}; if the `metric` settings are wrong, correct them and rerun: the "
        "finished trials are re-read from their kept evidence, not retrained."
    ) in remedy
    assert "lossz" not in remedy


def test_an_unreadable_objective_in_memory_names_the_trainer_and_the_metric_settings(
    tmp_path: Path,
) -> None:
    """With `storage: null` nothing continues, so the step names both causes and the rerun."""
    remedy = _remedy(tmp_path, [_KEY_MISSING], persistent=False)

    assert remedy.endswith(
        f"{_UNREADABLE}: fix the trainer's output or the `metric` settings, then rerun."
    )
    assert "metric bounds" not in remedy


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "persistent"])
@pytest.mark.parametrize(
    "reason",
    [
        "non-zero exit code 1",
        f"{METRIC_EXTRACTOR_LABEL} returned non-finite value: nan",
        "constraint extractor 'vram': Key 'peak' not found in r.json (failed at 'peak').",
        f"non-zero exit code 1; stderr: {METRIC_EXTRACTOR_LABEL}: Key 'x' not found",
    ],
    ids=["crash", "non-finite", "constraint-extractor", "quoted-stderr"],
)
def test_other_failures_are_not_read_as_an_unreadable_objective(
    tmp_path: Path, persistent: bool, reason: str
) -> None:
    """Only a reason that starts with the metric extractor's label counts, not a value that was read or a quoted line."""
    assert _UNREADABLE not in _remedy(tmp_path, [reason, reason], persistent=persistent)


def _write_result_trainer(directory: Path, *, keys: list[str]) -> Path:
    """Write a trainer that records 1.0 under each of ``keys`` in the file it is given."""
    return write_trainer(
        directory,
        f"""
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        with open(args.out, "w") as f:
            json.dump(dict.fromkeys({keys!r}, 1.0), f)
        """,
    )


@pytest.mark.integration
def test_an_aborted_phase_with_an_unreadable_objective_gives_steps_that_work(
    tmp_path: Path,
) -> None:
    """The failure-streak abort leads with the same sentence, and each branch it names runs."""
    trainer = _write_result_trainer(tmp_path, keys=["loss"])

    def experiment(name: str, key: str, n_trials: int) -> Experiment:
        return make_experiment(
            experiment=name,
            persistent=tmp_path,
            trainer=trainer,
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key=key)),
            phases=[
                Phase(
                    name="p",
                    n_trials=n_trials,
                    max_consecutive_failures=2,
                    sampler=Sampler(type="random", seed=0),
                    search_space={"x": IntParam(type="int", low=0, high=10)},
                )
            ],
        )

    wrong = experiment("t", "lossz", 4)
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(wrong)
    message = _operational_error_text(aborted.value)
    assert message.index(_UNREADABLE) < message.index(
        "not retrained. Otherwise, fix the cause, then raise phases[p].n_trials"
    )
    assert "correct them and rerun" in message

    def states_of(config: Experiment) -> list[str]:
        study = addressed_phase_study(config)
        return [trial.state.name for trial in study.get_trials(deepcopy=False)]

    corrected = experiment("t", "loss", 4)
    run_experiment(corrected)
    assert states_of(corrected) == ["COMPLETE"] * 4

    # The other branch, in an experiment whose only study is the aborted one.
    unfixed = experiment("u", "lossz", 4)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(unfixed)
    _write_result_trainer(tmp_path, keys=["loss", "lossz"])
    run_experiment(experiment("u", "lossz", 6))
    states = states_of(unfixed)
    assert states.count("FAIL") == 2
    assert states.count("COMPLETE") == 4


@pytest.fixture
def envelope() -> dict[str, object]:
    """An envelope bound to the context ``make_trial_context`` builds."""
    return {
        "schema_version": 1,
        "status": "complete",
        "generation_id": "generation-test",
        "attempt_id": "attempt-test",
        "overrides_sha256": "a" * 64,
        "objective": {"name": "eval_loss", "split": "validation", "value": 0.25},
        "evaluation": {"policy": "final", "checkpoint": "final.pt", "step": 10},
    }


def _envelope_error(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    objective_name: str = "eval_loss_typo",
    split: str = "validation",
    policy: str = "final",
    checkpoint: str | None = None,
    expected_step: int | None = None,
) -> str:
    """Return what the envelope extractor refuses with when it expects objective ``eval_loss_typo``."""
    (tmp_path / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    extractor = JsonEnvelopeExtractor(
        type="json_envelope",
        objective_name=objective_name,
        split=split,
        policy=policy,
        checkpoint=checkpoint,
        expected_step=expected_step,
    )
    with pytest.raises(ExtractorError) as refused:
        run_extractor(make_trial_context(tmp_path), extractor)
    return str(refused.value)


def test_an_envelope_objective_mismatch_says_what_the_envelope_reports(
    tmp_path: Path, envelope: dict[str, object]
) -> None:
    """A misspelled `objective_name` is corrected from the name the trainer wrote."""
    assert _envelope_error(tmp_path, envelope) == (
        f"JSON envelope at {tmp_path / 'result.json'} does not report objective "
        "'eval_loss_typo'; it reports 'eval_loss'."
    )


def test_an_envelope_without_an_objective_name_does_not_claim_one(
    tmp_path: Path, envelope: dict[str, object]
) -> None:
    """A missing name is not quoted back as `None`."""
    envelope["objective"] = {"split": "validation", "value": 0.25}

    assert _envelope_error(tmp_path, envelope) == (
        f"JSON envelope at {tmp_path / 'result.json'} does not report objective 'eval_loss_typo'."
    )


@pytest.mark.parametrize(
    ("expected", "field"),
    [
        pytest.param({"split": "test"}, "split 'test'; it reports 'validation'", id="split"),
        pytest.param({"policy": "best"}, "policy 'best'; it reports 'final'", id="policy"),
        pytest.param(
            {"checkpoint": "best.pt"},
            "checkpoint 'best.pt'; it reports 'final.pt'",
            id="checkpoint",
        ),
        pytest.param({"expected_step": 20}, "step 20; it reports 10", id="step"),
    ],
)
def test_an_envelope_mismatch_says_what_the_envelope_reports(
    tmp_path: Path, envelope: dict[str, object], expected: dict[str, object], field: str
) -> None:
    """Every field a config pins is corrected from the value the trainer wrote."""
    assert _envelope_error(tmp_path, envelope, objective_name="eval_loss", **expected) == (
        f"JSON envelope at {tmp_path / 'result.json'} does not report {field}."
    )


def test_an_envelope_without_a_split_does_not_claim_one(
    tmp_path: Path, envelope: dict[str, object]
) -> None:
    """A missing split is not quoted back as `None`."""
    envelope["objective"] = {"name": "eval_loss", "value": 0.25}

    assert _envelope_error(tmp_path, envelope, objective_name="eval_loss") == (
        f"JSON envelope at {tmp_path / 'result.json'} does not report split 'validation'."
    )
