"""Selection-time confound verdicts: the tri-state, each check, and the persisted block."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config.models import Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.confound import (
    CONFOUND_CHECKS,
    CONFOUND_VERDICTS,
    _assess_population,
    _confound_summary,
    _describe_flagged,
    _flagged_checks,
    _validate_confound_block,
)
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.evidence import _validate_objective_provenance
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.paths import _generation_winner_path
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.read import read_status
from phasesweep.engine.repeats import ReduceName, ranked_group
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    Winner,
)
from phasesweep.evidence.evaluation import EVIDENCE_PROVENANCE_SCHEMA_VERSION
from phasesweep.runtime.json import JsonObject
from tests.conftest import log_regex_provenance, make_experiment, write_constant_trainer

_COMPLETE = optuna.trial.TrialState.COMPLETE
_FAIL = optuna.trial.TrialState.FAIL
_RUNNING = optuna.trial.TrialState.RUNNING


def _envelope(
    checkpoint: str = "final.pt", step: int = 1000, *, tokens: int | None = None
) -> dict[str, Any]:
    """Return a current-format ``json_envelope`` objective provenance record.

    With ``tokens``, the extractor declared ``evaluation_axis: tokens`` and the
    trainer reported that position beside its step.
    """
    progress = (
        {"axis": "step", "value": step} if tokens is None else {"axis": "tokens", "value": tokens}
    )
    return {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
        "extractor": {"kind": "json_envelope", "config_sha256": "0" * 64},
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "source": {
            "kind": "file",
            "path": "result.json",
            "size_bytes": 0,
            "sha256": "0" * 64,
            "evaluation": {
                "objective_name": "loss",
                "split": "validation",
                "policy": "final",
                "checkpoint": checkpoint,
                "step": step,
                "progress": progress,
            },
        },
    }


@pytest.mark.parametrize(
    "progress",
    [
        None,
        {"axis": "step"},
        {"axis": "", "value": 1},
        {"axis": "step", "value": None},
        {"axis": "step", "value": -1},
        {"axis": "step", "value": True},
    ],
    ids=["missing", "no-value", "no-axis", "null", "negative", "bool"],
)
def test_envelope_provenance_requires_an_evaluation_axis_position(progress):
    record = _envelope()
    record["source"]["evaluation"]["progress"] = progress
    with pytest.raises(TrialEvidenceMissingError, match="no evaluation-axis position"):
        _validate_objective_provenance(record, subject="Trial 0")


def _wandb(value: float | None = 1000, axis: str = "eval/step") -> dict[str, Any]:
    """Return a current-format ``wandb`` objective provenance record."""
    located = {"axis": axis, "value": value}
    return {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
        "extractor": {"kind": "wandb", "config_sha256": "0" * 64},
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "source": {"kind": "wandb", "metric_key": "eval/loss", "evaluation": {"progress": located}},
        "remote_capture": {
            "kind": "wandb",
            "base_url": "https://api.wandb.ai",
            "entity": "e",
            "project": "p",
            "run_id": "attempt",
            "run_state": "finished",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "values": {"eval/loss": 0.5},
            "present_keys": [],
        },
    }


def test_wandb_provenance_may_lack_a_position_but_not_its_axis_record():
    # null: history could not say where the summary value was logged.
    _validate_objective_provenance(_wandb(None), subject="Trial 0")
    missing = _wandb()
    del missing["source"]["evaluation"]
    with pytest.raises(TrialEvidenceMissingError, match="no evaluation-axis record"):
        _validate_objective_provenance(missing, subject="Trial 0")


def _json() -> JsonObject:
    """Return a current-format ``json`` objective provenance record."""
    return {**log_regex_provenance(), "extractor": {"kind": "json", "config_sha256": "0" * 64}}


def _trials(*specs: tuple[optuna.trial.TrialState, float | None, int]) -> list[Any]:
    """Build numbered frozen trials from ``(state, value, x)`` specs."""
    study = optuna.create_study()
    for state, value, x in specs:
        study.add_trial(
            optuna.trial.create_trial(
                state=state,
                value=value,
                params={"x": x},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
            )
        )
    return study.get_trials(deepcopy=False)


def _assess(
    trials: list[Any],
    ranked: list[Any],
    *,
    provenance: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assess ``ranked`` with the lowest value winning, as minimize selection does."""
    winner = min(ranked, key=lambda trial: (trial.value, trial.number))
    records = provenance or {trial.number: log_regex_provenance() for trial in ranked}
    return _assess_population(trials, ranked, winner, provenance=records)


def test_verdicts_are_three_distinct_values_and_not_checked_is_never_ok() -> None:
    """``n/a`` means not checked; nothing may collapse it into ``ok``."""
    assert CONFOUND_VERDICTS == ("ok", "heterogeneous", "n/a")
    assert len(set(CONFOUND_VERDICTS)) == 3

    trials = _trials((_COMPLETE, 0.5, 1))
    block = _assess(trials, trials)

    assert block["checks"]["evaluation_point"]["verdict"] == "n/a"
    assert block["checks"]["tie"]["verdict"] == "n/a"
    assert _flagged_checks(block) == []
    assert _validate_confound_block(block, winner_trial=0)["checks"]["tie"]["verdict"] == "n/a"
    masquerading = copy.deepcopy(block)
    masquerading["checks"]["tie"]["detail"] = {"tied_trials": []}
    with pytest.raises(ValueError, match="n/a without a reason"):
        _validate_confound_block(masquerading, winner_trial=0)


def test_fixed_token_sweep_compares_tokens_not_steps_or_checkpoints() -> None:
    """Batch sizes differ, so steps and step-named checkpoints do; tokens match."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2), (_COMPLETE, 0.3, 3))
    records = {
        number: _envelope(f"checkpoint-{steps}", steps, tokens=2_000_000)
        for number, steps in enumerate([4000, 2000, 1000])
    }
    block = _assess(trials, trials, provenance=records)

    assert block["checks"]["evaluation_point"] == {
        "verdict": "ok",
        "detail": {"axis": "tokens", "values": [{"value": 2_000_000, "trials": [0, 1, 2]}]},
    }


def test_fixed_token_budget_reached_within_a_batch_is_one_point() -> None:
    """Each batch size stops within a batch of the budget; that is no confound."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2), (_COMPLETE, 0.3, 3))
    tokens = [1_990_656, 1_996_800, 1_992_704]
    records = {number: _envelope(tokens=count) for number, count in enumerate(tokens)}
    block = _assess(trials, trials, provenance=records)

    check = block["checks"]["evaluation_point"]
    assert check["verdict"] == "ok"
    assert [group["value"] for group in check["detail"]["values"]] == sorted(tokens)
    assert _validate_confound_block(block, winner_trial=2) == block


@pytest.mark.parametrize(
    ("records", "axis", "groups"),
    [
        pytest.param(
            {0: _envelope(step=1000), 1: _envelope(step=1000), 2: _envelope(step=400)},
            "step",
            [{"value": 400, "trials": [2]}, {"value": 1000, "trials": [0, 1]}],
            id="envelope-step",
        ),
        pytest.param(
            {0: _envelope(tokens=2000), 1: _envelope(tokens=2000), 2: _envelope(tokens=800)},
            "tokens",
            [{"value": 800, "trials": [2]}, {"value": 2000, "trials": [0, 1]}],
            id="truncated-token-budget",
        ),
        pytest.param(
            {0: _wandb(1000), 1: _wandb(1000), 2: _wandb(400)},
            "eval/step",
            [{"value": 400, "trials": [2]}, {"value": 1000, "trials": [0, 1]}],
            id="wandb-step-metric",
        ),
    ],
)
def test_evaluation_point_disagreement_is_heterogeneous(
    records: dict[int, dict[str, Any]], axis: str, groups: list[dict[str, Any]]
) -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2), (_COMPLETE, 0.3, 3))
    block = _assess(trials, trials, provenance=records)

    check = block["checks"]["evaluation_point"]
    assert check == {"verdict": "heterogeneous", "detail": {"axis": axis, "values": groups}}
    assert _flagged_checks(block) == ["evaluation_point"]
    values = ", ".join(str(group["value"]) for group in groups)
    assert f"2 distinct {axis} values ({values})" in _describe_flagged(block)[0]


_BINDING_TEXT = (
    "json_envelope (report_objective), wandb (evaluation_axis) or mlflow (selected step) "
    "would bind one"
)


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        pytest.param(
            None,
            f"log_regex objectives report no evaluation position; {_BINDING_TEXT}",
            id="log_regex",
        ),
        pytest.param(
            {number: _json() for number in range(3)},
            f"json objectives report no evaluation position; {_BINDING_TEXT}",
            id="json",
        ),
        pytest.param(
            {0: _wandb(1000), 1: _wandb(None), 2: _wandb(None)},
            "trial(s) 1, 2 recorded no eval/step position",
            id="wandb-unlocated",
        ),
    ],
)
def test_evaluation_point_is_not_checked_without_every_position(
    records: dict[int, dict[str, Any]] | None, reason: str
) -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2), (_COMPLETE, 0.3, 3))
    block = _assess(trials, trials, provenance=records)

    check = block["checks"]["evaluation_point"]
    assert check["verdict"] == "n/a"
    assert reason in check["reason"]


def test_evaluation_point_heterogeneous_line_mentions_training_length() -> None:
    """A flagged position is expected when the search space changes training length."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
    block = _assess(trials, trials, provenance={0: _envelope(step=400), 1: _envelope(step=1000)})

    (line,) = _describe_flagged(block)

    assert "2 distinct step values (400, 1000)" in line
    assert line.endswith(
        "Expected when this phase's search space changes training length "
        "(for example epochs or max_steps); otherwise the trials were not trained to one budget"
    )


@pytest.mark.parametrize(
    ("ranked", "infeasible", "failed", "verdict"),
    [
        pytest.param(1, 0, 1, "heterogeneous", id="last-standing"),
        pytest.param(2, 0, 2, "heterogeneous", id="half-failed"),
        pytest.param(3, 0, 2, "ok", id="majority-ranked"),
        # Constraints define what may win; excluding by them is no bias.
        pytest.param(2, 5, 1, "ok", id="minority-feasible"),
        pytest.param(1, 0, 0, "ok", id="single-trial"),
    ],
)
def test_survivorship_flags_failures_at_least_matching_the_ranked(
    ranked: int, infeasible: int, failed: int, verdict: str
) -> None:
    specs = (
        [(_COMPLETE, 0.5 + index, index) for index in range(ranked + infeasible)]
        + [(_FAIL, None, 50 + index) for index in range(failed)]
        + [(_RUNNING, None, 99)]
    )
    trials = _trials(*specs)
    block = _assess(trials, trials[:ranked])

    assert block["checks"]["survivorship"] == {
        "verdict": verdict,
        "detail": {
            "ranked": ranked,
            "infeasible": infeasible,
            "incomplete_group": 0,
            "failed": failed,
        },
    }
    assert block["ranked_trials"] == list(range(ranked))
    assert block["ranked_groups"] is None


def test_tie_between_distinct_params_is_heterogeneous() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_COMPLETE, 0.9, 3))
    block = _assess(trials, trials)

    assert block["checks"]["tie"] == {
        "verdict": "heterogeneous",
        "detail": {"tied_trials": [1], "within_noise": None},
    }
    assert "trial(s) 1 matched" in _describe_flagged(block)[0]


def test_tie_between_repeats_of_one_config_is_ok() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 1))
    block = _assess(trials, trials)

    assert block["checks"]["tie"] == {
        "verdict": "ok",
        "detail": {"tied_trials": [], "within_noise": None},
    }


def _seeded_trials(
    *specs: tuple[optuna.trial.TrialState, float | None, int, int],
) -> list[optuna.trial.FrozenTrial]:
    """Build numbered frozen trials from ``(state, value, x, seed)`` specs."""
    study = optuna.create_study()
    for state, value, x, seed in specs:
        study.add_trial(
            optuna.trial.create_trial(
                state=state,
                value=value,
                params={"x": x, "seed": seed},
                distributions={
                    "x": optuna.distributions.IntDistribution(0, 100),
                    "seed": optuna.distributions.CategoricalDistribution([0, 1, 2]),
                },
            )
        )
    return study.get_trials(deepcopy=False)


def _assess_groups(
    trials: list[optuna.trial.FrozenTrial],
    groups: list[list[int]],
    *,
    incomplete: tuple[int, ...] = (),
    reduce: ReduceName = "mean",
) -> dict[str, object]:
    """Assess complete repeat groups by ``reduce``, the lowest aggregate winning as minimize does."""
    by_number = {trial.number: trial for trial in trials}
    ranked_groups = [
        ranked_group(
            [(number, float(by_number[number].value or 0.0)) for number in members],
            {"x": by_number[members[0]].params["x"]},
            reduce,
        )
        for members in groups
    ]
    best = min(ranked_groups, key=lambda group: (group.aggregate, group.numbers[0]))
    ranked = [by_number[number] for members in groups for number in members]
    return _assess_population(
        trials,
        ranked,
        by_number[best.representative],
        provenance={trial.number: log_regex_provenance() for trial in ranked},
        groups=ranked_groups,
        incomplete_group=incomplete,
    )


def test_repeats_tie_flags_groups_within_noise() -> None:
    """A group whose mean sits within two standard errors of the winner's is flagged."""
    trials = _seeded_trials(
        (_COMPLETE, 0.50, 1, 0),
        (_COMPLETE, 0.60, 1, 1),
        (_COMPLETE, 0.52, 2, 0),
        (_COMPLETE, 0.62, 2, 1),
        (_COMPLETE, 0.90, 3, 0),
        (_COMPLETE, 0.91, 3, 1),
    )
    block = _assess_groups(trials, [[0, 1], [2, 3], [4, 5]])

    assert block["ranked_groups"] == [[0, 1], [2, 3], [4, 5]]
    assert block["checks"]["tie"] == {
        "verdict": "heterogeneous",
        "detail": {"tied_trials": [], "within_noise": [3]},
    }
    assert _validate_confound_block(block, winner_trial=1) == block
    (line,) = _describe_flagged(block)
    assert "repeat group(s) ending at trial(s) 3 are within two standard errors" in line
    assert _confound_summary(block, winner_trial=1)["tie"] == {
        "verdict": "heterogeneous",
        "tied_trials": 0,
        "within_noise": 1,
    }


def test_repeats_exact_group_tie_is_within_noise_and_clear_groups_are_ok() -> None:
    trials = _seeded_trials(
        (_COMPLETE, 0.5, 1, 0),
        (_COMPLETE, 0.5, 1, 1),
        (_COMPLETE, 0.5, 2, 0),
        (_COMPLETE, 0.5, 2, 1),
    )
    tied = _assess_groups(trials, [[0, 1], [2, 3]])
    assert tied["checks"]["tie"]["detail"] == {"tied_trials": [3], "within_noise": [3]}
    assert "lowest draw position won" in _describe_flagged(tied)[0]

    clear = _seeded_trials(
        (_COMPLETE, 0.50, 1, 0),
        (_COMPLETE, 0.51, 1, 1),
        (_COMPLETE, 0.90, 2, 0),
        (_COMPLETE, 0.91, 2, 1),
    )
    block = _assess_groups(clear, [[0, 1], [2, 3]])
    assert block["checks"]["tie"] == {
        "verdict": "ok",
        "detail": {"tied_trials": [], "within_noise": []},
    }


def test_repeats_median_reducer_ties_equal_medians_and_still_flags_by_standard_error() -> None:
    """Equal medians are an exact tie under the median reducer, and the SD still sets within_noise."""
    trials = _seeded_trials(
        (_COMPLETE, 1.0, 1, 0),
        (_COMPLETE, 1.0, 1, 1),
        (_COMPLETE, 1.0, 1, 2),
        (_COMPLETE, 1.0, 2, 0),
        (_COMPLETE, 1.0, 2, 1),
        (_COMPLETE, 3.0, 2, 2),
    )
    groups = [[0, 1, 2], [3, 4, 5]]

    median = _assess_groups(trials, groups, reduce="median")
    assert median["checks"]["tie"]["detail"] == {"tied_trials": [5], "within_noise": [5]}
    assert _validate_confound_block(median, winner_trial=2) == median
    mean = _assess_groups(trials, groups)
    assert mean["checks"]["tie"]["detail"] == {"tied_trials": [], "within_noise": [5]}


def test_repeats_single_group_leaves_tie_unchecked() -> None:
    """One group is one unit: there is nothing to tie with, though its members compare."""
    trials = _seeded_trials((_COMPLETE, 0.5, 1, 0), (_COMPLETE, 0.6, 1, 1))
    block = _assess_groups(trials, [[0, 1]])

    assert block["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "only one repeat group was ranked",
    }
    assert _validate_confound_block(block, winner_trial=1) == block
    assert _confound_summary(block, winner_trial=1)["tie"] == {
        "verdict": "n/a",
        "tied_trials": None,
        "within_noise": None,
    }


def test_survivorship_counts_feasible_trials_in_incomplete_groups() -> None:
    """A feasible trial whose group never completed is not an infeasible one."""
    trials = _seeded_trials(
        (_COMPLETE, 0.5, 1, 0),
        (_COMPLETE, 0.6, 1, 1),
        (_COMPLETE, 0.4, 2, 0),
        (_FAIL, None, 2, 1),
        (_COMPLETE, 0.9, 3, 0),
        (_COMPLETE, 9.9, 3, 1),
    )
    # Trial 5 broke a bound, so trial 4 is a feasible member of an incomplete group.
    block = _assess_groups(trials, [[0, 1]], incomplete=(2, 4))

    assert block["checks"]["survivorship"] == {
        "verdict": "ok",
        "detail": {"ranked": 2, "infeasible": 1, "incomplete_group": 2, "failed": 1},
    }
    assert _validate_confound_block(block, winner_trial=1) == block


def test_survivorship_discounts_a_failure_only_when_its_retry_chain_completed() -> None:
    """A failed attempt is replaced when a retry of it, or of a retry of it, completed."""
    study = optuna.create_study()
    # (state, x, rerun_of): cell 1 fails, its retry fails, and the next retry completes; cell 2
    # fails and its retry fails; cell 3 fails and nothing retries it.
    for state, x, rerun_of in [
        (_FAIL, 1, None),
        (_FAIL, 2, None),
        (_FAIL, 3, None),
        (_FAIL, 1, 0),
        (_FAIL, 2, 1),
        (_COMPLETE, 1, 3),
    ]:
        study.add_trial(
            optuna.trial.create_trial(
                state=state,
                value=None if state == _FAIL else 0.5,
                params={"x": x},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
                user_attrs={} if rerun_of is None else {RERUN_OF_ATTR: rerun_of},
            )
        )
    trials = study.get_trials(deepcopy=False)
    block = _assess(trials, trials[5:])

    assert block["checks"]["survivorship"] == {
        "verdict": "heterogeneous",
        "detail": {"ranked": 1, "infeasible": 0, "incomplete_group": 0, "failed": 3},
    }
    assert _validate_confound_block(block, winner_trial=5) == block


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        pytest.param(("ranked_groups",), [[0, 1, 2], [3, 4]], "one size", id="uneven-groups"),
        pytest.param(("ranked_groups",), [[4, 5], [0, 1], [2, 3]], "ordered", id="group-order"),
        pytest.param(("ranked_groups",), [[0, 1], [2, 3]], "cover each ranked", id="lost-group"),
        pytest.param(("ranked_groups",), [[0], [1]], "at least two", id="singleton-groups"),
        pytest.param(("ranked_groups",), [], "neither null", id="empty-groups"),
        pytest.param(
            ("checks", "tie"),
            {"verdict": "ok", "detail": {"tied_trials": [], "within_noise": None}},
            "no within_noise",
            id="noise-null",
        ),
        pytest.param(
            ("checks", "tie", "detail", "within_noise"), [2], "last member", id="noise-first"
        ),
        pytest.param(
            ("checks", "tie", "detail", "within_noise"), [1], "tying with itself", id="noise-self"
        ),
        pytest.param(
            ("checks", "tie"),
            {"verdict": "heterogeneous", "detail": {"tied_trials": [3], "within_noise": []}},
            "within_noise omits",
            id="tie-outside-noise",
        ),
    ],
)
def test_validator_rejects_malformed_repeat_groups(
    path: tuple[str, ...], value: object, error: str
) -> None:
    trials = _seeded_trials(
        (_COMPLETE, 0.50, 1, 0),
        (_COMPLETE, 0.60, 1, 1),
        (_COMPLETE, 0.52, 2, 0),
        (_COMPLETE, 0.62, 2, 1),
        (_COMPLETE, 0.90, 3, 0),
        (_COMPLETE, 0.91, 3, 1),
    )
    block = _assess_groups(trials, [[0, 1], [2, 3], [4, 5]])
    assert _validate_confound_block(block, winner_trial=1) == block
    with pytest.raises(ValueError, match=error):
        _validate_confound_block(_mutated(path, value, block), winner_trial=1)


def test_validator_requires_the_winner_to_name_its_group() -> None:
    """The highest-numbered member names a group; a lower member is not a winner."""
    trials = _seeded_trials((_COMPLETE, 0.5, 1, 0), (_COMPLETE, 0.6, 1, 1))
    block = _assess_groups(trials, [[0, 1]])
    with pytest.raises(ValueError, match="no group whose last member is the winner, trial 0"):
        _validate_confound_block(block, winner_trial=0)


def test_validator_refuses_group_evidence_without_repeat_groups() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
    noisy = _mutated(
        ("checks", "tie"),
        {"verdict": "heterogeneous", "detail": {"tied_trials": [], "within_noise": [0]}},
        _assess(trials, trials),
    )
    with pytest.raises(ValueError, match="within_noise groups without repeat groups"):
        _validate_confound_block(noisy, winner_trial=1)
    counted = _mutated(
        ("checks", "survivorship", "detail", "incomplete_group"), 1, _assess(trials, trials)
    )
    with pytest.raises(ValueError, match="incomplete_group trials without repeats"):
        _validate_confound_block(counted, winner_trial=1)


def test_assessed_block_survives_yaml_and_validation_unchanged() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_FAIL, None, 3))
    block = _assess(trials, trials[:2], provenance={0: _envelope(), 1: _envelope(step=5)})

    assert _validate_confound_block(yaml.safe_load(yaml.safe_dump(block)), winner_trial=0) == block


def _mutated(
    path: tuple[Any, ...], value: Any, block: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a valid block with the value at ``path`` replaced, or deleted for ``...``."""
    if block is None:
        trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
        block = _assess(trials, trials)
    target = block
    for key in path[:-1]:
        target = target[key]
    if value is ...:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return block


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("schema_version",), 1, id="schema-version"),
        pytest.param(("ranked_groups",), ..., id="missing-ranked-groups"),
        pytest.param(("checks", "tie", "detail", "within_noise"), ..., id="tie-without-noise"),
        pytest.param(("schema_version",), True, id="schema-version-bool"),
        pytest.param(("ranked_trials",), [1, 0], id="ranked-unsorted"),
        pytest.param(("ranked_trials",), [], id="ranked-empty"),
        pytest.param(("checks", "tie"), ..., id="missing-check"),
        pytest.param(("checks", "speed"), {"verdict": "ok", "detail": {}}, id="unknown-check"),
        pytest.param(("checks", "tie", "verdict"), "clean", id="unknown-verdict"),
        pytest.param(("checks", "evaluation_point", "reason"), ..., id="n/a-without-reason"),
        pytest.param(
            ("checks", "survivorship"),
            {"verdict": "n/a", "reason": "skipped"},
            id="survivorship-n/a",
        ),
        pytest.param(("checks", "survivorship", "detail"), ..., id="checked-without-detail"),
        pytest.param(("checks", "survivorship", "detail", "failed"), -1, id="negative-count"),
        pytest.param(("checks", "survivorship", "detail", "failed"), True, id="bool-count"),
        pytest.param(("checks", "tie", "detail", "tied_trials"), ["1"], id="string-trial"),
        pytest.param(
            ("checks", "evaluation_point"),
            {"verdict": "ok", "detail": {"step": [{"value": 1000, "trials": [0, 1]}]}},
            id="axis-less-evidence",
        ),
        pytest.param(
            ("checks", "evaluation_point"),
            {"verdict": "ok", "detail": {"axis": "", "values": [{"value": 1, "trials": [0]}]}},
            id="empty-axis",
        ),
        pytest.param(
            ("checks", "evaluation_point"),
            {"verdict": "ok", "detail": {"axis": "step", "values": [{"value": -1, "trials": [0]}]}},
            id="negative-position",
        ),
    ],
)
def test_validator_rejects_malformed_blocks(path: tuple[Any, ...], value: Any) -> None:
    with pytest.raises(ValueError, match="confound"):
        # Trial 1, at 0.4, won the default block.
        _validate_confound_block(_mutated(path, value), winner_trial=1)


def _all_checked() -> dict[str, Any]:
    """Assess two ranked trials that tie on distinct params beside one failure."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_FAIL, None, 3))
    return _assess(trials, trials[:2], provenance={0: _envelope(), 1: _envelope()})


def _evaluation(verdict: str, *groups: tuple[int, list[int]]) -> dict[str, Any]:
    values = [{"value": value, "trials": trials} for value, trials in groups]
    return {"verdict": verdict, "detail": {"axis": "step", "values": values}}


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        pytest.param(
            ("checks", "evaluation_point", "verdict"), "heterogeneous", "contradicts", id="eval"
        ),
        pytest.param(
            ("checks", "survivorship", "verdict"), "heterogeneous", "contradicts", id="survivorship"
        ),
        pytest.param(("checks", "tie", "verdict"), "ok", "contradicts", id="tie"),
        pytest.param(
            ("checks", "evaluation_point"),
            _evaluation("ok", (1000, [0])),
            "cover each ranked trial once",
            id="eval-missing-trial",
        ),
        pytest.param(
            ("checks", "evaluation_point"),
            _evaluation("heterogeneous", (400, [0, 1]), (1000, [1])),
            "cover each ranked trial once",
            id="eval-trial-twice",
        ),
        pytest.param(
            ("checks", "evaluation_point"),
            _evaluation("heterogeneous", (1000, [0]), (400, [1])),
            "ascending and distinct",
            id="eval-descending",
        ),
        pytest.param(("checks", "survivorship", "detail", "ranked"), 3, "ranked count", id="count"),
        pytest.param(
            ("checks", "tie", "detail", "tied_trials"), [7], "not ranked", id="tie-unranked"
        ),
        pytest.param(
            ("checks", "tie"), {"verdict": "n/a", "reason": "skipped"}, "several", id="tie-n/a"
        ),
        pytest.param(("ranked_trials",), [0], "only one was ranked", id="single-ranked"),
    ],
)
def test_validator_rejects_verdicts_that_contradict_their_evidence(
    path: tuple[Any, ...], value: Any, error: str
) -> None:
    """An unpublished winner.yaml has no manifest hash, so its block must hold together."""
    block = _all_checked()
    assert _validate_confound_block(block, winner_trial=0) == block
    assert [block["checks"][name]["verdict"] for name in CONFOUND_CHECKS] == [
        "ok",
        "ok",
        "heterogeneous",
    ]
    with pytest.raises(ValueError, match=error):
        _validate_confound_block(_mutated(path, value, _all_checked()), winner_trial=0)


def test_validator_anchors_the_block_to_its_own_winner() -> None:
    """A block that does not describe the winner beside it cannot report its position."""
    block = _all_checked()  # trial 0 won, tying with trial 1
    with pytest.raises(ValueError, match="does not include the winner, trial 5"):
        _validate_confound_block(block, winner_trial=5)
    with pytest.raises(ValueError, match="tying with itself"):
        _validate_confound_block(block, winner_trial=1)


def test_summary_keeps_only_verdicts_and_numbers() -> None:
    """The agent-visible projection drops every reported or configured string."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_FAIL, None, 3))
    block = _assess(
        trials,
        trials[:2],
        provenance={
            0: _envelope(checkpoint="/abs/ckpt.pt", tokens=2000),
            1: _envelope(tokens=1000),
        },
    )

    summary = _confound_summary(block, winner_trial=0)

    assert summary == {
        "flagged": ["evaluation_point", "tie"],
        "evaluation_point": {
            "verdict": "heterogeneous",
            "distinct_values": 2,
            "winner_value": 2000,
        },
        "survivorship": {
            "verdict": "ok",
            "ranked": 2,
            "infeasible": 0,
            "incomplete_group": 0,
            "failed": 1,
        },
        "tie": {"verdict": "heterogeneous", "tied_trials": 1, "within_noise": None},
    }
    rendered = json.dumps(summary)
    assert "/abs/ckpt.pt" not in rendered
    assert "tokens" not in rendered


def test_summary_reports_unchecked_counts_as_null() -> None:
    trials = _trials((_COMPLETE, 0.5, 1))
    summary = _confound_summary(_assess(trials, trials), winner_trial=0)

    assert summary["evaluation_point"] == {
        "verdict": "n/a",
        "distinct_values": None,
        "winner_value": None,
    }
    assert summary["tie"] == {"verdict": "n/a", "tied_trials": None, "within_noise": None}


def test_validator_rejects_an_absent_block() -> None:
    with pytest.raises(ValueError, match="confound block is missing"):
        _validate_confound_block(None, winner_trial=0)


def test_confound_block_never_moves_a_downstream_fingerprint() -> None:
    """An observation must never invalidate a study or block a top-up."""
    experiment = make_experiment(
        phases=[
            Phase(name="parent", n_trials=1),
            Phase(name="child", n_trials=1, inherits=["parent"]),
        ]
    )
    child = experiment.phases[1]
    trials = _trials((_COMPLETE, 0.5, 1), (_FAIL, None, 2))
    flagged = _assess(trials, trials[:1])
    clean = _assess(trials[:1], trials[:1])
    assert _flagged_checks(flagged) != _flagged_checks(clean)

    fingerprints = {
        _phase_fingerprint(
            experiment,
            child,
            {
                "parent": Winner(
                    trial_number=0,
                    params={"x": 1},
                    effective_overrides={"x": 1},
                    metric=0.5,
                    confound=block,
                )
            },
        )
        for block in (flagged, clean, None)
    }
    assert len(fingerprints) == 1


def _candidate_attrs(number: int, provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the user attrs a selection candidate carries."""
    return {
        FEASIBLE_ATTR: True,
        GENERATION_ID_ATTR: "generation-test",
        ATTEMPT_ID_ATTR: f"attempt-{number}",
        OBJECTIVE_PROVENANCE_ATTR: json.dumps(provenance),
        TRAINER_ENV_DIGEST_ATTR: "a" * 64,
        TRAINER_INPUT_ATTR: {"schema_version": 1},
    }


def _seeded_phase(*, n_trials: int) -> Phase:
    """Return phase ``p`` sampling the ``x`` the hand-built trials record."""
    return Phase(
        name="p",
        n_trials=n_trials,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )


def test_selection_records_the_block_and_logs_each_flagged_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    experiment = make_experiment(phases=[_seeded_phase(n_trials=3)])
    study = optuna.create_study(direction="minimize")
    for number, (value, step) in enumerate([(0.2, 1000), (0.3, 400)]):
        study.add_trial(
            optuna.trial.create_trial(
                state=_COMPLETE,
                value=value,
                params={"x": number},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                user_attrs=_candidate_attrs(number, _envelope(step=step)),
            )
        )
    study.add_trial(optuna.trial.create_trial(state=_FAIL))
    study.add_trial(optuna.trial.create_trial(state=_FAIL))

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        selected = select_winner(study, experiment, phase_name="p")

    assert selected.trial_number == 0
    assert _flagged_checks(selected.confound) == ["evaluation_point", "survivorship"]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.selection" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith("[p] potential confounds in the 2 ranked trials")
    assert "2 distinct step values (400, 1000)" in warnings[0]
    assert "2 failed trials against 2 ranked" in warnings[0]


@pytest.mark.parametrize(
    "record",
    [_json(), log_regex_provenance(), _wandb(None), _envelope()],
    ids=["json", "log_regex", "wandb-unlocated", "envelope"],
)
@pytest.mark.parametrize("count", [1, 2])
def test_selection_warns_when_multiple_ranked_trials_have_no_evaluation_comparison(
    caplog: pytest.LogCaptureFixture, record: JsonObject, count: int
) -> None:
    experiment = make_experiment(phases=[_seeded_phase(n_trials=count)])
    study = optuna.create_study(direction="minimize")
    for number in range(count):
        study.add_trial(
            optuna.trial.create_trial(
                state=_COMPLETE,
                value=0.2 + number * 0.1,
                params={"x": number},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                user_attrs=_candidate_attrs(number, record),
            )
        )

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        selected = select_winner(study, experiment, phase_name="p")

    assert selected.trial_number == 0
    check = selected.confound["checks"]["evaluation_point"]
    warnings = [
        item.getMessage()
        for item in caplog.records
        if item.name == "phasesweep.engine.selection"
        and item.levelno == logging.WARNING
        and "evaluation_point" in item.getMessage()
    ]
    extractor = record["extractor"]
    assert isinstance(extractor, dict)
    if count == 2 and extractor["kind"] != "json_envelope":
        assert check["verdict"] == "n/a"
        assert len(warnings) == 1
        assert "not checked" in warnings[0]
        assert check["reason"] in warnings[0]
    else:
        assert warnings == []


def test_unreadable_loser_provenance_leaves_selection_to_the_winner() -> None:
    """Confound verdicts are advisory: a losing trial's bad record cannot fail selection."""
    experiment = make_experiment(phases=[_seeded_phase(n_trials=2)])
    study = optuna.create_study(direction="minimize")
    for number, value in enumerate([0.2, 0.9]):
        attrs = _candidate_attrs(number, _envelope())
        if number == 1:
            attrs[OBJECTIVE_PROVENANCE_ATTR] = "{not json"
        study.add_trial(
            optuna.trial.create_trial(
                state=_COMPLETE,
                value=value,
                params={"x": number},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                user_attrs=attrs,
            )
        )

    selected = select_winner(study, experiment, phase_name="p")

    assert selected.trial_number == 0
    assert selected.confound["checks"]["evaluation_point"] == {
        "verdict": "n/a",
        "reason": "trial(s) 1 have unreadable objective provenance",
    }
    _validate_confound_block(selected.confound, winner_trial=selected.trial_number)


def _published_winner(experiment: Any) -> tuple[Path, dict[str, Any]]:
    """Return the published winner path and payload for phase ``p``."""
    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "ok"
    assert pointer.generation_id is not None
    path = _generation_winner_path(experiment, pointer.generation_id, "p")
    return path, yaml.safe_load(path.read_text())


@pytest.mark.integration
def test_published_block_is_manifest_covered(tmp_path: Path) -> None:
    """Editing a verdict after publication is tampering, not a correction."""
    experiment = make_experiment(persistent=tmp_path, trainer=write_constant_trainer(tmp_path))
    run_experiment(experiment)
    path, payload = _published_winner(experiment)
    # The constant trainer ties both trials with different params.
    block = _validate_confound_block(payload["confound"], winner_trial=payload["trial_number"])
    assert _flagged_checks(block) == ["tie"]

    payload["confound"]["checks"]["tie"] = {
        "verdict": "ok",
        "detail": {"tied_trials": [], "within_noise": None},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    assert _resolve_publication_pointer(experiment).state == "failed"
    assert read_status(experiment)["publication_integrity"] == "failed"


@pytest.mark.integration
def test_top_up_reselection_publishes_a_fresh_block_for_its_larger_population(
    tmp_path: Path,
) -> None:
    """Verdicts belong to the selection, not to the trial it re-selects."""
    experiment = make_experiment(persistent=tmp_path, trainer=write_constant_trainer(tmp_path))
    run_experiment(experiment)
    _, first = _published_winner(experiment)

    phase = experiment.phases[0].model_copy(update={"n_trials": 3})
    run_experiment(experiment.model_copy(update={"phases": [phase]}))
    _, second = _published_winner(experiment)

    assert second["winner_source"] == first["winner_source"]
    assert first["confound"]["ranked_trials"] == [0, 1]
    assert second["confound"]["ranked_trials"] == [0, 1, 2]
    assert second["confound"]["checks"]["tie"]["detail"] == {
        "tied_trials": [1, 2],
        "within_noise": None,
    }


def _inheritance_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return inheritance warnings logged since the last clear."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.run" and "inherits the winner" in record.getMessage()
    ]


@pytest.mark.integration
def test_inheriting_phases_are_warned_and_from_phase_carries_the_block(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The flag reaches every phase that builds on it, whichever path resolved the parent."""
    caplog.set_level(logging.WARNING, logger="phasesweep")
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        phases=[
            Phase(
                name="parent",
                n_trials=2,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            ),
            Phase(
                name="child",
                n_trials=1,
                inherits=["parent"],
                sampler=Sampler(type="random", seed=0),
                search_space={"y": IntParam(type="int", low=0, high=10)},
            ),
        ],
    )
    first = run_experiment(experiment)
    in_process = _inheritance_warnings(caplog)
    caplog.clear()
    second = run_experiment(experiment, from_phase="child")
    resumed = _inheritance_warnings(caplog)

    # The constant trainer ties both parent trials with different params.
    assert first["parent"].confound is not None
    assert _flagged_checks(first["parent"].confound) == ["tie"]
    assert second["parent"].confound == first["parent"].confound
    assert set(CONFOUND_CHECKS) == set(second["parent"].confound["checks"])
    for warnings in (in_process, resumed):
        assert len(warnings) == 1
        assert warnings[0].startswith("[child] inherits the winner of phase 'parent'")
        assert "tie: trial(s) 1 matched" in warnings[0]
