"""Selection-time confound verdicts: the tri-state, each check, and the persisted block."""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.confound import (
    CONFOUND_CHECKS,
    CONFOUND_VERDICTS,
    _assess_population,
    _confound_summary,
    _describe_flagged,
    _evaluation_point_remedy,
    _flagged_checks,
    _noise_remedy,
    _validate_confound_block,
)
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.evidence import _validate_objective_provenance
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.paths import _generation_winner_path
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.read import read_status
from phasesweep.engine.repeats import ReduceName, RepeatAxis, ranked_group
from phasesweep.engine.selection import (
    select_winner,
    tie_group_comparisons,
)
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
from phasesweep.evidence.models import MlflowExtractor
from phasesweep.runtime.json import JsonObject
from tests.conftest import (
    log_regex_provenance,
    make_experiment,
    single_trial_confound,
    write_constant_trainer,
)
from tests.ledger_fixtures import published_generation_id, winner_payload

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


def _trials(
    *specs: tuple[optuna.trial.TrialState, float | None, int]
    | tuple[optuna.trial.TrialState, float | None, int, int],
) -> list[optuna.trial.FrozenTrial]:
    """Build numbered frozen trials from ``(state, value, x[, seed])`` specs."""
    study = optuna.create_study()
    for state, value, x, *seed in specs:
        params = {"x": x}
        distributions: dict[str, optuna.distributions.BaseDistribution] = {
            "x": optuna.distributions.IntDistribution(0, 100)
        }
        if seed:
            params["seed"] = seed[0]
            distributions["seed"] = optuna.distributions.CategoricalDistribution([0, 1, 2])
        study.add_trial(
            optuna.trial.create_trial(
                state=state,
                value=value,
                params=params,
                distributions=distributions,
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
    short, furthest = groups[0]["value"], groups[1]["value"]
    assert _describe_flagged(block, winner_trial=2)[0] == (
        f"evaluation_point: the winner, trial 2, was measured at {axis} {short}; 1 ranked trial "
        f"fell more than 1% short of the furthest, {furthest}: trial 2 at {short}"
    )


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        pytest.param(
            None,
            "log_regex objectives record no evaluation position",
            id="log_regex",
        ),
        pytest.param(
            {number: _json() for number in range(3)},
            "json objectives record no evaluation position",
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


@pytest.mark.parametrize("select", ["min", "max", "last"])
def test_mlflow_best_value_objective_leaves_evaluation_point_unchecked(select: str) -> None:
    """The step of a best value says where it fell, not how far the trial trained."""
    experiment = make_experiment(phases=[_seeded_phase(n_trials=2)])
    experiment = experiment.model_copy(
        update={
            "metric": experiment.metric.model_copy(
                update={"extractor": MlflowExtractor(type="mlflow", metric_key="x", select=select)}
            )
        }
    )
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

    check = select_winner(study, experiment, phase_name="p").confound["checks"]["evaluation_point"]

    if select == "last":
        assert check["verdict"] == "heterogeneous"
    else:
        assert check == {
            "verdict": "n/a",
            "reason": f"mlflow select: {select} records the step where the best value fell, "
            "not how far each trial trained",
        }


def test_evaluation_point_line_names_the_winner_and_the_short_trials() -> None:
    """The line says where the winner and each short trial stand, and caps the list."""
    trials = _trials(*[(_COMPLETE, 0.5 - index / 100, index) for index in range(8)])
    steps = [1000, 400, 500, 600, 700, 800, 900, 995]
    block = _assess(
        trials, trials, provenance={n: _envelope(step=step) for n, step in enumerate(steps)}
    )

    (line,) = _describe_flagged(block, winner_trial=7)

    assert line == (
        "evaluation_point: the winner, trial 7, was measured at step 995; 6 ranked trials fell "
        "more than 1% short of the furthest, 1000: trial 1 at 400, trial 2 at 500, trial 3 at "
        "600, trial 4 at 700, trial 5 at 800 and 1 more"
    )


@pytest.mark.parametrize("kind", ["json_envelope", "wandb", "mlflow"])
def test_evaluation_point_remedy_uses_the_current_reader(kind: str) -> None:
    """Each reader's remedy is for a training-length phase and names that reader's way to report progress."""
    remedy = _evaluation_point_remedy(kind)
    assert remedy.startswith("For a phase that searches training length, ")
    assert "actual_steps / planned_steps" in remedy
    if kind == "wandb":
        assert "one wandb.log() call" in remedy
        assert "evaluation_axis" in remedy
        assert "json_envelope" not in remedy
    elif kind == "mlflow":
        assert "step=int(10000 *" in remedy
        assert "separate metric key" in remedy
        assert "metric.extractor.metric_key" in remedy
        assert "select: last" in remedy
        assert "evaluation_axis" not in remedy
    else:
        assert "report_objective" in remedy
        assert "evaluation_axis" in remedy


@pytest.mark.parametrize("kind", ["json_envelope", "wandb"])
def test_evaluation_point_remedy_keeps_an_existing_relative_axis(kind: str) -> None:
    """A fraction below 1 is a trial that stopped before its own planned steps, not a budget gap."""
    remedy = _evaluation_point_remedy(kind, axis="schedule_fraction")

    assert remedy == (
        "A schedule_fraction below 1 is a trial that stopped before its planned steps; its trial "
        "log says why."
    )


@pytest.mark.parametrize(
    ("ranked", "infeasible", "failed", "verdict"),
    [
        pytest.param(1, 0, 1, "heterogeneous", id="last-standing"),
        # One failed configuration among many still went unmeasured; no ratio hides it.
        pytest.param(9, 0, 1, "heterogeneous", id="one-of-ten"),
        # Constraints define what may win; excluding by them is no bias.
        pytest.param(2, 5, 0, "ok", id="minority-feasible"),
        pytest.param(1, 0, 0, "ok", id="single-trial"),
    ],
)
def test_survivorship_flags_any_configuration_without_a_result(
    ranked: int, infeasible: int, failed: int, verdict: str
) -> None:
    """Each configuration counts once, and a still-running one is left to the completion record."""
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
            "failed": failed,
            "untried": 0,
            "failed_trials": list(range(ranked + infeasible, ranked + infeasible + failed)),
        },
    }
    assert block["ranked_trials"] == list(range(ranked))
    assert block["ranked_groups"] is None


def test_survivorship_counts_a_configuration_once_however_often_it_ran() -> None:
    """Repeated failures of one configuration are one failed configuration, named by the latest."""
    trials = _trials(
        (_COMPLETE, 0.5, 1),
        (_FAIL, None, 2),
        (_FAIL, None, 2),
        (_COMPLETE, 0.4, 3),
        (_FAIL, None, 3),
    )
    detail = _assess(trials, [trials[0], trials[3]])["checks"]["survivorship"]["detail"]

    assert detail == {
        "ranked": 2,
        "infeasible": 0,
        "failed": 1,
        "untried": 0,
        "failed_trials": [2],
    }


def test_survivorship_warning_names_each_failed_configuration() -> None:
    """The warning lists failures by trial and params and caps the list."""
    trials = _trials(
        (_COMPLETE, 0.5, 1),
        *[(_FAIL, None, 10 + index) for index in range(6)],
    )
    block = _assess_population(
        trials,
        trials[:1],
        trials[0],
        provenance={0: log_regex_provenance()},
        untried_cells=[{"x": 99}],
    )

    (line,) = _describe_flagged(block, winner_trial=0, trials=trials, untried_cells=[{"x": 99}])

    assert line == (
        "survivorship: 7 of 8 configurations have no result (trial 1 x=10 failed, trial 2 x=11 "
        "failed, trial 3 x=12 failed, trial 4 x=13 failed, trial 5 x=14 failed and 2 more)"
    )


def test_survivorship_counts_a_grid_cell_no_trial_reached() -> None:
    """A full grid that skipped a cell flags it; with repeats the cell joins its group."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
    plain = _assess_population(
        trials,
        trials,
        trials[1],
        provenance={trial.number: log_regex_provenance() for trial in trials},
        untried_cells=[{"x": 3}],
    )
    assert plain["checks"]["survivorship"] == {
        "verdict": "heterogeneous",
        "detail": {"ranked": 2, "infeasible": 0, "failed": 0, "untried": 1, "failed_trials": []},
    }
    (line,) = _describe_flagged(plain, winner_trial=1, untried_cells=[{"x": 3}])
    assert line == "survivorship: 1 of 3 configurations has no result (x=3 was never tried)"

    members = _trials((_COMPLETE, 0.5, 1, 0), (_COMPLETE, 0.6, 1, 1), (_COMPLETE, 0.5, 2, 0))
    grouped = _assess_groups(
        members, [[0, 1]], incomplete=(2,), untried_cells=[{"x": 2, "seed": 1}]
    )
    assert grouped["checks"]["survivorship"]["detail"] == {
        "ranked": 1,
        "infeasible": 0,
        "failed": 0,
        "untried": 1,
        "failed_trials": [],
    }


def test_tie_between_distinct_params_is_heterogeneous() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_COMPLETE, 0.9, 3))
    block = _assess(trials, trials)

    assert block["checks"]["tie"] == {
        "verdict": "heterogeneous",
        "detail": {"tied_trials": [1], "within_noise": None, "pooled_sd": None, "df": None},
    }
    (line,) = _describe_flagged(block, winner_trial=0)
    assert line == "tie: trial 1 matched the winner's value exactly with different params"


def test_noise_remedy_accompanies_only_a_flagged_tie() -> None:
    """A winner clear of every other configuration carries no tie remedy; a flagged tie does."""
    clear = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.9, 2))
    tied = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2))

    clear_tie = _confound_summary(
        _assess(clear, clear), winner_trial=0, extractor_kind="json_envelope", persistent=True
    )["tie"]
    tied_tie = _confound_summary(
        _assess(tied, tied), winner_trial=0, extractor_kind="json_envelope", persistent=True
    )["tie"]

    assert (clear_tie["verdict"], clear_tie["remedy"]) == ("ok", None)
    assert (tied_tie["verdict"], tied_tie["remedy"]) == ("heterogeneous", _noise_remedy())


@pytest.mark.parametrize("values", [(0.5, 0.5), (0.5, 0.6, 0.7)])
def test_tie_between_repeats_of_one_config_is_unchecked(values: tuple[float, ...]) -> None:
    """Multiple trials of one configuration carry no configuration gap or noise advice."""
    trials = _trials(*[(_COMPLETE, value, 1) for value in values])
    block = _assess(trials, trials)

    assert block["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "only one configuration was ranked",
    }
    assert _validate_confound_block(block, winner_trial=0) == block
    assert _confound_summary(
        block, winner_trial=0, extractor_kind="json_envelope", persistent=True
    )["tie"] == {
        "verdict": "n/a",
        "remedy": None,
        "tied_trials": None,
        "within_noise": None,
        "pooled_sd": None,
        "df": None,
    }


def _assess_groups(
    trials: list[optuna.trial.FrozenTrial],
    groups: list[list[int]],
    *,
    incomplete: tuple[int, ...] = (),
    reduce: ReduceName = "mean",
    untried_cells: tuple[dict[str, object], ...] | list[dict[str, object]] = (),
) -> dict[str, object]:
    """Assess complete grid repeat groups over ``seed`` by ``reduce``, lowest aggregate winning."""
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
        axis=RepeatAxis(key="seed", values=(0, 1, 2), reduce=reduce, slot_size=None),
        groups=ranked_groups,
        incomplete_group=incomplete,
        untried_cells=untried_cells,
    )


def _noise_gaps(
    trials: list[optuna.trial.FrozenTrial], groups: list[list[int]]
) -> dict[int, tuple[float, float]]:
    """Return what selection hands the tie line: each group within noise's gap and threshold."""
    by_number = {trial.number: trial for trial in trials}
    ranked = [
        ranked_group(
            [(number, float(by_number[number].value or 0.0)) for number in members],
            {"x": by_number[members[0]].params["x"]},
            "mean",
        )
        for members in groups
    ]
    best = min(ranked, key=lambda group: (group.aggregate, group.numbers[0]))
    return {
        comparison["trial_number"]: (comparison["gap"], comparison["threshold"])
        for comparison in tie_group_comparisons(best, ranked)
    }


def test_repeats_tie_pools_the_spread_of_every_ranked_group() -> None:
    """Three groups of two seeds share one noise band, worked by hand.

    Variances 0.02, 0.02 and 0.005 pool to 0.015 on df = 3, so pooled_sd = sqrt(0.015) =
    0.122474. Student's t at 3 df is 3.182446 and the standard error of a difference of two
    two-seed means is pooled_sd * sqrt(1/2 + 1/2), so the threshold is 0.389768: the 0.1 gap to
    group 3 is within noise and the 0.55 gap to group 5 is not.
    """
    trials = _trials(
        (_COMPLETE, 1.0, 1, 0),
        (_COMPLETE, 1.2, 1, 1),
        (_COMPLETE, 1.1, 2, 0),
        (_COMPLETE, 1.3, 2, 1),
        (_COMPLETE, 1.6, 3, 0),
        (_COMPLETE, 1.7, 3, 1),
    )
    groups = [[0, 1], [2, 3], [4, 5]]
    block = _assess_groups(trials, groups)

    assert block["ranked_groups"] == groups
    tie = block["checks"]["tie"]
    assert tie["verdict"] == "heterogeneous"
    assert tie["detail"] == {
        "tied_trials": [],
        "within_noise": [3],
        "pooled_sd": pytest.approx(math.sqrt(0.015)),
        "df": 3,
    }
    gaps = _noise_gaps(trials, groups)
    assert gaps == {3: (pytest.approx(0.1), pytest.approx(3.182446 * math.sqrt(0.015)))}
    assert _validate_confound_block(block, winner_trial=1) == block
    (line,) = _describe_flagged(block, winner_trial=1, noise_gaps=gaps)
    assert line == (
        "tie: 1 group is within noise of the winner, threshold 0.389768 (sd 0.122474 pooled "
        "over 3 groups, df 3): group 3 at gap 0.1"
    )
    assert _confound_summary(
        block, winner_trial=1, extractor_kind="json_envelope", persistent=True
    )["tie"] == {
        "verdict": "heterogeneous",
        "remedy": None,
        "tied_trials": 0,
        "within_noise": 1,
        "pooled_sd": pytest.approx(math.sqrt(0.015)),
        "df": 3,
    }


def test_pooled_band_holds_a_quiet_pair_the_per_pair_rule_called_clear() -> None:
    """Two zero-spread groups 0.05 apart are within the spread a third group shows.

    Each group's own sd is 0, so comparing the pair on their own spreads gives a threshold of 0
    and calls the 0.05 gap clear. Pooled with the third group's variance of 0.08 the band is
    sqrt(0.08 / 3) = 0.163299 on df 3, a threshold of 3.182446 * 0.163299 = 0.519691.
    """
    trials = _trials(
        (_COMPLETE, 1.00, 1, 0),
        (_COMPLETE, 1.00, 1, 1),
        (_COMPLETE, 1.05, 2, 0),
        (_COMPLETE, 1.05, 2, 1),
        (_COMPLETE, 2.00, 3, 0),
        (_COMPLETE, 2.40, 3, 1),
    )
    groups = [[0, 1], [2, 3], [4, 5]]
    block = _assess_groups(trials, groups)

    assert block["checks"]["tie"]["detail"]["within_noise"] == [3]
    assert block["checks"]["tie"]["detail"]["pooled_sd"] == pytest.approx(math.sqrt(0.08 / 3))
    assert _noise_gaps(trials, groups)[3][1] == pytest.approx(0.519691, abs=1e-6)


def test_repeats_exact_group_tie_is_within_noise_and_clear_groups_are_ok() -> None:
    trials = _trials(
        (_COMPLETE, 0.5, 1, 0),
        (_COMPLETE, 0.5, 1, 1),
        (_COMPLETE, 0.5, 2, 0),
        (_COMPLETE, 0.5, 2, 1),
    )
    groups = [[0, 1], [2, 3]]
    tied = _assess_groups(trials, groups)
    assert tied["checks"]["tie"]["detail"] == {
        "tied_trials": [3],
        "within_noise": [3],
        "pooled_sd": 0.0,
        "df": 2,
    }
    (line,) = _describe_flagged(tied, winner_trial=1, noise_gaps=_noise_gaps(trials, groups))
    assert line.endswith("(sd 0 pooled over 2 groups, df 2): group 3 at gap 0")

    clear = _trials(
        (_COMPLETE, 0.50, 1, 0),
        (_COMPLETE, 0.51, 1, 1),
        (_COMPLETE, 0.90, 2, 0),
        (_COMPLETE, 0.91, 2, 1),
    )
    block = _assess_groups(clear, groups)
    assert block["checks"]["tie"] == {
        "verdict": "ok",
        "detail": {
            "tied_trials": [],
            "within_noise": [],
            "pooled_sd": pytest.approx(0.01 / math.sqrt(2)),
            "df": 2,
        },
    }


def test_repeats_median_reducer_flags_only_an_exact_tie_and_judges_no_noise() -> None:
    """A median's sampling spread is unknown at a few seeds, so median groups get no band.

    Equal medians are still an exact tie and flag the check. Without one the check is n/a,
    never ok, because seed noise was not judged. The same members ranked by mean compare
    against the pooled band.
    """
    trials = _trials(
        (_COMPLETE, 1.0, 1, 0),
        (_COMPLETE, 1.0, 1, 1),
        (_COMPLETE, 1.0, 1, 2),
        (_COMPLETE, 1.0, 2, 0),
        (_COMPLETE, 1.0, 2, 1),
        (_COMPLETE, 3.0, 2, 2),
    )
    groups = [[0, 1, 2], [3, 4, 5]]

    median = _assess_groups(trials, groups, reduce="median")
    assert median["checks"]["tie"] == {
        "verdict": "heterogeneous",
        "detail": {"tied_trials": [5], "within_noise": None, "pooled_sd": None, "df": None},
    }
    assert _validate_confound_block(median, winner_trial=2) == median
    assert _describe_flagged(median, winner_trial=2) == [
        "tie: group 5 matched the winner's median exactly"
    ]
    assert _confound_summary(
        median, winner_trial=2, extractor_kind="json_envelope", persistent=True
    )["tie"] == {
        "verdict": "heterogeneous",
        "remedy": None,
        "tied_trials": 1,
        "within_noise": None,
        "pooled_sd": None,
        "df": None,
    }
    # The tied group is listed for show-winners and MCP at gap 0, without a band to judge it by.
    winner = ranked_group([(0, 1.0), (1, 1.0), (2, 1.0)], {"x": 1}, "median")
    tied = ranked_group([(3, 1.0), (4, 1.0), (5, 3.0)], {"x": 2}, "median")
    assert tie_group_comparisons(winner, [winner, tied]) == [
        {"trial_number": 5, "params": {"x": 2}, "gap": 0.0, "combined_se": None, "threshold": None}
    ]
    apart = _trials(
        (_COMPLETE, 1.0, 1, 0),
        (_COMPLETE, 1.0, 1, 1),
        (_COMPLETE, 1.0, 1, 2),
        (_COMPLETE, 1.0, 2, 0),
        (_COMPLETE, 2.0, 2, 1),
        (_COMPLETE, 3.0, 2, 2),
    )
    apart_tied = ranked_group([(3, 1.0), (4, 2.0), (5, 3.0)], {"x": 2}, "median")
    assert tie_group_comparisons(winner, [winner, apart_tied]) == []
    assert _assess_groups(apart, groups, reduce="median")["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "reduce: median has no noise band, and no group matched the winner's median "
        "exactly",
    }
    mean = _assess_groups(trials, groups)
    assert mean["checks"]["tie"]["detail"]["within_noise"] == [5]
    assert mean["checks"]["tie"]["detail"]["df"] == 4


def test_repeats_single_group_leaves_tie_unchecked() -> None:
    """One group is one unit: there is nothing to tie with, though its members compare."""
    trials = _trials((_COMPLETE, 0.5, 1, 0), (_COMPLETE, 0.6, 1, 1))
    block = _assess_groups(trials, [[0, 1]])

    assert block["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "only one repeat group was ranked",
    }
    assert _validate_confound_block(block, winner_trial=1) == block
    assert _confound_summary(
        block, winner_trial=1, extractor_kind="json_envelope", persistent=True
    )["tie"] == {
        "verdict": "n/a",
        "remedy": None,
        "tied_trials": None,
        "within_noise": None,
        "pooled_sd": None,
        "df": None,
    }


def test_survivorship_counts_repeat_groups_as_configurations() -> None:
    """A group with a failed member failed, and one with an out-of-bounds member is infeasible."""
    trials = _trials(
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
        "verdict": "heterogeneous",
        "detail": {"ranked": 1, "infeasible": 1, "failed": 1, "untried": 0, "failed_trials": [3]},
    }
    assert _validate_confound_block(block, winner_trial=1) == block


def test_survivorship_leaves_out_the_failed_attempts_of_a_fixed_configuration_once_measured() -> (
    None
):
    """A fixed configuration samples no params, and its retry carries no link to the failure."""
    study = optuna.create_study()
    for state in (_FAIL, _FAIL, _COMPLETE):
        study.add_trial(
            optuna.trial.create_trial(state=state, value=None if state == _FAIL else 0.5)
        )
    trials = study.get_trials(deepcopy=False)

    assert _assess(trials, trials[2:])["checks"]["survivorship"] == {
        "verdict": "ok",
        "detail": {"ranked": 1, "infeasible": 0, "failed": 0, "untried": 0, "failed_trials": []},
    }


def test_survivorship_discounts_a_failure_only_when_its_retry_chain_completed() -> None:
    """A retried cell counts once: by its measured result, else by its latest failed attempt."""
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
        "detail": {
            "ranked": 1,
            "infeasible": 0,
            "failed": 2,
            "untried": 0,
            "failed_trials": [2, 4],
        },
    }
    assert _validate_confound_block(block, winner_trial=5) == block


@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param([[0], [1]], "fewer than two", id="singleton-groups"),
        pytest.param([], "neither null", id="empty-groups"),
        pytest.param([[1, 0], [2, 3]], "ascending", id="unsorted-members"),
    ],
)
def test_validator_rejects_malformed_repeat_groups(value: object, error: str) -> None:
    """The repeats validation and the group readers index ``ranked_groups`` as member lists."""
    trials = _trials(
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
        _validate_confound_block(_mutated(("ranked_groups",), value, block), winner_trial=1)


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
        pytest.param(("checks", "survivorship", "detail", "untried"), ..., id="missing-untried"),
        pytest.param(
            ("checks", "survivorship", "detail", "incomplete_group"), 0, id="old-count-field"
        ),
        pytest.param(
            ("checks", "survivorship", "detail", "failed_trials"), [3, 1], id="unsorted-failures"
        ),
        pytest.param(("checks", "tie", "detail", "tied_trials"), ["1"], id="string-trial"),
        pytest.param(("checks", "tie", "detail", "pooled_sd"), ..., id="tie-without-pooled-sd"),
        pytest.param(("checks", "tie", "detail", "pooled_sd"), -0.1, id="negative-pooled-sd"),
        pytest.param(("checks", "tie", "detail", "pooled_sd"), True, id="bool-pooled-sd"),
        pytest.param(("checks", "tie", "detail", "df"), 0, id="zero-df"),
        pytest.param(("checks", "tie", "detail", "df"), 2.0, id="float-df"),
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
    ("path", "value"),
    [
        pytest.param(("checks", "evaluation_point", "verdict"), "heterogeneous", id="eval"),
        pytest.param(("checks", "survivorship", "verdict"), "ok", id="survivorship"),
        pytest.param(("checks", "tie", "verdict"), "ok", id="tie"),
        pytest.param(
            ("checks", "evaluation_point"), _evaluation("ok", (1000, [0])), id="eval-one-trial"
        ),
        pytest.param(("checks", "survivorship", "detail", "ranked"), 3, id="count"),
        pytest.param(("checks", "tie", "detail", "tied_trials"), [7], id="tie-unranked"),
        pytest.param(("checks", "tie"), {"verdict": "n/a", "reason": "skipped"}, id="tie-n/a"),
        pytest.param(("ranked_trials",), [0], id="single-ranked"),
    ],
)
def test_validator_reads_recorded_verdicts_without_rejudging_them(
    path: tuple[str, ...], value: object
) -> None:
    """A verdict another rule would reach, or counts that disagree with the ranked trials, still read.

    The winner file's manifest hash catches tampering, so a reader that re-judged the block
    would only turn a changed rule into an unreadable publication.
    """
    block = _mutated(path, value, _all_checked())
    assert _validate_confound_block(block, winner_trial=0) == block


def test_validator_anchors_the_block_to_its_own_winner() -> None:
    """A block that does not rank the winner beside it cannot report its position."""
    block = _all_checked()  # trial 0 won, tying with trial 1
    with pytest.raises(ValueError, match="does not include the winner, trial 5"):
        _validate_confound_block(block, winner_trial=5)


def test_summary_exposes_the_evaluation_axis_without_checkpoint_paths() -> None:
    """The agent-visible projection retains the evaluation axis and excludes checkpoint paths."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_FAIL, None, 3))
    block = _assess(
        trials,
        trials[:2],
        provenance={
            0: _envelope(checkpoint="/abs/ckpt.pt", tokens=2000),
            1: _envelope(tokens=1000),
        },
    )

    summary = _confound_summary(
        block, winner_trial=0, extractor_kind="json_envelope", persistent=True
    )

    assert summary == {
        "flagged": ["evaluation_point", "survivorship", "tie"],
        "evaluation_point": {
            "verdict": "heterogeneous",
            "axis": "tokens",
            "remedy": _evaluation_point_remedy("json_envelope", axis="tokens"),
            "distinct_values": 2,
            "winner_value": 2000,
        },
        "survivorship": {
            "verdict": "heterogeneous",
            "remedy": (
                "Fix the failure, then raise the phase's n_trials and rerun: a grid or fixed "
                "configuration retries its failed cells, and another sampler draws new "
                "configurations."
            ),
            "ranked": 2,
            "infeasible": 0,
            "failed": 1,
            "untried": 0,
            "failed_trials": [2],
        },
        "tie": {
            "verdict": "heterogeneous",
            "remedy": _noise_remedy(),
            "tied_trials": 1,
            "within_noise": None,
            "pooled_sd": None,
            "df": None,
        },
    }
    rendered = json.dumps(summary)
    assert "/abs/ckpt.pt" not in rendered
    assert summary["evaluation_point"]["axis"] == "tokens"
    # The tie remedy asks for coverage, never for seeds.
    assert _noise_remedy().endswith("compare more distinct values around them.")
    assert "repeats" not in _noise_remedy()


@pytest.mark.parametrize(
    ("failed", "untried", "persistent", "remedy"),
    [
        (0, 0, True, None),
        (0, 2, True, "Raise the phase's n_trials and rerun to run its untried cells."),
        (
            1,
            0,
            False,
            "Fix the failure and rerun; storage: null keeps no completed trials between runs.",
        ),
    ],
)
def test_survivorship_remedy_follows_the_configurations_left_and_the_storage(
    failed: int, untried: int, persistent: bool, remedy: str | None
) -> None:
    """A clean check has no remedy; untried cells need only more trials; nothing persists without storage."""
    block = single_trial_confound(0)
    block["checks"]["survivorship"] = {
        "verdict": "heterogeneous" if failed or untried else "ok",
        "detail": {
            "ranked": 1,
            "infeasible": 0,
            "failed": failed,
            "untried": untried,
            "failed_trials": [3] if failed else [],
        },
    }

    summary = _confound_summary(
        block, winner_trial=0, extractor_kind="json_envelope", persistent=persistent
    )

    survivorship = summary["survivorship"]
    assert isinstance(survivorship, dict)
    assert survivorship["remedy"] == remedy


def test_summary_reports_unchecked_counts_as_null() -> None:
    trials = _trials((_COMPLETE, 0.5, 1))
    summary = _confound_summary(
        _assess(trials, trials), winner_trial=0, extractor_kind="json_envelope", persistent=True
    )

    assert summary["evaluation_point"] == {
        "verdict": "n/a",
        "axis": None,
        "remedy": None,
        "distinct_values": None,
        "winner_value": None,
    }
    assert summary["tie"] == {
        "verdict": "n/a",
        "remedy": None,
        "tied_trials": None,
        "within_noise": None,
        "pooled_sd": None,
        "df": None,
    }


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
    for x in (5, 6):
        study.add_trial(
            optuna.trial.create_trial(
                state=_FAIL,
                params={"x": x},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
            )
        )

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        selected = select_winner(study, experiment, phase_name="p")

    assert selected.trial_number == 0
    assert selected.confound["checks"]["survivorship"]["detail"]["failed_trials"] == [2, 3]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.selection" and record.levelno == logging.WARNING
    ]
    # The failed trials at x=5 and x=6 count as tried, so the winner's x=0 is the lowest of four.
    assert warnings[1:] == [
        "[p] no trial ran below the winner's x=0 (tried 0..6, at the configured lower bound)"
    ]
    # One warning per phase: each flagged check's sentence, stating the fact alone.
    assert warnings[0] == (
        "[p] evaluation_point: the winner, trial 0, was measured at step 1000; 1 ranked trial "
        "fell more than 1% short of the furthest, 1000: trial 1 at 400; survivorship: 2 of 4 "
        "configurations have no result (trial 2 x=5 failed, trial 3 x=6 failed)"
    )


@pytest.mark.parametrize(
    "record",
    [_json(), log_regex_provenance(), _wandb(None), _envelope()],
    ids=["json", "log_regex", "wandb-unlocated", "envelope"],
)
@pytest.mark.parametrize("count", [1, 2])
def test_selection_retains_unchecked_verdict_without_repeating_reader_disclosure(
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


def _published_winner(experiment: Experiment) -> tuple[Path, JsonObject]:
    """Return the published winner path and payload for phase ``p``."""
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    path = _generation_winner_path(experiment, generation_id, "p")
    return path, winner_payload(experiment, generation_id)


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
        "detail": {"tied_trials": [], "within_noise": None, "pooled_sd": None, "df": None},
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
        "pooled_sd": None,
        "df": None,
    }


def _confound_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the selection confound warnings logged since the last clear."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.selection"
        and any(f"{check}: " in record.getMessage() for check in CONFOUND_CHECKS)
    ]


@pytest.mark.integration
def test_inheriting_phases_keep_verdicts_without_repeating_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Inherited verdicts remain intact in both launch paths without repeated warnings."""
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
    in_process = _confound_warnings(caplog)
    caplog.clear()
    second = run_experiment(experiment, from_phase="child")
    resumed = _confound_warnings(caplog)

    # The constant trainer ties both parent trials with different params.
    assert first["parent"].confound is not None
    assert _flagged_checks(first["parent"].confound) == ["tie"]
    assert second["parent"].confound == first["parent"].confound
    assert set(CONFOUND_CHECKS) == set(second["parent"].confound["checks"])
    # Only the parent's own selection warns; the child inherits its winner silently, and the
    # resume that skips the parent replays its verdicts without warning again.
    assert len(in_process) == 1
    assert in_process[0].startswith("[parent] tie: trial 1 matched the winner's value exactly")
    assert resumed == []
