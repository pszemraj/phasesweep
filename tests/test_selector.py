from __future__ import annotations

import json

import optuna
import pytest

from phasesweep.config.models import Constraint, Phase
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.repeats import RepeatAxis
from phasesweep.engine.selection import NoFeasibleTrialError, select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    constraint_attr,
)
from phasesweep.evidence.evaluation import EVIDENCE_PROVENANCE_SCHEMA_VERSION
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import make_experiment

# Allocation writes both attrs on every trial before its trainer runs.
_ALLOCATION_ATTRS = {TRAINER_ENV_DIGEST_ATTR: "a" * 64, TRAINER_INPUT_ATTR: {"schema_version": 1}}


def _objective_provenance_json() -> str:
    """Return a structurally valid current-format local objective record."""
    return json.dumps(
        {
            "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
            "extractor": {"kind": "log_regex", "config_sha256": "0" * 64},
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "source": {
                "kind": "file",
                "path": "stdout.log",
                "size_bytes": 0,
                "sha256": "0" * 64,
            },
        }
    )


def _make_exp(constraints=None, *, goal: str = "minimize"):
    exp = make_experiment(
        storage=None,
        trial_command="echo",
        constraints=constraints or [],
        phases=[Phase(name="p", n_trials=1, search_space={})],
    )
    return exp.model_copy(update={"metric": exp.metric.model_copy(update={"goal": goal})})


def _make_study():
    return optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))


def _add_trial(
    study,
    value,
    *,
    feasible=True,
    constraint_vals=None,
    params=None,
    extra_user_attrs=None,
):
    distributions: dict = {}
    pvals: dict = {}
    for k, v in (params or {}).items():
        distributions[k] = optuna.distributions.IntDistribution(low=int(v), high=int(v))
        pvals[k] = int(v)
    user_attrs = {
        FEASIBLE_ATTR: feasible,
        GENERATION_ID_ATTR: "generation-test",
        ATTEMPT_ID_ATTR: f"attempt-{len(study.trials)}",
        OBJECTIVE_PROVENANCE_ATTR: _objective_provenance_json(),
        **_ALLOCATION_ATTRS,
    }
    user_attrs.update(extra_user_attrs or {})
    for cn, cv in (constraint_vals or {}).items():
        user_attrs[constraint_attr(cn)] = cv
    trial = optuna.trial.create_trial(
        params=pvals,
        distributions=distributions,
        value=value,
        user_attrs=user_attrs,
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(trial)


class _TrialOrderStudy:
    """Small test double exposing trials in a deliberate non-Optuna order."""

    study_name = "t::p"

    def __init__(self, trials):
        self._trials = trials

    def get_trials(self, *, deepcopy: bool):
        return list(self._trials)


def test_argmin_over_feasible():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.5, params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})
    _add_trial(study, 0.3, params={"x": 3})

    w = select_winner(study, exp)
    assert w.metric == pytest.approx(0.1)
    assert w.params == {"x": 2}


def test_excludes_infeasible():
    exp = _make_exp(
        constraints=[
            Constraint(
                name="bytes",
                max=100,
                extractor=JsonExtractor(type="json", path="r.json", key="bytes"),
            )
        ]
    )
    study = _make_study()
    # Best metric but violates constraint
    _add_trial(study, 0.05, feasible=False, constraint_vals={"bytes": 200}, params={"x": 1})
    _add_trial(study, 0.20, feasible=True, constraint_vals={"bytes": 50}, params={"x": 2})
    _add_trial(study, 0.30, feasible=True, constraint_vals={"bytes": 75}, params={"x": 3})

    w = select_winner(study, exp)
    assert w.metric == pytest.approx(0.20)
    assert w.params == {"x": 2}


def test_no_feasible_raises():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.1, feasible=False, params={"x": 1})
    with pytest.raises(NoFeasibleTrialError):
        select_winner(study, exp)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        (OBJECTIVE_PROVENANCE_ATTR, "{not-json"),
        (OBJECTIVE_PROVENANCE_ATTR, json.dumps([])),
        (GATES_ATTR, "{not-json"),
        (GATES_ATTR, json.dumps([{"type": "required_file"}])),
    ],
)
def test_corrupt_winner_evidence_attrs_fail_closed(attr: str, value: str) -> None:
    """Present-but-corrupt evidence is not treated as absent legacy state."""
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.1, extra_user_attrs={attr: value})

    with pytest.raises(TrialEvidenceMissingError, match="(corrupt|malformed)"):
        select_winner(study, exp)


@pytest.mark.parametrize(
    ("goal", "first_metric", "second_metric", "expected_x"),
    [
        pytest.param("minimize", 0.1 + 1e-15, 0.1, 2, id="minimize-worse-by-one-ulp-scale"),
        pytest.param("maximize", 0.1 - 1e-15, 0.1, 2, id="maximize-worse-by-one-ulp-scale"),
        pytest.param("minimize", 3e-13, 1e-13, 2, id="minimize-tiny-scale"),
        pytest.param("maximize", 3e-13, 1e-13, 1, id="maximize-tiny-scale"),
    ],
)
def test_metric_ordering_preserves_representable_differences(
    goal: str, first_metric: float, second_metric: float, expected_x: int
) -> None:
    """Any representable metric difference decides the winner, never the trial number.

    1e-13 and 3e-13 are 2e-13 apart, which an absolute 1e-12 tie epsilon would
    swallow — the lower draw position would win regardless of which result was
    actually better.
    """
    exp = _make_exp(goal=goal)
    study = _make_study()
    _add_trial(study, first_metric, params={"x": 1})
    _add_trial(study, second_metric, params={"x": 2})

    winner = select_winner(study, exp)

    assert winner.params == {"x": expected_x}
    assert winner.trial_number == expected_x - 1


def test_exact_tie_is_anchored_to_optimum_not_iteration_order():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 1.0, params={"x": 0})
    _add_trial(study, 0.0, params={"x": 1})
    _add_trial(study, 0.0, params={"x": 2})
    trials = list(reversed(study.get_trials(deepcopy=False)))

    w = select_winner(_TrialOrderStudy(trials), exp)  # type: ignore[arg-type]

    assert w.trial_number == 1
    assert w.params == {"x": 1}


def _add_interrupted(study, params: dict[str, int]) -> None:  # noqa: ANN001
    """Add a FAIL trial that recorded its params before an interruption stopped it."""
    study.add_trial(
        optuna.trial.create_trial(
            params=params,
            distributions={
                name: optuna.distributions.IntDistribution(low=value, high=value)
                for name, value in params.items()
            },
            user_attrs={
                TRIAL_OUTCOME_ATTR: {
                    "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
                    "sequence": len(study.trials) + 1,
                    "outcome": "interrupted",
                }
            },
            state=optuna.trial.TrialState.FAIL,
        )
    )


def test_exact_tie_goes_to_the_lowest_draw_position_not_the_lowest_trial_number() -> None:
    """A rerun keeps its source's draw position, so it wins a tie its source would have won."""
    study = _make_study()
    _add_trial(study, 1.0, params={"x": 0})
    _add_interrupted(study, {"x": 1})
    _add_trial(study, 0.0, params={"x": 2})
    _add_trial(study, 0.0, params={"x": 1}, extra_user_attrs={RERUN_OF_ATTR: 1})

    winner = select_winner(study, _make_exp())

    assert winner.trial_number == 3
    assert winner.params == {"x": 1}
    assert winner.confound["checks"]["tie"]["detail"]["tied_trials"] == [2]


def test_rejects_nan_constraint_values_defensively():
    """If a NaN somehow reaches user_attrs, the selector must reject it."""
    study = optuna.create_study(study_name="t", direction="minimize")

    # Trial 0: clean, feasible.
    t0 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t0.set_user_attr(FEASIBLE_ATTR, True)
    t0.set_user_attr(GENERATION_ID_ATTR, "generation-test")
    t0.set_user_attr(ATTEMPT_ID_ATTR, "attempt-0")
    t0.set_user_attr(OBJECTIVE_PROVENANCE_ATTR, _objective_provenance_json())
    for key, value in _ALLOCATION_ATTRS.items():
        t0.set_user_attr(key, value)
    t0.set_user_attr(constraint_attr("size"), 100.0)
    study.tell(t0, 0.5)

    # Trial 1: NaN constraint value but mistakenly marked feasible.
    t1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t1.set_user_attr(FEASIBLE_ATTR, True)
    t1.set_user_attr(GENERATION_ID_ATTR, "generation-test")
    t1.set_user_attr(ATTEMPT_ID_ATTR, "attempt-1")
    t1.set_user_attr(OBJECTIVE_PROVENANCE_ATTR, _objective_provenance_json())
    for key, value in _ALLOCATION_ATTRS.items():
        t1.set_user_attr(key, value)
    t1.set_user_attr(constraint_attr("size"), float("nan"))
    study.tell(t1, 0.1)  # Better metric, but invalid.

    exp = _make_exp(
        [
            Constraint(
                name="size",
                extractor=JsonExtractor(type="json", path="r.json", key="s"),
                max=1000.0,
            )
        ]
    )
    sel = select_winner(study, exp)
    assert sel.trial_number == 0, (
        "NaN-constraint trial 1 must be rejected even though metric was lower"
    )


def _with_bounds(exp, **bounds: float):
    """Return ``exp`` with the given metric ``min``/``max`` viability bounds."""
    return exp.model_copy(update={"metric": exp.metric.model_copy(update=bounds)})


def test_metric_bounds_are_rechecked_at_selection() -> None:
    """A stored value outside today's bounds cannot win, whatever its feasible attr says."""
    exp = _with_bounds(_make_exp(), min=0.2, max=2.0)
    study = _make_study()
    # Every trial carries feasible=True, as if the bounds were set or tightened later.
    _add_trial(study, 0.1, params={"x": 1})
    _add_trial(study, 3.0, params={"x": 2})
    _add_trial(study, 0.5, params={"x": 3})
    _add_trial(study, 2.0, params={"x": 4})

    winner = select_winner(study, exp)

    assert (winner.trial_number, winner.metric) == (2, pytest.approx(0.5))
    survivorship = winner.confound["checks"]["survivorship"]["detail"]
    assert survivorship["ranked"] == 2
    assert survivorship["infeasible"] == 2


def test_metric_bounds_no_feasible_message_names_bound_and_best_value() -> None:
    """When the bounds leave nothing, the error names them and the best value seen."""
    study = _make_study()
    _add_trial(study, 5.0, params={"x": 1})
    _add_trial(study, 3.5, params={"x": 2}, feasible=False)
    _add_trial(study, 7.0, params={"x": 3})
    study.add_trial(
        optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL, user_attrs={"x": 1})
    )

    with pytest.raises(NoFeasibleTrialError) as excinfo:
        select_winner(study, _with_bounds(_make_exp(), max=2.0))
    assert str(excinfo.value) == (
        "No feasible completed trials in phase. The phase's trials.csv records each "
        "failed trial's reason; trainer output is in its trial_* directory. "
        "Metric bounds: metric.max=2.0. The best value among COMPLETE trials with a finite "
        "value is 3.5 (trial 1)."
    )

    with pytest.raises(NoFeasibleTrialError, match=r"metric\.min=6\.0.*is 7\.0 \(trial 2\)"):
        select_winner(study, _with_bounds(_make_exp(goal="maximize"), min=6.0, max=6.5))

    unbounded_study = _make_study()
    _add_trial(unbounded_study, 0.1, feasible=False, params={"x": 1})
    with pytest.raises(NoFeasibleTrialError) as unbounded:
        select_winner(unbounded_study, _make_exp())
    assert "Metric bounds" not in str(unbounded.value)


def test_metric_bounds_note_without_a_completed_value() -> None:
    """Bounds are still named when no trial recorded a finite value to compare."""
    study = _make_study()
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))

    with pytest.raises(NoFeasibleTrialError, match=r"metric\.min=1\.0, metric\.max=2\.0.*finite"):
        select_winner(study, _with_bounds(_make_exp(), min=1.0, max=2.0))


def _axis(reduce: str = "mean", *, slot_size: int | None = None) -> RepeatAxis:
    """Return a ``seed`` repeat axis over three values, slotted for seeded random."""
    assert reduce in ("mean", "median")
    return RepeatAxis(key="seed", values=(0, 1, 2), reduce=reduce, slot_size=slot_size)


def _add_group(study, x: int, values: list[float], **kwargs) -> None:  # noqa: ANN001
    """Add one configuration's trials, one per seed in order."""
    for seed, value in enumerate(values):
        _add_trial(study, value, params={"x": x, "seed": seed}, **kwargs)


def _add_failed(study, x: int, seed: int) -> None:  # noqa: ANN001
    """Add a FAIL trial that recorded its params before failing."""
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": x, "seed": seed},
            distributions={
                "x": optuna.distributions.IntDistribution(low=x, high=x),
                "seed": optuna.distributions.IntDistribution(low=seed, high=seed),
            },
            state=optuna.trial.TrialState.FAIL,
        )
    )


@pytest.mark.parametrize(("reduce", "winner_x"), [("mean", 2), ("median", 1)])
def test_repeats_rank_complete_groups_by_reducer(reduce: str, winner_x: int) -> None:
    study = _make_study()
    _add_group(study, 1, [0.1, 0.2, 0.9])  # mean 0.4, median 0.2
    _add_group(study, 2, [0.3, 0.3, 0.3])  # mean 0.3, median 0.3

    winner = select_winner(study, _make_exp(), repeats=_axis(reduce))

    assert winner.params == {"x": winner_x}
    assert winner.metric == pytest.approx(0.2 if reduce == "median" else 0.3)
    assert winner.trial_number == (2 if winner_x == 1 else 5)
    assert winner.repeats is not None
    assert [member["trial_number"] for member in winner.repeats["members"]] == (
        [0, 1, 2] if winner_x == 1 else [3, 4, 5]
    )
    assert winner.confound["ranked_groups"] == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize("member", ["failed", "infeasible", "missing"])
def test_repeats_group_with_failed_or_infeasible_member_cannot_rank(member: str) -> None:
    """A configuration whose best members look unbeatable still loses without all of them."""
    study = _make_study()
    _add_trial(study, 0.01, params={"x": 1, "seed": 0})
    _add_trial(study, 0.01, params={"x": 1, "seed": 1})
    if member == "failed":
        _add_failed(study, 1, 2)
    elif member == "infeasible":
        _add_trial(study, 0.01, feasible=False, params={"x": 1, "seed": 2})
    _add_group(study, 2, [0.5, 0.5, 0.5])

    winner = select_winner(study, _make_exp(), repeats=_axis())

    assert winner.params == {"x": 2}
    assert winner.metric == pytest.approx(0.5)
    survivorship = winner.confound["checks"]["survivorship"]["detail"]
    assert survivorship["ranked"] == 3
    assert survivorship["incomplete_group"] == 2
    assert winner.confound["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "only one repeat group was ranked",
    }


def test_repeats_collapsed_member_outside_metric_bounds_excludes_its_configuration() -> None:
    """One collapsed seed cannot drag a configuration's mean into first place."""
    study = _make_study()
    _add_group(study, 1, [0.0, 0.3, 0.3])  # a leaked or collapsed seed scores exactly 0
    _add_group(study, 2, [0.25, 0.25, 0.25])

    unbounded = select_winner(study, _make_exp(), repeats=_axis())
    bounded = select_winner(study, _with_bounds(_make_exp(), min=0.01), repeats=_axis())

    assert unbounded.params == {"x": 1}
    assert bounded.params == {"x": 2}
    survivorship = bounded.confound["checks"]["survivorship"]["detail"]
    assert (survivorship["infeasible"], survivorship["incomplete_group"]) == (1, 2)


def test_repeats_without_complete_group_names_incomplete_groups() -> None:
    study = _make_study()
    _add_trial(study, 0.1, params={"x": 1, "seed": 0})
    _add_trial(study, 0.1, params={"x": 1, "seed": 1})
    _add_trial(study, 0.0, params={"x": 2, "seed": 0})
    _add_trial(study, 0.2, params={"x": 2, "seed": 1})
    _add_trial(study, 0.2, params={"x": 2, "seed": 2})

    with pytest.raises(NoFeasibleTrialError) as excinfo:
        select_winner(study, _with_bounds(_make_exp(), min=0.01), repeats=_axis())

    assert str(excinfo.value).startswith(
        "No complete repeat group: a configuration ranks only when all 3 trials (one per "
        "repeats.values of 'seed') are COMPLETE, feasible and within bounds; 2 group(s) were "
        "incomplete. The phase's trials.csv labels each trial's repeat_group and records "
        "each failed trial's reason. Metric bounds: metric.min=0.01."
    )


def test_repeats_tie_goes_to_lowest_group_and_highest_member_names_it() -> None:
    """Exact aggregate ties go to the group whose first member ran first."""
    study = _make_study()
    for x, seed, value in [(1, 0, 0.2), (2, 0, 0.3), (1, 1, 0.4), (2, 1, 0.3), (2, 2, 0.3)]:
        _add_trial(study, value, params={"x": x, "seed": seed})
    _add_trial(study, 0.3, params={"x": 1, "seed": 2})

    winner = select_winner(study, _make_exp(), repeats=_axis())

    assert winner.params == {"x": 1}
    assert winner.trial_number == 5
    assert winner.attempt_id == "attempt-5"
    assert winner.confound["ranked_groups"] == [[0, 2, 5], [1, 3, 4]]
    tie = winner.confound["checks"]["tie"]
    assert tie["verdict"] == "heterogeneous"
    assert tie["detail"] == {"tied_trials": [4], "within_noise": [4]}


def test_repeats_tie_goes_to_the_group_with_the_lowest_member_draw_position() -> None:
    """A group completed by a rerun of the first trial wins a tie against a group numbered earlier."""
    study = _make_study()
    _add_interrupted(study, {"x": 2, "seed": 0})
    _add_group(study, 1, [0.3, 0.3, 0.3])
    _add_trial(study, 0.3, params={"x": 2, "seed": 1})
    _add_trial(study, 0.3, params={"x": 2, "seed": 2})
    _add_trial(study, 0.3, params={"x": 2, "seed": 0}, extra_user_attrs={RERUN_OF_ATTR: 0})

    winner = select_winner(study, _make_exp(), repeats=_axis())

    assert winner.params == {"x": 2}
    assert winner.trial_number == 6
    assert winner.confound["checks"]["tie"]["detail"]["tied_trials"] == [3]


def test_repeats_within_noise_flags_a_close_runner_up_and_clears_a_distant_one() -> None:
    study = _make_study()
    _add_group(study, 1, [0.50, 0.60, 0.55])
    _add_group(study, 2, [0.52, 0.62, 0.57])
    _add_group(study, 3, [0.90, 0.91, 0.92])

    winner = select_winner(study, _make_exp(), repeats=_axis())

    assert winner.params == {"x": 1}
    assert winner.confound["checks"]["tie"] == {
        "verdict": "heterogeneous",
        "detail": {"tied_trials": [], "within_noise": [5]},
    }


def test_seeded_random_duplicate_configs_rank_as_separate_slots() -> None:
    """Two slots that drew the same params are two groups, never one oversized group."""
    study = _make_study()
    _add_group(study, 1, [0.4, 0.4, 0.4])
    _add_group(study, 1, [0.3, 0.3, 0.3])

    slotted = select_winner(study, _make_exp(), repeats=_axis(slot_size=3))

    assert slotted.trial_number == 5
    assert slotted.metric == pytest.approx(0.3)
    assert slotted.confound["ranked_groups"] == [[0, 1, 2], [3, 4, 5]]
    # The other slot repeats the winner's configuration, so it is not a tie candidate.
    assert slotted.confound["checks"]["tie"]["detail"] == {"tied_trials": [], "within_noise": []}
    with pytest.raises(NoFeasibleTrialError, match="1 group\\(s\\) were incomplete"):
        select_winner(study, _make_exp(), repeats=_axis())
