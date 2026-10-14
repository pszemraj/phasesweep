from __future__ import annotations

import json

import optuna
import pytest

from phasesweep.config import Constraint, JsonExtractor, Phase
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.selection import NoFeasibleTrialError, select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    constraint_attr,
)
from phasesweep.evidence.evaluation import EVIDENCE_PROVENANCE_SCHEMA_VERSION
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

    Regression for review v0.5.17 / finding H: 1e-13 and 3e-13 are 2e-13 apart,
    which the removed absolute 1e-12 tie epsilon swallowed — the lower trial
    number won regardless of which result was actually better.
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
