from __future__ import annotations

import optuna
import pytest

from phasesweep.config import (
    Constraint,
    Experiment,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
)
from phasesweep.engine.selection import WINNER_TIE_EPS, NoFeasibleTrialError, select_winner
from phasesweep.engine.state import FEASIBLE_ATTR, constraint_attr
from tests.conftest import make_experiment


def _make_exp(constraints=None, *, goal: str = "minimize"):
    exp = make_experiment(
        storage=":memory:",
        trial_command="echo",
        constraints=constraints or [],
        phases=[Phase(name="p", n_trials=1, search_space={})],
    )
    return exp.model_copy(update={"metric": exp.metric.model_copy(update={"goal": goal})})


def _make_study():
    return optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))


def _add_trial(study, value, *, feasible=True, constraint_vals=None, params=None):
    distributions: dict = {}
    pvals: dict = {}
    for k, v in (params or {}).items():
        distributions[k] = optuna.distributions.IntDistribution(low=int(v), high=int(v))
        pvals[k] = int(v)
    user_attrs = {FEASIBLE_ATTR: feasible}
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


def test_tie_break_lower_trial_number():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.1, params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})
    w = select_winner(study, exp)
    # First trial wins on tie
    assert w.params == {"x": 1}


def test_near_tie_within_eps_prefers_lower_trial_number_minimize():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.1 + (WINNER_TIE_EPS / 2), params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})

    w = select_winner(study, exp)

    assert w.params == {"x": 1}


def test_metric_difference_beyond_eps_wins_minimize():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.1 + (WINNER_TIE_EPS * 2), params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})

    w = select_winner(study, exp)

    assert w.params == {"x": 2}


def test_near_tie_within_eps_prefers_lower_trial_number_maximize():
    exp = _make_exp(goal="maximize")
    study = _make_study()
    _add_trial(study, 0.1 - (WINNER_TIE_EPS / 2), params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})

    w = select_winner(study, exp)

    assert w.params == {"x": 1}


def test_metric_difference_beyond_eps_wins_maximize():
    exp = _make_exp(goal="maximize")
    study = _make_study()
    _add_trial(study, 0.1 - (WINNER_TIE_EPS * 2), params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})

    w = select_winner(study, exp)

    assert w.params == {"x": 2}


def test_rejects_nan_constraint_values_defensively(tmp_path):
    """If a NaN somehow made it into user_attrs (legacy study), selector must reject."""
    db = tmp_path / "s.db"
    storage = f"sqlite:///{db}"
    study = optuna.create_study(study_name="t", storage=storage, direction="minimize")

    # Trial 0: clean, feasible.
    t0 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t0.set_user_attr(FEASIBLE_ATTR, True)
    t0.set_user_attr(constraint_attr("size"), 100.0)
    study.tell(t0, 0.5)

    # Trial 1: legacy NaN constraint value but mistakenly marked feasible.
    t1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t1.set_user_attr(FEASIBLE_ATTR, True)
    t1.set_user_attr(constraint_attr("size"), float("nan"))
    study.tell(t1, 0.1)  # Better metric, but invalid.

    exp = Experiment(
        experiment="t",
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        constraints=[
            Constraint(
                name="size",
                extractor=JsonExtractor(type="json", path="r.json", key="s"),
                max=1000.0,
            )
        ],
        phases=[
            Phase(name="a", n_trials=1, search_space={"x": IntParam(type="int", low=0, high=10)}),
        ],
    )
    sel = select_winner(study, exp)
    assert sel.trial_number == 0, (
        "NaN-constraint trial 1 must be rejected even though metric was lower"
    )
