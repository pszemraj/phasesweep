from __future__ import annotations

import optuna
import pytest

from phasesweep.config import (
    Constraint,
    JsonExtractor,
    Phase,
)
from phasesweep.selector import NoFeasibleTrialError, select_winner
from tests.conftest import make_experiment


def _make_exp(constraints=None):
    return make_experiment(
        storage=":memory:",
        trial_command="echo",
        constraints=constraints or [],
        phases=[Phase(name="p", n_trials=1, search_space={})],
    )


def _make_study():
    return optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))


def _add_trial(study, value, *, feasible=True, constraint_vals=None, params=None):
    distributions: dict = {}
    pvals: dict = {}
    for k, v in (params or {}).items():
        distributions[k] = optuna.distributions.IntDistribution(low=int(v), high=int(v))
        pvals[k] = int(v)
    user_attrs = {"phasesweep_feasible": feasible}
    for cn, cv in (constraint_vals or {}).items():
        user_attrs[f"constraint:{cn}"] = cv
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
