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
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    constraint_attr,
)
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
    user_attrs = {
        FEASIBLE_ATTR: feasible,
        GENERATION_ID_ATTR: "generation-test",
        ATTEMPT_ID_ATTR: f"attempt-{len(study.trials)}",
    }
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


def test_tie_break_lower_trial_number():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 0.1, params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})
    w = select_winner(study, exp)
    # First trial wins on tie
    assert w.params == {"x": 1}


@pytest.mark.parametrize(
    ("goal", "first_delta", "expected_x"),
    [
        pytest.param("minimize", WINNER_TIE_EPS / 2, 1, id="minimize-within-epsilon"),
        pytest.param("minimize", WINNER_TIE_EPS * 2, 2, id="minimize-beyond-epsilon"),
        pytest.param("maximize", -(WINNER_TIE_EPS / 2), 1, id="maximize-within-epsilon"),
        pytest.param("maximize", -(WINNER_TIE_EPS * 2), 2, id="maximize-beyond-epsilon"),
    ],
)
def test_metric_tie_epsilon(goal: str, first_delta: float, expected_x: int) -> None:
    exp = _make_exp(goal=goal)
    study = _make_study()
    _add_trial(study, 0.1 + first_delta, params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})

    winner = select_winner(study, exp)

    assert winner.params == {"x": expected_x}


def test_near_tie_band_is_anchored_to_optimum_not_iteration_order():
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, WINNER_TIE_EPS * 1.5, params={"x": 0})
    _add_trial(study, WINNER_TIE_EPS * 0.75, params={"x": 1})
    _add_trial(study, 0.0, params={"x": 2})
    trials = list(reversed(study.get_trials(deepcopy=False)))

    w = select_winner(_TrialOrderStudy(trials), exp)  # type: ignore[arg-type]

    assert w.trial_number == 1
    assert w.params == {"x": 1}


def test_rejects_nan_constraint_values_defensively(tmp_path):
    """If a NaN somehow made it into user_attrs (legacy study), selector must reject."""
    db = tmp_path / "s.db"
    storage = f"sqlite:///{db}"
    study = optuna.create_study(study_name="t", storage=storage, direction="minimize")

    # Trial 0: clean, feasible.
    t0 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t0.set_user_attr(FEASIBLE_ATTR, True)
    t0.set_user_attr(GENERATION_ID_ATTR, "generation-test")
    t0.set_user_attr(ATTEMPT_ID_ATTR, "attempt-0")
    t0.set_user_attr(constraint_attr("size"), 100.0)
    study.tell(t0, 0.5)

    # Trial 1: legacy NaN constraint value but mistakenly marked feasible.
    t1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t1.set_user_attr(FEASIBLE_ATTR, True)
    t1.set_user_attr(GENERATION_ID_ATTR, "generation-test")
    t1.set_user_attr(ATTEMPT_ID_ATTR, "attempt-1")
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
