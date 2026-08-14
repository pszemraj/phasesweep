from __future__ import annotations

import json
import logging

import optuna
import pytest

from phasesweep.config import (
    Constraint,
    Experiment,
    IntParam,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
)
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.selection import NoFeasibleTrialError, select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
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


def _add_trial(
    study,
    value,
    *,
    feasible=True,
    constraint_vals=None,
    params=None,
    env_digest=None,
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
    }
    if env_digest is not None:
        user_attrs[TRAINER_ENV_DIGEST_ATTR] = env_digest
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
    ("goal", "first_delta", "expected_x"),
    [
        pytest.param("minimize", 1e-15, 2, id="minimize-worse-by-one-ulp-scale"),
        pytest.param("maximize", -1e-15, 2, id="maximize-worse-by-one-ulp-scale"),
    ],
)
def test_metric_ordering_preserves_representable_differences(
    goal: str, first_delta: float, expected_x: int
) -> None:
    """Any representable metric difference decides the winner."""
    exp = _make_exp(goal=goal)
    study = _make_study()
    _add_trial(study, 0.1 + first_delta, params={"x": 1})
    _add_trial(study, 0.1, params={"x": 2})

    winner = select_winner(study, exp)

    assert winner.params == {"x": expected_x}


@pytest.mark.parametrize(
    ("goal", "expected_x"),
    [pytest.param("minimize", 2, id="minimize"), pytest.param("maximize", 1, id="maximize")],
)
def test_tiny_scale_objectives_rank_by_value_not_trial_number(goal: str, expected_x: int) -> None:
    """Objectives below the old 1e-12 epsilon must still rank by value.

    Regression for review v0.5.17 / finding H: 1e-13 and 3e-13 are 2e-13 apart,
    which the removed absolute tie epsilon swallowed — the lower trial number
    won regardless of which result was actually better.
    """
    exp = _make_exp(goal=goal)
    study = _make_study()
    _add_trial(study, 3e-13, params={"x": 1})
    _add_trial(study, 1e-13, params={"x": 2})

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


def test_selection_warns_when_candidates_span_environments(caplog):
    """A top-up run under a different environment mixes environments in one ranking."""
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 1.0, params={"x": 0}, env_digest="a" * 64)
    _add_trial(study, 0.5, params={"x": 1}, env_digest="b" * 64)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        winner = select_winner(study, exp, phase_name="p")

    assert winner.trial_number == 1
    warnings = [r for r in caplog.records if "environment" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].getMessage().startswith("[p] ")
    assert "2 distinct" in warnings[0].getMessage()


def test_selection_is_quiet_when_candidates_share_one_environment(caplog):
    """One environment across the compared trials is the normal case; stay silent."""
    exp = _make_exp()
    study = _make_study()
    _add_trial(study, 1.0, params={"x": 0}, env_digest="a" * 64)
    _add_trial(study, 0.5, params={"x": 1}, env_digest="a" * 64)
    # A trial recorded before the digest existed carries no digest and must not
    # be counted as a second environment.
    _add_trial(study, 0.9, params={"x": 2})

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        select_winner(study, exp, phase_name="p")

    assert not [r for r in caplog.records if "environment" in r.getMessage()]


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
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
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
