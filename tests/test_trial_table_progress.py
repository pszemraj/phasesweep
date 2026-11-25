"""A ``status --trials`` phase heading reads finished trials of the target, then the rows that do not count.

The tables here come from the read model, so the count is the engine's one rule for what goes toward
``n_trials``, not a tally of the rows the renderer happens to see.
"""

from __future__ import annotations

import optuna
import pytest
from optuna.distributions import IntDistribution

from phasesweep.config.models import Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.read import TrialTables, _trial_table
from phasesweep.engine.state import (
    DURATION_ATTR,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
)
from phasesweep.runtime.json import JsonValue
from phasesweep.trial_table import render_trial_tables
from tests.conftest import make_experiment, trial_outcome
from tests.test_trial_table import _metric

_STATE = optuna.trial.TrialState
_SIGINT = "orchestrator interrupted by SIGINT"


def _trial(
    state: optuna.trial.TrialState,
    x: int,
    value: float | None = None,
    *,
    seed: int | None = None,
    **attrs: JsonValue,
) -> optuna.trial.FrozenTrial:
    """Build one trial of phase ``learning_rate`` with its parameter ``x`` and, for repeats, ``seed``."""
    params = {"x": x} if seed is None else {"x": x, "seed": seed}
    return optuna.trial.create_trial(
        state=state,
        value=value,
        params=params,
        distributions={name: IntDistribution(0, 9) for name in params},
        user_attrs={DURATION_ATTR: 0.5, **attrs},
    )


def _complete(
    x: int, value: float, *, seed: int | None = None, **attrs: JsonValue
) -> optuna.trial.FrozenTrial:
    """Build a COMPLETE trial inside every bound, which is what counts toward the target."""
    return _trial(_STATE.COMPLETE, x, value, seed=seed, **attrs)


def _interrupted(x: int) -> optuna.trial.FrozenTrial:
    """Build the FAIL trial that stale recovery or a Ctrl-C leaves, which reruns instead of counting."""
    outcome = trial_outcome(1, "interrupted", cause=_SIGINT)
    return _trial(_STATE.FAIL, x, **{TRIAL_OUTCOME_ATTR: outcome})


def _heading_and_rows(
    trials: list[optuna.trial.FrozenTrial], width: int, *, n_trials: int, repeats: bool = False
) -> list[str]:
    """Render one phase through the read model and return the lines from its heading on."""
    phase = Phase(
        name="learning_rate",
        n_trials=n_trials,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=9)},
        **({"repeats": Repeats(key="seed", values=[0, 1, 2])} if repeats else {}),
    )
    study = optuna.create_study()
    for trial in trials:
        study.add_trial(trial)
    experiment = make_experiment(phases=[phase], metric=_metric())
    table = _trial_table(experiment, phase, study.get_trials(deepcopy=False), None)
    tables = TrialTables(
        experiment="t",
        published_generation_id=None,
        publication_integrity="absent",
        publication_error=None,
        phases=(table,),
    )
    rendered = render_trial_tables(tables, experiment, csv_root=None, width=width, color=False)
    return rendered.splitlines()[3:]


@pytest.mark.parametrize("width", [80, 60])
def test_a_finished_phase_reads_its_target_with_no_parenthesis(width: int) -> None:
    """A phase whose every trial finished says so, with nothing to list after the count."""
    lines = _heading_and_rows([_complete(3, 0.5), _complete(4, 0.3)], width, n_trials=2)

    assert lines == [
        "learning_rate: 2 of 2 trials, eval_loss (lower is better)",
        "          trial  eval_loss  x  state     time",
        "              0        0.5  3  COMPLETE  0.5s",
        "              1        0.3  4  COMPLETE  0.5s",
    ]


@pytest.mark.parametrize(
    ("width", "heading"),
    [
        (80, ["learning_rate: 2 of 4 trials (1 interrupted), eval_loss (lower is better)"]),
        (60, ["learning_rate: 2 of 4 trials (1 interrupted), eval_loss", "  (lower is better)"]),
    ],
)
def test_an_interrupted_trial_is_named_beside_the_progress_it_does_not_add_to(
    width: int, heading: list[str]
) -> None:
    """The interrupted row stays in the table but not in the count, so the phase is still short."""
    lines = _heading_and_rows(
        [_complete(3, 0.5), _interrupted(5), _complete(4, 0.3)], width, n_trials=4
    )

    assert lines == [
        *heading,
        "          trial  eval_loss  x  state        time",
        "              0        0.5  3  COMPLETE     0.5s",
        "              2        0.3  4  COMPLETE     0.5s",
        "              1          -  5  INTERRUPTED  0.5s",
        f"          reason: {_SIGINT}",
    ]


def test_an_interrupted_trial_its_rerun_replaced_is_not_named_as_pending() -> None:
    """The rerun counts and the trial it replaced stays a row, but the heading names no pending interruption."""
    lines = _heading_and_rows(
        [_interrupted(5), _complete(5, 0.4, **{RERUN_OF_ATTR: 0}), _complete(4, 0.3)],
        80,
        n_trials=2,
    )

    assert lines == [
        "learning_rate: 2 of 2 trials, eval_loss (lower is better)",
        "          trial  eval_loss  x  state        time  rerun of",
        "              2        0.3  4  COMPLETE     0.5s         -",
        "              0          -  5  INTERRUPTED  0.5s         -",
        f"          reason: {_SIGINT}",
        "              1        0.4  5  COMPLETE     0.5s         0",
    ]


@pytest.mark.parametrize("width", [80, 60])
def test_a_repeats_phase_counts_every_member_trial_toward_its_target(width: int) -> None:
    """Six members of two configurations are six of six trials, not two configurations."""
    members = [
        _complete(x, value, seed=seed)
        for x, values in ((3, (0.5, 0.7, 0.9)), (1, (1.0, 2.0, 1.5)))
        for seed, value in enumerate(values)
    ]

    lines = _heading_and_rows(members, width, n_trials=6, repeats=True)

    assert lines[:8] == [
        "learning_rate: 6 of 6 trials, eval_loss (lower is better)",
        "          trial  eval_loss  x  seed  group  state     time",
        "              3          1  1     0      5  COMPLETE  0.5s",
        "              4          2  1     1      5  COMPLETE  0.5s",
        "              5        1.5  1     2      5  COMPLETE  0.5s",
        "              0        0.5  3     0      2  COMPLETE  0.5s",
        "              1        0.7  3     1      2  COMPLETE  0.5s",
        "              2        0.9  3     2      2  COMPLETE  0.5s",
    ]


@pytest.mark.parametrize(
    ("width", "heading"),
    [
        (
            80,
            [
                "learning_rate: 1 of 4 trials (2 running, 1 interrupted), eval_loss (lower is",
                "  better)",
            ],
        ),
        (
            60,
            [
                "learning_rate: 1 of 4 trials (2 running, 1 interrupted),",
                "  eval_loss (lower is better)",
            ],
        ),
    ],
)
def test_running_trials_are_listed_before_interrupted_ones_in_a_fixed_order(
    width: int, heading: list[str]
) -> None:
    """A RUNNING trial belongs to no live run here and does not count, and the states keep one order."""
    lines = _heading_and_rows(
        [_complete(3, 0.5), _trial(_STATE.RUNNING, 4), _trial(_STATE.RUNNING, 6), _interrupted(2)],
        width,
        n_trials=4,
    )

    assert lines[: len(heading)] == heading
    assert lines[len(heading) :] == [
        "          trial  eval_loss  x  state        time",
        "              3          -  2  INTERRUPTED  0.5s",
        f"          reason: {_SIGINT}",
        "              0        0.5  3  COMPLETE     0.5s",
        "              1          -  4  RUNNING      0.5s",
        "              2          -  6  RUNNING      0.5s",
    ]
