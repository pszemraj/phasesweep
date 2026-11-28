"""Trial-state names and durable trainer-completion evidence."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from phasesweep.engine.repeats import RepeatMember, RepeatsRecord, spread
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    DURATION_ATTR,
    FINISHED_TRIAL_STATES,
    RETURN_CODE_ATTR,
    TRIAL_STATE_NAMES,
    _completed_trainer_duration,
    _stands_under_winner,
    _winning_trials,
)
from tests.conftest import make_experiment, new_phase_study, phase_trials


def test_trial_state_names_match_optuna_trial_state() -> None:
    """``TRIAL_STATE_NAMES`` names Optuna's ``TrialState`` members except ``PRUNED``, plus ``INFEASIBLE`` and ``INTERRUPTED``."""
    optuna_names = {state.name for state in optuna.trial.TrialState}
    assert set(TRIAL_STATE_NAMES) == (optuna_names - {"PRUNED"}) | {"INFEASIBLE", "INTERRUPTED"}
    assert len(TRIAL_STATE_NAMES) == len(set(TRIAL_STATE_NAMES))


def test_terminal_trial_states_match_is_finished() -> None:
    """``FINISHED_TRIAL_STATES`` names the states Optuna reports as finished, except ``PRUNED``, plus ``INFEASIBLE``."""
    finished_names = {state.name for state in optuna.trial.TrialState if state.is_finished()}
    assert set(FINISHED_TRIAL_STATES) == (finished_names - {"PRUNED"}) | {"INFEASIBLE"}
    assert set(FINISHED_TRIAL_STATES) < set(TRIAL_STATE_NAMES)


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        pytest.param(10**400, None, id="positive-overflow"),
        pytest.param(-(10**400), None, id="negative-overflow"),
        pytest.param(float("nan"), None, id="nan"),
        pytest.param(-1.0, None, id="negative"),
        pytest.param(0, 0.0, id="zero"),
        pytest.param(1.25, 1.25, id="finite"),
    ],
)
def test_completed_trainer_duration_requires_representable_time(
    tmp_path: Path, duration: int | float, expected: float | None
) -> None:
    """Reopened trainer metadata supplies completion evidence only with a finite duration."""
    experiment = make_experiment(persistent=tmp_path)
    trial = new_phase_study(experiment, "p").ask()
    trial.set_user_attr(RETURN_CODE_ATTR, 0)
    trial.set_user_attr(DURATION_ATTR, duration)

    (recorded,) = phase_trials(experiment)

    assert _completed_trainer_duration(recorded, None) == expected


def _winning_study(attempts: list[str]) -> list[optuna.trial.FrozenTrial]:
    """Record one completed trial per attempt id, numbered in order, and read them back."""
    study = optuna.create_study()
    for attempt in attempts:
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=0.0,
                user_attrs={ATTEMPT_ID_ATTR: attempt},
            )
        )
    return study.get_trials()


def test_one_winning_trials_function_serves_the_summary_and_the_trial_table() -> None:
    """A repeats winner stands on every member at its own attempt; any other on its one trial."""
    trials = _winning_study(["attempt-a", "attempt-b", "attempt-c"])
    members: list[RepeatMember] = [
        {
            "trial_number": number,
            "repeat": number,
            "metric": 0.5,
            "generation_id": "g",
            "attempt_id": attempt,
            "objective_provenance": {},
            "trainer_input": {},
            "constraints": {},
            "observed": {},
        }
        for number, attempt in ((0, "attempt-a"), (2, "attempt-c"))
    ]
    repeats: RepeatsRecord = {
        "key": "seed",
        "reduce": "mean",
        "spread": spread([0.5, 0.5]),
        "members": members,
    }

    single = _winning_trials(1, "attempt-b", None)
    group = _winning_trials(2, "attempt-c", repeats)
    unchecked = _winning_trials(1, None, None)

    assert single == {1: "attempt-b"}
    assert group == {0: "attempt-a", 2: "attempt-c"}
    assert [_stands_under_winner(trial, single) for trial in trials] == [False, True, False]
    assert [_stands_under_winner(trial, group) for trial in trials] == [True, False, True]
    assert [_stands_under_winner(trial, unchecked) for trial in trials] == [False, True, False]
    assert not _stands_under_winner(trials[1], {1: "attempt-other"})
