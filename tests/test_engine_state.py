"""Trial-state names and durable trainer-completion evidence."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from phasesweep.engine.state import (
    DURATION_ATTR,
    FINISHED_TRIAL_STATES,
    RETURN_CODE_ATTR,
    TRIAL_STATE_NAMES,
    _completed_trainer_duration,
)
from tests.conftest import make_experiment, phase_trials
from tests.recovery_helpers import new_phase_study


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
