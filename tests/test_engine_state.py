"""engine.state's trial-state constants must agree with Optuna's own enum."""

from __future__ import annotations

import optuna

from phasesweep.engine.state import TERMINAL_TRIAL_STATES, TRIAL_STATE_NAMES


def test_trial_state_names_match_optuna_trial_state() -> None:
    """``TRIAL_STATE_NAMES`` names exactly Optuna's ``TrialState`` members."""
    assert set(TRIAL_STATE_NAMES) == {state.name for state in optuna.trial.TrialState}
    assert len(TRIAL_STATE_NAMES) == len(set(TRIAL_STATE_NAMES))


def test_terminal_trial_states_match_is_finished() -> None:
    """``TERMINAL_TRIAL_STATES`` names exactly the states Optuna reports as finished."""
    finished_names = {state.name for state in optuna.trial.TrialState if state.is_finished()}
    assert set(TERMINAL_TRIAL_STATES) == finished_names
    assert set(TERMINAL_TRIAL_STATES) < set(TRIAL_STATE_NAMES)
