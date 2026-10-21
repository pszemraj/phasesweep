"""engine.state's trial-state constants must agree with Optuna's own enum."""

from __future__ import annotations

import optuna

from phasesweep.engine.state import TERMINAL_TRIAL_STATES, TRIAL_STATE_NAMES


def test_trial_state_names_match_optuna_trial_state() -> None:
    """``TRIAL_STATE_NAMES`` names Optuna's ``TrialState`` members except ``PRUNED``, plus ``INFEASIBLE`` and ``INTERRUPTED``."""
    optuna_names = {state.name for state in optuna.trial.TrialState}
    assert set(TRIAL_STATE_NAMES) == (optuna_names - {"PRUNED"}) | {"INFEASIBLE", "INTERRUPTED"}
    assert len(TRIAL_STATE_NAMES) == len(set(TRIAL_STATE_NAMES))


def test_terminal_trial_states_match_is_finished() -> None:
    """``TERMINAL_TRIAL_STATES`` names the states Optuna reports as finished, except ``PRUNED``, plus ``INFEASIBLE``."""
    finished_names = {state.name for state in optuna.trial.TrialState if state.is_finished()}
    assert set(TERMINAL_TRIAL_STATES) == (finished_names - {"PRUNED"}) | {"INFEASIBLE"}
    assert set(TERMINAL_TRIAL_STATES) < set(TRIAL_STATE_NAMES)
