"""Interrupted trials: the one outcome label, what stale recovery records, and how the streak treats them."""

from __future__ import annotations

import optuna
import pytest

from phasesweep.engine.attempts import _record_stale_trial_outcome
from phasesweep.engine.state import (
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    _parsed_trial_outcome,
    _stale_recovery_outcome,
    _trial_interrupted,
)
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.runtime.json import JsonObject

_STALE_CAUSE = "stale RUNNING trial recovered after its orchestrator stopped"
_UNCOMMITTED_CAUSE = "orchestrator stopped before Optuna committed the terminal trial state"


def _outcome(sequence: int, outcome: str, **extra: str | JsonObject) -> JsonObject:
    """Build a current-schema outcome record.

    :param int sequence: Completion sequence.
    :param str outcome: Outcome label.
    :param str | JsonObject extra: Further fields, such as ``cause`` or an abort record.
    :return JsonObject: The record as the objective writes it.
    """
    record: JsonObject = {
        "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
        "sequence": sequence,
        "outcome": outcome,
    }
    record.update(extra)
    return record


def _abort(sequence: int) -> JsonObject:
    """Build a well-formed abort record for the outcome at ``sequence``.

    :param int sequence: Completion sequence of the outcome that trips it.
    :return JsonObject: The abort record.
    """
    return {
        "schema_version": 1,
        "completion_sequence": sequence,
        "trial_target": 4,
        "policy": "max_consecutive_failures",
        "cause": "two consecutive failures",
    }


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        pytest.param(None, "interrupted", id="no-record"),
        pytest.param(_outcome(1, "success"), "interrupted", id="uncommitted-success"),
        pytest.param(_outcome(1, "pruned"), "interrupted", id="uncommitted-prune"),
        pytest.param(_outcome(1, "interrupted"), "interrupted", id="interrupted"),
        pytest.param(_outcome(1, "failure"), "failure", id="failure"),
        pytest.param(
            _outcome(1, "failure", **{TRIAL_OUTCOME_ABORT_KEY: _abort(1)}),
            "failure",
            id="failure-with-abort",
        ),
        pytest.param(_outcome(1, "fatal", policy="unsafe_process_cleanup"), "fatal", id="fatal"),
    ],
)
def test_stale_recovery_keeps_failures_and_interrupts_everything_else(
    recorded: JsonObject | None, expected: str
) -> None:
    """Recovery keeps a recorded failure or fatal error and turns anything unproven into an interruption."""
    parsed = None if recorded is None else _parsed_trial_outcome(recorded)
    assert recorded is None or parsed is not None
    assert _stale_recovery_outcome(parsed) == expected


def test_cancelled_is_no_longer_an_outcome() -> None:
    """The old ``cancelled`` label and the previous outcome schema are both rejected."""
    assert _parsed_trial_outcome(_outcome(1, "cancelled")) is None
    previous = _outcome(1, "interrupted")
    previous["schema_version"] = TRIAL_OUTCOME_SCHEMA_VERSION - 1
    assert _parsed_trial_outcome(previous) is None


def _running_trial(study: optuna.Study, outcome: JsonObject | None) -> optuna.trial.FrozenTrial:
    """Leave one RUNNING trial in ``study``, optionally with an outcome its objective recorded.

    :param optuna.Study study: Study to add the trial to.
    :param JsonObject | None outcome: Recorded outcome, or ``None`` for none.
    :return optuna.trial.FrozenTrial: The RUNNING trial.
    """
    trial = study.ask()
    if outcome is not None:
        trial.set_user_attr(TRIAL_OUTCOME_ATTR, outcome)
    return study._storage.get_trial(trial._trial_id)


def _finished(study: optuna.Study, sequence: int, outcome: str) -> None:
    """Add a terminal trial carrying ``outcome`` at ``sequence``.

    :param optuna.Study study: Study to add the trial to.
    :param int sequence: Completion sequence.
    :param str outcome: Outcome label.
    """
    state = (
        optuna.trial.TrialState.COMPLETE if outcome == "success" else optuna.trial.TrialState.FAIL
    )
    study.add_trial(
        optuna.trial.create_trial(
            state=state,
            value=0.0 if outcome == "success" else None,
            user_attrs={TRIAL_OUTCOME_ATTR: _outcome(sequence, outcome)},
        )
    )


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        pytest.param(None, _outcome(3, "interrupted", cause=_STALE_CAUSE), id="no-record"),
        pytest.param(
            _outcome(3, "success"),
            _outcome(3, "interrupted", cause=_UNCOMMITTED_CAUSE),
            id="uncommitted-success",
        ),
        pytest.param(
            _outcome(3, "interrupted", cause="orchestrator interrupted by signal 2"),
            _outcome(3, "interrupted", cause="orchestrator interrupted by signal 2"),
            id="signal",
        ),
        pytest.param(
            _outcome(3, "failure", cause="exit 1"),
            _outcome(3, "failure", cause="exit 1"),
            id="failure",
        ),
        pytest.param(
            _outcome(2, "interrupted"),
            _outcome(3, "interrupted", cause=_STALE_CAUSE),
            id="duplicate-sequence",
        ),
    ],
)
def test_stale_recovery_records_interrupted_unless_the_objective_recorded_a_failure(
    recorded: JsonObject | None, expected: JsonObject
) -> None:
    """A stale trial is recorded as interrupted, and an interruption never extends the streak."""
    study = optuna.create_study()
    _finished(study, 1, "failure")
    _finished(study, 2, "failure")
    trial = _running_trial(study, recorded)

    _record_stale_trial_outcome(study, trial)
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

    recovered = study.trials[trial.number]
    assert recovered.user_attrs[TRIAL_OUTCOME_ATTR] == expected
    assert _trial_interrupted(recovered) is (expected["outcome"] == "interrupted")
    streak = 3 if expected["outcome"] == "failure" else 2
    assert _load_phase_policy_state(study).consecutive_failures == streak


def test_a_running_trial_is_not_interrupted_until_it_is_terminal() -> None:
    """Only a terminal trial counts as interrupted, even when its objective already recorded the label."""
    study = optuna.create_study()
    trial = _running_trial(study, _outcome(1, "interrupted"))
    assert _trial_interrupted(trial) is False
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    assert _trial_interrupted(study.trials[trial.number]) is True
