"""Interrupted trials: the one outcome label, what stale recovery records, how they are counted, and their reruns."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import optuna
import pytest
from optuna.distributions import IntDistribution

from phasesweep import run_experiment
from phasesweep.engine import read_trial_tables
from phasesweep.engine.attempts import _record_stale_trial_outcome
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.optuna import (
    _draw_positions,
    _finished_trial_count,
    _trial_count_state,
    _trial_state_label,
)
from phasesweep.engine.paths import _phase_dir
from phasesweep.engine.state import (
    PHASE_FINGERPRINT_ATTR,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    _parsed_trial_outcome,
    _stale_recovery_outcome,
    _trial_interrupted,
)
from phasesweep.engine.study_policy import (
    _load_phase_policy_state,
    _record_reruns,
    _rerun_sources,
)
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import make_experiment, write_param_echo_trainer
from tests.recovery_helpers import fabricate_stale_trial

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


def _interrupted(number_of_params: int = 0) -> optuna.trial.FrozenTrial:
    """Build an interrupted trial template, as stale recovery leaves one.

    :param int number_of_params: How many of the params ``a``, ``b`` it recorded.
    :return optuna.trial.FrozenTrial: A FAIL trial whose outcome is ``interrupted``.
    """
    names = ["a", "b"][:number_of_params]
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.FAIL,
        params={name: 1 for name in names},
        distributions={name: IntDistribution(0, 3) for name in names},
        system_attrs={"grid_id": 4},
        user_attrs={TRIAL_OUTCOME_ATTR: _outcome(1, "interrupted")},
    )


def test_an_interrupted_trial_is_counted_apart_and_never_toward_the_target() -> None:
    """Interrupted trials count as ``INTERRUPTED``, never as finished, and reruns draw no new position."""
    study = optuna.create_study()
    _finished(study, 2, "failure")
    study.add_trial(_interrupted())
    _record_reruns(study)
    trials = study.get_trials(deepcopy=False)

    assert [_trial_count_state(t) for t in trials] == ["FAIL", "INTERRUPTED", "WAITING"]
    assert [_trial_state_label(t) for t in trials] == ["FAIL", "INTERRUPTED", "WAITING"]
    assert _finished_trial_count(trials) == 1
    assert _draw_positions(trials) == {0: 0, 1: 1, 2: 1}


def test_each_interrupted_trial_gets_exactly_one_waiting_rerun_carrying_its_draw() -> None:
    """A rerun copies its source's params and system attrs, and a second pass schedules nothing."""
    study = optuna.create_study()
    study.add_trial(_interrupted(2))
    study.add_trial(_interrupted(0))

    assert _record_reruns(study) == [(0, 2), (1, 3)]
    assert _record_reruns(study) == []

    rerun = study.get_trials(deepcopy=False)[2]
    assert rerun.state == optuna.trial.TrialState.WAITING
    assert rerun.params == {"a": 1, "b": 1}
    assert rerun.system_attrs == {"grid_id": 4}
    assert rerun.user_attrs == {RERUN_OF_ATTR: 0}
    assert _rerun_sources(study.get_trials(deepcopy=False)) == []


def test_an_interrupted_rerun_gets_its_own_rerun_and_its_source_is_not_rescheduled() -> None:
    """Chains stay linear: only the last interrupted member of a chain is rescheduled."""
    study = optuna.create_study()
    study.add_trial(_interrupted(2))
    _record_reruns(study)
    rerun = study.ask()
    rerun.set_user_attr(TRIAL_OUTCOME_ATTR, _outcome(2, "interrupted"))
    study.tell(rerun, state=optuna.trial.TrialState.FAIL)

    assert _record_reruns(study) == [(1, 2)]
    assert _draw_positions(study.get_trials(deepcopy=False)) == {0: 0, 1: 0, 2: 0}


def test_a_stale_trial_reruns_its_own_draw_before_any_new_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run after a crash reruns the dead trial's draw first and evaluates what an uninterrupted run would."""
    trainer = write_param_echo_trainer(tmp_path)
    control = make_experiment(persistent=tmp_path / "control", trainer=trainer, n_trials=3)
    exp = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=3)
    study, trial_dir, _number = fabricate_stale_trial(exp, "p", attempt_id="killed-attempt")
    study.set_user_attr(PHASE_FINGERPRINT_ATTR, _phase_fingerprint(exp, exp.phases[0], {}))
    write_attempt_lifecycle(trial_dir, attempt_id="killed-attempt", state="allocated")

    caplog.set_level(logging.INFO, logger="phasesweep")
    winner = run_experiment(exp)["p"]
    lines = [record.getMessage() for record in caplog.records]
    expected = run_experiment(control)["p"]

    trials = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(exp.resolved_storage)
    ).get_trials(deepcopy=False)
    controls = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(control.resolved_storage)
    ).get_trials(deepcopy=False)
    assert [t.state.name for t in trials] == ["FAIL", "COMPLETE", "COMPLETE", "COMPLETE"]
    assert _trial_interrupted(trials[0])
    assert trials[1].user_attrs[RERUN_OF_ATTR] == 0
    assert [t.params for t in trials[1:]] == [t.params for t in controls]
    assert (winner.params, winner.metric) == (expected.params, expected.metric)
    assert winner.completion["finished_trials"] == 3

    assert "phase=p study=t::p finished=0 remaining=3 reruns=1 n_jobs=1" in lines
    assert f"phase=p RERUN trial=1 of=0 reason={_STALE_CAUSE}" in lines
    (rerun_line,) = (line for line in lines if line.startswith("phase=p TRIAL trial=1 "))
    assert " rerun_of=0 (1/3 finished)" in rerun_line
    with (_phase_dir(exp, "p") / "trials.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["state"] for row in rows] == ["INTERRUPTED", "COMPLETE", "COMPLETE", "COMPLETE"]
    assert [row[f"user_attr:{RERUN_OF_ATTR}"] for row in rows] == ["", "0", "", ""]
    (table,) = read_trial_tables(exp).phases
    assert "rerun_of" in table.columns
    assert [(row["state"], row["rerun_of"]) for row in table.rows] == [
        ("INTERRUPTED", None),
        ("COMPLETE", 0),
        ("COMPLETE", None),
        ("COMPLETE", None),
    ]
