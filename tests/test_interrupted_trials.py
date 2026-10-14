"""Interrupted trials: the one outcome label, what stale recovery records, how they are counted, and their reruns."""

from __future__ import annotations

import csv
import logging
import os
import signal
import threading
import time
from pathlib import Path

import optuna
import pytest
from optuna.distributions import IntDistribution

from phasesweep import run_experiment
from phasesweep.config import Experiment
from phasesweep.engine import phase as phase_module
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
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    PHASE_FINGERPRINT_ATTR,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    _outcome_sequence_counts,
    _parsed_trial_outcome,
    _recovered_trial_state,
    _stale_recovery_outcome,
    _trial_interrupted,
)
from phasesweep.engine.study_policy import (
    _load_phase_policy_state,
    _record_reruns,
    _rerun_sources,
)
from phasesweep.engine.workers import _await_worker_trials, _WorkerTrials
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import make_experiment, write_param_echo_trainer, write_trainer
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


def test_the_outcome_parser_rejects_an_unknown_label_and_another_schema_version() -> None:
    """A label outside the outcome vocabulary, such as ``cancelled``, and any other schema version parse as nothing."""
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
        pytest.param(
            _outcome(2, "failure", cause="exit 1"),
            _outcome(3, "interrupted", cause=_STALE_CAUSE),
            id="duplicate-sequence-failure",
        ),
    ],
)
def test_stale_recovery_records_interrupted_unless_the_objective_recorded_a_failure(
    recorded: JsonObject | None, expected: JsonObject
) -> None:
    """A stale trial is recorded as interrupted, as the dry run and status predict, and an interruption never extends the streak."""
    study = optuna.create_study()
    _finished(study, 1, "failure")
    _finished(study, 2, "failure")
    trial = _running_trial(study, recorded)
    predicted = _recovered_trial_state(trial, _outcome_sequence_counts(study.get_trials()))

    _record_stale_trial_outcome(study, trial)
    assert predicted == ("INTERRUPTED" if expected["outcome"] == "interrupted" else "FAIL")
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


_SIGINT_CAUSE = f"orchestrator interrupted by signal {int(signal.SIGINT)}"


def _phase_trials(experiment: Experiment) -> list[optuna.trial.FrozenTrial]:
    """Read the trials of the experiment's phase ``p``.

    :param Experiment experiment: Experiment whose ledger is read.
    :return list[optuna.trial.FrozenTrial]: Its trials in number order.
    """
    storage = _resolve_storage(experiment.resolved_storage)
    return optuna.load_study(study_name="t::p", storage=storage).get_trials(deepcopy=False)


def _cause(trial: optuna.trial.FrozenTrial) -> str | None:
    """Return the cause a trial's outcome record carries.

    :param optuna.trial.FrozenTrial trial: Trial whose record is read.
    :return str | None: The recorded cause, or ``None`` without a record.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    return None if parsed is None else parsed.cause


@pytest.mark.parametrize("serviced", [True, False], ids=["shutdown", "no-shutdown"])
def test_a_trainer_that_fails_while_a_shutdown_is_serviced_is_interrupted_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serviced: bool
) -> None:
    """A worker sees its trainer die when the handler kills it; that counts as an interruption, not a failure."""
    launched = tmp_path / "launched"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys
        from pathlib import Path
        Path({str(launched)!r}).touch()
        sys.exit(1)
        """,
    )
    exp = make_experiment(
        persistent=tmp_path, trainer=trainer, n_trials=1, max_consecutive_failures=1
    )
    # The shutdown starts once the trainer runs, as the handler's kill sweep does.
    monkeypatch.setattr(
        phase_module,
        "servicing_shutdown_signal",
        lambda: signal.SIGINT if serviced and launched.exists() else None,
    )

    expected = (
        (RuntimeError, "stopped after 0/1")
        if serviced
        else (NoFeasibleTrialError, "aborted after 1 consecutive failures")
    )
    with pytest.raises(expected[0], match=expected[1]):
        run_experiment(exp)

    (trial,) = _phase_trials(exp)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(exp.resolved_storage))
    assert launched.exists()
    if serviced:
        assert _trial_state_label(trial) == "INTERRUPTED"
        assert _cause(trial) == _SIGINT_CAUSE
        assert _load_phase_policy_state(study).consecutive_failures == 0
        assert _record_reruns(study) == [(0, 1)]
    else:
        assert _trial_state_label(trial) == "FAIL"
        assert _load_phase_policy_state(study).consecutive_failures == 1
        assert _record_reruns(study) == []


@pytest.mark.parametrize("refusing_check", [1, 2], ids=["before-allocation", "inside-the-lease"])
def test_no_trainer_starts_while_a_shutdown_is_serviced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusing_check: int
) -> None:
    """A trial that reaches either launch gate during a shutdown is interrupted by its signal and starts nothing."""
    launched = tmp_path / "launched"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        from pathlib import Path
        Path({str(launched)!r}).touch()
        print("x=1")
        """,
    )
    exp = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=1)
    checks = 0

    def servicing() -> int | None:
        nonlocal checks
        checks += 1
        return signal.SIGINT if checks >= refusing_check else None

    monkeypatch.setattr(phase_module, "servicing_shutdown_signal", servicing)
    with pytest.raises(RuntimeError, match="stopped after 0/1"):
        run_experiment(exp)

    (trial,) = _phase_trials(exp)
    assert not launched.exists()
    assert _trial_state_label(trial) == "INTERRUPTED"
    assert _cause(trial) == _SIGINT_CAUSE


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_a_shutdown_waits_for_parallel_workers_to_record_their_own_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A signal during Optuna's final join lets each worker record its interruption before reconciliation reaps."""
    reached = tmp_path / "reached"
    reached.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import os, time
        from pathlib import Path
        Path({str(reached)!r}, os.environ["PHASESWEEP_TRIAL_ID"]).touch()
        time.sleep(60)
        """,
    )
    exp = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=2,
        n_jobs=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        timeout_seconds_per_trial=60.0,
    )
    record_outcome = phase_module._record_outcome

    def slow_interruption(
        run: phase_module._PhaseExecution,
        trial: optuna.Trial,
        outcome: str,
        *,
        cause: str | None = None,
        fatal_policy: str | None = None,
    ) -> None:
        # Holds each worker past the moment a reconciliation that did not wait would reap.
        if outcome == "interrupted":
            time.sleep(1.0)
        record_outcome(run, trial, outcome, cause=cause, fatal_policy=fatal_policy)

    monkeypatch.setattr(phase_module, "_record_outcome", slow_interruption)

    def interrupt_once_both_trainers_run() -> None:
        deadline = time.monotonic() + 30.0
        while len(list(reached.iterdir())) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    sender = threading.Thread(target=interrupt_once_both_trainers_run, daemon=True)
    caplog.set_level(logging.INFO, logger="phasesweep")
    sender.start()
    with pytest.raises(PhaseSweepShutdown):
        run_experiment(exp)
    sender.join(timeout=5.0)

    trials = _phase_trials(exp)
    assert [_trial_state_label(trial) for trial in trials] == ["INTERRUPTED", "INTERRUPTED"]
    assert [_cause(trial) for trial in trials] == [_SIGINT_CAUSE, _SIGINT_CAUSE]
    assert not any("FATAL OBJECTIVE ERROR" in record.getMessage() for record in caplog.records)


class _ObservedWorkers(_WorkerTrials):
    """Worker tracker that reports when the main thread starts waiting on it."""

    def __init__(self) -> None:
        """Track objectives and record the first wait."""
        super().__init__()
        self.waiting = threading.Event()

    def wait_idle(self) -> frozenset[int]:
        """Report the wait, then wait as the real tracker does.

        :return frozenset[int]: The abandoned trials.
        """
        self.waiting.set()
        return super().wait_idle()


def test_the_shutdown_wait_holds_until_a_running_worker_tells_its_trial() -> None:
    """The main thread returns only after the worker's objective ended and Optuna told the trial."""
    study = optuna.create_study()
    trial = study.ask()
    workers = _ObservedWorkers()
    entered = threading.Event()

    def worker() -> None:
        with workers.track(trial.number):
            entered.set()
            # Finish only once the main thread is waiting on this objective.
            assert workers.waiting.wait(10.0)
        study.tell(trial, state=optuna.trial.TrialState.FAIL)

    running = threading.Thread(target=worker)
    running.start()
    assert entered.wait(10.0)
    _await_worker_trials(workers, study)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    running.join(10.0)


def test_the_shutdown_wait_skips_a_trial_its_objective_abandoned() -> None:
    """A trial a shutdown left RUNNING on purpose is stale recovery's, so the wait does not hold for it."""
    study = optuna.create_study()
    trial = study.ask()
    workers = _WorkerTrials()
    with pytest.raises(SystemExit), workers.track(trial.number):
        raise SystemExit(130)
    started = time.monotonic()
    _await_worker_trials(workers, study)
    assert time.monotonic() - started < 1.0


def test_the_shutdown_wait_gives_up_on_a_running_trial_no_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RUNNING trial no objective runs any more (its tell failed) bounds the wait instead of hanging it."""
    monkeypatch.setattr("phasesweep.engine.workers._WORKER_SETTLE_SECONDS", 0.2)
    study = optuna.create_study()
    study.ask()
    started = time.monotonic()
    _await_worker_trials(_WorkerTrials(), study)
    assert 0.2 <= time.monotonic() - started < 2.0
