"""Interrupted trials: the one outcome label, what stale recovery records, how they are counted, and their reruns."""

from __future__ import annotations

import csv
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from optuna.distributions import IntDistribution

import phasesweep.engine.attempts as attempts_module
import phasesweep.runtime.reaper as reaper
from phasesweep import run_experiment
from phasesweep.engine import phase as phase_module
from phasesweep.engine.attempts import (
    _AttemptRecord,
    _PreflightCleanupReport,
    _record_stale_trial_outcome,
)
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.guards import _reconcile_existing_studies
from phasesweep.engine.optuna import (
    _draw_positions,
    _finished_trial_count,
    _trial_count_state,
)
from phasesweep.engine.paths import _phase_dir
from phasesweep.engine.preview import _preview_ledger
from phasesweep.engine.read import read_trial_tables
from phasesweep.engine.selection import NoFeasibleTrialError, PhaseAbortRefusedError
from phasesweep.engine.state import (
    FEASIBLE_ATTR,
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
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.runtime.json import JsonObject, json_object
from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle
from phasesweep.runtime.reaper import read_boot_id
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import (
    make_experiment,
    phase_study,
    phase_trials,
    trial_outcome,
    write_experiment_config,
    write_param_echo_trainer,
    write_trainer,
)
from tests.recovery_helpers import fabricate_stale_trial, write_trial_identity

_STALE_CAUSE = "stale RUNNING trial recovered after its orchestrator stopped"
_UNCOMMITTED_CAUSE = "orchestrator stopped before Optuna committed the terminal trial state"


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
        pytest.param(trial_outcome(1, "success"), "interrupted", id="uncommitted-success"),
        pytest.param(trial_outcome(1, "interrupted"), "interrupted", id="interrupted"),
        pytest.param(trial_outcome(1, "failure"), "failure", id="failure"),
        pytest.param(
            trial_outcome(1, "failure", **{TRIAL_OUTCOME_ABORT_KEY: _abort(1)}),
            "failure",
            id="failure-with-abort",
        ),
        pytest.param(
            trial_outcome(1, "fatal", policy="unsafe_process_cleanup"), "fatal", id="fatal"
        ),
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
    assert _parsed_trial_outcome(trial_outcome(1, "cancelled")) is None
    previous = trial_outcome(1, "interrupted")
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
            user_attrs={TRIAL_OUTCOME_ATTR: trial_outcome(sequence, outcome)},
        )
    )


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        pytest.param(None, trial_outcome(3, "interrupted", cause=_STALE_CAUSE), id="no-record"),
        pytest.param(
            trial_outcome(3, "success"),
            trial_outcome(3, "interrupted", cause=_UNCOMMITTED_CAUSE),
            id="uncommitted-success",
        ),
        pytest.param(
            trial_outcome(3, "interrupted", cause="orchestrator interrupted by signal 2"),
            trial_outcome(3, "interrupted", cause="orchestrator interrupted by signal 2"),
            id="signal",
        ),
        pytest.param(
            trial_outcome(3, "failure", cause="exit 1"),
            trial_outcome(3, "failure", cause="exit 1"),
            id="failure",
        ),
        pytest.param(
            trial_outcome(2, "interrupted"),
            trial_outcome(3, "interrupted", cause=_STALE_CAUSE),
            id="duplicate-sequence",
        ),
        pytest.param(
            trial_outcome(2, "failure", cause="exit 1"),
            trial_outcome(3, "interrupted", cause=_STALE_CAUSE),
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


def test_an_interruption_between_two_failures_leaves_the_streak_at_two() -> None:
    """Failure, stale-recovered interruption, failure replays as a streak of 2: the interruption neither counts nor resets it."""
    study = optuna.create_study()
    _finished(study, 1, "failure")
    stale = _running_trial(study, None)
    _record_stale_trial_outcome(study, stale)
    study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
    _finished(study, 3, "failure")

    assert _trial_interrupted(study.trials[stale.number])
    assert _load_phase_policy_state(study).consecutive_failures == 2


def test_a_running_trial_is_not_interrupted_until_it_is_terminal() -> None:
    """Only a terminal trial counts as interrupted, even when its objective already recorded the label."""
    study = optuna.create_study()
    trial = _running_trial(study, trial_outcome(1, "interrupted"))
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
        user_attrs={TRIAL_OUTCOME_ATTR: trial_outcome(1, "interrupted")},
    )


def test_an_interrupted_trial_is_counted_apart_and_never_toward_the_target() -> None:
    """Interrupted trials count as ``INTERRUPTED``, never as finished, and reruns draw no new position."""
    study = optuna.create_study()
    _finished(study, 2, "failure")
    study.add_trial(_interrupted())
    _record_reruns(study)
    trials = study.get_trials(deepcopy=False)

    assert [_trial_count_state(t) for t in trials] == ["FAIL", "INTERRUPTED", "WAITING"]
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
    rerun.set_user_attr(TRIAL_OUTCOME_ATTR, trial_outcome(2, "interrupted"))
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

    trials = phase_trials(exp)
    controls = phase_trials(control)
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
        else (NoFeasibleTrialError, "1 consecutive failed/infeasible trials")
    )
    with pytest.raises(expected[0], match=expected[1]):
        run_experiment(exp)

    (trial,) = phase_trials(exp)
    study = phase_study(exp)
    assert launched.exists()
    if serviced:
        assert _trial_count_state(trial) == "INTERRUPTED"
        assert _cause(trial) == _SIGINT_CAUSE
        assert _load_phase_policy_state(study).consecutive_failures == 0
        assert _record_reruns(study) == [(0, 1)]
    else:
        assert _trial_count_state(trial) == "FAIL"
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

    (trial,) = phase_trials(exp)
    assert not launched.exists()
    assert _trial_count_state(trial) == "INTERRUPTED"
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

    trials = phase_trials(exp)
    assert [_trial_count_state(trial) for trial in trials] == ["INTERRUPTED", "INTERRUPTED"]
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


@pytest.mark.integration
@pytest.mark.parametrize(("return_code", "feasible"), [(0, True), (0, False), (7, True)])
@pytest.mark.parametrize("sdk_worker", [False, True])
def test_completed_orphan_evidence_survives_runner_sigkill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    feasible: bool,
    sdk_worker: bool,
) -> None:
    """A surviving guardian saves trainer exit evidence; only zero exits survive resume as COMPLETE."""
    started = tmp_path / "started"
    release = tmp_path / "release"
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        args, _ = parser.parse_known_args()
        with Path({str(launches)!r}).open("a") as handle:
            handle.write("launch\\n")
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            time.sleep(0.02)
        Path(args.out).write_text(json.dumps({{"x": 1.0}}))
        print("x=1.0")
        raise SystemExit({return_code})
    """,
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=1)
    if not feasible:
        experiment = experiment.model_copy(
            update={"metric": experiment.metric.model_copy(update={"max": 0.5})}
        )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    worker_started = tmp_path / "worker-started"
    worker = write_trainer(
        tmp_path / "worker.py",
        f"""
        import time
        from pathlib import Path
        Path({str(worker_started)!r}).touch()
        time.sleep(60)
    """,
    )
    if sdk_worker:
        release.touch()
    runner_code = """
import sys, time
from pathlib import Path
from phasesweep.config.io import load_experiment
from phasesweep import run_experiment
from phasesweep.engine import phase
from phasesweep.runtime.remote_worker import run_worker

original_extract = phase.extract_trial_result

def extract_with_worker(**kwargs):
    \"\"\"Keep an SDK worker active until recovery terminates its process slot.\"\"\"
    ctx = kwargs["executed"].ctx
    run_worker({}, Path(sys.argv[3]), "fake-sdk", statuses=frozenset({"ok"}), run_id=ctx.attempt_id, trial_dir=ctx.trial_dir, environment={}, deadline=time.monotonic() + 60)
    return original_extract(**kwargs)

if sys.argv[2] == "1":
    phase.extract_trial_result = extract_with_worker
run_experiment(load_experiment(sys.argv[1]))
"""
    orphan_identity = None
    with (tmp_path / "runner.log").open("w") as output:
        runner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                runner_code,
                str(config),
                "1" if sdk_worker else "0",
                str(worker),
            ],
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 15
            reached = worker_started if sdk_worker else started
            while not reached.exists() and time.monotonic() < deadline:
                assert runner.poll() is None, (tmp_path / "runner.log").read_text()
                time.sleep(0.02)
            assert reached.exists()
            runner.kill()
            runner.wait(timeout=5)
            release.touch()
            (trial,) = phase_trials(experiment)
            directory = Path(trial.user_attrs["phasesweep_trial_dir"])
            attempt = trial.user_attrs["phasesweep_attempt_id"]
            orphan_identity = reaper.read_stale_process_identity(
                directory, expected_attempt_id=attempt
            )
            if sdk_worker:
                lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=attempt)
                assert lifecycle is not None and lifecycle.state == "launching"
                assert reaper._process_group_alive(orphan_identity.pgid)
            else:
                while time.monotonic() < deadline:
                    lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=attempt)
                    if lifecycle is not None and lifecycle.state == "exited":
                        break
                    time.sleep(0.02)
                assert lifecycle is not None and lifecycle.state == "exited"
                assert lifecycle.return_code == return_code
                assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds > 0
            assert (directory / "r.json").exists()
            if return_code == 0:
                assert (
                    _recovered_trial_state(trial, _outcome_sequence_counts([trial]))
                    == "EVIDENCE_PENDING"
                )
                with pytest.raises(
                    PhaseSweepError, match="completed trainer evidence awaiting recovery"
                ):
                    _preview_ledger(experiment, from_phase=None)

                def lose_terminal_commit(*_args: object, **_kwargs: object) -> None:
                    raise RuntimeError("interrupted terminal commit")

                with monkeypatch.context() as interrupted:
                    interrupted.setattr(optuna.Study, "tell", lose_terminal_commit)
                    with pytest.raises(StudyStorageUnavailableError, match="retained evidence"):
                        _reconcile_existing_studies(
                            experiment, cleanup_report=_PreflightCleanupReport()
                        )
                assert phase_trials(experiment)[0].state.name == "RUNNING"
                if not feasible:
                    interrupted_study = phase_study(experiment)
                    pending = interrupted_study.get_trials()[0]
                    parsed_pending = _parsed_trial_outcome(pending.user_attrs[TRIAL_OUTCOME_ATTR])
                    assert parsed_pending is not None
                    saved_outcome = json_object(pending.user_attrs[TRIAL_OUTCOME_ATTR])
                    saved_abort = _abort(parsed_pending.sequence)
                    saved_abort["trial_target"] = 1
                    saved_outcome[TRIAL_OUTCOME_ABORT_KEY] = saved_abort
                    optuna.Trial(interrupted_study, pending._trial_id).set_user_attr(
                        TRIAL_OUTCOME_ATTR, saved_outcome
                    )
            _reconcile_existing_studies(experiment, cleanup_report=_PreflightCleanupReport())
            (recovered,) = phase_trials(experiment)
            assert recovered.state.name == ("COMPLETE" if return_code == 0 else "FAIL")
            if return_code == 0:
                assert recovered.value == 1.0
                assert recovered.user_attrs[FEASIBLE_ATTR] is feasible
                if feasible:
                    assert _cause(recovered) is None
                    assert run_experiment(experiment)["p"].metric == 1.0
                    assert run_experiment(experiment)["p"].metric == 1.0
                else:
                    assert (
                        _cause(recovered)
                        == "recovered evidence violates metric or constraint bounds"
                    )
                    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
                        run_experiment(experiment)
                assert len(phase_trials(experiment)) == 1
            else:
                assert _trial_interrupted(recovered)
            assert launches.read_text() == "launch\n"
            assert not reaper._process_group_alive(orphan_identity.pgid)
        finally:
            release.touch()
            if runner.poll() is None:
                runner.kill()
            runner.wait(timeout=5)
            if orphan_identity is not None:
                reaper.cleanup_stale_trial_process(orphan_identity, grace_seconds=0.2)


@pytest.mark.parametrize(
    "case",
    ["live", "leader_exited", "group_exited", "prior_boot", "foreign_absent", "reused_group"],
)
def test_recovery_preview_names_trial_group_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    """Inspection and confirmation share every signal-target and absence decision."""
    write_trial_identity(tmp_path, attempt_id="preview", pid=12345, starttime=1)
    monkeypatch.setattr(
        reaper, "read_boot_id", lambda: "old" if case == "prior_boot" else read_boot_id()
    )
    monkeypatch.setattr(reaper, "same_pid_namespace", lambda _: case != "foreign_absent")
    monkeypatch.setattr(reaper, "recorded_process_blocker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(reaper, "is_pid_alive", lambda _: case == "live")
    monkeypatch.setattr(reaper, "read_proc_starttime", lambda _: 1)
    monkeypatch.setattr(reaper.os, "getpgid", lambda _: 12345)
    monkeypatch.setattr(reaper, "_process_group_exists", lambda _: case != "group_exited")
    monkeypatch.setattr(reaper, "_process_group_alive", lambda _: case != "group_exited")
    monkeypatch.setattr(
        reaper,
        "_read_proc_stat",
        lambda _: SimpleNamespace(pgrp=12345, starttime=2) if case == "reused_group" else None,
    )
    signalled: list[int] = []
    monkeypatch.setattr(
        reaper, "_terminate_process_group", lambda pgid, **_: signalled.append(pgid) or True
    )
    record = _AttemptRecord("preview", tmp_path, "generation", "p", 0)
    attempts_module._resolve_attempt(
        record,
        context="RUNNING trial 0",
        evidence_action=OperatorAction.RESTORE_TREE,
        lifecycle_settles=True,
        confirm=False,
    )
    assert not signalled
    should_signal = case in {"live", "leader_exited"}
    assert ("--confirm will terminate it if still alive" in caplog.text) is should_signal
    assert "pid=12345 pgid=12345" in caplog.text
    if case == "leader_exited":
        assert "stored PGID fallback" in caplog.text
    if case == "reused_group":
        assert "unrelated group will be spared" in caplog.text
        with pytest.raises(attempts_module.ProcessCleanupUncertainError):
            attempts_module._resolve_attempt(
                record,
                context="RUNNING trial 0",
                evidence_action=OperatorAction.RESTORE_TREE,
                lifecycle_settles=True,
                confirm=True,
            )
    else:
        attempts_module._resolve_attempt(
            record,
            context="RUNNING trial 0",
            evidence_action=OperatorAction.RESTORE_TREE,
            lifecycle_settles=True,
            confirm=True,
        )
    assert signalled == ([12345] if should_signal else [])
