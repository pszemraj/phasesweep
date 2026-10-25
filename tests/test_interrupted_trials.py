"""Interrupted trials: the one outcome label, what stale recovery records, how they are counted, and their reruns."""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml
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
from phasesweep.engine.paths import (
    _attempts_dir,
    _experiment_dir,
    _generation_config_snapshot_path,
    _phase_dir,
)
from phasesweep.engine.preview import _preview_ledger
from phasesweep.engine.read import read_trial_tables
from phasesweep.engine.selection import NoFeasibleTrialError, PhaseAbortRefusedError
from phasesweep.engine.state import (
    DURATION_ATTR,
    FEASIBLE_ATTR,
    PHASE_FINGERPRINT_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
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
from phasesweep.errors import OPERATOR_STEPS, OperatorAction, PhaseSweepError
from phasesweep.runtime.json import JsonObject, json_object
from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle
from phasesweep.runtime.reaper import read_boot_id
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import (
    invoke_cli_boundary,
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
@pytest.mark.parametrize("trainer_return_code", [0, 1])
def test_a_shutdown_waits_for_parallel_workers_to_record_their_own_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    trainer_return_code: int,
) -> None:
    """A signal during Optuna's final join lets each worker record its interruption before reconciliation reaps."""
    reached = tmp_path / "reached"
    reached.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import os, signal, sys, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, lambda *_: sys.exit({trainer_return_code}))
        print("x=0.25", flush=True)
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
    exp = exp.model_copy(update={"trial_command": "exec " + exp.trial_command})
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
@pytest.mark.parametrize(
    "cancellation",
    ["intent-write-fails", "completed-before-cancellation", "handles-term", "ignores-term"],
)
@pytest.mark.parametrize("procfs_available", [True, False], ids=["procfs", "no-procfs"])
def test_guardian_cancellation_preserves_evidence_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancellation: str,
    procfs_available: bool,
) -> None:
    """Guardians reap cancelled children without procfs and retain leases when intent cannot persist."""
    from phasesweep.runtime import process

    trainer_completed = cancellation == "completed-before-cancellation"
    intent_write_fails = cancellation == "intent-write-fails"
    cancelled = cancellation in {"handles-term", "ignores-term"}
    started, release, failed, signalled = (
        tmp_path / name for name in ("started", "release", "failed", "signalled")
    )
    publish = tmp_path / "publish"
    publication_timer: threading.Timer | None = None
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import signal, time
        from pathlib import Path
        def terminate(*_):
            Path({str(signalled)!r}).touch()
            if {cancelled!r}:
                raise SystemExit(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN if {cancellation == "ignores-term"!r} else terminate)
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            time.sleep(0.01)
        """,
    )
    guardian_script = write_trainer(
        tmp_path / "guardian.py",
        f"""
        import runpy, time
        from pathlib import Path
        namespace = runpy.run_path({process._SUPERVISOR_SCRIPT_PATH!r})
        original = namespace["write_lifecycle_record"]
        def fail_intent(path: Path, *, attempt_id: str, state: str, return_code: int | None, cleanup_confirmed: bool | None, duration_seconds: float | None) -> None:
            if state == "terminating" and {intent_write_fails!r}:
                Path({str(failed)!r}).touch()
                raise OSError("intent write unavailable")
            if state == "exited" and {trainer_completed!r}:
                Path({str(failed)!r}).touch()
                while not Path({str(publish)!r}).exists():
                    time.sleep(0.01)
            original(path, attempt_id=attempt_id, state=state, return_code=return_code, cleanup_confirmed=cleanup_confirmed, duration_seconds=duration_seconds)
        namespace["main"].__globals__["write_lifecycle_record"] = fail_intent
        namespace["main"].__globals__["_GROUP_TERM_GRACE_SECONDS"] = 0.2
        if not {procfs_available!r}:
            def no_procfs(path):
                raise FileNotFoundError(path)
            namespace["os"].scandir = no_procfs
        raise SystemExit(namespace["main"]())
        """,
    )
    monkeypatch.setattr(process, "_SUPERVISOR_SCRIPT_PATH", str(guardian_script))
    process.write_attempt_lifecycle(tmp_path, attempt_id="held-lease", state="launching")
    lease = tmp_path / "gpu.lock"
    with lease.open("w") as held, (tmp_path / "guardian.log").open("w") as output:
        fcntl.flock(held, fcntl.LOCK_EX)
        guardian, pgid, ack_write, status_read = process._spawn_blocked_supervisor(
            stdout=output,
            stderr=output,
            gpu_lease_fds=(held.fileno(),),
            lifecycle_args=(str(tmp_path), "held-lease", str(time.monotonic())),
        )
    try:
        process._write_all(
            ack_write,
            process._encode_launch_payload(
                "exec " + shlex.join([sys.executable, str(trainer)]), os.environ.copy(), None
            ),
        )
        os.close(ack_write)
        ack_write = -1
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            assert guardian.poll() is None
            time.sleep(0.01)
        assert started.exists()
        if trainer_completed:
            release.touch()
        else:
            guardian.send_signal(signal.SIGTERM)
        if cancelled:
            assert guardian.wait(timeout=5) == -signal.SIGTERM
            assert signalled.exists() is (cancellation == "handles-term")
        else:
            while not failed.exists() and time.monotonic() < deadline:
                assert guardian.poll() is None
                time.sleep(0.01)
            assert failed.exists()
            assert guardian.poll() is None
            assert reaper._process_group_alive(pgid) is not trainer_completed
            with lease.open("w") as contender, pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if trainer_completed:
                publication_timer = threading.Timer(0.2, publish.touch)
                publication_timer.start()
                assert reaper._terminate_process_groups(
                    (pgid,), grace_seconds=1.0, guardians={pgid: guardian}
                ) == {pgid: True}
                lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="held-lease")
                assert lifecycle is not None and lifecycle.state == "exited"
                assert lifecycle.return_code == 0
            else:
                release.touch()
            assert guardian.wait(timeout=5) == 0
            assert not signalled.exists()
        with lease.open("w") as contender:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="held-lease")
        assert lifecycle is not None and lifecycle.state == "exited"
        assert lifecycle.return_code == (-signal.SIGTERM if cancelled else 0)
        assert lifecycle.cleanup_confirmed
        assert not reaper._process_group_alive(pgid)
    finally:
        release.touch()
        publish.touch()
        if publication_timer is not None:
            publication_timer.cancel()
            publication_timer.join()
        if ack_write >= 0:
            os.close(ack_write)
        os.close(status_read)
        reaper._terminate_process_group(pgid, grace_seconds=0.1)
        if guardian.poll() is None:
            guardian.kill()
        guardian.wait(timeout=5)
        process._unregister(pgid)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("return_code", "feasible", "recovery_stage"),
    [
        (0, True, "finished"),
        (0, False, "finished"),
        (7, True, "finished"),
        (0, True, "sdk"),
        (0, False, "sdk"),
        (0, True, "sdk-interrupt"),
        (0, True, "sdk-interrupt-parallel"),
        (0, True, "shutdown-trainer"),
        (0, True, "timeout-trainer"),
        (0, True, "terminate"),
        (0, True, "mcp-terminate"),
        (0, True, "mcp-finished"),
        (0, True, "interrupt-intent"),
        (0, True, "interrupt-termination"),
        (0, True, "interrupt-extraction"),
        (0, True, "interrupt-extraction-signal"),
        (0, True, "missing-snapshot"),
        (0, True, "corrupt-snapshot"),
        (0, True, "altered-snapshot"),
        (0, True, "mcp-missing-snapshot"),
        (0, True, "mcp-corrupt-snapshot"),
        (0, True, "mcp-altered-snapshot"),
    ],
)
def test_completed_orphan_evidence_survives_runner_sigkill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    return_code: int,
    feasible: bool,
    recovery_stage: str,
) -> None:
    """Recovery preserves completed training across crashes and rejects a trainer it terminates."""
    sdk_worker = recovery_stage in {"sdk", "sdk-interrupt", "sdk-interrupt-parallel"}
    shutdown_sdk = recovery_stage.startswith("sdk-interrupt")
    guardian_cancel = recovery_stage in {"shutdown-trainer", "timeout-trainer"}
    terminated = recovery_stage in {
        "terminate",
        "mcp-terminate",
        "interrupt-intent",
        "interrupt-termination",
        "shutdown-trainer",
        "timeout-trainer",
    }
    started = tmp_path / "started"
    release = tmp_path / "release"
    launches = tmp_path / "launches"
    runner_pid = tmp_path / "runner-pid"
    termination_intent = tmp_path / "termination-intent"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, os, signal, sys, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        args, _ = parser.parse_known_args()
        def terminate(*_):
            if {guardian_cancel!r}:
                lifecycle = Path(args.out).parent / "attempt_lifecycle.json"
                Path({str(termination_intent)!r}).write_text(lifecycle.read_text())
                os.kill(int(Path({str(runner_pid)!r}).read_text()), signal.SIGKILL)
            sys.exit(0)
        signal.signal(signal.SIGTERM, terminate)
        with Path({str(launches)!r}).open("a") as handle:
            handle.write("launch\\n")
        Path(args.out).write_text(json.dumps({{"x": 0.25}}))
        print("x=0.25", flush=True)
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            time.sleep(0.02)
        Path(args.out).write_text(json.dumps({{"x": 1.0}}))
        print("x=1.0")
        raise SystemExit({return_code})
    """,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=1,
        n_jobs=2 if recovery_stage == "sdk-interrupt-parallel" else 1,
        timeout_seconds_per_trial=1.0 if recovery_stage == "timeout-trainer" else 30.0,
    )
    if terminated:
        experiment = experiment.model_copy(
            update={"trial_command": "exec " + experiment.trial_command}
        )
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
import logging, sys
from dataclasses import replace
from pathlib import Path
from phasesweep.config.io import load_experiment
from phasesweep import run_experiment
from phasesweep.engine import trial
from phasesweep.engine.attempts import _PreflightCleanupReport
from phasesweep.engine.guards import _reconcile_existing_studies
from phasesweep.evidence.evaluation import _capture_mlflow
from phasesweep.evidence.mlflow import compile_query
from phasesweep.evidence.models import MlflowExtractor
from phasesweep.tracking import mlflow, session
from phasesweep.tracking.models import MlflowSettings

logging.basicConfig(level=logging.WARNING)
original_extract = trial.run_extractor
original_worker = mlflow.run_worker

def fake_sdk_worker(request, worker_path, backend, **kwargs):
    \"\"\"Replace the SDK process while retaining its real supervisor and controller.\"\"\"
    return original_worker(request, Path(sys.argv[3]), backend, **kwargs)

def extract_with_worker(ctx, cfg, **kwargs):
    \"\"\"Exercise the production MLflow capture and extractor-error conversion.\"\"\"
    settings = MlflowSettings(tracking_uri="http://localhost:5000", experiment_id="1")
    binding = {
        "schema_version": 1,
        "tracking_uri": settings.tracking_uri,
        "experiment_id": "1",
        "run_id": "native-id",
        "run_mode": settings.run_mode,
        "experiment": ctx.experiment,
        "phase": ctx.phase,
        "generation_id": ctx.generation_id,
        "attempt_id": ctx.attempt_id,
        "overrides_sha256": ctx.overrides_sha256,
    }
    session.write_record(ctx.trial_dir / "mlflow-binding.json", binding)
    session.write_record(
        ctx.trial_dir / "mlflow-expected.json",
        {k: v for k, v in binding.items() if k != "run_id"},
    )
    remote = experiment.model_copy(update={
        "mlflow": settings,
        "metric": experiment.metric.model_copy(update={
            "extractor": MlflowExtractor(type="mlflow", metric_key="x", select="last"),
        }),
    })
    _capture_mlflow(
        replace(ctx, mlflow_settings=settings.target(), mlflow_query=compile_query(remote, [])),
        deadline=kwargs.get("deadline"),
    )
    return original_extract(ctx, cfg, **kwargs)

if sys.argv[2] in {"1", "recovery", "recovery-signal"}:
    trial.run_extractor = extract_with_worker
    mlflow.run_worker = fake_sdk_worker
experiment = load_experiment(sys.argv[1])
if sys.argv[2] == "recovery":
    _reconcile_existing_studies(experiment, cleanup_report=_PreflightCleanupReport())
else:
    run_experiment(
        experiment,
        generation_id=None if sys.argv[2] == "recovery-signal" else "orphan-generation",
    )
"""
    orphan_identity = None
    recovery_runner: subprocess.Popen[bytes] | None = None
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
        runner_pid.write_text(str(runner.pid))
        try:
            deadline = time.monotonic() + 15
            reached = worker_started if sdk_worker else started
            while not reached.exists() and time.monotonic() < deadline:
                assert runner.poll() is None, (tmp_path / "runner.log").read_text()
                time.sleep(0.02)
            assert reached.exists()
            (trial,) = phase_trials(experiment)
            directory = Path(trial.user_attrs["phasesweep_trial_dir"])
            attempt = trial.user_attrs["phasesweep_attempt_id"]
            orphan_identity = reaper.read_stale_process_identity(
                directory, expected_attempt_id=attempt
            )
            if shutdown_sdk or recovery_stage == "shutdown-trainer":
                runner.send_signal(signal.SIGINT)
            elif recovery_stage != "timeout-trainer":
                runner.kill()
            runner.wait(timeout=5)
            if shutdown_sdk:
                assert runner.returncode == 128 + signal.SIGINT
            elif guardian_cancel:
                assert runner.returncode == -signal.SIGKILL
                intent = json_object(json.loads(termination_intent.read_text()))
                assert intent["state"] == "terminating"
                assert intent["return_code"] == -signal.SIGTERM
                assert intent["cleanup_confirmed"] is False
            if not terminated:
                release.touch()
            (trial,) = phase_trials(experiment)
            if (sdk_worker and not shutdown_sdk) or (terminated and not guardian_cancel):
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
                if shutdown_sdk:
                    assert trial.state.name == "RUNNING"
                    assert TRIAL_OUTCOME_ATTR not in trial.user_attrs
                    assert trial.user_attrs[RETURN_CODE_ATTR] == 0
                    assert trial.user_attrs[DURATION_ATTR] > 0
                    assert (_attempts_dir(experiment) / f"{attempt}.json").is_file()
                else:
                    assert lifecycle.return_code == (
                        -signal.SIGTERM if guardian_cancel else return_code
                    )
                    assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds > 0
            assert (directory / "r.json").exists()
            if terminated and not guardian_cancel:
                terminate_group = reaper._terminate_process_group

                def terminate_and_wait_for_guardian(
                    pgid: int,
                    *,
                    grace_seconds: float,
                ) -> bool:
                    """Expose a handled SIGTERM's real zero exit before recovery reads evidence.

                    :param int pgid: Trainer group selected for termination.
                    :param float grace_seconds: SIGTERM grace before escalation.
                    :return bool: Confirmed cleanup after the guardian records the trainer exit.
                    """
                    assert terminate_group(pgid, grace_seconds=grace_seconds)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        exited = read_attempt_lifecycle(directory, expected_attempt_id=attempt)
                        if exited is not None and exited.state == "exited":
                            break
                        time.sleep(0.02)
                    assert exited is not None and exited.state == "exited"
                    assert exited.return_code == 0
                    assert exited.cleanup_confirmed
                    assert exited.duration_seconds is not None and exited.duration_seconds > 0
                    assert (directory / "r.json").read_text() == '{"x": 0.25}'
                    return True

                monkeypatch.setattr(
                    reaper, "_terminate_process_group", terminate_and_wait_for_guardian
                )
            if recovery_stage.endswith("snapshot"):
                snapshot = _generation_config_snapshot_path(
                    _experiment_dir(experiment), "orphan-generation"
                )
                snapshot_bytes = snapshot.read_bytes()
                if recovery_stage.endswith("missing-snapshot"):
                    snapshot.unlink()
                elif recovery_stage.endswith("altered-snapshot"):
                    saved_config = yaml.safe_load(snapshot_bytes)
                    saved_config["metric"]["max"] = 0.5
                    snapshot.write_text(yaml.safe_dump(saved_config))
                else:
                    snapshot.write_text("invalid: [unterminated\n")
                with pytest.raises(PhaseSweepError, match="configuration snapshot") as refusal:
                    _reconcile_existing_studies(
                        experiment, cleanup_report=_PreflightCleanupReport()
                    )
                assert refusal.value.action is OperatorAction.RESTORE_TREE
                if recovery_stage.endswith("altered-snapshot"):
                    assert "snapshot hash mismatch" in str(refusal.value)
                argv = ["run", str(config)]
                if recovery_stage.startswith("mcp-"):
                    from tests.mcp_helpers import make_mcp_app, stage_dead_run, write_srv_catalog

                    _app, registry, store = make_mcp_app(write_srv_catalog(tmp_path, config))
                    stage_dead_run(
                        store,
                        "orphan-generation",
                        config,
                        "srv",
                        cleanup_uncertain=False,
                    )
                    argv = [
                        "mcp",
                        "recover-run",
                        "--state-dir",
                        str(registry.state_dir),
                        "--run-id",
                        "orphan-generation",
                        "--confirm",
                    ]
                with monkeypatch.context() as boundary:
                    assert invoke_cli_boundary(argv, boundary) == 1
                failure_text = capsys.readouterr().err
                assert "configuration snapshot" in failure_text
                assert "fix: " + OPERATOR_STEPS[OperatorAction.RESTORE_TREE] in failure_text
                assert "internal error" not in failure_text
                pending = phase_trials(experiment)[0]
                assert pending.state.name == "RUNNING"
                assert TRIAL_OUTCOME_ATTR not in pending.user_attrs
                assert launches.read_text() == "launch\n"
                snapshot.write_bytes(snapshot_bytes)
            if recovery_stage in {"interrupt-extraction", "interrupt-extraction-signal"}:
                recovery_runner = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        runner_code,
                        str(config),
                        (
                            "recovery-signal"
                            if recovery_stage == "interrupt-extraction-signal"
                            else "recovery"
                        ),
                        str(worker),
                    ],
                    stdout=output,
                    stderr=output,
                )
                deadline = time.monotonic() + 15
                while not worker_started.exists() and time.monotonic() < deadline:
                    assert recovery_runner.poll() is None, (tmp_path / "runner.log").read_text()
                    time.sleep(0.02)
                assert worker_started.exists()
                pending = phase_trials(experiment)[0]
                assert pending.user_attrs[RETURN_CODE_ATTR] == 0
                assert pending.user_attrs[DURATION_ATTR] == lifecycle.duration_seconds
                if recovery_stage == "interrupt-extraction-signal":
                    recovery_runner.send_signal(signal.SIGINT)
                else:
                    recovery_runner.kill()
                recovery_runner.wait(timeout=5)
                if recovery_stage == "interrupt-extraction-signal":
                    assert recovery_runner.returncode == 128 + signal.SIGINT
                    pending = phase_trials(experiment)[0]
                    assert pending.state.name == "RUNNING"
                    assert TRIAL_OUTCOME_ATTR not in pending.user_attrs
                    assert (_attempts_dir(experiment) / f"{attempt}.json").is_file()
                orphan_identity = reaper.read_stale_process_identity(
                    directory, expected_attempt_id=attempt
                )
                lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=attempt)
                assert lifecycle is not None and lifecycle.state == (
                    "exited" if recovery_stage == "interrupt-extraction-signal" else "launching"
                )
                trial = pending
            if shutdown_sdk or recovery_stage == "interrupt-extraction-signal":
                log_text = (tmp_path / "runner.log").read_text()
                assert (
                    log_text.count("Trial 0 in phase p finished training; its result is retained.")
                    == 1
                )
                assert (
                    "The next phasesweep run with the same config will extract the result without retraining."
                    in log_text
                )
            if return_code == 0 and not terminated:
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
            if recovery_stage in {"interrupt-intent", "interrupt-termination"}:
                if recovery_stage == "interrupt-intent":
                    original_intent = attempts_module._record_recovery_interruption

                    def interrupt_intent(
                        study: optuna.Study, stale: optuna.trial.FrozenTrial
                    ) -> None:
                        """Stop recovery after its interruption intent commits, before any signal.

                        :param optuna.Study study: Owning phase study.
                        :param optuna.trial.FrozenTrial stale: Interrupted trainer trial.
                        :raises KeyboardInterrupt: After the durable interruption commits.
                        """
                        original_intent(study, stale)
                        raise KeyboardInterrupt

                    target = attempts_module
                    name = "_record_recovery_interruption"
                    replacement = interrupt_intent
                else:
                    original_terminate = reaper._terminate_process_group

                    def interrupt_termination(
                        pgid: int,
                        *,
                        grace_seconds: float,
                    ) -> bool:
                        """Stop recovery after SIGTERM cleanup and before trial settlement.

                        :param int pgid: Trainer group selected for termination.
                        :param float grace_seconds: SIGTERM grace before escalation.
                        :raises KeyboardInterrupt: After real group cleanup completes.
                        :return bool: Never returns because recovery is interrupted.
                        """
                        assert original_terminate(pgid, grace_seconds=grace_seconds)
                        raise KeyboardInterrupt

                    target = reaper
                    name = "_terminate_process_group"
                    replacement = interrupt_termination
                with monkeypatch.context() as interrupted:
                    interrupted.setattr(target, name, replacement)
                    with pytest.raises(KeyboardInterrupt):
                        _reconcile_existing_studies(
                            experiment, cleanup_report=_PreflightCleanupReport()
                        )
                pending = phase_trials(experiment)[0]
                assert pending.state.name == "RUNNING"
                assert _recovered_trial_state(pending, _outcome_sequence_counts([pending])) == (
                    "INTERRUPTED"
                )
                assert _cause(pending) == _STALE_CAUSE
                assert reaper._process_group_alive(orphan_identity.pgid) is (
                    recovery_stage == "interrupt-intent"
                )
            if recovery_stage in {"mcp-terminate", "mcp-finished"}:
                from tests.mcp_helpers import make_mcp_app, stage_dead_run, write_srv_catalog
                from tests.recovery_helpers import recover_run_cli

                _app, registry, store = make_mcp_app(write_srv_catalog(tmp_path, config))
                stage_dead_run(
                    store,
                    "orphan-generation",
                    config,
                    "srv",
                    cleanup_uncertain=False,
                )
                recovered_run = recover_run_cli(
                    registry.state_dir, "orphan-generation", confirm=True
                )
                assert recovered_run.exit_code == 0, recovered_run.output
                expected_kept = 0 if terminated else 1
                assert f"kept {expected_kept} completed trial result(s)" in recovered_run.output
                if expected_kept:
                    assert "Kept completed trial result: phase='p' trial=0 value=1" in caplog.text
            else:
                cleanup_report = _PreflightCleanupReport()
                _reconcile_existing_studies(experiment, cleanup_report=cleanup_report)
                assert cleanup_report.retained_results == int(return_code == 0 and not terminated)
            (recovered,) = phase_trials(experiment)
            assert recovered.state.name == (
                "COMPLETE" if return_code == 0 and not terminated else "FAIL"
            )
            if return_code == 0 and not terminated:
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
            if terminated:
                lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=attempt)
                assert lifecycle is not None and lifecycle.return_code == (
                    -signal.SIGTERM if guardian_cancel else 0
                )
                assert recovered.value is None
                release.touch()
                assert run_experiment(experiment)["p"].metric == 1.0
                rerun = phase_trials(experiment)[1]
                assert rerun.user_attrs[RERUN_OF_ATTR] == 0
                assert launches.read_text() == "launch\nlaunch\n"
        finally:
            release.touch()
            if runner.poll() is None:
                runner.kill()
            runner.wait(timeout=5)
            if recovery_runner is not None:
                if recovery_runner.poll() is None:
                    recovery_runner.kill()
                recovery_runner.wait(timeout=5)
            if orphan_identity is not None:
                if (directory / reaper.PROCESS_IDENTITY_FILE).exists():
                    orphan_identity = reaper.read_stale_process_identity(
                        directory, expected_attempt_id=attempt
                    )
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
