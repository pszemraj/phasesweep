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
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml
from optuna.distributions import IntDistribution

import phasesweep.engine.attempts as attempts_module
import phasesweep.runtime.reaper as reaper
from phasesweep import run_experiment
from phasesweep.config.models import Experiment
from phasesweep.engine import phase as phase_module
from phasesweep.engine.attempts import (
    _AttemptRecord,
    _fail_stale_running_trial,
    _PreflightCleanupReport,
    _record_stale_trial_outcome,
)
from phasesweep.engine.errors import (
    TRACKER_UNREADABLE_STEP,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrackingTargetUnavailableError,
    TrialEvidenceMissingError,
)
from phasesweep.engine.fingerprints import _fresh_identity, _stamp_identity
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
    _last_successful_generation_path,
    _phase_dir,
    _trial_dir_for,
)
from phasesweep.engine.preview import _preview_ledger
from phasesweep.engine.read import read_trial_tables
from phasesweep.engine.run import TerminalReport
from phasesweep.engine.selection import NoFeasibleTrialError, PhaseAbortRefusedError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    FEASIBLE_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    _parsed_trial_outcome,
    _recovered_trial_state,
    _trial_attempt_dir,
    _trial_interrupted,
)
from phasesweep.engine.study_policy import (
    _load_phase_policy_state,
    _record_reruns,
    _rerun_sources,
)
from phasesweep.engine.workers import _await_worker_trials, _WorkerTrials
from phasesweep.errors import OPERATOR_STEPS, OperatorAction, PhaseSweepError
from phasesweep.evidence.wandb import WANDB_SDK_MISSING
from phasesweep.runtime.json import JsonObject, JsonValue, json_object
from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle
from phasesweep.runtime.reaper import read_boot_id
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownLaunchRefusedError
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    phase_study,
    phase_trials,
    trial_outcome,
    use_guardian_timing,
    write_experiment_config,
    write_param_echo_trainer,
    write_trainer,
)
from tests.recovery_helpers import fabricate_stale_trial, write_trial_identity
from tests.tracking_helpers import SwitchableWandb, switchable_wandb, wandb_objective_experiment

_STALE_CAUSE = "stale RUNNING trial recovered after its orchestrator stopped"


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


def test_the_outcome_parser_rejects_an_unknown_label_and_another_schema_version() -> None:
    """A label outside the outcome vocabulary, such as ``cancelled``, and any other schema version parse as nothing."""
    assert _parsed_trial_outcome(trial_outcome(1, "cancelled")) is None
    previous = trial_outcome(1, "interrupted")
    previous["schema_version"] = TRIAL_OUTCOME_SCHEMA_VERSION - 1
    assert _parsed_trial_outcome(previous) is None


@pytest.mark.parametrize("metric", [None, True, "1", float("inf"), float("nan")])
def test_outcome_parser_requires_a_finite_numeric_success_metric(metric: JsonValue) -> None:
    """A successful outcome must carry a finite numeric measurement."""
    payload = trial_outcome(1, "success", metric=metric)
    assert _parsed_trial_outcome(payload) is None
    del payload["metric"]
    assert _parsed_trial_outcome(payload) is None


def test_outcome_parser_distinguishes_measured_and_unmeasured_failure() -> None:
    """Zero is a valid measurement; failures, fatal errors, and interruptions can lack one."""
    assert _parsed_trial_outcome(trial_outcome(1, "success", metric=0.0)) is not None
    measured = _parsed_trial_outcome(trial_outcome(2, "failure", metric=0.0))
    assert measured is not None and measured.metric == 0.0
    for outcome in ("failure", "fatal", "interrupted"):
        parsed = _parsed_trial_outcome(trial_outcome(3, outcome))
        assert parsed is not None and parsed.metric is None
    for outcome in ("fatal", "interrupted"):
        assert _parsed_trial_outcome(trial_outcome(3, outcome, metric=1.0)) is None


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
    "recorded",
    [
        pytest.param(None, id="no-record"),
        pytest.param(
            trial_outcome(3, "interrupted", cause="orchestrator interrupted by SIGINT"),
            id="signal",
        ),
        pytest.param(trial_outcome(3, "failure", cause="exit 1"), id="failure"),
        pytest.param(
            trial_outcome(3, "failure", cause="exit 1", abort=_abort(3)),
            id="failure-with-abort",
        ),
        pytest.param(
            trial_outcome(3, "fatal", cause="cleanup unconfirmed", policy="unsafe_process_cleanup"),
            id="fatal",
        ),
    ],
)
def test_stale_recovery_records_interrupted_unless_the_objective_recorded_a_failure(
    recorded: JsonObject | None,
) -> None:
    """Settlement keeps the exact recorded verdict or records an interruption for an unknown outcome."""
    study = optuna.create_study()
    _finished(study, 1, "failure")
    _finished(study, 2, "failure")
    trial = _running_trial(study, deepcopy(recorded))
    predicted = _recovered_trial_state(trial, _parsed_trial_outcome(recorded))
    expected = (
        recorded if recorded is not None else trial_outcome(3, "interrupted", cause=_STALE_CAUSE)
    )

    assert _fail_stale_running_trial(study, trial, where="could not settle trial")

    recovered = study.trials[trial.number]
    interrupted = expected["outcome"] == "interrupted"
    assert predicted == ("INTERRUPTED" if interrupted else "FAIL")
    assert recovered.state == optuna.trial.TrialState.FAIL
    assert recovered.user_attrs[TRIAL_OUTCOME_ATTR] == expected
    assert _trial_interrupted(recovered) is interrupted
    policy = _load_phase_policy_state(study)
    assert policy.consecutive_failures == (2 if interrupted else 3)
    assert policy.abort == expected.get(TRIAL_OUTCOME_ABORT_KEY)


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


@pytest.mark.parametrize("holder", ["peer", "itself"])
def test_recovery_refuses_duplicate_sequences_without_erasing_outcomes(holder: str) -> None:
    """Disputed completion order leaves both saved measurements and aborts intact."""
    phase = make_experiment(max_consecutive_failures=2).phases[0]
    study = optuna.create_study(study_name="example::p")
    _finished(study, 1, "failure")
    failed = _running_trial(study, trial_outcome(2, "failure", metric=3.0, abort=_abort(2)))
    succeeded = _running_trial(study, trial_outcome(2, "success", metric=4.0))
    pending = _running_trial(study, None) if holder == "peer" else failed
    before = [trial.user_attrs for trial in study.get_trials()]
    with pytest.raises(
        StudySchemaMismatchError,
        match="study 'example::p', trials 1 and 2 both use completion sequence 2",
    ):
        _record_stale_trial_outcome(study, pending, "failure", cause="crashed", policy_phase=phase)
    assert [trial.user_attrs for trial in study.get_trials()] == before
    assert all(trial.state == optuna.trial.TrialState.RUNNING for trial in (failed, succeeded))


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
    study, _trial_dir, _number, attempt_dir = fabricate_stale_trial(
        exp, "p", attempt_id="killed-attempt"
    )
    _stamp_identity(study, _fresh_identity(exp, exp.phases[0], {}))
    write_attempt_lifecycle(attempt_dir, attempt_id="killed-attempt", state="allocated")

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

    assert "p  3 trials to run, 1 of them a rerun, 1 at a time" in lines
    assert f"p  trial 1  RERUN of trial 0  reason: {_STALE_CAUSE}" in lines
    (rerun_line,) = (line for line in lines if line.startswith("p  trial 1  COMPLETE  "))
    assert "  1/3  " in rerun_line and rerun_line.endswith("  rerun of 0")
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


_SIGINT_CAUSE = "orchestrator interrupted by SIGINT"


def _cause(trial: optuna.trial.FrozenTrial) -> str | None:
    """Return the cause a trial's outcome record carries.

    :param optuna.trial.FrozenTrial trial: Trial whose record is read.
    :return str | None: The recorded cause, or ``None`` without a record.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    return None if parsed is None else parsed.cause


@pytest.mark.parametrize("serviced", [True, False], ids=["shutdown", "no-shutdown"])
def test_a_trainer_that_fails_while_a_shutdown_is_serviced_keeps_its_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serviced: bool
) -> None:
    """A trainer that crashed by itself keeps its failure and stderr reason though a shutdown is serviced after it."""
    launched = tmp_path / "launched"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys
        from pathlib import Path
        Path({str(launched)!r}).touch()
        print("CUDA error: out of memory", file=sys.stderr)
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

    with pytest.raises(NoFeasibleTrialError, match="1 consecutive failed/infeasible trials"):
        run_experiment(exp)

    (trial,) = phase_trials(exp)
    study = phase_study(exp)
    assert launched.exists()
    assert _trial_count_state(trial) == "FAIL"
    assert _cause(trial) == "non-zero exit code 1; stderr: CUDA error: out of memory"
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
    with (_phase_dir(exp, "p") / "trials.csv").open(newline="") as handle:
        assert [row[f"user_attr:{FAILURE_REASON_ATTR}"] for row in csv.DictReader(handle)] == [
            _SIGINT_CAUSE,
            _SIGINT_CAUSE,
        ]
    (table,) = read_trial_tables(exp).phases
    assert [row["reason"] for row in table.rows] == [_SIGINT_CAUSE, _SIGINT_CAUSE]
    assert "the trainer exited with code" not in caplog.text
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
    use_guardian_timing(
        monkeypatch,
        tmp_path,
        settle_seconds=0.2,
        term_grace_seconds=0.2,
        patch=f"""
        import time
        from pathlib import Path
        original = guardian.write_lifecycle_record
        def fail_intent(path: Path, *, attempt_id: str, state: str, ended_by: str | None, return_code: int | None, cleanup_confirmed: bool | None, duration_seconds: float | None) -> None:
            if state == "terminating" and {intent_write_fails!r}:
                Path({str(failed)!r}).touch()
                raise OSError("intent write unavailable")
            if state == "exited" and {trainer_completed!r}:
                Path({str(failed)!r}).touch()
                while not Path({str(publish)!r}).exists():
                    time.sleep(0.01)
            original(path, attempt_id=attempt_id, state=state, ended_by=ended_by, return_code=return_code, cleanup_confirmed=cleanup_confirmed, duration_seconds=duration_seconds)
        guardian.write_lifecycle_record = fail_intent
        if not {procfs_available!r}:
            def no_procfs(path):
                raise FileNotFoundError(path)
            guardian.os.scandir = no_procfs
        """,
    )
    process.write_attempt_lifecycle(tmp_path, attempt_id="held-lease", state="launching")
    lease = tmp_path / "gpu.lock"
    with lease.open("w") as held, (tmp_path / "guardian.log").open("w") as output:
        fcntl.flock(held, fcntl.LOCK_EX)
        guardian, pgid, ack_write, status_read = process._spawn_blocked_supervisor(
            stdout=output,
            stderr=output,
            gpu_lease_fds=(held.fileno(),),
            lifecycle_args=(str(tmp_path), "held-lease", str(time.monotonic())),
            label="trial",
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
        assert (lifecycle.ended_by, lifecycle.return_code) == (
            ("cancelled", None) if cancelled else ("trainer", 0)
        )
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
    direct_recovery = recovery_stage in {"interrupt-intent", "interrupt-termination"}
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
        from phasesweep.engine.paths import _attempt_dir
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        args, _ = parser.parse_known_args()
        def terminate(*_):
            print("stopping on SIGTERM", file=sys.stderr, flush=True)
            if {guardian_cancel!r}:
                attempt_dir = _attempt_dir(
                    Path(args.out).parent.parent, os.environ["PHASESWEEP_ATTEMPT_ID"]
                )
                lifecycle = attempt_dir / "attempt_lifecycle.json"
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
    tracking_dir = session.attempt_tracking_dir(
        ctx.trial_dir.parent.parent, ctx.generation_id, ctx.attempt_id
    )
    session.write_record(tracking_dir / session.BINDING_FILE, binding)
    session.write_record(
        tracking_dir / session.EXPECTED_BINDING_FILE,
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
            attempt_dir = _trial_attempt_dir(trial)
            assert attempt_dir is not None
            attempt = trial.user_attrs["phasesweep_attempt_id"]
            orphan_identity = reaper.read_stale_process_identity(
                attempt_dir, expected_attempt_id=attempt
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
                assert (intent["ended_by"], intent["return_code"]) == (
                    "timed_out" if recovery_stage == "timeout-trainer" else "cancelled",
                    None,
                )
                assert intent["cleanup_confirmed"] is False
            if not terminated:
                release.touch()
            (trial,) = phase_trials(experiment)
            if (sdk_worker and not shutdown_sdk) or (terminated and not guardian_cancel):
                lifecycle = read_attempt_lifecycle(attempt_dir, expected_attempt_id=attempt)
                assert lifecycle is not None and lifecycle.state == "launching"
                assert reaper._process_group_alive(orphan_identity.pgid)
            else:
                while time.monotonic() < deadline:
                    lifecycle = read_attempt_lifecycle(attempt_dir, expected_attempt_id=attempt)
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
                    assert lifecycle.return_code == (None if guardian_cancel else return_code)
                    assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds > 0
            assert (directory / "r.json").exists()
            if direct_recovery:
                # Only the unguarded fallback records its interruption in Optuna before
                # it signals. These cases exercise losing the recovery process there.
                guardian_pid = orphan_identity.guardian_pid
                os.kill(guardian_pid, signal.SIGKILL)
                stop_deadline = time.monotonic() + 5
                while reaper.is_same_live_process(guardian_pid, orphan_identity.guardian_starttime):
                    assert time.monotonic() < stop_deadline, "guardian still owns cleanup"
                    time.sleep(0.02)
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
                    attempt_dir, expected_attempt_id=attempt
                )
                lifecycle = read_attempt_lifecycle(attempt_dir, expected_attempt_id=attempt)
                assert lifecycle is not None and lifecycle.state == (
                    "exited" if recovery_stage == "interrupt-extraction-signal" else "launching"
                )
                trial = pending
            if shutdown_sdk or recovery_stage == "interrupt-extraction-signal":
                log_text = (tmp_path / "runner.log").read_text()
                assert (
                    log_text.count(
                        "Trial 0 in phase p finished training; its result is retained. The "
                        "next run extracts it without retraining."
                    )
                    == 1
                )
            if return_code == 0 and not terminated:
                assert (
                    _recovered_trial_state(
                        trial, _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
                    )
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
                pending = phase_trials(experiment)[0]
                assert pending.state.name == "RUNNING"
                assert pending.user_attrs[TRIAL_OUTCOME_ATTR]["metric"] == 1.0
                saved_outcome = json_object(pending.user_attrs[TRIAL_OUTCOME_ATTR])
                if recovery_stage == "finished":
                    objective = attempt_dir / "stdout.log"
                    original = objective.read_bytes()
                    objective.unlink()
                    with pytest.raises(TrialEvidenceMissingError, match="missing or unreadable"):
                        run_experiment(experiment)
                    objective.write_bytes(original.replace(b"x=1.0", b"x=0.1"))
                    with pytest.raises(
                        TrialEvidenceMissingError,
                        match="now hashes to .*: its bytes have changed",
                    ) as refused:
                        run_experiment(experiment)
                    assert refused.value.remediation is not None
                    assert "restore the files" in refused.value.remediation
                    assert "set a new `experiment` name" in refused.value.remediation
                    unchanged = phase_trials(experiment)[0]
                    assert unchanged.state.name == "RUNNING"
                    assert unchanged.user_attrs == pending.user_attrs
                    assert not _last_successful_generation_path(experiment).exists()
                    objective.write_bytes(original)
                if not feasible:
                    interrupted_study = phase_study(experiment)
                    pending = interrupted_study.get_trials()[0]
                    parsed_pending = _parsed_trial_outcome(pending.user_attrs[TRIAL_OUTCOME_ATTR])
                    assert parsed_pending is not None
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
                assert _recovered_trial_state(
                    pending, _parsed_trial_outcome(pending.user_attrs.get(TRIAL_OUTCOME_ATTR))
                ) == ("INTERRUPTED")
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
                assert "terminated 0 live process group(s)" in recovered_run.output
                expected_kept = 0 if terminated else 1
                assert f"kept {expected_kept} completed trial result(s)" in recovered_run.output
                if expected_kept:
                    assert any(
                        line.endswith("  recovered") and "p  trial 0  COMPLETE  " in line
                        for line in caplog.text.splitlines()
                    )
            else:
                cleanup_report = _PreflightCleanupReport()
                _reconcile_existing_studies(experiment, cleanup_report=cleanup_report)
                assert cleanup_report.retained_results == int(return_code == 0 and not terminated)
            (recovered,) = phase_trials(experiment)
            with (_phase_dir(experiment, "p") / "trials.csv").open(newline="") as handle:
                (row,) = csv.DictReader(handle)
            assert row["state"] == _trial_count_state(recovered)
            assert recovered.state.name == (
                "COMPLETE" if return_code == 0 and not terminated else "FAIL"
            )
            if return_code == 0 and not terminated:
                assert recovered.value == 1.0
                assert recovered.user_attrs[TRIAL_OUTCOME_ATTR] == saved_outcome
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
            elif recovery_stage in {"finished", "timeout-trainer"}:
                # A crash or a per-trial timeout is the trainer's own failure, kept though
                # the runner died before recording it.
                assert not _trial_interrupted(recovered)
                # A timeout reads as the live one does, without the trainer's stderr.
                assert (
                    _cause(recovered) == "timeout after the phase's timeout_seconds_per_trial"
                    if recovery_stage == "timeout-trainer"
                    else (_cause(recovered) or "").startswith("non-zero exit code 7")
                )
            else:
                assert _trial_interrupted(recovered)
            assert launches.read_text() == "launch\n"
            assert not reaper._process_group_alive(orphan_identity.pgid)
            if terminated:
                lifecycle = read_attempt_lifecycle(attempt_dir, expected_attempt_id=attempt)
                assert lifecycle is not None and lifecycle.return_code is None
                assert lifecycle.ended_by == (
                    "timed_out" if recovery_stage == "timeout-trainer" else "cancelled"
                )
                assert recovered.value is None
            if terminated and recovery_stage != "timeout-trainer":
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
                if (attempt_dir / reaper.PROCESS_IDENTITY_FILE).exists():
                    orphan_identity = reaper.read_stale_process_identity(
                        attempt_dir, expected_attempt_id=attempt
                    )
                reaper.cleanup_stale_trial_process(orphan_identity, grace_seconds=0.2)


def _assert_evidence_pending(experiment: Experiment, trial: optuna.trial.FrozenTrial) -> None:
    """Assert that ``trial`` keeps its finished training, unrecorded, for a later extraction.

    :param Experiment experiment: Experiment owning the attempt registry.
    :param optuna.trial.FrozenTrial trial: Trial whose tracker gave no answer.
    """
    assert trial.state.name == "RUNNING"
    assert trial.user_attrs[RETURN_CODE_ATTR] == 0
    assert trial.user_attrs[DURATION_ATTR] > 0
    assert FAILURE_REASON_ATTR not in trial.user_attrs
    assert TRIAL_OUTCOME_ATTR not in trial.user_attrs
    assert (_attempts_dir(experiment) / f"{trial.user_attrs[ATTEMPT_ID_ATTR]}.json").is_file()
    assert _recovered_trial_state(trial, None) == "EVIDENCE_PENDING"


def _assert_restored_tracker_keeps_training(
    experiment: Experiment,
    tracker: SwitchableWandb,
    launches: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run again with the tracker answering: every pending result is kept and nothing retrains.

    :param Experiment experiment: Experiment whose trials are all pending.
    :param SwitchableWandb tracker: The fake W&B server, switched back up here.
    :param Path launches: Trainer launch log, one line per launch.
    :param pytest.LogCaptureFixture caplog: Log capture holding the recovery lines.
    """
    pending = phase_trials(experiment)
    tracker.switch("up")
    assert run_experiment(experiment)["p"].metric == 1.0
    recovered = phase_trials(experiment)
    assert [trial.state.name for trial in recovered] == ["COMPLETE"] * len(pending)
    for trial in pending:
        assert any(
            line.endswith("  recovered") and f"p  trial {trial.number}  COMPLETE  " in line
            for line in caplog.text.splitlines()
        )
    assert launches.read_text().splitlines() == ["launch"] * len(pending)


@pytest.mark.integration
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_unreadable_tracker_leaves_finished_training_pending_until_it_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    wandb_worker_sdk: Callable[[str], None],
    n_jobs: int,
) -> None:
    """A tracker that gives no answer stops the run with each finished trial pending; its next answer is kept.

    The run's reconciliation learns of the outage from the live stop, so it reads no tracker,
    and its pending evidence names only attempts. Until the tracker answers, the run cannot
    know whether new trainers are needed and leaves their launch checks to a later pass.
    """
    import phasesweep.engine.guards as guards_module

    caplog.set_level(logging.INFO)
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, launches = wandb_objective_experiment(tmp_path, n_trials=n_jobs, n_jobs=n_jobs)
    trainer_cwd = tmp_path / "trainer-cwd"
    trainer_cwd.mkdir()
    experiment = experiment.model_copy(
        update={"execution": experiment.execution.model_copy(update={"cwd": str(trainer_cwd)})}
    )
    reconciliations: list[_PreflightCleanupReport] = []
    reconcile = guards_module._reconcile_existing_studies

    def recording_reconcile(
        *args: object, cleanup_report: _PreflightCleanupReport, **kwargs: object
    ):
        reconciliations.append(cleanup_report)
        return reconcile(*args, cleanup_report=cleanup_report, **kwargs)

    monkeypatch.setattr(guards_module, "_reconcile_existing_studies", recording_reconcile)
    tracker.switch("denied")
    with pytest.raises(TrackingTargetUnavailableError) as refused:
        run_experiment(experiment)
    assert refused.value.remediation == TRACKER_UNREADABLE_STEP
    trials = phase_trials(experiment)
    assert len(trials) == n_jobs
    for trial in trials:
        _assert_evidence_pending(experiment, trial)
        retained = f"Trial {trial.number} in phase p finished training; its result is retained."
        assert caplog.text.count(retained) == 1
    # Each trial read the tracker once; reconciliation did not spend another read.
    assert tracker.lookups("denied") == n_jobs
    (reconciliation,) = reconciliations
    assert set(reconciliation.unreadable_evidence) == {
        trial.user_attrs[ATTEMPT_ID_ATTR] for trial in trials
    }
    assert reconciliation.unreadable_tracker == str(refused.value)
    trainer_cwd.rmdir()
    with pytest.raises(TrackingTargetUnavailableError, match="tracker could not be read"):
        run_experiment(experiment)
    _assert_restored_tracker_keeps_training(experiment, tracker, launches, caplog)
    # Only a failed run reconciles.
    assert len(reconciliations) == 1


@pytest.mark.integration
def test_a_pending_trial_whose_tracker_sdk_is_missing_names_installing_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A recovery read that finds the tracker SDK missing says the tracker could not be read and gives a step that installs it."""
    caplog.set_level(logging.INFO)
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, launches = wandb_objective_experiment(tmp_path, n_trials=1)
    tracker.switch("denied")
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(experiment)
    wandb_worker_sdk("raise ImportError('No module named wandb')")
    with pytest.raises(TrackingTargetUnavailableError) as refused:
        run_experiment(experiment)
    assert str(refused.value) == (
        "1 completed trial(s) finished training, but the tracker could not be read: "
        f"metric extractor: {WANDB_SDK_MISSING}"
    )
    assert refused.value.remediation == (
        "Restore access to the tracker or install its SDK, then run the same config again; the "
        "finished training is kept. If the tracker is gone for good, set a new `experiment` name "
        "to start a fresh study."
    )
    assert "its tracker could not be read: metric extractor: W&B evidence requires" in caplog.text
    (trial,) = phase_trials(experiment)
    _assert_evidence_pending(experiment, trial)
    # With the SDK back, the next run keeps the training.
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    _assert_restored_tracker_keeps_training(experiment, tracker, launches, caplog)


@pytest.mark.integration
@pytest.mark.parametrize("path", ["live", "recovered"])
def test_finished_training_whose_tracker_answers_a_failure_fails_once(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None], path: str
) -> None:
    """A tracker's failure verdict on finished training records a failure with its reason, whether the trial was live or pending, and nothing trains again."""
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, launches = wandb_objective_experiment(tmp_path, n_trials=1)
    if path == "recovered":
        tracker.switch("denied")
        with pytest.raises(TrackingTargetUnavailableError):
            run_experiment(experiment)
        (pending,) = phase_trials(experiment)
        _assert_evidence_pending(experiment, pending)
    tracker.switch("crashed")
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)
    (trial,) = phase_trials(experiment)
    assert trial.state.name == "FAIL"
    assert not _trial_interrupted(trial)
    reason = trial.user_attrs[FAILURE_REASON_ATTR]
    assert "crashed" in reason
    outcome = trial.user_attrs[TRIAL_OUTCOME_ATTR]
    assert (outcome["outcome"], outcome["cause"]) == ("failure", reason)
    assert launches.read_text().splitlines() == ["launch"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "failure_stage",
    ["before-outcome", "after-gate-outcome", "partial-before-failure"],
)
def test_recovered_evidence_failure_survives_a_journal_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Keep finished training and its failure-limit abort across a failed recovery write."""
    from phasesweep.evidence.models import RequiredFileGate
    from phasesweep.evidence.wandb import WandbRunTerminalError, WandbSetupError
    from phasesweep.runtime.time import utc_now_iso

    experiment, launches = wandb_objective_experiment(
        tmp_path, n_trials=1, max_consecutive_failures=1
    )
    gate_failure = failure_stage == "after-gate-outcome"
    partial_success = failure_stage == "partial-before-failure"
    if gate_failure:
        (phase,) = experiment.phases
        gate = RequiredFileGate(type="required_file", path="receipt")
        experiment = experiment.model_copy(
            update={"phases": [phase.model_copy(update={"gates": [gate]})]}
        )

    tracker_down = True
    reads = 0

    def summary(*, run_id: str, **_options: object) -> JsonObject:
        nonlocal reads
        reads += 1
        if tracker_down:
            raise WandbSetupError(run_id, "HTTP 503")
        if not gate_failure and not partial_success:
            raise WandbRunTerminalError(run_id, "crashed")
        return {"values": {"x": 4.0}, "present_keys": [], "retrieved_at": utc_now_iso()}

    monkeypatch.setattr("phasesweep.evidence.wandb.require_wandb_sdk", lambda: None)
    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", summary)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(experiment)
    tracker_down = False

    original_set = optuna.Trial.set_user_attr
    writes_failed = False
    failed_write = TRIAL_OUTCOME_ATTR if failure_stage == "before-outcome" else FAILURE_REASON_ATTR

    def fail_journal(trial: optuna.Trial, key: str, value: JsonValue) -> None:
        nonlocal writes_failed
        writes_failed = writes_failed or key == failed_write
        if writes_failed:
            raise OSError("journal write failed")
        original_set(trial, key, value)

    if partial_success:
        failed_write = TRIAL_OUTCOME_ATTR
        with monkeypatch.context() as journal:
            journal.setattr(optuna.Trial, "set_user_attr", fail_journal)
            with pytest.raises(StudyStorageUnavailableError, match="could not be committed"):
                run_experiment(experiment)
        (partial,) = phase_trials(experiment)
        assert OBJECTIVE_PROVENANCE_ATTR in partial.user_attrs
        assert TRIAL_OUTCOME_ATTR not in partial.user_attrs
        partial_success = False
        writes_failed = False
        failed_write = FAILURE_REASON_ATTR

    with monkeypatch.context() as journal:
        journal.setattr(optuna.Trial, "set_user_attr", fail_journal)
        with pytest.raises(StudyStorageUnavailableError, match="could not be committed"):
            run_experiment(experiment)

    (pending,) = phase_trials(experiment)
    assert FAILURE_REASON_ATTR not in pending.user_attrs
    if failed_write == FAILURE_REASON_ATTR:
        assert pending.state == optuna.trial.TrialState.FAIL
        assert OBJECTIVE_PROVENANCE_ATTR in pending.user_attrs
        assert TRIAL_OUTCOME_ATTR in pending.user_attrs
        assert pending.user_attrs[TRIAL_OUTCOME_ATTR].get("metric") is None
    else:
        assert pending.state == optuna.trial.TrialState.RUNNING
        assert TRIAL_OUTCOME_ATTR not in pending.user_attrs

    reads_before_retry = reads
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
        run_experiment(experiment)
    (failed,) = phase_trials(experiment)
    outcome = failed.user_attrs[TRIAL_OUTCOME_ATTR]
    assert failed.state == optuna.trial.TrialState.FAIL
    assert outcome["outcome"] == "failure"
    assert outcome.get("metric") is None
    assert ("receipt is missing" if gate_failure else "crashed") in outcome["cause"]
    assert outcome[TRIAL_OUTCOME_ABORT_KEY]["consecutive_failures"] == 1
    # A committed rejection never re-reads its evidence.
    assert reads == reads_before_retry + (0 if failed_write == FAILURE_REASON_ATTR else 1)
    assert launches.read_text().splitlines() == ["launch"]
    assert not _last_successful_generation_path(experiment).exists()


@pytest.mark.integration
@pytest.mark.parametrize("path", ["live", "recovered"])
@pytest.mark.parametrize("verdict", ["failure", "infeasible"])
def test_a_recovered_verdict_trips_the_failure_limit_as_a_live_one_does(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None], verdict: str, path: str
) -> None:
    """A failure or infeasible value the tracker reports for finished training counts toward max_consecutive_failures whether the trial was live or pending: its outcome carries the same abort, and the phase publishes nothing."""
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    # The second trial's verdict trips the limit: a crashed run after a success, or a second
    # value above the bound.
    final_mode = "crashed" if verdict == "failure" else "up"
    limit = 1 if verdict == "failure" else 2
    experiment, launches = wandb_objective_experiment(
        tmp_path,
        n_trials=2,
        switch_at_launch=(tracker, 2, "denied" if path == "recovered" else final_mode),
        max_consecutive_failures=limit,
        metric_max=None if verdict == "failure" else 0.5,
    )
    if path == "recovered":
        with pytest.raises(TrackingTargetUnavailableError):
            run_experiment(experiment)
        _assert_evidence_pending(experiment, phase_trials(experiment)[1])
        tracker.switch(final_mode)
    with pytest.raises(PhaseAbortRefusedError if path == "recovered" else NoFeasibleTrialError):
        run_experiment(experiment)
    first, second = phase_trials(experiment)
    assert second.state.name == ("FAIL" if verdict == "failure" else "COMPLETE")
    outcome = second.user_attrs[TRIAL_OUTCOME_ATTR]
    assert outcome["outcome"] == "failure"
    assert outcome.get("metric") == (second.value if verdict == "infeasible" else None)
    assert outcome[TRIAL_OUTCOME_ABORT_KEY] == {
        "schema_version": 1,
        "policy": "max_consecutive_failures",
        "completion_sequence": 2,
        "trial_target": 2,
        "cause": f"{limit} consecutive failed/infeasible trials reached "
        f"max_consecutive_failures={limit}.",
        "threshold": limit,
        "consecutive_failures": limit,
    }
    assert TRIAL_OUTCOME_ABORT_KEY not in first.user_attrs[TRIAL_OUTCOME_ATTR]
    assert not _last_successful_generation_path(experiment).exists()
    # Either way the next run refuses the recorded abort and trains nothing.
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
        run_experiment(experiment)
    assert launches.read_text().splitlines() == ["launch"] * 2


@pytest.mark.integration
def test_tracker_outage_preserves_a_parallel_peers_failure_limit_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer's definitive failure trips the limit during a tracker outage and blocks publication after recovery."""
    from phasesweep.evidence.wandb import WandbRunTerminalError, WandbSetupError
    from phasesweep.runtime.time import utc_now_iso

    experiment, launches = wandb_objective_experiment(
        tmp_path, n_trials=2, n_jobs=2, max_consecutive_failures=1
    )
    both_reading = threading.Barrier(2)
    launches_stopped = threading.Event()
    original_stop_launches = phase_module._PhaseExecution.stop_launches

    def stop_launches(run: phase_module._PhaseExecution) -> None:
        original_stop_launches(run)
        launches_stopped.set()

    def outage(*, run_id: str, **_options: object) -> JsonObject:
        both_reading.wait(10)
        if _trial_dir_for(experiment, experiment.phases[0].name, 0, attempt_id=run_id).is_dir():
            raise WandbSetupError(run_id, "HTTP 401")
        # The peer's verdict comes after the outage has already stopped new launches.
        assert launches_stopped.wait(10)
        raise WandbRunTerminalError(run_id, "crashed")

    monkeypatch.setattr(phase_module._PhaseExecution, "stop_launches", stop_launches)
    monkeypatch.setattr("phasesweep.evidence.wandb.require_wandb_sdk", lambda: None)
    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", outage)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(experiment)

    pending, failed = phase_trials(experiment)
    _assert_evidence_pending(experiment, pending)
    assert failed.state == optuna.trial.TrialState.FAIL
    outcome = failed.user_attrs[TRIAL_OUTCOME_ATTR]
    abort = outcome[TRIAL_OUTCOME_ABORT_KEY]
    assert abort["policy"] == "max_consecutive_failures"
    assert abort["completion_sequence"] == outcome["sequence"] == 1

    monkeypatch.setattr(
        "phasesweep.evidence.evaluation.poll_wandb_summary",
        lambda **_options: {
            "values": {"x": 1.0},
            "present_keys": [],
            "retrieved_at": utc_now_iso(),
        },
    )
    monkeypatch.setattr("phasesweep.evidence.evaluation.locate_wandb_position", lambda **_: None)
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
        run_experiment(experiment)

    settled, failed = phase_trials(experiment)
    assert settled.state == optuna.trial.TrialState.COMPLETE
    assert settled.value == 1.0
    assert settled.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] == 2
    assert failed.user_attrs[TRIAL_OUTCOME_ATTR][TRIAL_OUTCOME_ABORT_KEY] == abort
    assert not _last_successful_generation_path(experiment).exists()
    assert launches.read_text().splitlines() == ["launch"] * 2


@pytest.mark.parametrize("first", ["cleanup", "left-running"])
def test_unconfirmed_cleanup_outranks_a_parallel_trial_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    """Under n_jobs above 1, a worker's unconfirmed cleanup stops the run whichever worker records first.

    One worker leaves its trial RUNNING for an unreadable tracker and its stop reaches the
    optimize loop first; the other cannot confirm its trainer group gone. The leaked group
    may hold the GPU, so the cleanup error is what the run raises.
    """
    from phasesweep.engine.errors import tracker_unreadable
    from phasesweep.errors import UnsafeProcessCleanupError

    experiment = make_experiment(persistent=tmp_path, n_trials=3, n_jobs=2)
    cleanup_recorded = threading.Event()
    left_running_raised = threading.Event()

    def stubbed_objective(run: phase_module._PhaseExecution, trial: optuna.Trial) -> float:
        if trial.number == 0:
            if first == "cleanup":
                assert cleanup_recorded.wait(10)
            left_running = phase_module._TrialLeftRunning(tracker_unreadable("tracker down"))
            run.record_fatal(left_running)
            left_running_raised.set()
            raise left_running
        cleanup_error = UnsafeProcessCleanupError("trainer group may still be alive")
        if first == "cleanup":
            # The launch path records unconfirmed cleanup before it releases the lease.
            run.record_fatal(cleanup_error)
            cleanup_recorded.set()
        assert left_running_raised.wait(10)
        # The trial left RUNNING finishes first, so the optimize loop surfaces it.
        time.sleep(0.2)
        raise cleanup_error

    monkeypatch.setattr(phase_module, "_execute_objective", stubbed_objective)
    with pytest.raises(UnsafeProcessCleanupError, match="trainer group may still be alive"):
        run_experiment(experiment)
    states = {trial.number: trial.state.name for trial in phase_trials(experiment)}
    assert states[1] == "FAIL"


def _in_memory_wandb_sweep(
    tmp_path: Path, tracker: SwitchableWandb, *, n_trials: int, deny_at_launch: int
) -> tuple[Experiment, Path]:
    """Build a ``storage: null`` sweep whose W&B objective answers once each trainer exits.

    :param Path tmp_path: Directory holding the workdir, trainer and launch log.
    :param SwitchableWandb tracker: The fake W&B server the trainer switches.
    :param int n_trials: The phase's trial target.
    :param int deny_at_launch: Launch number at which the trainer switches W&B to ``denied``.
    :return tuple[Experiment, Path]: The experiment and its trainer launch log.
    """
    from phasesweep.config.models import Metric
    from phasesweep.evidence.models import WandbExtractor

    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        from pathlib import Path
        launches = Path({str(launches)!r})
        with launches.open("a") as log:
            log.write("launch\\n")
        if len(launches.read_text().splitlines()) == {deny_at_launch}:
            Path({str(tracker.mode)!r}).write_text("denied")
        """,
    )
    remote = WandbExtractor(
        type="wandb",
        base_url="https://example.test",
        entity="e",
        project="p",
        metric_key="x",
        poll_seconds=0.01,
        timeout_seconds=30,
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        metric=Metric(extractor=remote),
        n_trials=n_trials,
    )
    assert not experiment.persistent
    return experiment, launches


@pytest.mark.integration
def test_in_memory_run_fails_a_trial_whose_tracker_is_unreadable_and_promises_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """With storage: null nothing outlives the run, so the trial fails with the tracker's reason and no later read is promised."""
    caplog.set_level(logging.INFO)
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, launches = _in_memory_wandb_sweep(tmp_path, tracker, n_trials=1, deny_at_launch=1)
    with pytest.raises(NoFeasibleTrialError) as refused:
        run_experiment(experiment)
    assert not isinstance(refused.value, TrackingTargetUnavailableError)
    assert "trial 0  FAIL" in caplog.text
    assert "reason: metric extractor: W&B authentication/setup failed" in caplog.text
    assert "retained" not in caplog.text
    assert "kept" not in str(refused.value) + str(refused.value.remediation)
    # The next run starts afresh: nothing was kept, so the trainer runs again.
    tracker.switch("up")
    assert run_experiment(experiment)["p"].metric == 1.0
    assert launches.read_text().splitlines() == ["launch"] * 2


@pytest.mark.integration
def test_in_memory_sweep_keeps_its_finished_trials_when_the_tracker_turns_unreadable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """An in-memory sweep fails the trials its tracker cannot answer about and still publishes the rest."""
    caplog.set_level(logging.INFO)
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, launches = _in_memory_wandb_sweep(tmp_path, tracker, n_trials=6, deny_at_launch=5)
    winners = run_experiment(experiment)
    assert winners["p"].metric == 1.0
    assert launches.read_text().splitlines() == ["launch"] * 6
    assert _last_successful_generation_path(experiment).is_file()
    assert caplog.text.count("  COMPLETE  ") == 4
    assert caplog.text.count("reason: metric extractor: W&B authentication/setup failed") == 2


@pytest.mark.parametrize("storage", ["journal", "memory"])
def test_shutdown_during_extraction_promises_kept_training_only_with_persistent_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    storage: str,
) -> None:
    """Finished training a shutdown stops before extraction is kept only by a persistent study."""
    trainer = write_trainer(tmp_path / "trainer.py", "pass\n")
    experiment = make_experiment(
        persistent=tmp_path if storage == "journal" else None,
        workdir=tmp_path / "runs",
        trainer=trainer,
        n_trials=1,
    )

    def interrupted(**_kwargs: object) -> None:
        raise ShutdownLaunchRefusedError(signal.SIGTERM)

    monkeypatch.setattr(phase_module, "extract_trial_result", interrupted)
    with pytest.raises(PhaseSweepShutdown):
        run_experiment(experiment)
    retained = "Trial 0 in phase p finished training; its result is retained."
    lost = "Trial 0 in phase p finished training, but its result was not extracted"
    assert (retained in caplog.text) is (storage == "journal")
    assert (lost in caplog.text) is (storage == "memory")


_RUN_CONFIG = """
import sys
from phasesweep import run_experiment
from phasesweep.config.io import load_experiment
run_experiment(load_experiment(sys.argv[1]), generation_id="orphan-generation")
"""


@pytest.mark.integration
@pytest.mark.parametrize("recovery", ["run", "mcp-recover-run"])
def test_recovery_defers_a_killed_runs_finished_trial_while_its_tracker_is_unreadable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    wandb_worker_sdk: Callable[[str], None],
    recovery: str,
) -> None:
    """Recovery leaves the trial pending after one lookup of the unreadable tracker; the next answer is kept."""
    caplog.set_level(logging.INFO)
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, launches = wandb_objective_experiment(tmp_path, n_trials=1)
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    tracker.switch("hang")
    reader = None
    with (tmp_path / "runner.log").open("w") as output:
        runner = subprocess.Popen(
            [sys.executable, "-c", _RUN_CONFIG, str(config)], stdout=output, stderr=output
        )
    try:
        deadline = time.monotonic() + 30
        while not tracker.hung.exists() and time.monotonic() < deadline:
            assert runner.poll() is None, (tmp_path / "runner.log").read_text()
            time.sleep(0.02)
        assert tracker.hung.exists()
        (trial,) = phase_trials(experiment)
        attempt_dir = _trial_attempt_dir(trial)
        assert attempt_dir is not None
        reader = reaper.read_stale_process_identity(
            attempt_dir, expected_attempt_id=trial.user_attrs[ATTEMPT_ID_ATTR]
        )
        runner.kill()
        runner.wait(timeout=5)
        tracker.switch("denied")
        if recovery == "run":
            reports: list[TerminalReport] = []
            with pytest.raises(
                TrackingTargetUnavailableError, match="1 completed trial"
            ) as refused:
                run_experiment(experiment, terminal_callback=reports.append)
            assert refused.value.remediation == TRACKER_UNREADABLE_STEP
            (report,) = reports
            assert report.cleanup_confirmed
            assert report.failure_stage == "preflight"
        else:
            from tests.mcp_helpers import make_mcp_app, stage_dead_run, write_srv_catalog
            from tests.recovery_helpers import recover_run_cli

            _app, registry, store = make_mcp_app(write_srv_catalog(tmp_path, config))
            stage_dead_run(store, "orphan-generation", config, "srv", cleanup_uncertain=False)
            recovered_run = recover_run_cli(registry.state_dir, "orphan-generation", confirm=True)
            assert recovered_run.exit_code == 0, recovered_run.output
            assert "kept 0 completed trial result(s)" in recovered_run.output
            # The run's own refusal and recover-run end with the same step.
            assert "Left 1 completed trial(s) pending" in recovered_run.output
            assert f"). {TRACKER_UNREADABLE_STEP}\n" in recovered_run.output
        assert not reaper._process_group_alive(reader.pgid)
        (trial,) = phase_trials(experiment)
        _assert_evidence_pending(experiment, trial)
        assert tracker.lookups("denied") == 1
        _assert_restored_tracker_keeps_training(experiment, tracker, launches, caplog)
    finally:
        if runner.poll() is None:
            runner.kill()
        runner.wait(timeout=5)
        if reader is not None:
            reaper.cleanup_stale_trial_process(reader, grace_seconds=0.2)


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
    monkeypatch.setattr(reaper, "is_pid_alive", lambda pid: pid == 12345 and case == "live")
    monkeypatch.setattr(reaper, "read_proc_starttime", lambda _: 1)
    monkeypatch.setattr(reaper.os, "getpgid", lambda _: 12345)
    monkeypatch.setattr(reaper, "_process_group_exists", lambda _: case != "group_exited")
    monkeypatch.setattr(reaper, "_process_group_alive", lambda _: case != "group_exited")
    monkeypatch.setattr(
        reaper,
        "_read_proc_stat",
        lambda entry: (
            SimpleNamespace(pgrp=12345, starttime=2)
            if entry.name == "12345" and case == "reused_group"
            else None
        ),
    )
    signalled: list[int] = []
    monkeypatch.setattr(
        reaper, "_terminate_process_group", lambda pgid, **_: signalled.append(pgid) or True
    )
    record = _AttemptRecord("preview", tmp_path, tmp_path, "generation", "p", 0)
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
    assert (
        "wait for the trainer to exit successfully to keep its result" in caplog.text
    ) is should_signal
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
