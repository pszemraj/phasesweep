"""Terminal delivery preserves engine outcome and cleanup authority."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml
from optuna.trial import TrialState, create_trial

from phasesweep.config.models import Experiment
from phasesweep.engine import guards
from phasesweep.engine import phase as phase_module
from phasesweep.engine.errors import TrackingSyncRefusedError
from phasesweep.engine.generation import _recorded_generation
from phasesweep.engine.ledger import (
    claim_ledger,
    open_phase_study,
    read_phase_trials,
    validate_ledger,
)
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_dir,
    _generation_record_path,
    _trial_dir_for,
)
from phasesweep.engine.read import experiment_status, read_status
from phasesweep.engine.run import run_experiment
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    DURATION_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OVERRIDES_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _trial_interrupted,
    constraint_attr,
)
from phasesweep.errors import (
    PhaseSweepError,
    ProcessCleanupUncertainError,
    UnsafeProcessCleanupError,
)
from phasesweep.runtime import process
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import _publish
from phasesweep.tracking.session import (
    BINDING_FILE,
    INHERITS_FILE,
    DeliveryTally,
    read_record,
    tracking_root,
    write_record,
)
from tests.conftest import lose_record_writes, patch_path_method_failure, trial_outcome
from tests.tracking_helpers import (
    FakeWorkerProcess,
    RunnerKilled,
    delivery_record,
    killed_sweep,
    leak_tracking_worker,
    leave_cleanup_uncertain_attempt,
    mlflow_binding,
    mlflow_run,
    native_experiment,
    remote_mlflow_example,
)


@pytest.mark.parametrize("requested", ["FINISHED", "FAILED", "KILLED"])
@pytest.mark.parametrize("actual", ["RUNNING", "FINISHED", "FAILED", "KILLED"])
def test_publish_applies_the_owned_run_outcome(actual: str, requested: str) -> None:
    """The retained managed outcome overrides a different trainer status and retries idempotently."""
    pytest.importorskip("mlflow")
    binding = mlflow_binding()
    calls = []
    client = SimpleNamespace(
        get_run=lambda _: mlflow_run(binding, status=actual),
        log_batch=lambda *args, **kwargs: calls.append("batch"),
        log_artifact=lambda *args, **kwargs: calls.append("artifact"),
        set_terminated=lambda run_id, status: calls.append(status),
    )
    _publish(
        client,
        {
            "name": "generation",
            "binding": binding,
            "projection": publication._projection(metrics={}, params={}, tags={}, summary={}),
            "finish": requested,
        },
    )
    assert calls == ["batch", "artifact", *([requested] if actual != requested else [])]


def test_trainer_owned_run_is_never_terminated():
    pytest.importorskip("mlflow")
    binding = {**mlflow_binding(), "run_mode": "trainer"}
    client = SimpleNamespace(get_run=lambda _: mlflow_run(binding, status="RUNNING"))
    with pytest.raises(ValueError, match="Only runs PhaseSweep created"):
        _publish(client, {"binding": binding, "projection": {}, "finish": "FAILED"})


@pytest.mark.parametrize("outcome", ["completed", "uncertain"])
def test_managed_creation_before_launch_recovers_the_unlaunched_draw(
    tmp_path, monkeypatch, outcome
):
    """Creation workers never become trainer evidence, and an unlaunched draw is retried."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, sampler={"type": "random", "seed": 123})
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    created = {}
    search_generations = []

    def create_before_trainer(request, *, operation_id, directory, environment, deadline):
        binding = request.get("binding", {})
        if request["action"] == "find" and binding.get("attempt_id") == created.get("attempt_id"):
            search_generations.append(directory.parents[1].name)
            return {"found": True, "binding": created}
        if request["action"] == "create" and binding.get("phase") != "generation" and not created:
            created.update(respond(request))
            with Path(os.devnull).open("w") as sink:
                process.run_supervised(
                    "true",
                    env=dict(os.environ),
                    stdout=sink,
                    stderr=sink,
                    timeout=10,
                    trial_dir=directory,
                    attempt_id=operation_id,
                )
            if outcome == "uncertain":
                process.write_attempt_lifecycle(
                    directory,
                    attempt_id=operation_id,
                    state="terminating",
                    return_code=-15,
                    cleanup_confirmed=False,
                )
                raise UnsafeProcessCleanupError("creation worker cleanup is uncertain")
            raise RunnerKilled()
        return respond(
            request,
            operation_id=operation_id,
            directory=directory,
            environment=environment,
            deadline=deadline,
        )

    monkeypatch.setattr(session, "request_operation", create_before_trainer)
    with monkeypatch.context() as stopped:
        stopped.setattr(guards, "_reconcile_existing_studies", lambda *args, **kwargs: None)
        if outcome == "completed":
            stopped.setattr(phase_module, "_record_outcome", lambda *args, **kwargs: None)
        with pytest.raises(RunnerKilled if outcome == "completed" else UnsafeProcessCleanupError):
            run_experiment(experiment)
    (original,) = read_phase_trials(validate_ledger(experiment), "depth").trials
    assert original.state == TrialState.RUNNING
    trial_dir = Path(original.user_attrs[TRIAL_DIR_ATTR])
    lifecycle = process.read_attempt_lifecycle(trial_dir, expected_attempt_id=created["attempt_id"])
    assert lifecycle is not None and lifecycle.state == "allocated"
    assert experiment_status(experiment)["tracking_delivery"]["pending"] == 1

    reports = []
    run_experiment(experiment, terminal_callback=reports.append)
    original, retried = read_phase_trials(validate_ledger(experiment), "depth").trials
    assert _trial_interrupted(original)
    assert retried.state == TrialState.COMPLETE
    assert retried.params == original.params
    assert retried.user_attrs[RERUN_OF_ATTR] == original.number
    assert search_generations == [reports[0].generation_id]
    assert session.read_binding(trial_dir) == created
    assert _sent(uploads, [created["attempt_id"]]) == ["KILLED"]
    assert publication.delivery_status(experiment)["pending"] == 0


@pytest.mark.parametrize("appears_late", [True, False], ids=["late-run", "abandoned"])
def test_interrupted_summary_creation_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, appears_late: bool
) -> None:
    """Recovery binds an interrupted summary or clears its expired, unmatched intent."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    created: dict[str, str] = {}
    appeared = False

    def create_summary(request, **kwargs):
        if request["action"] == "create" and not created:
            assert request["binding"]["phase"] == "generation"
            created.update(respond(request))
            raise RunnerKilled()
        if request["action"] == "find":
            assert request["binding"]["generation_id"] == created["generation_id"]
            return {"found": True, "binding": created} if appeared else {"found": False}
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", create_summary)
    with pytest.raises(RunnerKilled):
        run_experiment(experiment)
    previous = created["generation_id"]
    assert experiment_status(experiment)["tracking_delivery"] == {"delivered": 0, "pending": 1}
    assert publication.sync_tracking(experiment, previous) == {"delivered": 0, "pending": 1}

    [pending] = session.pending_creation_records(experiment, previous)
    os.utime(pending, (0, 0))
    counts = {"delivered": 0, "pending": 0}
    if appears_late:
        appeared = True
        run_experiment(experiment)
        counts["delivered"] = 1
    else:
        assert publication.sync_tracking(experiment, previous) == counts

    expected = [(previous, "KILLED")] if appears_late else []
    assert [(gid, state) for gid, state in _summaries(uploads) if gid == previous] == expected
    assert publication.sync_tracking(experiment, previous) == counts
    assert experiment_status(experiment)["tracking_delivery"]["pending"] == 0
    assert not pending.exists()


def test_failed_trial_finishes_its_managed_run(tmp_path, monkeypatch):
    """A trial that never reaches evidence capture still ends its RUNNING run."""
    experiment = native_experiment(tmp_path).model_copy(
        update={
            "trial_command": shlex.join([sys.executable, "-c", "raise SystemExit(1)"])
            + " {config_path}"
        }
    )

    def prepare(experiment, *, phase, generation_id, attempt_id, directory, **kwargs):
        binding = {
            **mlflow_binding(),
            "phase": phase,
            "generation_id": generation_id,
            "attempt_id": attempt_id,
        }
        write_record(directory / "mlflow-expected.json", binding)
        write_record(directory / BINDING_FILE, binding)
        return binding

    requests = []
    monkeypatch.setattr("phasesweep.tracking.session.require_tracking_target", lambda *a, **k: None)
    monkeypatch.setattr("phasesweep.tracking.session.prepare_attempt", prepare)
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)
    assert requests and {r["finish"] for r in requests} == {"FAILED"}
    assert {r["binding"]["phase"] for r in requests} == {"depth"}


@pytest.mark.parametrize("trainer", ["accepted", "failed"])
@pytest.mark.parametrize(
    ("n_jobs", "error_class"),
    [
        (1, UnsafeProcessCleanupError),
        (2, UnsafeProcessCleanupError),
        (2, ProcessCleanupUncertainError),
    ],
    ids=["serial", "parallel", "parallel-unreadable-lifecycle"],
)
def test_unresolved_attempt_upload_worker_fails_the_run(
    tmp_path, monkeypatch, n_jobs, trainer, error_class
):
    """Optuna drops callback errors in its final threaded batch; the fatal channel must not."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["phases"][0].update(n_jobs=n_jobs, allow_no_gpu_isolation=True)
    if trainer == "failed":
        config["trial_command"] = (
            shlex.join([sys.executable, "-c", "raise SystemExit(1)"]) + " {config_path}"
        )
    experiment = Experiment.model_validate(config)
    error = error_class("upload worker still unresolved")
    uploaded_phases = []

    def unresolved(*args, **kwargs):
        if args[2]["name"].startswith("attempt-"):
            uploaded_phases.append(args[2]["binding"]["phase"])
            raise error
        return session.call_tracking(*args, **kwargs)

    monkeypatch.setattr(publication, "call_tracking", unresolved)
    reports = []
    with pytest.raises(error_class) as caught:
        run_experiment(experiment, terminal_callback=reports.append)
    assert caught.value is error
    [report] = reports
    assert report.primary_error is error
    assert not report.cleanup_confirmed
    assert report.winners is None
    assert read_status(experiment)["published_generation_id"] is None
    # The inheriting phase never launched.
    assert set(uploaded_phases) == {"depth"}


def _add_bound_attempt(
    experiment, study, phase, generation, attempt, number, *, trial, run_mode="managed"
):
    """Persist a fresh trial and its native binding in the same attempt directory."""
    directory = _trial_dir_for(
        experiment, phase, number, generation_id=generation, attempt_id=attempt
    )
    write_record(
        directory / BINDING_FILE,
        {**mlflow_binding(), "phase": phase, "attempt_id": attempt, "run_mode": run_mode},
    )
    trial.user_attrs.update({ATTEMPT_ID_ATTR: attempt, GENERATION_ID_ATTR: generation})
    study.add_trial(trial)
    return directory


def test_failed_generation_reports_each_unrecorded_attempt_by_its_trial(tmp_path, monkeypatch):
    """The Optuna outcome, never a missing record, decides what an unrecorded attempt reports."""
    experiment = native_experiment(tmp_path, storage=f"journal:///{tmp_path}/study.journal")
    generation = "a" * 32
    write_record(
        tracking_root(experiment) / generation / "generation-binding.json",
        {**mlflow_binding(), "phase": "generation", "attempt_id": generation},
    )
    ledger = claim_ledger(validate_ledger(experiment))
    study = open_phase_study(ledger, experiment.phases[0])
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    attempts = {
        "1": ("managed", TrialState.RUNNING),  # no outcome until recovery records one
        "2": ("managed", TrialState.FAIL),  # its callback already recorded it
        "3": ("trainer", TrialState.FAIL),  # PhaseSweep never ends a trainer's run
        "4": ("managed", TrialState.COMPLETE),  # accepted; its record was never saved
    }
    for number, (digit, (mode, state)) in enumerate(attempts.items()):
        attempt = digit * 32
        directory = _add_bound_attempt(
            experiment,
            study,
            "depth",
            generation,
            attempt,
            number,
            run_mode=mode,
            trial=create_trial(
                state=state,
                value=0.5 if state == TrialState.COMPLETE else None,
                user_attrs={
                    OVERRIDES_ATTR: json.dumps({"model.depth": 2}),
                    FEASIBLE_ATTR: True,
                    DURATION_ATTR: 3.0,
                    constraint_attr("cost"): 1.0,
                },
            ),
        )
        write_record(directory / INHERITS_FILE, {"schema_version": 1, "inherits": {}})
    write_record(
        tracking_root(experiment) / generation / "delivery" / f"attempt-{'2' * 32}.json",
        delivery_record(generation, state="delivered"),
    )
    requests = []
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))
    publication.deliver_closeout(experiment, ledger, generation, error=KeyboardInterrupt())
    # An interrupted run only saves its records; the next run or sync-tracking sends them.
    assert requests == []
    saved = [read_record(path) for path in publication._pending_records(experiment, generation)]
    assert [(r["binding"]["attempt_id"][0], r["finish"]) for r in saved] == [
        ("4", None),
        ("a", "KILLED"),
    ]
    accepted = saved[0]["projection"]
    assert accepted["metrics"] == {
        "phasesweep.objective": 0.5,
        "phasesweep.feasible": 1.0,
        "phasesweep.duration_seconds": 3.0,
        "phasesweep.constraint.cost": 1.0,
    }
    assert accepted["params"] == {"phasesweep.override.model.depth": "2"}
    assert "phasesweep.outcome" not in accepted["tags"]


@pytest.mark.parametrize("storage", ["journal", "memory"])
def test_failed_invocation_leaves_a_pending_trials_managed_run_open_only_when_persistent(
    tmp_path, monkeypatch, storage
):
    """A failed run ends the run of a RUNNING trial only when no later recovery can settle it.

    With persistent storage, ending it FAILED would read as a verdict when recovery captures
    the run, discarding the training; left open, recovery's accepted result reaches it instead.
    An in-memory trial is gone when the process exits, so its run ends with the invocation.
    """
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    if storage == "journal":
        config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    phase = experiment.phases[0]
    generation = "a" * 32
    write_record(
        tracking_root(experiment) / generation / "generation-binding.json",
        {**mlflow_binding(), "phase": "generation", "attempt_id": generation},
    )
    ledger = claim_ledger(validate_ledger(experiment))
    study = open_phase_study(ledger, phase)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    attempt = "1" * 32
    directory = _add_bound_attempt(
        experiment,
        study,
        phase.name,
        generation,
        attempt,
        0,
        trial=create_trial(
            state=TrialState.RUNNING,
            user_attrs={
                OVERRIDES_ATTR: json.dumps({"model.depth": 2}),
                RETURN_CODE_ATTR: 0,
                DURATION_ATTR: 3.0,
            },
        ),
    )
    write_record(directory / INHERITS_FILE, {"schema_version": 1, "inherits": {}})
    requests = []
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))
    publication.deliver_closeout(experiment, ledger, generation, error=RuntimeError("outage"))
    delivery = tracking_root(experiment) / generation / "delivery"
    if storage == "memory":
        assert [(r["binding"]["attempt_id"], r["finish"]) for r in requests] == [
            (attempt, "FAILED"),
            (generation, "FAILED"),
        ]
        assert read_record(delivery / f"attempt-{attempt}.json")["state"] == "delivered"
        return
    assert [(r["binding"]["attempt_id"], r["finish"]) for r in requests] == [(generation, "FAILED")]
    assert not (delivery / f"attempt-{attempt}.json").exists()
    # Recovery later accepts the completed training.
    (pending,) = study.get_trials()
    optuna.Trial(study, pending._trial_id).set_user_attr(FEASIBLE_ATTR, True)
    study.tell(pending.number, 0.5)
    publication._rebuild_attempt_intents(experiment, validate_ledger(experiment), [generation])
    record = read_record(delivery / f"attempt-{attempt}.json")
    assert record["finish"] is None
    assert record["projection"]["metrics"]["phasesweep.objective"] == 0.5


def test_recovered_cleanup_settles_only_the_attempt_of_its_own_study(tmp_path, monkeypatch):
    """Trial numbers are study-local, so one study's recovery never settles another study's attempt."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    generation = "a" * 32
    ledger = claim_ledger(validate_ledger(experiment))
    # Trial 0 failed with unconfirmed cleanup in both studies; only depth's was recovered.
    for phase, digit, recovered in (("depth", "1", [0]), ("rate", "2", [])):
        study = open_phase_study(ledger, next(p for p in experiment.phases if p.name == phase))
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
        if recovered:
            study.set_user_attr(CLEANUP_RECOVERED_TRIALS_ATTR, recovered)
        attempt = digit * 32
        _add_bound_attempt(
            experiment,
            study,
            phase,
            generation,
            attempt,
            0,
            trial=create_trial(state=TrialState.FAIL, user_attrs={CLEANUP_CONFIRMED_ATTR: False}),
        )
    publication._rebuild_attempt_intents(experiment, validate_ledger(experiment), [generation])
    delivery = tracking_root(experiment) / generation / "delivery"
    assert [path.name for path in delivery.glob("attempt-*.json")] == [f"attempt-{'1' * 32}.json"]
    assert read_record(delivery / f"attempt-{'1' * 32}.json")["finish"] == "FAILED"


def test_interrupted_attempt_finishes_its_managed_run_killed(tmp_path, monkeypatch):
    """An interrupted trial's managed run ends KILLED; a failed one's ends FAILED."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    phase = experiment.phases[0]
    generation = "a" * 32
    study = open_phase_study(claim_ledger(validate_ledger(experiment)), phase)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    for number, (digit, outcome) in enumerate((("1", "interrupted"), ("2", "failure"))):
        attempt = digit * 32
        record = trial_outcome(1, outcome)
        _add_bound_attempt(
            experiment,
            study,
            phase.name,
            generation,
            attempt,
            number,
            trial=create_trial(state=TrialState.FAIL, user_attrs={TRIAL_OUTCOME_ATTR: record}),
        )
    publication._rebuild_attempt_intents(experiment, validate_ledger(experiment), [generation])
    delivery = tracking_root(experiment) / generation / "delivery"
    assert read_record(delivery / f"attempt-{'1' * 32}.json")["finish"] == "KILLED"
    assert read_record(delivery / f"attempt-{'2' * 32}.json")["finish"] == "FAILED"


@pytest.mark.parametrize("storage", ["memory", "journal"])
def test_unsaved_accepted_attempt_keeps_its_accepted_projection(tmp_path, monkeypatch, storage):
    """A record write that fails after Optuna accepts a trial never turns it into a failure."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    if storage == "journal":
        config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    actual_write = publication.write_record
    failed = []

    def fail_first_accepted(path, record):
        if not failed and path.name.startswith("attempt-") and record["state"] == "pending":
            failed.append(path)
            raise OSError("tracking disk full")
        actual_write(path, record)

    monkeypatch.setattr(publication, "write_record", fail_first_accepted)
    reports = []
    with pytest.raises(OSError, match="tracking disk full"):
        run_experiment(experiment, terminal_callback=reports.append)
    [path] = failed
    record = read_record(path)
    assert record["finish"] is None
    assert record["projection"]["metrics"]["phasesweep.objective"] == 1.0
    assert "phasesweep.outcome" not in record["projection"]["tags"]
    assert [r["projection"] for r in uploads if r["name"] == path.stem] == [record["projection"]]
    failure = read_record(
        tracking_root(experiment) / reports[0].generation_id / "delivery" / "generation.json"
    )
    assert failure["finish"] == "FAILED"
    if storage == "journal":
        # Resuming selects that very trial and leaves its record as it is.
        assert run_experiment(experiment)["depth"].attempt_id == record["binding"]["attempt_id"]
        assert read_record(path) == record


def test_sync_rebuilds_accepted_attempt_records_from_the_ledger(tmp_path, monkeypatch):
    """A controller that dies between Optuna's commit and its callback saved no record."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    monkeypatch.setattr(publication, "publish_trial", lambda *args: None)
    run_experiment(experiment)
    [generation] = [path.name for path in tracking_root(experiment).iterdir() if path.is_dir()]
    delivery = tracking_root(experiment) / generation / "delivery"
    assert not list(delivery.glob("attempt-*.json"))
    journal = (tmp_path / "study.journal").read_bytes()

    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 7}
    assert (tmp_path / "study.journal").read_bytes() == journal
    records = [read_record(path) for path in sorted(delivery.glob("attempt-*.json"))]
    assert len(records) == 4
    for record in records:
        assert record["finish"] is None
        assert record["projection"]["metrics"]["phasesweep.objective"] == 1.0
    rate = [r for r in records if r["binding"]["phase"] == "rate"]
    depth_runs = {r["binding"]["run_id"] for r in records if r["binding"]["phase"] == "depth"}
    assert all(r["projection"]["tags"]["phasesweep.inherits.depth"] in depth_runs for r in rate)


def test_the_run_that_confirms_an_attempts_cleanup_finishes_its_managed_run(tmp_path, monkeypatch):
    """Recovery never rewrites the failed trial; the study's ledger settles its attempt, and the run that confirmed it sends the outcome once."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, sampler={"type": "random", "seed": 123})
    attempt = leave_cleanup_uncertain_attempt(Experiment.model_validate(config), monkeypatch)
    # The same config with a larger budget, so the resumed invocation is ordinary preflight.
    config["phases"][0]["n_trials"] = 2
    experiment = Experiment.model_validate(config)
    [generation] = [path.name for path in tracking_root(experiment).iterdir() if path.is_dir()]
    delivery = tracking_root(experiment) / generation / "delivery"
    record = delivery / f"attempt-{attempt}.json"
    journal = tmp_path / "study.journal"
    before = journal.read_bytes()

    # While its cleanup is unresolved, the attempt reports nothing; only the failed run's summary
    # run is finished, as that run recorded.
    uploads.clear()
    assert publication.sync_tracking(experiment) == {"pending": 0, "delivered": 1}
    assert not record.exists()
    assert [(r["name"], r["finish"]) for r in uploads] == [("generation", "FAILED")]
    assert journal.read_bytes() == before

    run_experiment(experiment)
    read = read_phase_trials(validate_ledger(experiment), "depth")
    failed = next(trial for trial in read.trials if trial.user_attrs[ATTEMPT_ID_ATTR] == attempt)
    assert failed.state == TrialState.FAIL
    assert failed.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    assert failed.number in read.cleanup_recovered
    [upload] = [r for r in uploads if r["binding"]["attempt_id"] == attempt]
    assert upload["finish"] == "FAILED"
    assert read_record(record)["state"] == "delivered"
    before = journal.read_bytes()

    counts = publication.sync_tracking(experiment)
    assert counts["pending"] == 0
    assert publication.sync_tracking(experiment) == counts
    # Neither sync delivers it again or rewrites the ledger.
    assert [r for r in uploads if r["binding"]["attempt_id"] == attempt] == [upload]
    assert journal.read_bytes() == before


def _fail_engine(tmp_path, monkeypatch, error):
    """Fail after generation preparation with a durable owned summary binding."""
    experiment = native_experiment(tmp_path)

    def fail(*args, generation, **kwargs):
        generation_id = generation.generation_id
        binding = {**mlflow_binding(), "phase": "generation", "generation_id": generation_id}
        write_record(tracking_root(experiment) / generation_id / "generation-binding.json", binding)
        raise error

    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail)
    return experiment


@pytest.mark.parametrize("failure", ["ordinary", "signal", "keyboard"])
def test_failure_projection_outage_preserves_error_and_retries(tmp_path, monkeypatch, failure):
    """Terminal failures retain status intent and original error despite upload outage.

    An ordinary failure sends its summary record once; a shutdown only saves it.
    """
    error = (
        PhaseSweepShutdown(
            signal.SIGTERM,
            ShutdownCleanupReport(signum=signal.SIGTERM, cleanup_confirmed=True, child_pgids=()),
        )
        if failure == "signal"
        else KeyboardInterrupt()
        if failure == "keyboard"
        else RuntimeError("private trainer failure")
    )
    experiment = _fail_engine(tmp_path, monkeypatch, error)
    requests = []

    def outage(*args, **kwargs):
        requests.append(args[2])
        raise RuntimeError("Transport disconnected")

    monkeypatch.setattr(publication, "call_tracking", outage)
    reports = []
    with pytest.raises(type(error)) as caught:
        run_experiment(experiment, terminal_callback=reports.append)
    assert caught.value is error
    assert reports[0].cleanup_confirmed
    generation = reports[0].generation_id
    record_path = tracking_root(experiment) / generation / "delivery/generation.json"
    record = read_record(record_path)
    assert record["state"] == "pending"
    assert record["finish"] == ("FAILED" if failure == "ordinary" else "KILLED")
    assert "private trainer failure" not in str(record)
    assert record["projection"]["metrics"] == {}
    assert len(requests) == (1 if failure == "ordinary" else 0)
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))
    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 1}
    assert requests[0] == requests[-1]
    # sync-tracking sends the saved summary record as it is and never rebuilds it.
    assert read_record(record_path) == {**record, "state": "delivered"}


def test_uncertain_cleanup_never_enqueues_terminal_projection(tmp_path, monkeypatch):
    """Even a later successful reconciliation cannot erase original cleanup uncertainty."""
    error = ProcessCleanupUncertainError("trainer still unresolved")
    experiment = _fail_engine(tmp_path, monkeypatch, error)
    monkeypatch.setattr(publication, "call_tracking", lambda *args: pytest.fail("No upload"))
    with pytest.raises(ProcessCleanupUncertainError) as caught:
        run_experiment(experiment)
    assert caught.value is error
    assert publication.delivery_status(experiment) == {"pending": 0, "delivered": 0}


def test_failure_publisher_cleanup_uncertainty_remains_actionable(tmp_path, monkeypatch):
    """An unresolved tracking worker changes cleanup evidence, never the primary report."""
    error = RuntimeError("trainer failed")
    experiment = _fail_engine(tmp_path, monkeypatch, error)

    def unsafe(*args):
        raise UnsafeProcessCleanupError("publisher still unresolved")

    monkeypatch.setattr(publication, "call_tracking", unsafe)
    reports = []
    with pytest.raises(ProcessCleanupUncertainError) as caught:
        run_experiment(experiment, terminal_callback=reports.append)
    assert caught.value.__cause__ is error
    assert reports[0].primary_error is error
    assert not reports[0].cleanup_confirmed
    assert publication.delivery_status(experiment) == {"pending": 1, "delivered": 0}


def test_failure_persistence_error_cannot_replace_primary(tmp_path, monkeypatch):
    error = RuntimeError("trainer failed")
    experiment = _fail_engine(tmp_path, monkeypatch, error)
    monkeypatch.setattr(
        publication,
        "deliver_closeout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("delivery filesystem read-only")
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        run_experiment(experiment)
    assert caught.value is error


def test_each_summary_record_is_sent_after_the_records_of_its_generation(tmp_path: Path) -> None:
    """Records go generation by generation, each summary-run record after its generation's others.

    A summary run MLflow shows ended therefore had the attempt outcomes and winner references it
    summarizes sent first, including a winner record whose name sorts after ``generation``.
    """
    experiment = native_experiment(tmp_path)
    root = tracking_root(experiment)
    stems = {
        "1" * 32: ["winner-rate", "generation", "attempt-b", "attempt-a"],
        "2" * 32: ["generation", "attempt-c"],
    }
    for generation, names in stems.items():
        (root / generation / "delivery").mkdir(parents=True)
        for name in names:
            write_record(
                root / generation / "delivery" / f"{name}.json", delivery_record(generation)
            )
    assert [
        f"{path.parent.parent.name[0]}/{path.stem}"
        for path in publication._pending_records(experiment, None)
    ] == [
        "1/attempt-a",
        "1/attempt-b",
        "1/winner-rate",
        "1/generation",
        "2/attempt-c",
        "2/generation",
    ]


def test_missing_binding_never_creates_or_discovers_a_summary(tmp_path, monkeypatch):
    """An unresolved create response cannot identify a run to finish by guessing."""
    experiment = native_experiment(tmp_path)
    monkeypatch.setattr(publication, "call_tracking", lambda *args: pytest.fail("No remote work"))
    publication.deliver_closeout(
        experiment, validate_ledger(experiment), "a" * 32, error=RuntimeError("create unresolved")
    )
    assert publication.delivery_status(experiment) == {"pending": 0, "delivered": 0}


@pytest.mark.parametrize("recorded_state", ["published", "failed"])
def test_failure_publisher_preserves_historical_generation_outcome(
    tmp_path, monkeypatch, recorded_state
):
    """A newer publication cannot turn an older success into a failure or hide an older failure."""
    from phasesweep.engine.generation import _write_generation_state
    from phasesweep.engine.paths import _last_successful_generation_path
    from tests.ledger_fixtures import materialize, published_generation_id

    fixture = materialize("current-journal", tmp_path / "publication", mode="tree")
    experiment = fixture.experiment.model_copy(
        update={"mlflow": native_experiment(tmp_path).mlflow}
    )
    ledger = validate_ledger(experiment)
    newer_generation = published_generation_id(experiment)
    assert newer_generation is not None
    generation = "a" * 32
    assert generation != newer_generation
    _write_generation_state(
        experiment,
        generation_id=generation,
        state=recorded_state,
        from_phase=None,
        publish_current=False,
    )
    write_record(
        tracking_root(experiment) / generation / "generation-binding.json",
        {**mlflow_binding(), "phase": "generation", "attempt_id": generation},
    )
    pointer = _last_successful_generation_path(experiment)
    pointer_before = pointer.read_bytes()
    requests = []
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))

    publication.deliver_closeout(experiment, ledger, generation, error=RuntimeError("late error"))

    assert pointer.read_bytes() == pointer_before
    assert published_generation_id(experiment) == newer_generation
    assert (
        _recorded_generation(_generation_record_path(experiment, generation)).state
        == recorded_state
    )
    if recorded_state == "published":
        assert requests == []
        assert publication.delivery_status(experiment, generation) == {"pending": 0, "delivered": 0}
    else:
        (request,) = requests
        assert request["finish"] == "FAILED"
        assert request["projection"]["summary"] == {
            "generation_id": generation,
            "status": "FAILED",
            "error_class": "RuntimeError",
        }
        assert publication.delivery_status(experiment, generation) == {"pending": 0, "delivered": 1}


def _fail_mirror(monkeypatch, failure):
    """Break the first winner delivery after the pointer commit in one way.

    :return BaseException | None: The injected error, or ``None`` for a failed write.
    """
    if failure in {"intent", "acknowledgement"}:
        state = "pending" if failure == "intent" else "delivered"
        actual_write = publication.write_record

        def read_only(path, record):
            if path.name == "winner-depth.json" and record["state"] == state:
                raise PermissionError("tracking filesystem became read-only")
            actual_write(path, record)

        monkeypatch.setattr(publication, "write_record", read_only)
        return None
    error = (
        UnsafeProcessCleanupError("upload worker still unresolved")
        if failure == "worker"
        else PhaseSweepShutdown(
            signal.SIGTERM,
            ShutdownCleanupReport(signum=signal.SIGTERM, cleanup_confirmed=True, child_pgids=()),
        )
    )
    actual_call = publication.call_tracking

    def interrupted(*args, **kwargs):
        if args[2]["name"] == "winner-depth":
            raise error
        return actual_call(*args, **kwargs)

    monkeypatch.setattr(publication, "call_tracking", interrupted)
    return error


@pytest.mark.parametrize("failure", ["intent", "acknowledgement", "worker", "shutdown"])
def test_mirror_failure_after_publication_keeps_the_result(tmp_path, monkeypatch, failure):
    """Publication, returned winners, and the terminal report agree; sync finishes the mirror."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    experiment = Experiment.model_validate(config)
    error = _fail_mirror(monkeypatch, failure)
    reports = []
    if failure == "shutdown":
        with pytest.raises(PhaseSweepShutdown) as caught:
            run_experiment(experiment, terminal_callback=reports.append)
        assert caught.value is error and caught.value.published_result_committed
    else:
        winners = run_experiment(experiment, terminal_callback=reports.append)
        assert set(winners) == {"depth", "rate"}
    [report] = reports
    generation = report.generation_id
    assert report.primary_error is None
    assert set(report.winners) == {"depth", "rate"}
    # Only a worker that may still be running is cleanup evidence.
    assert report.cleanup_confirmed is (failure != "worker")
    assert report.cleanup_error is (error if failure == "worker" else None)
    assert read_status(experiment)["published_generation_id"] == generation
    assert _recorded_generation(_generation_record_path(experiment, generation)).state == (
        "published"
    )
    assert (
        publication.delivery_status(experiment, generation)
        == {
            "intent": {"pending": 0, "delivered": 4},
            "acknowledgement": {"pending": 1, "delivered": 6},
            "worker": {"pending": 3, "delivered": 4},
            "shutdown": {"pending": 3, "delivered": 4},
        }[failure]
    )
    monkeypatch.setattr(publication, "write_record", session.write_record)
    monkeypatch.setattr(publication, "call_tracking", session.call_tracking)
    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 7}
    latest = {request["name"]: request for request in uploads}
    assert latest["winner-depth"]["projection"]["tags"]["phasesweep.winner"] == "true"
    winner_run = latest["winner-depth"]["binding"]["run_id"]
    assert latest["generation"]["projection"]["tags"]["phasesweep.winner.depth"] == winner_run
    if failure == "acknowledgement":
        # The frozen projection MLflow already accepted is resent unchanged.
        resent = [r for r in uploads if r["name"] == "winner-depth"]
        assert resent == [latest["winner-depth"]] * 2


@pytest.mark.parametrize("edit", ["append", "rewire"])
def test_rebuilt_winner_references_keep_the_generation_graph(tmp_path, monkeypatch, edit):
    """Sync rebuilds an older generation from its own phase plan and lineage, not today's."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    if edit == "rewire":
        config["phases"][1]["inherits"] = []
    _fail_mirror(monkeypatch, "intent")
    reports = []
    run_experiment(Experiment.model_validate(config), terminal_callback=reports.append)
    generation = reports[0].generation_id
    monkeypatch.setattr(publication, "write_record", session.write_record)
    if edit == "append":
        config["phases"].append(
            {
                "name": "momentum",
                "inherits": ["rate"],
                "n_trials": 1,
                "sampler": {"type": "grid"},
                "search_space": {"optimizer.momentum": {"type": "categorical", "choices": [0.9]}},
            }
        )
    else:
        config["phases"][1]["inherits"] = ["depth"]
    edited = Experiment.model_validate(config)
    if edit == "append":
        # Extending a finished experiment reuses its studies and publishes anew.
        assert list(run_experiment(edited)) == ["depth", "rate", "momentum"]

    assert publication.sync_tracking(edited, generation) == {"pending": 0, "delivered": 7}
    delivery = tracking_root(edited) / generation / "delivery"
    assert sorted(path.name for path in delivery.glob("[!a]*.json")) == [
        "generation.json",
        "winner-depth.json",
        "winner-rate.json",
    ]
    depth_run = read_record(delivery / "winner-depth.json")["binding"]["run_id"]
    rate_tags = read_record(delivery / "winner-rate.json")["projection"]["tags"]
    expected = {"phasesweep.inherits.depth": depth_run} if edit == "append" else {}
    assert {k: v for k, v in rate_tags.items() if "inherits" in k} == expected
    summary_tags = read_record(delivery / "generation.json")["projection"]["tags"]
    assert sorted(summary_tags) == ["phasesweep.winner.depth", "phasesweep.winner.rate"]


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["intent", "worker"])
def test_mcp_status_agrees_with_a_mirror_failure_after_publication(tmp_path, monkeypatch, failure):
    """A detached run records success; an unresolved upload worker also needs recovery."""
    from phasesweep.mcp.runs import RunStore
    from tests.mcp_helpers import run_claimed_runner

    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    _fail_mirror(monkeypatch, failure)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    store = RunStore(tmp_path / "state")
    run_id = "mirror-" + "a" * 32
    assert run_claimed_runner(store, run_id, config_path, cwd=tmp_path) == 0

    status = json.loads(store.status_path(run_id).read_text())
    assert status["returncode"] == 0
    assert status["failure"] is None and status["error_class"] is None
    assert status["cleanup_confirmed"] is (failure == "intent")
    assert status["result_snapshot_state"] == "complete"
    assert [w["phase"] for w in status["result_snapshot"]["winners"]] == ["depth", "rate"]
    assert read_status(Experiment.model_validate(config))["published_generation_id"] == run_id
    handle = store.get(run_id)
    assert handle is not None
    assert store.observe(handle).cleanup_recovery_required is (failure == "worker")


@pytest.mark.integration
def test_sync_rebuilds_a_generation_whose_record_a_post_commit_crash_lost(tmp_path, monkeypatch):
    """A controller that dies right after the pointer commit loses neither its proof nor its mirror.

    The first invocation dies before it writes the lifecycle record and before
    the in-process mirror. The pointer alone keeps the generation published, a
    later no-work replay leaves it there and never ends its summary run as a
    killed run's, and ``sync-tracking`` rebuilds its winner and summary uploads.
    """
    from phasesweep.engine.paths import _last_successful_generation_path

    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"  # the replay must find the studies
    experiment = Experiment.model_validate(config)
    pointer_path = _last_successful_generation_path(experiment)

    with monkeypatch.context() as crash:
        lose_record_writes(crash, OSError("controller died after the pointer commit"))
        crash.setattr(publication, "deliver_closeout", lambda *args, **kwargs: None)
        run_experiment(experiment)
    first = yaml.safe_load(pointer_path.read_text())["generation_id"]
    first_delivery = tracking_root(experiment) / first / "delivery"
    intents = ["winner-depth.json", "winner-rate.json", "generation.json"]
    assert not _generation_record_path(experiment, first).exists()
    assert len(list(first_delivery.glob("attempt-*.json"))) == 4
    assert not any((first_delivery / name).exists() for name in intents)
    uploads.clear()  # the attempt uploads are already delivered; only sync's remain to count

    reports = []
    run_experiment(experiment, terminal_callback=reports.append)
    assert not _generation_dir(_experiment_dir(experiment), reports[0].generation_id).exists()
    assert yaml.safe_load(pointer_path.read_text())["generation_id"] == first
    assert not _generation_record_path(experiment, first).exists()
    assert publication._published(experiment, first)

    assert publication.sync_tracking(experiment) == {"pending": 0, "delivered": 7}
    assert all((first_delivery / name).exists() for name in intents)
    assert read_record(first_delivery / "generation.json")["finish"] == "FINISHED"
    assert sorted(request["name"] for request in uploads) == [
        "generation",
        "winner-depth",
        "winner-rate",
    ]


@pytest.mark.integration
def test_sync_leaves_superseded_publication_unresolved_when_its_record_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The history record resolves a superseded publication before any terminal intent is saved."""
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, gpu_policy="none", sampler={"type": "random", "seed": 1})
    experiment = Experiment.model_validate(config)

    def offline(*args: object, **kwargs: object) -> None:
        raise OSError("tracker offline")

    with monkeypatch.context() as without_mirror:
        without_mirror.setattr(publication, "deliver_closeout", lambda *args, **kwargs: None)
        without_mirror.setattr(publication, "_deliver", offline)
        run_experiment(experiment, generation_id="older")
    with monkeypatch.context() as older_left_for_sync:
        # The newer run's closeout would otherwise deliver the older generation's records.
        older_left_for_sync.setattr(
            publication, "deliver_ended_generations", lambda *args, **kwargs: None
        )
        run_experiment(experiment, generation_id="newer")
    assert read_status(experiment)["published_generation_id"] == "newer"
    lifecycle = _generation_record_path(experiment, "older")
    assert _recorded_generation(lifecycle).state == "published"
    summary = tracking_root(experiment) / "older" / "delivery" / "generation.json"
    uploads.clear()
    read_text = Path.read_text
    reads: list[Path] = []

    def unreadable_once(path: Path, *args, **kwargs) -> str:
        if path == lifecycle:
            reads.append(path)
            if len(reads) == 1:
                raise PermissionError("temporary read failure")
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "read_text", unreadable_once)
        with pytest.raises(
            TrackingSyncRefusedError, match="generation record .* is unreadable"
        ) as refusal:
            publication.sync_tracking(experiment, "older")
    assert reads == [lifecycle]
    assert not summary.exists()
    assert len(uploads) == 1 and uploads[0]["name"].startswith("attempt-")
    assert "Delivered records in scope: 1" in str(refusal.value)

    assert publication.sync_tracking(experiment, "older") == {"pending": 0, "delivered": 3}
    assert read_record(summary)["finish"] == "FINISHED"
    reads.clear()
    with monkeypatch.context() as fault:
        fault.setattr(Path, "read_text", unreadable_once)
        assert publication.sync_tracking(experiment, "older") == {"pending": 0, "delivered": 3}
    assert read_record(summary)["finish"] == "FINISHED"


@pytest.mark.parametrize(
    "reply,unreadable",
    [(None, False), ("unknown", False), ("unknown", True), ("ok", False), ("ok", True)],
    ids=["lost-response", "unknown", "unknown-unreadable", "ok", "ok-unreadable"],
)
def test_sync_binds_and_finishes_a_run_created_after_its_attempt_gave_up(
    tmp_path, monkeypatch, reply, unreadable
):
    """A creation never acknowledged is searched for, not resent, and sync binds the run it made late.

    The first attempt's creation is lost outright (``None``) or answered by the worker process
    through the real response reader, which may then fail to read the answer.
    """
    from phasesweep.tracking.mlflow import OperationOutcomeUnknownError

    real_operation = worker.request_operation  # ``remote_mlflow_example`` replaces it
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    late: dict[str, str] = {}
    # A worker that reported success made its run before the controller looked; others, only later.
    appeared = {"late": reply == "ok"}
    creations: list[str] = []
    if reply is not None:
        FakeWorkerProcess(
            monkeypatch,
            unreadable=["create"] if unreadable else [],
            create=[
                lambda: (
                    {"status": "ok", "result": dict(late)}
                    if reply == "ok"
                    else {
                        "status": "unknown",
                        "message": "Creation may still commit; reply unavailable.",
                    }
                )
            ],
        )

    def first_attempt_creation(request, **kwargs):
        binding = request.get("binding", {})
        if request["action"] == "create":
            creations.append(binding["attempt_id"])
        if request["action"] == "create" and binding["phase"] != "generation" and not late:
            late.update(binding, run_id="late-run")
            if reply is None:
                raise OperationOutcomeUnknownError("response lost")
            return real_operation(request, **kwargs)
        if request["action"] == "find":
            found = appeared["late"] and binding["attempt_id"] == late.get("attempt_id")
            return {"found": True, "binding": late} if found else {"found": False}
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", first_attempt_creation)
    reports = []
    run_experiment(experiment, terminal_callback=reports.append)
    generation = reports[0].generation_id
    pattern = f"*/trial_*__generation_{generation}__attempt_{late['attempt_id']}"
    [attempt_directory] = _experiment_dir(experiment).glob(pattern)
    pending = attempt_directory / "mlflow-binding.pending.json"
    # An existing run is bound before sync, directly or by the search that follows an unreadable
    # reply. A run that does not exist yet leaves its pending record for sync to resolve.
    assert pending.exists() is (reply != "ok")
    assert (attempt_directory / BINDING_FILE).exists() is (reply == "ok")
    if reply != "ok":
        assert publication.sync_tracking(experiment, generation) == {"pending": 1, "delivered": 6}
        assert pending.exists()

    appeared["late"] = True
    uploads.clear()
    journal = tmp_path / "study.journal"
    before = journal.read_bytes()
    # Three accepted attempts, two winners, the summary, and the late run, readable reply or not.
    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 7}
    assert journal.read_bytes() == before
    assert not pending.exists()
    assert session.read_binding(attempt_directory)["run_id"] == "late-run"
    assert creations.count(late["attempt_id"]) == 1
    # Sync finishes a run it bound with its failed trial; an accepted attempt was delivered earlier.
    assert [r["finish"] for r in uploads if r["binding"]["run_id"] == "late-run"] == (
        [] if reply == "ok" else ["FAILED"]
    )


@pytest.mark.parametrize(
    "answer,upload",
    [
        ("accepted", "up"),
        ("objective-missing", "up"),
        ("accepted", "offline"),
        ("accepted", "unsaved"),
    ],
    ids=["accepted", "objective-missing", "upload-offline", "record-unsaved"],
)
def test_recovery_delivers_the_outcome_it_records_for_a_pending_trial(
    tmp_path, monkeypatch, caplog, answer, upload
):
    """A pending trial's managed run receives the outcome recovery commits.

    Accepted evidence sends its accepted projection. Evidence the tracker answers with a
    failure records the finished trial failed with that reason, as a live trial would be, and
    trains nothing again; the run then fails before any target check, so it only saves the
    outcome, and sync-tracking ends the run FAILED. An upload that fails, or a record that
    cannot be saved, leaves the outcome for sync-tracking and the run still publishes its result.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError
    from phasesweep.tracking.mlflow import OperationUnavailableError

    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, sampler={"type": "random", "seed": 123})
    config["mlflow"].update(timeout_seconds=6, poll_seconds=1)
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    refused_upload: dict[str, str] = {}

    def capture_after_outage(request, **kwargs):
        if request["action"] == "capture":
            if request["settings"]["timeout_seconds"] < 10:
                raise OperationUnavailableError("tracking server did not answer in time")
            assert request["settings"]["poll_seconds"] == 3
            if answer == "objective-missing":
                run = mlflow_run(request["binding"], metrics={"cost": 1.0})
                return worker._capture(SimpleNamespace(get_run=lambda _: run), request)
        if request["action"] == "publish" and request["name"] == refused_upload.get("name"):
            raise OSError("upload refused")
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", capture_after_outage)
    monkeypatch.setattr(worker, "request_operation", capture_after_outage)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(experiment)
    (pending,) = read_phase_trials(validate_ledger(experiment), "depth").trials
    assert pending.state == TrialState.RUNNING
    attempt = pending.user_attrs[ATTEMPT_ID_ATTR]
    generation = pending.user_attrs[GENERATION_ID_ATTR]

    config["mlflow"].update(timeout_seconds=30, poll_seconds=3)
    experiment = Experiment.model_validate(config)
    if upload == "offline":
        refused_upload["name"] = f"attempt-{attempt}"
    delivery = tracking_root(experiment) / generation / "delivery"
    if upload == "unsaved":
        delivery.chmod(0o500)
    uploads.clear()
    if answer == "accepted":
        run_experiment(experiment)
    else:
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(experiment)
    (recovered,) = read_phase_trials(validate_ledger(experiment), "depth").trials
    assert recovered.state == (TrialState.COMPLETE if answer == "accepted" else TrialState.FAIL)
    record_path = delivery / f"attempt-{attempt}.json"
    if answer == "objective-missing":
        assert [r for r in uploads if r["name"] == f"attempt-{attempt}"] == []
        assert read_record(record_path)["state"] == "pending"
        publication.sync_tracking(experiment)
    if upload == "unsaved":
        assert read_status(experiment)["published_generation_id"] not in {None, generation}
        assert not record_path.exists()
        assert "phasesweep sync-tracking" in caplog.text
        delivery.chmod(0o700)
        uploads.clear()
        publication.sync_tracking(experiment)
    if upload == "offline":
        assert read_record(record_path)["state"] == "pending"
        assert (
            f"MLflow: attempt record attempt-{attempt} of generation {generation} is pending (upload "
            "refused)." in caplog.text
        )
        refused_upload.clear()
        uploads.clear()
        publication.sync_tracking(experiment)
    [delivered] = [r for r in uploads if r["name"] == f"attempt-{attempt}"]
    record = read_record(record_path)
    assert record["state"] == "delivered"
    assert delivered["projection"] == record["projection"]
    if answer == "accepted":
        assert delivered["finish"] is None
        assert delivered["projection"]["metrics"]["phasesweep.objective"] == 1.0
    else:
        assert not _trial_interrupted(recovered)
        cause = recovered.user_attrs[TRIAL_OUTCOME_ATTR]["cause"]
        assert "MLflow run finished without metric 'eval/loss'" in cause
        assert delivered["finish"] == "FAILED"
        assert delivered["projection"]["tags"] == {"phasesweep.outcome": "FAILED"}
    # Nothing is left for sync-tracking to send again for that attempt.
    uploads.clear()
    publication.sync_tracking(experiment)
    assert [r for r in uploads if r["name"] == f"attempt-{attempt}"] == []


def _sent(uploads, attempts):
    """Return the finish status of each upload to one of ``attempts``, in upload order."""
    names = {f"attempt-{attempt}" for attempt in attempts}
    return [request["finish"] for request in uploads if request["name"] in names]


def _summaries(uploads):
    """Return the generation and finish status of each summary-run upload, in upload order."""
    return [
        (request["projection"]["summary"]["generation_id"], request["finish"])
        for request in uploads
        if request["name"] == "generation"
    ]


@pytest.mark.parametrize("in_flight", [True, False], ids=["in-flight", "between-trials"])
@pytest.mark.parametrize("caller", ["own-run", "next-run", "recover-run", "sync-tracking"])
def test_a_killed_runs_summary_run_is_finished_by_recovery_never_by_itself(
    tmp_path, monkeypatch, caller, in_flight
):
    """A run killed outright leaves its summary run open until the next run, recover-run or sync-tracking handles it.

    It ends KILLED, as its run recorded no failure, and is sent once, whether the run was
    killed with its trials in flight or between them, with no trial left to settle. A run's
    delivery of ended generations never finishes that run's own summary run, which its terminal
    delivery owns, and a dry run writes and sends nothing.
    """
    experiment, uploads, _attempts = killed_sweep(
        tmp_path, monkeypatch, n_jobs=1, in_flight=in_flight
    )
    [generation] = [path.name for path in tracking_root(experiment).iterdir() if path.is_dir()]
    record = tracking_root(experiment) / generation / "delivery" / "generation.json"
    ledger = validate_ledger(experiment)
    uploads.clear()
    if caller == "own-run":
        tally = DeliveryTally()
        publication.deliver_ended_generations(experiment, ledger, generation, tally)
        assert tally.delivered == 0
        assert not record.exists()
        assert uploads == []
        return
    if caller == "next-run":
        tree = sorted(tracking_root(experiment).rglob("*"))
        run_experiment(experiment, dry_run=True)
        assert (sorted(tracking_root(experiment).rglob("*")), uploads) == (tree, [])
        reports = []
        run_experiment(experiment, terminal_callback=reports.append)
        # Only a trial killed in flight is rerun, so only then does the run bind its own
        # summary run, which its publication finishes.
        own = [(reports[0].generation_id, "FINISHED")] if in_flight else []
        assert sorted(_summaries(uploads)) == sorted([(generation, "KILLED"), *own])
        uploads.clear()
        run_experiment(experiment)
        assert uploads == []
    else:
        if caller == "recover-run":
            tally = DeliveryTally()
            publication.deliver_ended_generations(
                experiment, ledger, generation, tally, recovering=True
            )
            assert tally.delivered == 1
            assert publication.delivery_status(experiment)["pending"] == 0
        else:
            # A finished attempt's accepted outcome was delivered before the kill.
            delivered = 1 if in_flight else 2
            assert publication.sync_tracking(experiment) == {"delivered": delivered, "pending": 0}
        publication.sync_tracking(experiment)
        assert _summaries(uploads) == [(generation, "KILLED")]
    assert read_record(record)["projection"]["summary"]["error_class"] is None


@pytest.mark.parametrize("caller", ["next-run", "sync-tracking"])
def test_a_killed_in_memory_sweeps_attempt_runs_end_with_its_summary_run(
    tmp_path, monkeypatch, caller
):
    """An in-memory sweep killed outright has its managed attempt runs ended KILLED once their cleanup is confirmed.

    Its trials were gone with the process, so nothing else could ever end those runs. The next
    run's preflight confirms the attempts' cleanup, so that run ends them with the summary run.
    sync-tracking confirms no trainer's cleanup: it ends the summary run and leaves each attempt,
    whose trainer may still log to its run, to the next run. Each outcome is sent once.
    """
    experiment, uploads, attempts = killed_sweep(tmp_path, monkeypatch, n_jobs=2, persistent=False)
    [killed] = [path.name for path in tracking_root(experiment).iterdir() if path.is_dir()]
    uploads.clear()

    if caller == "sync-tracking":
        assert publication.sync_tracking(experiment) == {"delivered": 1, "pending": 0}
        assert _sent(uploads, attempts) == []
    run_experiment(experiment)

    assert _sent(uploads, attempts) == ["KILLED", "KILLED"]
    assert [entry for entry in _summaries(uploads) if entry[0] == killed] == [(killed, "KILLED")]
    uploads.clear()
    run_experiment(experiment)
    publication.sync_tracking(experiment)
    assert _sent(uploads, attempts) == []
    assert [entry for entry in _summaries(uploads) if entry[0] == killed] == []


@pytest.mark.parametrize(
    ("storage", "recorded", "finish"),
    [
        (None, None, "KILLED"),
        (None, "failed", "FAILED"),
        (None, "published", None),
        ("journal", None, None),
    ],
    ids=["in-memory-killed", "in-memory-failed", "in-memory-published", "persistent"],
)
def test_only_an_in_memory_generation_ends_an_attempt_its_ledger_lost(
    tmp_path, storage, recorded, finish
):
    """A bound attempt the ledger does not hold ends with its ended generation's status only without persistent storage.

    It ends KILLED, or with the status the generation's recorded failure maps to. A published
    generation's attempt is left alone, and with persistent storage a missing trial stays
    unresolved.
    """
    from phasesweep.engine.generation import _write_generation_state

    journal = f"journal:///{tmp_path}/study.journal"
    experiment = native_experiment(tmp_path, storage=journal if storage else None)
    ledger = claim_ledger(validate_ledger(experiment))
    generation = "a" * 32
    if recorded is not None:
        _write_generation_state(
            experiment,
            generation_id=generation,
            state=recorded,
            from_phase=None,
            publish_current=False,
            error_class="RuntimeError" if recorded == "failed" else None,
        )
    write_record(
        tracking_root(experiment) / generation / "generation-binding.json",
        {**mlflow_binding(), "phase": "generation", "attempt_id": generation},
    )
    attempt = "1" * 32
    directory = _trial_dir_for(experiment, "depth", 0, generation_id=generation, attempt_id=attempt)
    binding = {**mlflow_binding(), "phase": "depth", "attempt_id": attempt}
    write_record(directory / BINDING_FILE, binding)

    publication._rebuild_attempt_intents(experiment, ledger, [generation])

    record = tracking_root(experiment) / generation / "delivery" / f"attempt-{attempt}.json"
    assert (read_record(record)["finish"] if record.exists() else None) == finish


@pytest.mark.integration
@pytest.mark.parametrize("problem", ["unreadable", "malformed"])
def test_unusable_generation_record_leaves_mlflow_outcomes_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, problem: str
) -> None:
    """Recovery defers an unusable lifecycle's outcomes while sync delivers readable generations."""
    from phasesweep.engine.generation import _write_generation_state

    generation = "failed-run"
    experiment, uploads, [attempt] = killed_sweep(
        tmp_path, monkeypatch, n_jobs=1, generation_id=generation, persistent=False
    )
    _write_generation_state(
        experiment,
        generation_id=generation,
        state="failed",
        from_phase=None,
        publish_current=False,
        error_class="RuntimeError",
    )
    lifecycle = _generation_record_path(experiment, generation)
    original = lifecycle.read_bytes()
    delivery = tracking_root(experiment) / generation / "delivery"
    healthy = "healthy-run"

    def offline(*args: object, **kwargs: object) -> None:
        raise OSError("simulated upload outage")

    with monkeypatch.context() as fault:
        if problem == "unreadable":
            patch_path_method_failure(
                fault, lifecycle, "read_text", PermissionError("read refused")
            )
        else:
            lifecycle.write_text("state: [", encoding="utf-8")

        with fault.context() as unavailable:
            unavailable.setattr(publication, "call_tracking", offline)
            assert run_experiment(experiment, generation_id=healthy)["depth"].metric == 1.0
        assert f"{lifecycle} is {problem}" in caplog.text
        assert not delivery.exists()
        assert publication.delivery_status(experiment, healthy)["pending"] > 0
        with pytest.raises(TrackingSyncRefusedError, match=f"generation record .* is {problem}"):
            publication.sync_tracking(experiment, generation)
        assert not delivery.exists()
        uploads.clear()
        with pytest.raises(TrackingSyncRefusedError, match=f"generation record .* is {problem}"):
            publication.sync_tracking(experiment)
        assert _summaries(uploads) == [(healthy, "FINISHED")]
        assert publication.delivery_status(experiment, healthy)["pending"] == 0
        assert not delivery.exists()

    lifecycle.write_bytes(original)
    uploads.clear()
    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 2}
    assert _sent(uploads, [attempt]) == ["FAILED"]
    assert _summaries(uploads) == [(generation, "FAILED")]


@pytest.mark.parametrize("in_flight", [True, False], ids=["in-flight", "between-trials"])
def test_a_run_with_another_mlflow_target_saves_no_recovered_outcome(
    tmp_path, monkeypatch, caplog, in_flight
):
    """A run whose MLflow target is not the one the tree recorded saves no record of a killed sweep.

    The changed launch is refused, and the killed sweep's attempt and summary-run outcomes,
    whether recovery settled a trial or only found the summary run left open, wait unsaved
    under one warning. sync-tracking with the config the sweep ran with then delivers each once.
    """
    experiment, uploads, attempts = killed_sweep(
        tmp_path, monkeypatch, n_jobs=1, in_flight=in_flight
    )
    [generation] = [path.name for path in tracking_root(experiment).iterdir() if path.is_dir()]
    delivery = tracking_root(experiment) / generation / "delivery"
    recorded = sorted(delivery.glob("*.json"))
    moved = experiment.model_copy(
        update={
            "mlflow": experiment.mlflow.model_copy(update={"tracking_uri": "http://moved:5000"})
        }
    )
    uploads.clear()

    # The moved target changes the trainer environment of a rerun or the evidence of a kept trial.
    with pytest.raises(PhaseSweepError):
        run_experiment(moved)

    assert sorted(delivery.glob("*.json")) == recorded
    assert uploads == []
    assert "with the config the sweep ran with" in caplog.text
    with pytest.raises(TrackingSyncRefusedError, match="belongs to a different target"):
        publication.sync_tracking(moved)
    assert (sorted(delivery.glob("*.json")), uploads) == (recorded, [])
    assert publication.sync_tracking(experiment)["pending"] == 0
    assert _sent(uploads, attempts) == (["KILLED"] if in_flight else [])
    assert _summaries(uploads) == [(generation, "KILLED")]
    for path in delivery.glob("*.json"):
        assert read_record(path)["source"] == experiment.mlflow.evidence_identity()


def test_recovery_sends_nothing_to_a_tracker_that_is_down(tmp_path, monkeypatch):
    """Recovery commits what a killed run left RUNNING without contacting the tracker.

    While MLflow is down the next run makes one request, its target check, and leaves each
    settled attempt's record pending, and the killed run's summary-run record; sync-tracking
    then sends each exactly once.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError
    from phasesweep.tracking.mlflow import OperationUnavailableError

    experiment, uploads, attempts = killed_sweep(tmp_path, monkeypatch, n_jobs=3)
    respond = session.request_operation
    requests = []

    def down(request, **kwargs):
        requests.append(request["action"])
        raise OperationUnavailableError("tracking server unreachable")

    monkeypatch.setattr(session, "request_operation", down)
    monkeypatch.setattr(worker, "request_operation", down)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(experiment)
    assert requests == ["resolve"]
    trials = read_phase_trials(validate_ledger(experiment), "depth").trials
    assert all(_trial_interrupted(trial) for trial in trials)
    [generation] = {trial.user_attrs[GENERATION_ID_ATTR] for trial in trials}
    delivery = tracking_root(experiment) / generation / "delivery"
    assert [read_record(delivery / f"attempt-{a}.json")["state"] for a in attempts] == [
        "pending"
    ] * 3
    assert read_record(delivery / "generation.json")["state"] == "pending"

    monkeypatch.setattr(session, "request_operation", respond)
    monkeypatch.setattr(worker, "request_operation", respond)
    uploads.clear()
    publication.sync_tracking(experiment)
    assert _sent(uploads, attempts) == ["KILLED"] * 3
    assert _summaries(uploads) == [(generation, "KILLED")]
    uploads.clear()
    publication.sync_tracking(experiment)
    assert uploads == []


def test_a_reachable_tracker_receives_recovered_outcomes_after_its_check(tmp_path, monkeypatch):
    """A run that reaches MLflow sends each outcome its recovery committed once, after the target check.

    The killed run's summary run ends KILLED with them, and the run's own summary run still ends
    as its own publication does.
    """
    experiment, uploads, attempts = killed_sweep(tmp_path, monkeypatch, n_jobs=3)
    [killed] = [path.name for path in tracking_root(experiment).iterdir() if path.is_dir()]
    respond = session.request_operation
    requests = []

    def logged(request, **kwargs):
        requests.append(request["action"] + ":" + request.get("name", ""))
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", logged)
    monkeypatch.setattr(worker, "request_operation", logged)
    uploads.clear()
    reports = []
    run_experiment(experiment, terminal_callback=reports.append)
    assert _sent(uploads, attempts) == ["KILLED"] * 3
    first_send = min(requests.index(f"publish:attempt-{attempt}") for attempt in attempts)
    assert requests.index("resolve:") < first_send
    assert sorted(_summaries(uploads)) == sorted(
        [(killed, "KILLED"), (reports[0].generation_id, "FINISHED")]
    )
    uploads.clear()
    publication.sync_tracking(experiment)
    assert uploads == []


@pytest.mark.integration
def test_an_upload_workers_unconfirmed_cleanup_is_not_charged_to_the_recovered_attempt(
    tmp_path, monkeypatch
):
    """The worker sending a recovered outcome is what leaves cleanup uncertain, never the trial it reports.

    The outcome is sent right after the run's target check, so, as with a target check's own
    worker, the run stops there and starts no trial.
    """
    experiment, _uploads, [attempt] = killed_sweep(tmp_path, monkeypatch, n_jobs=1)
    leaked = leak_tracking_worker(monkeypatch, name=f"attempt-{attempt}")
    reports = []
    with pytest.raises(UnsafeProcessCleanupError):
        run_experiment(experiment, terminal_callback=reports.append)
    [report] = reports
    assert leaked
    assert not report.cleanup_confirmed
    assert attempt not in report.uncertain_attempt_ids
    assert report.failure_stage == "preflight"
    [recovered] = [
        trial
        for trial in read_phase_trials(validate_ledger(experiment), "depth").trials
        if trial.user_attrs[ATTEMPT_ID_ATTR] == attempt
    ]
    assert recovered.state == TrialState.FAIL


@pytest.mark.parametrize("stage", ["before-saving", "before-sending"])
def test_a_run_killed_before_delivering_a_recovered_outcome_leaves_it_for_sync(
    tmp_path, monkeypatch, stage
):
    """A runner killed between recovery's commit and the upload loses no outcome and sends none twice."""
    experiment, uploads, [attempt] = killed_sweep(tmp_path, monkeypatch, n_jobs=1)
    with monkeypatch.context() as patch:
        if stage == "before-saving":
            patch.setattr(
                publication,
                "_rebuild_attempt_intents",
                lambda *args, **kwargs: (_ for _ in ()).throw(RunnerKilled()),
            )
        else:
            deliver = publication._deliver

            def killed_at_send(experiment, path, environment, **kwargs):
                if path.stem == f"attempt-{attempt}":
                    raise RunnerKilled()
                return deliver(experiment, path, environment, **kwargs)

            patch.setattr(publication, "_deliver", killed_at_send)
        with pytest.raises(RunnerKilled):
            run_experiment(experiment)
    [recovered] = [
        trial
        for trial in read_phase_trials(validate_ledger(experiment), "depth").trials
        if trial.user_attrs[ATTEMPT_ID_ATTR] == attempt
    ]
    assert _trial_interrupted(recovered)
    record = (
        tracking_root(experiment)
        / recovered.user_attrs[GENERATION_ID_ATTR]
        / "delivery"
        / f"attempt-{attempt}.json"
    )
    assert record.exists() is (stage == "before-sending")
    uploads.clear()
    run_experiment(experiment)
    publication.sync_tracking(experiment)
    publication.sync_tracking(experiment)
    assert _sent(uploads, [attempt]) == ["KILLED"]
    assert read_record(record)["state"] == "delivered"


def _tree_listing(root):
    """List every entry below ``root`` with its size and modification time."""
    return {
        path.relative_to(root).as_posix(): (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in [root, *sorted(root.rglob("*"))]
    }


def test_sync_refuses_a_tree_of_another_format_as_run_does_and_writes_nothing(
    tmp_path, monkeypatch
):
    """sync-tracking checks the artifact-root binding under its lock before it writes anything."""
    from phasesweep.engine.errors import ArtifactRootSchemaMismatchError
    from phasesweep.engine.paths import _artifact_root_binding_path

    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    deliver = publication.call_tracking

    def offline(*args, **kwargs):
        raise OSError("tracking server offline")

    monkeypatch.setattr(publication, "call_tracking", offline)
    run_experiment(experiment)
    pending = publication.delivery_status(experiment)
    assert pending["pending"] == 7 and pending["delivered"] == 0
    binding_path = _artifact_root_binding_path(experiment)
    current = binding_path.read_text()
    binding = json.loads(current)
    binding["schema_version"] -= 1
    binding_path.write_text(json.dumps(binding) + "\n")
    monkeypatch.setattr(publication, "call_tracking", deliver)
    journal = tmp_path / "study.journal"
    before = (_tree_listing(_experiment_dir(experiment)), journal.read_bytes())

    with pytest.raises(ArtifactRootSchemaMismatchError) as run_refusal:
        run_experiment(experiment)
    uploads.clear()
    with pytest.raises(ArtifactRootSchemaMismatchError) as sync_refusal:
        publication.sync_tracking(experiment)
    assert str(sync_refusal.value) == str(run_refusal.value)
    assert "Nothing was written" in str(sync_refusal.value)
    assert uploads == []
    assert (_tree_listing(_experiment_dir(experiment)), journal.read_bytes()) == before

    # Restoring the binding lets the next sync deliver every pending upload.
    binding_path.write_text(current)
    assert publication.sync_tracking(experiment) == {"pending": 0, "delivered": 7}
    assert len(uploads) == 7
