"""Terminal delivery preserves engine outcome and cleanup authority."""

from __future__ import annotations

import json
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from optuna.trial import TrialState, create_trial

from phasesweep.config import Experiment
from phasesweep.engine import NoFeasibleTrialError, run_experiment
from phasesweep.engine.generation import _recorded_generation_state
from phasesweep.engine.ledger import claim_ledger, open_phase_study, validate_ledger
from phasesweep.engine.paths import _experiment_dir, _generation_record_path, _trial_dir_for
from phasesweep.engine.read import read_status
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    DURATION_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OVERRIDES_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    constraint_attr,
)
from phasesweep.errors import ProcessCleanupUncertainError, UnsafeProcessCleanupError
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import _capture, _publish
from phasesweep.tracking.session import (
    BINDING_FILE,
    INHERITS_FILE,
    read_record,
    tracking_root,
    write_record,
)
from tests.test_mlflow import _binding, _run, native_experiment


@pytest.mark.parametrize("requested", ["FINISHED", "FAILED", "KILLED"])
@pytest.mark.parametrize("actual", ["RUNNING", "FINISHED", "FAILED", "KILLED"])
def test_publish_finishes_only_a_running_owned_run(actual, requested):
    """A terminal run keeps its status, so a retry or later outcome never rewrites it."""
    pytest.importorskip("mlflow")
    binding = _binding()
    calls = []
    client = SimpleNamespace(
        get_run=lambda _: _run(binding, status=actual),
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
    assert calls == ["batch", "artifact", *([requested] if actual == "RUNNING" else [])]


def test_trainer_owned_run_is_never_terminated():
    pytest.importorskip("mlflow")
    binding = {**_binding(), "run_mode": "trainer"}
    client = SimpleNamespace(get_run=lambda _: _run(binding, status="RUNNING"))
    with pytest.raises(ValueError, match="Only runs PhaseSweep created"):
        _publish(client, {"binding": binding, "projection": {}, "finish": "FAILED"})


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
            **_binding(),
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
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_unresolved_attempt_upload_worker_fails_the_run(tmp_path, monkeypatch, n_jobs, trainer):
    """Optuna drops callback errors in its final threaded batch; the fatal channel must not."""
    config, _uploads = _remote_example(tmp_path, monkeypatch)
    config["phases"][0].update(n_jobs=n_jobs, allow_no_gpu_isolation=True)
    if trainer == "failed":
        config["trial_command"] = (
            shlex.join([sys.executable, "-c", "raise SystemExit(1)"]) + " {config_path}"
        )
    experiment = Experiment.model_validate(config)
    error = UnsafeProcessCleanupError("upload worker still unresolved")
    uploaded_phases = []

    def unresolved(*args, **kwargs):
        if args[2]["name"].startswith("attempt-"):
            uploaded_phases.append(args[2]["binding"]["phase"])
            raise error
        return session.call_tracking(*args, **kwargs)

    monkeypatch.setattr(publication, "call_tracking", unresolved)
    reports = []
    with pytest.raises(UnsafeProcessCleanupError) as caught:
        run_experiment(experiment, terminal_callback=reports.append)
    assert caught.value is error
    [report] = reports
    assert report.primary_error is error
    assert not report.cleanup_confirmed
    assert report.winners is None
    assert read_status(experiment)["published_generation_id"] is None
    # The inheriting phase never launched.
    assert set(uploaded_phases) == {"depth"}


def test_failed_generation_reports_each_unrecorded_attempt_by_its_trial(tmp_path, monkeypatch):
    """The Optuna outcome, never a missing record, decides what an unrecorded attempt reports."""
    experiment = native_experiment(tmp_path)
    generation = "a" * 32
    write_record(
        tracking_root(experiment) / generation / "generation-binding.json",
        {**_binding(), "phase": "generation", "attempt_id": generation},
    )
    ledger = claim_ledger(validate_ledger(experiment))
    study = open_phase_study(ledger, experiment.phases[0])
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    attempts = {
        "1": ("managed", TrialState.RUNNING),  # interrupted before its callback
        "2": ("managed", TrialState.FAIL),  # its callback already recorded it
        "3": ("trainer", TrialState.FAIL),  # PhaseSweep never ends a trainer's run
        "4": ("managed", TrialState.COMPLETE),  # accepted; its record was never saved
        "5": ("managed", TrialState.PRUNED),
    }
    for number, (digit, (mode, state)) in enumerate(attempts.items()):
        attempt = digit * 32
        directory = _trial_dir_for(
            experiment, "depth", number, generation_id=generation, attempt_id=attempt
        )
        binding = {**_binding(), "phase": "depth", "attempt_id": attempt, "run_mode": mode}
        write_record(directory / BINDING_FILE, binding)
        write_record(directory / INHERITS_FILE, {"schema_version": 1, "inherits": {}})
        study.add_trial(
            create_trial(
                state=state,
                value=0.5 if state == TrialState.COMPLETE else None,
                user_attrs={
                    ATTEMPT_ID_ATTR: attempt,
                    GENERATION_ID_ATTR: generation,
                    OVERRIDES_ATTR: json.dumps({"model.depth": 2}),
                    FEASIBLE_ATTR: True,
                    DURATION_ATTR: 3.0,
                    constraint_attr("cost"): 1.0,
                },
            )
        )
    write_record(
        tracking_root(experiment) / generation / "delivery" / f"attempt-{'2' * 32}.json",
        {"schema_version": 1, "generation_id": generation, "state": "delivered"},
    )
    requests = []
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))
    publication.publish_failed_generation(experiment, generation, KeyboardInterrupt(), ledger)
    assert [(r["binding"]["attempt_id"][0], r["finish"]) for r in requests] == [
        ("1", "KILLED"),
        ("4", None),
        ("5", "KILLED"),
        ("a", "KILLED"),
    ]
    accepted = requests[1]["projection"]
    assert accepted["metrics"] == {
        "phasesweep.objective": 0.5,
        "phasesweep.feasible": 1.0,
        "phasesweep.duration_seconds": 3.0,
        "phasesweep.constraint.cost": 1.0,
    }
    assert accepted["params"] == {"phasesweep.override.model.depth": "2"}
    assert "phasesweep.outcome" not in accepted["tags"]


@pytest.mark.parametrize("storage", ["memory", "journal"])
def test_unsaved_accepted_attempt_keeps_its_accepted_projection(tmp_path, monkeypatch, storage):
    """A record write that fails after Optuna accepts a trial never turns it into a failure."""
    config, uploads = _remote_example(tmp_path, monkeypatch)
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
    config, uploads = _remote_example(tmp_path, monkeypatch)
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


def _fail_engine(tmp_path, monkeypatch, error):
    """Fail after generation preparation with a durable owned summary binding."""
    experiment = native_experiment(tmp_path)

    def fail(*args, generation_id, **kwargs):
        binding = {**_binding(), "phase": "generation", "generation_id": generation_id}
        write_record(tracking_root(experiment) / generation_id / "generation-binding.json", binding)
        raise error

    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail)
    return experiment


@pytest.mark.parametrize("failure", ["ordinary", "signal", "keyboard"])
def test_failure_projection_outage_preserves_error_and_retries(tmp_path, monkeypatch, failure):
    """Terminal failures retain status intent and original error despite upload outage."""
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
    monkeypatch.setattr(publication, "call_tracking", lambda *args: requests.append(args[2]))
    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 1}
    assert requests[0] == requests[1]


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
        "publish_failed_generation",
        lambda *args: (_ for _ in ()).throw(PermissionError("delivery filesystem read-only")),
    )
    with pytest.raises(RuntimeError) as caught:
        run_experiment(experiment)
    assert caught.value is error


def test_missing_binding_never_creates_or_discovers_a_summary(tmp_path, monkeypatch):
    """An unresolved create response cannot identify a run to finish by guessing."""
    experiment = native_experiment(tmp_path)
    monkeypatch.setattr(publication, "call_tracking", lambda *args: pytest.fail("No remote work"))
    publication.publish_failed_generation(
        experiment, "a" * 32, RuntimeError("create unresolved"), validate_ledger(experiment)
    )
    assert publication.delivery_status(experiment) == {"pending": 0, "delivered": 0}


def _remote_example(tmp_path, monkeypatch):
    """Run the native example against in-process MLflow answers and a no-op trainer."""
    config = yaml.safe_load((Path(__file__).parents[1] / "examples/mlflow.yaml").read_text())
    config.update(
        workdir=str(tmp_path / "runs"),
        storage=None,
        trial_command=f"{sys.executable} -c pass {{config_path}}",
    )
    uploads = []

    def respond(request, **kwargs):
        if request["action"] == "resolve":
            return {"experiment_id": "1"}
        if request["action"] == "create":
            return {**request["binding"], "run_id": "run-" + request["binding"]["attempt_id"]}
        if request["action"] == "capture":
            run = _run(request["binding"], metrics={"eval/loss": 1.0, "cost": 1.0})
            run.data.params = {"depth": "1", "rate": "0"}
            run.data.tags["trainer"] = "none"
            point = SimpleNamespace(value=1.0, step=1, timestamp=1)
            client = SimpleNamespace(get_run=lambda _: run, get_metric_history=lambda *_: [point])
            return _capture(client, request)
        uploads.append(request)
        return {}

    monkeypatch.setattr(session, "request_operation", respond)
    monkeypatch.setattr(worker, "request_operation", respond)
    return config, uploads


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
    config, uploads = _remote_example(tmp_path, monkeypatch)
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
    assert _recorded_generation_state(_generation_record_path(experiment, generation)) == (
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
    config, _uploads = _remote_example(tmp_path, monkeypatch)
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

    config, _uploads = _remote_example(tmp_path, monkeypatch)
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
    assert store.cleanup_recovery_required(handle) is (failure == "worker")


def test_sync_rebuilds_a_generation_the_pointer_left_after_a_post_commit_crash(
    tmp_path, monkeypatch
):
    """A controller that dies right after the pointer commit loses neither its proof nor its mirror.

    The first invocation dies before it writes the lifecycle record and before
    the in-process mirror. A later no-work replay moves the pointer, which must
    first leave the record that keeps the generation published, so
    ``sync-tracking`` still rebuilds its winner and summary uploads.
    """
    from phasesweep.engine import generation as generation_ops
    from phasesweep.engine.paths import _last_successful_generation_path

    config, uploads = _remote_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"  # the replay must find the studies
    experiment = Experiment.model_validate(config)
    pointer_path = _last_successful_generation_path(experiment)

    def die_before_record(**kwargs):
        raise OSError("controller died after the pointer commit")

    with monkeypatch.context() as crash:
        crash.setattr(generation_ops, "_write_generation_record_once", die_before_record)
        crash.setattr(publication, "publish_generation", lambda *args: None)
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
    second = reports[0].generation_id
    assert second != first
    assert yaml.safe_load(pointer_path.read_text())["generation_id"] == second
    assert _recorded_generation_state(_generation_record_path(experiment, first)) == "published"
    assert publication._published(experiment, first)

    assert publication.sync_tracking(experiment) == {"pending": 0, "delivered": 7}
    assert all((first_delivery / name).exists() for name in intents)
    assert sorted(request["name"] for request in uploads) == [
        "generation",
        "winner-depth",
        "winner-rate",
    ]


def test_sync_binds_and_finishes_a_run_created_after_its_attempt_gave_up(tmp_path, monkeypatch):
    """A run a timed-out creation made late is bound by sync and finished with its failed trial."""
    from phasesweep.tracking.mlflow import OperationOutcomeUnknownError

    config, uploads = _remote_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    late: dict[str, str] = {}

    def lost_first_attempt_creation(request, **kwargs):
        binding = request.get("binding", {})
        if request["action"] == "create" and binding["phase"] != "generation" and not late:
            late.update(binding, run_id="late-run")
            raise OperationOutcomeUnknownError("response lost")
        if request["action"] == "find":
            appeared = binding["attempt_id"] == late.get("attempt_id") and kwargs.get("appeared")
            return {"found": True, "binding": late} if appeared else {"found": False}
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", lost_first_attempt_creation)
    reports = []
    run_experiment(experiment, terminal_callback=reports.append)
    generation = reports[0].generation_id
    [pending] = _experiment_dir(experiment).glob("*/trial_*/mlflow-binding.pending.json")
    assert not (pending.parent / BINDING_FILE).exists()

    def appeared(request, **kwargs):
        return lost_first_attempt_creation(request, **kwargs, appeared=True)

    monkeypatch.setattr(session, "request_operation", appeared)
    uploads.clear()
    # Three accepted attempts, two winners, the summary, and the late run.
    assert publication.sync_tracking(experiment, generation) == {"pending": 0, "delivered": 7}
    assert not pending.exists()
    assert session.read_binding(pending.parent)["run_id"] == "late-run"
    [finished] = [r for r in uploads if r["binding"]["run_id"] == "late-run"]
    assert finished["finish"] == "FAILED"
