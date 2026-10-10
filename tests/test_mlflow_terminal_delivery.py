"""Failed-generation delivery preserves engine outcome and cleanup authority."""

from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from phasesweep.engine import run_experiment
from phasesweep.errors import ProcessCleanupUncertainError, UnsafeProcessCleanupError
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from phasesweep.tracking import publication
from phasesweep.tracking.mlflow import _publish
from phasesweep.tracking.session import read_record, tracking_root, write_record
from tests.test_mlflow import _binding, _run, native_experiment


@pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
def test_terminal_acknowledgment_retry_does_not_reterminate(status):
    """A lost terminal response retries artifacts without reopening or changing status."""
    pytest.importorskip("mlflow")
    binding = {**_binding(), "phase": "generation"}
    uploaded = []
    client = SimpleNamespace(
        get_run=lambda _: _run(binding, status=status),
        log_batch=lambda *args, **kwargs: uploaded.append("batch"),
        log_artifact=lambda *args, **kwargs: uploaded.append("artifact"),
        set_terminated=lambda *args, **kwargs: pytest.fail("Already terminal"),
    )
    _publish(
        client,
        {
            "binding": binding,
            "projection": publication._projection(metrics={}, params={}, tags={}, summary={}),
            "finish": status,
        },
    )
    assert uploaded == ["batch", "artifact"]


@pytest.mark.parametrize(
    "actual,requested",
    [
        ("FAILED", "FINISHED"),
        ("KILLED", "FINISHED"),
        ("FAILED", "KILLED"),
        ("KILLED", "FAILED"),
        ("FINISHED", "FAILED"),
        ("FINISHED", "KILLED"),
    ],
)
def test_conflicting_terminal_status_never_uploads_or_changes(actual, requested):
    """A retry cannot rewrite another terminal outcome or publish its artifact."""
    pytest.importorskip("mlflow")
    binding = {**_binding(), "phase": "generation"}
    client = SimpleNamespace(get_run=lambda _: _run(binding, status=actual))
    with pytest.raises(ValueError, match="different terminal status"):
        _publish(client, {"binding": binding, "projection": {}, "finish": requested})


@pytest.mark.parametrize(
    "binding",
    [
        _binding(),
        {**_binding(), "phase": "generation", "run_mode": "trainer"},
    ],
)
def test_failure_delivery_cannot_terminate_attempt_or_trainer_owned_run(binding):
    pytest.importorskip("mlflow")
    client = SimpleNamespace(get_run=lambda _: _run(binding, status="RUNNING"))
    with pytest.raises(ValueError, match="Only PhaseSweep-owned"):
        _publish(client, {"binding": binding, "projection": {}, "finish": "FAILED"})


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
    publication.publish_failed_generation(experiment, "a" * 32, RuntimeError("create unresolved"))
    assert publication.delivery_status(experiment) == {"pending": 0, "delivered": 0}
