"""Every published generation delivers its own reading of the attempts it published."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import OperationUnavailableError, _publish
from phasesweep.tracking.records import OUTCOME_TAG, STATUS_BEFORE_DELIVERY
from phasesweep.tracking.session import summary_binding_path
from tests.tracking_helpers import mlflow_binding, mlflow_run, remote_mlflow_example


def _reread_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], list[JsonObject], str]:
    """Publish one trial read as ``eval/loss`` and return a config that rereads it as ``cost``.

    :return tuple[dict[str, object], list[JsonObject], str]: The rereading config, the uploads
        list and the published attempt's ID.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config.update(storage="auto", phases=config["phases"][:1])
    respond = worker.request_operation

    def capture_cost(request: JsonObject, **kwargs: object) -> JsonObject:
        result = respond(request, **kwargs)
        if request["action"] == "capture":
            for key, observation in result["observations"].items():
                if "cost" in key:
                    observation["value"] = 3.0
        return result

    monkeypatch.setattr(worker, "request_operation", capture_cost)
    monkeypatch.setattr(session, "request_operation", capture_cost)
    winner = run_experiment(Experiment.model_validate(config))["depth"]
    assert winner.attempt_id is not None and winner.metric == 1.0
    reread = deepcopy(config)
    reread["metric"]["extractor"]["metric_key"] = "cost"
    return reread, uploads, winner.attempt_id


def test_a_generation_that_launched_nothing_delivers_its_reading(tmp_path, monkeypatch):
    """A reread sends its objective, its winner tag and its own summary run without a launch."""
    config, uploads, attempt = _reread_setup(tmp_path, monkeypatch)
    uploads.clear()
    reports: list[TerminalReport] = []
    experiment = Experiment.model_validate(config)
    assert run_experiment(experiment, terminal_callback=reports.append)["depth"].metric == 3.0
    generation = reports[0].generation_id

    sent = {upload["name"]: upload for upload in uploads}
    assert sent[f"attempt-{attempt}"]["projection"]["metrics"]["phasesweep.objective"] == 3.0
    winner_tags = sent["winner-depth"]["projection"]["tags"]
    assert winner_tags[f"phasesweep.winner_generation.{generation}"] == "depth"
    assert sent["generation"]["binding"]["generation_id"] == generation
    assert summary_binding_path(experiment, generation).is_file()
    assert publication.delivery_counts(experiment).pending == 0


def test_an_outage_leaves_a_rereads_summary_run_for_sync_tracking(tmp_path, monkeypatch):
    """A silent tracker at the closeout keeps the reread published; sync-tracking creates its summary run."""
    config, uploads, attempt = _reread_setup(tmp_path, monkeypatch)
    respond = session.request_operation
    reports: list[TerminalReport] = []
    experiment = Experiment.model_validate(config)
    with monkeypatch.context() as outage:

        def silent(request: JsonObject, **kwargs: object) -> JsonObject:
            if request["action"] == "capture":
                return respond(request, **kwargs)
            raise OperationUnavailableError("tracking server did not answer in time")

        outage.setattr(session, "request_operation", silent)
        outage.setattr(worker, "request_operation", silent)
        assert run_experiment(experiment, terminal_callback=reports.append)["depth"].metric == 3.0
    generation = reports[0].generation_id
    assert not summary_binding_path(experiment, generation).is_file()
    # Both attempts of the phase and its winner reference, and the summary run to create.
    counts = publication.delivery_counts(experiment, generation)
    assert (counts.records, counts.creations) == (3, 1)

    uploads.clear()
    assert publication.sync_tracking(experiment, generation)["pending"] == 0
    assert summary_binding_path(experiment, generation).is_file()
    names = [upload["name"] for upload in uploads]
    assert len(names) == 4 and {f"attempt-{attempt}", "winner-depth"} <= set(names)
    assert names[-1] == "generation"


@pytest.mark.parametrize("delivered", [True, False], ids=["delivered-failed", "trainer-failed"])
def test_an_accepted_reading_finishes_only_a_run_phasesweep_failed(delivered):
    """A run PhaseSweep delivered FAILED is finished and loses that delivery's tags; a trainer's failure stands."""
    pytest.importorskip("mlflow")
    binding = mlflow_binding()
    run = mlflow_run(binding, status="FAILED")
    if delivered:
        run.data.tags.update({OUTCOME_TAG: "FAILED", STATUS_BEFORE_DELIVERY: "FINISHED"})
    calls: list[object] = []
    client = SimpleNamespace(
        get_run=lambda _: run,
        log_batch=lambda *args, **kwargs: calls.append("batch"),
        log_artifact=lambda *args, **kwargs: calls.append("artifact"),
        set_terminated=lambda run_id, status: calls.append(status),
        delete_tag=lambda run_id, key: calls.append(("delete", key)),
    )
    request = {
        "name": f"attempt-{binding['attempt_id']}",
        "binding": binding,
        "projection": publication._projection(
            metrics={"phasesweep.objective": 2.0}, params={}, tags={}, summary={}
        ),
        "finish": "FINISHED",
    }
    if not delivered:
        with pytest.raises(ValueError, match="ended unsuccessfully"):
            _publish(client, request)
        assert calls == []
        return
    _publish(client, request)
    assert calls == [
        "batch",
        "artifact",
        "FINISHED",
        ("delete", OUTCOME_TAG),
        ("delete", STATUS_BEFORE_DELIVERY),
    ]
