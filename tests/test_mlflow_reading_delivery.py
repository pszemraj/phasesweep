"""Every published generation delivers its own reading of the attempts it published."""

from __future__ import annotations

import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.read import read_status
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


def _run_shows(uploads: list[JsonObject], attempt: str) -> tuple[object, object, object]:
    """Fold the uploads to one attempt's run as MLflow keeps them.

    :return tuple[object, object, object]: The last status set, the objective point with the
        newest timestamp, and the metric of the attempt artifact uploaded last.
    """
    sent = [upload for upload in uploads if upload["binding"]["attempt_id"] == attempt]
    readings = [upload for upload in sent if upload["name"] == f"attempt-{attempt}"]
    status = [upload["finish"] for upload in readings if upload["finish"]][-1]
    newest = max(readings, key=lambda upload: upload["projection"]["timestamp"])
    objective = newest["projection"]["metrics"]["phasesweep.objective"]
    return status, objective, readings[-1]["projection"]["summary"]["metric"]


def _silent_tracker(outage: pytest.MonkeyPatch) -> None:
    """Make every MLflow request but a capture go unanswered."""
    respond = session.request_operation

    def silent(request: JsonObject, **kwargs: object) -> JsonObject:
        if request["action"] == "capture":
            return respond(request, **kwargs)
        raise OperationUnavailableError("tracking server did not answer in time")

    outage.setattr(session, "request_operation", silent)
    outage.setattr(worker, "request_operation", silent)


def test_a_run_ends_on_the_newest_of_two_pending_readings(tmp_path, monkeypatch):
    """Two readings queued in one outage are sent oldest first, whatever their generation IDs."""
    config, uploads, attempt = _reread_setup(tmp_path, monkeypatch)
    original = deepcopy(config)
    original["metric"]["extractor"]["metric_key"] = "eval/loss"
    older, newer = "f" * 32, "0" * 32  # the newer generation's ID sorts first
    with monkeypatch.context() as outage:
        _silent_tracker(outage)
        experiment = Experiment.model_validate(config)
        assert run_experiment(experiment, generation_id=older)["depth"].metric == 3.0
        newest = Experiment.model_validate(original)
        assert run_experiment(newest, generation_id=newer)["depth"].metric == 1.0

    assert publication.sync_tracking(newest)["pending"] == 0
    assert _run_shows(uploads, attempt) == ("FINISHED", 1.0, 1.0)
    winners = [upload for upload in uploads if upload["name"] == "winner-depth"]
    assert f"phasesweep.winner_generation.{newer}" in winners[-1]["projection"]["tags"]


def test_scoped_sync_delivers_an_equivalent_older_pending_attempt(tmp_path, monkeypatch):
    """The newest scope includes the pending attempt upload its reading shares."""
    config, uploads, attempt = _reread_setup(tmp_path, monkeypatch)
    experiment = Experiment.model_validate(config)
    newer_config = deepcopy(config)
    newer_config["metric"]["goal"] = "maximize"
    newest = Experiment.model_validate(newer_config)
    older, newer = "f" * 32, "0" * 32
    with monkeypatch.context() as outage:
        _silent_tracker(outage)
        assert run_experiment(experiment, generation_id=older)["depth"].metric == 3.0
        assert run_experiment(newest, generation_id=newer)["depth"].metric == 3.0
    name = f"attempt-{attempt}"
    earlier_path = session.delivery_record_path(experiment, older, name)
    later_path = session.delivery_record_path(experiment, newer, name)
    before = session.read_record(earlier_path)
    assert before["state"] == "pending"
    assert session.read_record(later_path)["state"] == "delivered"
    assert _run_shows(uploads, attempt) == ("FINISHED", 1.0, 1.0)
    assert publication.delivery_counts(newest, newer, shared=True).records == 3

    assert publication.sync_tracking(newest, newer)["pending"] == 0
    assert _run_shows(uploads, attempt) == ("FINISHED", 3.0, 3.0)
    assert session.read_record(earlier_path) == {**before, "state": "delivered"}
    attempts_sent = sum(request["name"] == name for request in uploads)
    assert publication.sync_tracking(newest)["pending"] == 0
    assert sum(request["name"] == name for request in uploads) == attempts_sent


def test_rebuilding_an_older_generations_records_keeps_the_newer_reading(tmp_path, monkeypatch):
    """Records rebuilt for a generation that has none left take its lifecycle record's time."""
    config, uploads, attempt = _reread_setup(tmp_path, monkeypatch)
    reports: list[TerminalReport] = []
    experiment = Experiment.model_validate(config)
    assert run_experiment(experiment, terminal_callback=reports.append)["depth"].metric == 3.0
    first = next(
        path.parent.parent.name
        for path in session.tracking_root(experiment).glob(f"*/delivery/attempt-{attempt}.json")
        if path.parent.parent.name != reports[0].generation_id
    )
    for path in (session.tracking_root(experiment) / first / "delivery").iterdir():
        path.unlink()
    before = len(uploads)

    assert publication.sync_tracking(experiment)["pending"] == 0
    rebuilt = session.read_record(
        session.tracking_root(experiment) / first / "delivery" / f"attempt-{attempt}.json"
    )
    assert rebuilt["state"] == "delivered"
    assert {upload["name"] for upload in uploads[before:]} == {"winner-depth", "generation"}
    assert _run_shows(uploads, attempt) == ("FINISHED", 3.0, 3.0)


def test_rebuilding_a_past_reading_excludes_trials_added_later(tmp_path, monkeypatch):
    """An older reading rebuilds its own failed and measured attempts after its study grows."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config.update(storage="auto", phases=config["phases"][:1])
    phase = config["phases"][0]
    phase["allow_partial_grid"] = True
    phase["search_space"]["model.depth"]["choices"] = [1, 2, 3]
    marker = tmp_path / "trainer-launched"
    trainer = (
        f"from pathlib import Path; marker = Path({str(marker)!r}); "
        "launched = marker.exists(); marker.touch(); raise SystemExit(0 if launched else 7)"
    )
    config["trial_command"] = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(trainer)} {{config_path}}"
    )
    original_id, historical_id, grown_id = (digit * 32 for digit in "123")
    run_experiment(Experiment.model_validate(config), generation_id=original_id)
    config["metric"]["extractor"]["metric_key"] = "cost"
    experiment = Experiment.model_validate(config)
    run_experiment(experiment, generation_id=historical_id)
    history = session.tracking_root(experiment) / historical_id / "delivery"
    retained = {path.name: session.read_record(path) for path in history.glob("attempt-*.json")}
    assert {record["finish"] for record in retained.values()} == {"FAILED", "FINISHED"}
    assert {record["binding"]["generation_id"] for record in retained.values()} == {original_id}

    phase["n_trials"] = 3
    grown = Experiment.model_validate(config)
    run_experiment(grown, generation_id=grown_id)
    for path in history.glob("attempt-*.json"):
        path.unlink()

    assert publication.sync_tracking(grown, historical_id)["pending"] == 0
    rebuilt = {path.name: session.read_record(path) for path in history.glob("attempt-*.json")}
    assert rebuilt.keys() == retained.keys()
    for name, record in rebuilt.items():
        assert record["finish"] == retained[name]["finish"]
        assert record["binding"] == retained[name]["binding"]


@pytest.mark.parametrize(
    "newest_first, same_values",
    [(False, False), (True, False), (True, True)],
    ids=["in-order", "scoped-newest-first", "scoped-coalesced"],
)
def test_the_winner_tag_moves_with_the_newest_reading(
    tmp_path, monkeypatch, newest_first, same_values
):
    """The latest reading owns the winner tag and artifact even when an older upload retries."""
    pytest.importorskip("mlflow")
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config.update(storage="auto", phases=config["phases"][:1])
    respond = worker.request_operation
    readings = iter([(1.0, 1.0), (2.0, 2.0)] if same_values else [(1.0, 3.0), (2.0, 2.0)])
    values: dict[str, tuple[float, float]] = {}
    runs: dict[str, SimpleNamespace] = {}
    artifacts: dict[tuple[str, str], object] = {}

    def client(binding: JsonObject) -> SimpleNamespace:
        run = runs.setdefault(binding["run_id"], mlflow_run(binding))
        return SimpleNamespace(
            get_run=lambda _: run,
            log_batch=lambda _, *, tags, **kwargs: run.data.tags.update(
                {tag.key: tag.value for tag in tags}
            ),
            log_artifact=lambda run_id, path, **kwargs: artifacts.__setitem__(
                (run_id, Path(path).name), json.loads(Path(path).read_text())
            ),
            set_terminated=lambda _, status: setattr(run.info, "status", status),
            delete_tag=lambda _, key: run.data.tags.pop(key),
        )

    def tracker(request: JsonObject, **kwargs: object) -> JsonObject:
        if request["action"] == "publish":
            uploads.append(request)
            return _publish(client(request["binding"]), request)
        result = respond(request, **kwargs)
        if request["action"] == "capture":
            attempt = request["binding"]["attempt_id"]
            loss, cost = values.setdefault(attempt, next(readings, (5.0, 5.0)))
            for key, observation in result["observations"].items():
                observation["value"] = cost if "cost" in key else loss
        return result

    monkeypatch.setattr(worker, "request_operation", tracker)
    monkeypatch.setattr(session, "request_operation", tracker)
    reports: list[TerminalReport] = []
    original = Experiment.model_validate(config)
    first = run_experiment(original, terminal_callback=reports.append)
    config["metric"]["extractor"]["metric_key"] = "cost"
    if same_values:
        config["metric"]["goal"] = "maximize"
    changed = Experiment.model_validate(config)
    if newest_first:
        with monkeypatch.context() as outage:
            _silent_tracker(outage)
            second = run_experiment(changed, terminal_callback=reports.append)
            run_experiment(original, terminal_callback=reports.append)
        if same_values:
            first_name = f"attempt-{first['depth'].attempt_id}"
            older_path = session.delivery_record_path(
                original, reports[1].generation_id, first_name
            )
            newer_path = session.delivery_record_path(
                original, reports[2].generation_id, first_name
            )
            older_record = session.read_record(older_path)
            assert older_record["state"] == "pending"
            assert session.read_record(newer_path)["state"] == "delivered"
        assert publication.sync_tracking(original, reports[2].generation_id)["pending"] == 0
        newest_artifacts = dict(artifacts)
        assert publication.sync_tracking(original)["pending"] == 0
        # The older generation owes its own summary. The shared attempt runs keep their artifacts.
        summary = artifacts.pop((f"run-{reports[1].generation_id}", "generation.json"))
        assert summary["generation_id"] == reports[1].generation_id
        assert artifacts == newest_artifacts
        if same_values:
            assert session.read_record(older_path) == {**older_record, "state": "delivered"}
    else:
        second = run_experiment(changed, terminal_callback=reports.append)
    earlier, later = first["depth"].attempt_id, second["depth"].attempt_id
    assert earlier is not None and later is not None and earlier != later

    tags = {attempt: runs[f"run-{attempt}"].data.tags for attempt in (earlier, later)}
    winner, loser = (earlier, later) if newest_first else (later, earlier)
    assert "phasesweep.winner" not in tags[loser]
    assert tags[winner]["phasesweep.winner"] == "true"
    assert tags[earlier][f"phasesweep.winner_generation.{reports[0].generation_id}"] == "depth"
    assert tags[later][f"phasesweep.winner_generation.{reports[1].generation_id}"] == "depth"


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


@pytest.mark.parametrize("scoped", [False, True], ids=["experiment", "generation"])
def test_sync_discovers_a_superseded_publication_with_no_tracking_directory(
    tmp_path, monkeypatch, scoped
):
    """A superseded MLflow publication can recover all delivery work from its lifecycle record."""
    config, uploads, attempt = _reread_setup(tmp_path, monkeypatch)
    reread = Experiment.model_validate(config)
    original = deepcopy(config)
    original["metric"]["extractor"]["metric_key"] = "eval/loss"
    restored = Experiment.model_validate(original)
    # A return to either initialized reading needs neither training nor capture workers.
    run_experiment(reread)
    run_experiment(restored)
    earlier, later = "a" * 32, "b" * 32
    earlier_tracking = session.tracking_root(reread) / earlier
    with monkeypatch.context() as without_mirror:
        without_mirror.setattr(publication, "deliver_closeout", lambda *args, **kwargs: None)
        assert run_experiment(reread, generation_id=earlier)["depth"].attempt_id == attempt
        assert read_status(reread)["published_generation_id"] == earlier
        assert not earlier_tracking.exists()
        assert run_experiment(restored, generation_id=later)["depth"].metric == 1.0
    assert read_status(restored)["published_generation_id"] == later
    assert publication._published(reread, earlier)
    assert not earlier_tracking.exists()

    uploads.clear()
    assert publication.sync_tracking(restored, earlier if scoped else None)["pending"] == 0
    delivery = earlier_tracking / "delivery"
    reading = session.read_record(delivery / f"attempt-{attempt}.json")
    assert reading["projection"]["metrics"]["phasesweep.objective"] == 3.0
    [summary] = [
        upload
        for upload in uploads
        if upload["name"] == "generation" and upload["binding"]["generation_id"] == earlier
    ]
    assert summary["finish"] == "FINISHED"
    assert summary_binding_path(reread, earlier).is_file()
    assert session.read_record(delivery / "winner-depth.json")["state"] == "delivered"


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
