"""Qualification against a temporary HTTP MLflow 3.17 server."""

from __future__ import annotations

import builtins
import http.client
import http.server
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import yaml

from phasesweep.config.models import Experiment
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.read import read_result, read_status
from phasesweep.engine.run import run_experiment
from phasesweep.tracking.publication import delivery_status, sync_tracking

pytestmark = pytest.mark.integration


@contextmanager
def _server(tmp_path):
    """Provision explicit SQLite and proxied artifact storage outside the tree."""
    pytest.importorskip("mlflow.server")
    from mlflow import MlflowClient

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uri = f"http://127.0.0.1:{port}"
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
                "--backend-store-uri",
                f"sqlite:///{tmp_path / 'mlflow.db'}",
                "--serve-artifacts",
                "--artifacts-destination",
                str(tmp_path / "artifacts"),
            ],
            stdout=log,
            stderr=log,
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
        try:
            stop = time.monotonic() + 45
            while True:
                try:
                    with urllib.request.urlopen(uri + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= stop:
                        pytest.fail((tmp_path / "server.log").read_text())
                    time.sleep(0.1)
            client = MlflowClient(tracking_uri=uri)
            yield uri, client, process
        finally:
            process.terminate()
            process.wait(timeout=15)


@pytest.fixture(scope="module")
def shared_server(tmp_path_factory):
    """Keep one HTTP server for cases that leave its process running."""
    with _server(tmp_path_factory.mktemp("mlflow-server")) as server:
        yield server


@pytest.fixture
def local_server(shared_server, tmp_path):
    """Give each case a fresh explicit experiment on the shared server."""
    uri, client, process = shared_server
    return uri, client, client.create_experiment(tmp_path.name), process


@pytest.fixture
def offline_server(tmp_path):
    """Give an offline replay its own server process to terminate."""
    with _server(tmp_path) as (uri, client, process):
        yield uri, client, client.create_experiment("phasesweep-native"), process


@pytest.fixture
def held_create(local_server, monkeypatch):
    """Front the real server with a proxy that holds its first run-creation POST.

    The held POST reaches the server only after the client timed out, so it
    commits late, as a request still executing server-side would. It is
    released after ``release_after_searches`` run searches, or by ``release``.
    """
    uri, client, experiment_id, _process = local_server
    upstream = urlsplit(uri)
    state = SimpleNamespace(
        lock=threading.Lock(),
        creates=0,
        searches=0,
        release_after_searches=None,
        release=threading.Event(),
        committed=threading.Event(),
        client=client,
        experiment_id=experiment_id,
    )

    class Proxy(http.server.BaseHTTPRequestHandler):
        def forward(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            creating = self.command == "POST" and self.path.endswith("/runs/create")
            searching = self.command == "POST" and self.path.endswith("/runs/search")
            held = False
            if creating:
                with state.lock:
                    state.creates += 1
                    held = state.creates == 1
            if held:
                assert state.release.wait(60), "The held creation was never released."
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=30)
            hop_by_hop = {"host", "connection"}
            connection.request(
                self.command,
                self.path,
                body=body,
                headers={k: v for k, v in self.headers.items() if k.lower() not in hop_by_hop},
            )
            reply = connection.getresponse()
            payload = reply.read()
            connection.close()
            if held:
                state.committed.set()
            try:
                self.send_response(reply.status)
                for key, value in reply.getheaders():
                    if key.lower() not in {"transfer-encoding", "connection", "content-length"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                pass  # The client gave up on the held request.
            if searching:
                with state.lock:
                    state.searches += 1
                    due = state.release_after_searches
                    if due is not None and state.searches >= due:
                        state.release.set()

        do_GET = do_POST = do_PUT = do_DELETE = forward

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.uri = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    try:
        yield state
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _late_creation(server, tmp_path, monkeypatch, *, timeout_seconds):
    """Prepare one attempt's creation through the proxy, with the worker run in-process.

    The stand-in reports outcomes to ``_create_bound`` as ``request_operation``
    does, but runs the worker's actions against the real SDK without a subprocess.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from phasesweep.engine.paths import _trial_dir_for
    from phasesweep.runtime.json import json_object
    from phasesweep.tracking import mlflow as worker
    from phasesweep.tracking import session
    from tests.test_mlflow import _binder, _identity, native_experiment

    calls = []

    def request_operation(request, **kwargs):
        calls.append(request["action"])
        client = MlflowClient(tracking_uri=request["settings"]["tracking_uri"])
        try:
            if request["action"] == "create":
                return json_object(worker._create(client, request))
            return json_object(worker._find(client, request))
        except MlflowException as exc:
            # The worker reports a server rejection as status "error".
            raise RuntimeError(str(exc)) from exc

    monkeypatch.setattr(session, "request_operation", request_operation)
    experiment = native_experiment(tmp_path)
    experiment = experiment.model_copy(
        update={
            "mlflow": experiment.mlflow.model_copy(
                update={
                    "tracking_uri": server.uri,
                    "experiment_id": server.experiment_id,
                    "experiment_name": None,
                    "timeout_seconds": timeout_seconds,
                    "poll_seconds": 0.05,
                }
            )
        }
    )
    identity = {
        **_identity(),
        "tracking_uri": server.uri,
        "experiment": experiment.experiment,
        "experiment_id": server.experiment_id,
    }
    trial_dir = _trial_dir_for(
        experiment,
        identity["phase"],
        0,
        generation_id=identity["generation_id"],
        attempt_id=identity["attempt_id"],
    )
    trial_dir.mkdir(parents=True)
    bind, path = _binder(trial_dir, "attempt", experiment=experiment, identity=identity)
    return bind, path, calls, experiment


def test_create_committing_after_a_timeout_binds_the_one_run(held_create, tmp_path, monkeypatch):
    """A timed-out creation that commits mid-reconciliation is bound, never re-sent."""
    from phasesweep.tracking import session

    held_create.release_after_searches = 2
    bind, path, calls, _experiment = _late_creation(
        held_create, tmp_path, monkeypatch, timeout_seconds=4
    )
    bound = bind()
    assert held_create.committed.wait(10)
    runs = held_create.client.search_runs([held_create.experiment_id])
    assert [run.info.run_id for run in runs] == [bound["run_id"]]
    assert runs[0].data.tags["phasesweep.attempt_id"] == bound["attempt_id"]
    assert held_create.creates == 1
    assert calls == ["create", "find"]
    assert session.read_binding(path.parent) == bound
    assert not path.with_name("mlflow-binding.pending.json").exists()


@pytest.mark.parametrize("recovery", ["restart", "sync"])
def test_create_committing_after_the_controller_gave_up_is_bound_later(
    held_create, tmp_path, monkeypatch, recovery
):
    """An unresolved creation keeps its record, and its late run is bound without a new POST."""
    from phasesweep.tracking import session

    bind, path, calls, experiment = _late_creation(
        held_create, tmp_path, monkeypatch, timeout_seconds=2
    )
    pending = path.with_name("mlflow-binding.pending.json")
    with pytest.raises(RuntimeError, match="unresolved"):
        bind()
    assert pending.exists() and not path.exists()
    assert held_create.client.search_runs([held_create.experiment_id]) == []
    held_create.release.set()
    assert held_create.committed.wait(10)
    if recovery == "restart":
        bound = bind()
    else:
        generation_id = "a" * 32
        session.resolve_pending_creations(experiment, {}, generation_id)
        bound = session.read_binding(path.parent)
    runs = held_create.client.search_runs([held_create.experiment_id])
    assert [run.info.run_id for run in runs] == [bound["run_id"]]
    assert held_create.creates == 1
    assert "create" not in calls[1:]
    assert not pending.exists()


@pytest.mark.parametrize("mode", ["managed", "trainer"])
def test_two_phases_native_http_then_offline(offline_server, tmp_path, monkeypatch, mode):
    """Both ownership modes publish four trials and replay without the SDK/server."""
    uri, client, experiment_id, process = offline_server
    config = yaml.safe_load((Path(__file__).parents[1] / "examples" / "mlflow.yaml").read_text())
    config.update(
        workdir=str(tmp_path / "work"), storage=f"journal:///{tmp_path / 'study.journal'}"
    )
    config["trial_command"] = (
        f"{sys.executable} -m phasesweep.examples.mlflow_train --config {{config_path}}"
    )
    config["mlflow"].update(tracking_uri=uri, run_mode=mode, timeout_seconds=15, poll_seconds=0.05)
    if mode == "trainer":
        parent = client.create_run(experiment_id)
        trainer = tmp_path / "trainer.py"
        source = (
            Path(__file__).parents[1] / "src" / "phasesweep" / "examples" / "mlflow_train.py"
        ).read_text()
        source = source.replace(
            'mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID"))',
            f'mlflow.start_run(tags={{"mlflow.parentRunId": "{parent.info.run_id}"}})',
        )
        trainer.write_text(source)
        config["trial_command"] = f"{sys.executable} {trainer} --config {{config_path}}"
    experiment = Experiment.model_validate(config)
    import phasesweep.tracking.publication as publication

    actual_call = publication.call_tracking
    uploads = []

    def first_upload_disconnected(*args, **kwargs):
        uploads.append(args[2]["action"])
        if len(uploads) == 1:
            raise RuntimeError("Simulated first result-upload outage")
        return actual_call(*args, **kwargs)

    monkeypatch.setattr(publication, "call_tracking", first_upload_disconnected)
    winners = run_experiment(experiment)
    assert winners["depth"].effective_overrides["model.depth"] == 2
    assert winners["depth"].metric == 9
    assert winners["rate"].effective_overrides == {"model.depth": 2, "optimizer.rate": 3}
    assert winners["rate"].metric == 0
    # Every confirmed creation removed its pending record.
    assert not list((tmp_path / "work").rglob("*.pending.json"))
    status = read_status(experiment)
    generation_id = status["published_generation_id"]
    assert delivery_status(experiment, generation_id) == {"pending": 1, "delivered": 6}
    assert sync_tracking(experiment, generation_id) == {"pending": 0, "delivered": 7}
    assert delivery_status(experiment, generation_id) == {"pending": 0, "delivered": 7}
    runs = client.search_runs(
        [experiment_id], filter_string="tags.`phasesweep.phase` != 'generation'"
    )
    assert len(runs) == 4
    summary = client.search_runs(
        [experiment_id], filter_string="tags.`phasesweep.phase` = 'generation'"
    )
    assert len(summary) == 1
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.metrics["phasesweep.objective"] == run.data.metrics["eval/loss"]
        assert run.data.metrics["phasesweep.feasible"] == 1
        assert run.data.params["phasesweep.override.model.depth"] in {"1", "2"}
        if mode == "managed":
            tags = run.data.tags
            # The trainer's PHASESWEEP_RUN_NAME names the managed run too.
            assert run.info.run_name == "-".join(
                [experiment.experiment]
                + [tags[f"phasesweep.{k}"] for k in ("phase", "trial_number", "attempt_id")]
            )
        assert run.data.tags["mlflow.parentRunId"] == (
            summary[0].info.run_id if mode == "managed" else parent.info.run_id
        )
    assert summary[0].info.status == "FINISHED"
    publication.publish_failed_generation(
        experiment, generation_id, RuntimeError("late error"), validate_ledger(experiment)
    )
    assert client.get_run(summary[0].info.run_id).info.status == "FINISHED"
    assert delivery_status(experiment, generation_id) == {"pending": 0, "delivered": 7}
    assert [a.path for a in client.list_artifacts(summary[0].info.run_id, "phasesweep")] == [
        "phasesweep/generation.json"
    ]
    rate_run = winners["rate"].objective_provenance["remote_capture"]["binding"]["run_id"]
    assert {a.path for a in client.list_artifacts(rate_run, "phasesweep")} == {
        f"phasesweep/attempt-{winners['rate'].attempt_id}.json",
        "phasesweep/winner-rate.json",
    }
    rate_capture = winners["rate"].objective_provenance["remote_capture"]
    assert rate_capture["binding"]["run_id"] != winners["rate"].attempt_id
    assert all(point["step"] == 1 for point in rate_capture["observations"].values())
    process.terminate()
    process.wait(timeout=15)
    before = [
        (w.phase, w.metric, w.params)
        for w in read_result(experiment, phase_names=[phase.name for phase in experiment.phases])[1]
    ]
    original_import = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise AssertionError("Offline scientific reads must not import MLflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    import phasesweep.tracking.mlflow as transport

    def no_network(*args, **kwargs):
        raise AssertionError("Offline scientific replay must not launch an SDK worker")

    monkeypatch.setattr(transport, "request_operation", no_network)
    replay = run_experiment(experiment)
    assert {k: w.metric for k, w in replay.items()} == {k: w.metric for k, w in winners.items()}
    assert [
        (w.phase, w.metric, w.params)
        for w in read_result(experiment, phase_names=[phase.name for phase in experiment.phases])[1]
    ] == before
    assert all(p["trials"]["COMPLETE"] == 2 for p in read_status(experiment)["phases"])


def test_blocked_tracking_worker_is_cleaned(tmp_path, monkeypatch):
    """A blocked SDK worker leaves confirmed process cleanup and no active entry."""
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.runtime.process import read_attempt_lifecycle
    from phasesweep.tracking.session import call_tracking, tracking_root
    from tests.test_mlflow import native_experiment

    sdk = tmp_path / "sdk/mlflow"
    sdk.mkdir(parents=True)
    (sdk / "__init__.py").write_text(
        "import time\nclass MlflowClient:\n    def __init__(self, **kwargs): pass\n    def get_experiment_by_name(self, name): time.sleep(30)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(sdk.parent))
    experiment = native_experiment(tmp_path).model_copy(
        update={
            "mlflow": native_experiment(tmp_path).mlflow.model_copy(update={"timeout_seconds": 1.0})
        }
    )
    with pytest.raises(TimeoutError):
        call_tracking(experiment, "a" * 32, {"action": "resolve"}, {})
    directories = list((tracking_root(experiment) / ("a" * 32) / "operations").iterdir())
    assert len(directories) == 1
    lifecycle = read_attempt_lifecycle(directories[0], expected_attempt_id=directories[0].name)
    assert lifecycle.state == "exited" and lifecycle.cleanup_confirmed
    assert not list(_attempts_dir(experiment).glob("*.json"))


@pytest.mark.parametrize("outage", ["none", "before_upload", "lost_ack"])
def test_failed_trainer_http_summary_delivery_and_sync(local_server, tmp_path, monkeypatch, outage):
    """A real failed trainer leaves no score and a retryable failed summary projection."""
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.engine.selection import NoFeasibleTrialError
    from phasesweep.tracking import publication
    from phasesweep.tracking.session import read_record, tracking_root

    uri, client, experiment_id, _process = local_server
    trainer = tmp_path / "failed_trainer.py"
    trainer.write_text(
        "import os\nimport mlflow\n"
        "with mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID']):\n"
        "    mlflow.log_metric('eval/loss', 0.01, step=500)\n"
        "    raise RuntimeError('failure after logging a tempting score')\n"
    )
    config = yaml.safe_load((Path(__file__).parents[1] / "examples/mlflow.yaml").read_text())
    config.update(
        workdir=str(tmp_path / "work"), storage=f"journal:///{tmp_path / 'study.journal'}"
    )
    config["trial_command"] = f"{sys.executable} {trainer} --config {{config_path}}"
    config["mlflow"].update(
        tracking_uri=uri,
        experiment_id=experiment_id,
        experiment_name=None,
        timeout_seconds=15,
        poll_seconds=0.05,
    )
    config["phases"][0].update(max_consecutive_failures=1)
    experiment = Experiment.model_validate(config)
    actual_call = publication.call_tracking

    def disconnected(*args, **kwargs):
        if outage == "lost_ack":
            actual_call(*args, **kwargs)
        raise RuntimeError("Simulated summary-upload outage")

    if outage != "none":
        monkeypatch.setattr(publication, "call_tracking", disconnected)
    reports = []
    with pytest.raises(NoFeasibleTrialError) as caught:
        run_experiment(experiment, terminal_callback=reports.append)
    assert reports[0].primary_error is caught.value
    assert reports[0].cleanup_confirmed
    generation = reports[0].generation_id
    status = read_status(experiment)
    assert status["published_generation_id"] is None
    assert status["phases"][0]["trials"]["FAIL"] == 1
    assert status["phases"][0]["trials"].get("COMPLETE", 0) == 0
    assert not list(_attempts_dir(experiment).glob("*.json"))
    summaries = client.search_runs(
        [experiment_id], filter_string="tags.`phasesweep.phase` = 'generation'"
    )
    attempts = client.search_runs(
        [experiment_id], filter_string="tags.`phasesweep.phase` != 'generation'"
    )
    assert len(summaries) == len(attempts) == 1
    assert attempts[0].info.status == "FAILED"
    assert attempts[0].data.metrics["eval/loss"] == 0.01
    assert "phasesweep.objective" not in attempts[0].data.metrics
    assert summaries[0].info.status == ("RUNNING" if outage == "before_upload" else "FAILED")
    record_path = tracking_root(experiment) / generation / "delivery/generation.json"
    before = read_record(record_path)
    assert before["finish"] == "FAILED"
    assert before["state"] == ("delivered" if outage == "none" else "pending")
    monkeypatch.setattr(publication, "call_tracking", actual_call)
    assert sync_tracking(experiment, generation) == {"pending": 0, "delivered": 2}
    assert read_record(record_path)["projection"] == before["projection"]
    assert client.get_run(summaries[0].info.run_id).info.status == "FAILED"
    attempt = client.get_run(attempts[0].info.run_id)
    assert (attempt.info.status, attempt.data.tags["phasesweep.outcome"]) == ("FAILED", "FAILED")
    assert "phasesweep/generation.json" in [
        a.path for a in client.list_artifacts(summaries[0].info.run_id, "phasesweep")
    ]
    assert read_status(experiment)["phases"] == status["phases"]
