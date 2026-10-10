"""Qualification against a temporary HTTP MLflow 3.17 server."""

from __future__ import annotations

import builtins
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

from phasesweep.config import Experiment
from phasesweep.engine import run_experiment
from phasesweep.engine.read import read_status, read_winners
from phasesweep.tracking.publication import delivery_status, sync_tracking

pytestmark = pytest.mark.integration


@pytest.fixture
def local_server(tmp_path):
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
            experiment_id = client.create_experiment("phasesweep-native")
            yield uri, client, experiment_id, process
        finally:
            process.terminate()
            process.wait(timeout=15)


@pytest.mark.parametrize("mode", ["managed", "trainer"])
def test_two_phases_native_http_then_offline(local_server, tmp_path, monkeypatch, mode):
    """Both ownership modes publish four trials and replay without the SDK/server."""
    uri, client, experiment_id, process = local_server
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
        assert run.data.tags["mlflow.parentRunId"] == (
            summary[0].info.run_id if mode == "managed" else parent.info.run_id
        )
    assert summary[0].info.status == "FINISHED"
    assert (
        client.list_artifacts(summary[0].info.run_id, "phasesweep")[0].path
        == "phasesweep/summary.json"
    )
    rate_capture = winners["rate"].objective_provenance["remote_capture"]
    assert rate_capture["binding"]["run_id"] != winners["rate"].attempt_id
    assert all(point["step"] == 1 for point in rate_capture["observations"].values())
    process.terminate()
    process.wait(timeout=15)
    before = [(w.phase, w.metric, w.params) for w in read_winners(experiment)]
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
    assert [(w.phase, w.metric, w.params) for w in read_winners(experiment)] == before
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
