"""Shared MLflow binding, worker, and experiment support for tracking tests."""

from __future__ import annotations

import dataclasses
import errno
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from optuna.trial import TrialState

import phasesweep.engine.attempts as attempts_module
import phasesweep.engine.cleanup as cleanup_module
import phasesweep.engine.phase as phase_module
from phasesweep.config.models import Experiment
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.run import run_experiment
from phasesweep.engine.state import ATTEMPT_ID_ATTR, CLEANUP_CONFIRMED_ATTR
from phasesweep.errors import ProcessCleanupUncertainError, UnsafeProcessCleanupError
from phasesweep.runtime import process
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import session
from phasesweep.tracking.mlflow import _capture


def _binding():
    return {
        "schema_version": 1,
        "tracking_uri": "http://localhost:5000",
        "experiment_id": "1",
        "run_id": "native-id",
        "experiment": "test",
        "phase": "p",
        "generation_id": "a" * 32,
        "attempt_id": "b" * 32,
        "overrides_sha256": "c" * 64,
        "run_mode": "managed",
    }


def _run(binding, *, status="FINISHED", metrics=None):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=binding["run_id"],
            experiment_id=binding["experiment_id"],
            lifecycle_stage="active",
            status=status,
        ),
        data=SimpleNamespace(
            tags=worker.identity_tags(binding), params={}, metrics=metrics or {"loss": 2.0}
        ),
    )


def _identity(generation_id="a" * 32, attempt_id="b" * 32):
    """Return the creation identity of one attempt, which names no native run yet."""
    binding = {**_binding(), "generation_id": generation_id, "attempt_id": attempt_id}
    return {k: v for k, v in binding.items() if k != "run_id"}


def native_experiment(tmp_path):
    """Build the complete native example in an isolated artifact namespace."""
    config = yaml.safe_load((Path(__file__).parents[1] / "examples/mlflow.yaml").read_text())
    config.update(workdir=str(tmp_path), storage=None)
    return Experiment.model_validate(config)


def _scripted_response(script, request):
    """Consume the next response for an action, evaluating deferred responses."""
    if not script.get(request["action"]):
        pytest.fail(f"Unscripted {request['action']} request.")
    response = script[request["action"]].pop(0)
    return response() if callable(response) else response


def _binder(tmp_path, slot, *, experiment=None, identity=None, directory=None):
    """Return the binding path and a ``_create_bound`` call for one creation in either slot."""
    experiment = experiment or native_experiment(tmp_path)
    identity = identity or _identity()
    if slot == "attempt":
        directory = directory or tmp_path
        path, slot_arguments = directory / "mlflow-binding.json", {"attempt_directory": directory}
    else:
        path = (
            session.tracking_root(experiment)
            / identity["generation_id"]
            / "generation-binding.json"
        )
        path.parent.mkdir(parents=True)
        slot_arguments = {}

    def bind(deadline=None):
        return session._create_bound(
            experiment, identity, path, {}, run_name="r", deadline=deadline, **slot_arguments
        )

    return bind, path


class _WorkerProcess:
    """A stand-in for the worker process behind the real controller and its response reader.

    Each request is answered with the next scripted response for its action. The controller's
    first read of the response to an action named in ``unreadable`` fails with EIO.
    Callable responses are evaluated when requested; non-MLflow processes run normally.
    """

    def __init__(self, monkeypatch, *, unreadable=(), **script):
        self.script = {action: list(responses) for action, responses in script.items()}
        self.unreadable = set(unreadable)
        self.requests = []
        self.real_supervised = process.run_supervised
        real_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if path.name == "response.json" and self.requests:
                action = self.requests[-1]["action"]
                if action in self.unreadable:
                    self.unreadable.discard(action)
                    raise OSError(errno.EIO, "injected response read failure", str(path))
            return real_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(process, "run_supervised", self)

    @property
    def actions(self):
        return [request["action"] for request in self.requests]

    def __call__(self, command, **kwargs):
        parts = shlex.split(command)
        if not any(part.endswith("/tracking/mlflow.py") for part in parts):
            return self.real_supervised(command, **kwargs)
        request_path, response_path = (Path(part) for part in parts[-2:])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.requests.append(request)
        response = _scripted_response(self.script, request)
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return process.ProcessResult(return_code=0, timed_out=False, pid=0, duration_seconds=0)


def _leave_cleanup_uncertain_attempt(experiment, monkeypatch, **run_kwargs):
    """Fail the one trial of ``depth`` with unconfirmed cleanup, recorded as recovered by nobody.

    The trainer really exits. The launch result reports its cleanup unconfirmed, and the engine's
    own reconciliation after the failure cannot prove the group gone, so the trial keeps its
    uncertainty and its attempt-registry entry stays for a later pass. The injected faults are
    undone before this returns.

    :return str: The failed trial's attempt id.
    """
    original_launch = phase_module.launch_trial

    def uncertain_launch(*args, **kwargs):
        executed = original_launch(*args, **kwargs)
        return dataclasses.replace(
            executed, process=dataclasses.replace(executed.process, cleanup_confirmed=False)
        )

    def unproven(*args, **kwargs):
        raise ProcessCleanupUncertainError("the process group may still be alive")

    with monkeypatch.context() as patch:
        patch.setattr(phase_module, "launch_trial", uncertain_launch)
        # The registry pass and the terminal-trial pass each resolve attempts through their own name.
        patch.setattr(attempts_module, "_resolve_attempt", unproven)
        patch.setattr(cleanup_module, "_resolve_attempt", unproven)
        with pytest.raises(UnsafeProcessCleanupError):
            run_experiment(experiment, **run_kwargs)
    read = read_phase_trials(validate_ledger(experiment), "depth")
    [trial] = read.trials
    assert trial.state == TrialState.FAIL
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    assert trial.number not in read.cleanup_recovered
    return trial.user_attrs[ATTEMPT_ID_ATTR]


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
