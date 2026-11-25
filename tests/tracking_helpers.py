"""Shared polling clocks, MLflow bindings, workers and experiment setup for tracking tests."""

from __future__ import annotations

import dataclasses
import errno
import json
import os
import shlex
import sys
import threading
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from optuna.trial import TrialState

import phasesweep.engine.attempts as attempts_module
import phasesweep.engine.cleanup as cleanup_module
import phasesweep.engine.generation as generation_module
import phasesweep.engine.guards as guards_module
import phasesweep.engine.phase as phase_module
from phasesweep.config.models import Experiment
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.run import run_experiment
from phasesweep.engine.state import ATTEMPT_ID_ATTR, CLEANUP_CONFIRMED_ATTR
from phasesweep.errors import ProcessCleanupUncertainError, UnsafeProcessCleanupError
from phasesweep.evidence.evaluation import TrialContext
from phasesweep.evidence.mlflow import compile_query
from phasesweep.runtime import process, shutdown
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import OperationOutcomeUnknownError, _capture
from phasesweep.tracking.records import RunBinding, RunIdentity
from tests.conftest import study_name


class ManualClock:
    """A monotonic clock whose sleeps advance instantly and record their durations."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        """Return the current simulated monotonic time in seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record a requested sleep and advance time without waiting.

        :param float seconds: Simulated sleep duration.
        """
        self.sleeps.append(seconds)
        self.now += seconds


def mlflow_binding():
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


def mlflow_run(binding, *, status="FINISHED", metrics=None):
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


def delivery_record(generation_id, *, state="pending", **fields):
    """Return a delivery record holding every field delivery reads, for a test that writes one."""
    return {
        "schema_version": 1,
        "generation_id": generation_id,
        "state": state,
        "source": {},
        "binding": {},
        "projection": {},
        "finish": None,
        **fields,
    }


def attempt_identity(generation_id="a" * 32, attempt_id="b" * 32):
    """Return the creation identity of one attempt, which names no native run yet."""
    binding = {**mlflow_binding(), "generation_id": generation_id, "attempt_id": attempt_id}
    return {k: v for k, v in binding.items() if k != "run_id"}


def native_experiment(tmp_path, *, storage=None):
    """Build the complete native example in an isolated artifact namespace, in memory unless ``storage`` names a ledger."""
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "src/phasesweep/templates/mlflow.yaml").read_text()
    )
    config.update(workdir=str(tmp_path), storage=storage)
    return Experiment.model_validate(config)


def mlflow_trial_context(tmp_path: Path) -> TrialContext:
    """Build the context of the native example's attempt in a scratch artifact tree.

    :param Path tmp_path: The attempt directory, also the example's workdir.
    :return TrialContext: Context carrying the MLflow settings and compiled query of a capture.
    """
    experiment = native_experiment(tmp_path)
    return TrialContext(
        experiment="test",
        phase="p",
        trial_id=0,
        generation_id="a" * 32,
        attempt_id="b" * 32,
        overrides_sha256="c" * 64,
        trial_dir=tmp_path / "p" / "trial_00000__test",
        attempt_dir=tmp_path,
        run_name="test",
        return_code=0,
        duration_seconds=1,
        wandb_run_id="b" * 32,
        mlflow_settings=experiment.mlflow.model_dump(),
        mlflow_query=compile_query(experiment, []),
    )


def bind_mlflow_attempt(tmp_path: Path) -> tuple[dict[str, str | int], TrialContext]:
    """Write the expected record and the binding of the native example's attempt.

    :param Path tmp_path: The attempt directory.
    :return tuple[dict[str, str | int], TrialContext]: The binding and the attempt's context.
    """
    binding = mlflow_binding()
    ctx = mlflow_trial_context(tmp_path)
    directory = session.attempt_tracking_dir(tmp_path, ctx.generation_id, ctx.attempt_id)
    session.write_record(
        directory / session.EXPECTED_BINDING_FILE,
        {k: v for k, v in binding.items() if k != "run_id"},
    )
    session.write_record(directory / session.BINDING_FILE, binding)
    return binding, ctx


def run_mlflow_worker_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer each supervised MLflow worker launch by running its entry point in this process.

    :param pytest.MonkeyPatch monkeypatch: Replaces ``process.run_supervised`` for the test.
    """

    def run_in_process(command, **kwargs):
        request_path, response_path = shlex.split(command)[-2:]
        assert worker.main([request_path, response_path]) == 0
        return process.ProcessResult(return_code=0, timed_out=False, pid=0, duration_seconds=0)

    monkeypatch.setattr(process, "run_supervised", run_in_process)


def scripted_response(script, request):
    """Consume the next response for an action, evaluating deferred responses."""
    if not script.get(request["action"]):
        pytest.fail(f"Unscripted {request['action']} request.")
    response = script[request["action"]].pop(0)
    return response() if callable(response) else response


def mlflow_binder(
    tmp_path: Path,
    slot: str,
    *,
    experiment: Experiment | None = None,
    identity: RunIdentity | None = None,
) -> tuple[partial[RunBinding], Path]:
    """Prepare a binding call at the location used by creation and reconciliation.

    :param Path tmp_path: Scratch root for the default experiment.
    :param str slot: Attempt or generation binding.
    :param Experiment | None experiment: Experiment that owns the binding.
    :param RunIdentity | None identity: Identity for the run to create.
    :return tuple[partial[RunBinding], Path]: Binding call and record destination.
    """
    experiment = experiment or native_experiment(tmp_path)
    identity = identity or attempt_identity()
    if slot == "attempt":
        directory = session.attempt_tracking_dir(
            _experiment_dir(experiment), identity["generation_id"], identity["attempt_id"]
        )
        path = directory / session.BINDING_FILE
    else:
        path = session.summary_binding_path(experiment, identity["generation_id"])
    path.parent.mkdir(parents=True, exist_ok=True)

    return partial(
        session._create_bound, experiment, identity, path, {}, run_name="r", deadline=None
    ), path


class FakeWorkerProcess:
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
        response = scripted_response(self.script, request)
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return process.ProcessResult(return_code=0, timed_out=False, pid=0, duration_seconds=0)


def leave_cleanup_uncertain_attempt(experiment, monkeypatch, **run_kwargs):
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
    read = read_phase_trials(validate_ledger(experiment), study_name(experiment, "depth"))
    [trial] = read.trials
    assert trial.state == TrialState.FAIL
    assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    assert trial.number not in read.cleanup_recovered
    return trial.user_attrs[ATTEMPT_ID_ATTR]


def remote_mlflow_example(tmp_path, monkeypatch):
    """Patch the MLflow request operations with in-process answers; return the native example config and the list collecting uploads."""
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "src/phasesweep/templates/mlflow.yaml").read_text()
    )
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
            run = mlflow_run(request["binding"], metrics={"eval/loss": 1.0, "cost": 1.0})
            run.data.params = {"depth": "1", "rate": "0"}
            run.data.tags["trainer"] = "none"
            point = SimpleNamespace(value=1.0, step=1, timestamp=1)
            earlier = {prior["run_id"]: mlflow_run(prior) for prior in request["earlier"]}
            client = SimpleNamespace(
                get_run=lambda run_id: earlier.get(run_id, run),
                get_metric_history=lambda *_: [point],
            )
            return _capture(client, request)
        uploads.append(request)
        return {}

    monkeypatch.setattr(session, "request_operation", respond)
    monkeypatch.setattr(worker, "request_operation", respond)
    return config, uploads


def leak_tracking_worker(monkeypatch, *, action="publish", name=None):
    """Leave the first MLflow worker of one request with its cleanup unconfirmed.

    The worker's process group really runs and exits, but its lifecycle stays ``terminating`` and
    its attempt-registry entry stays, as for a worker the controller lost track of, so a later pass
    resolves it from its recorded identity. Every other request reaches the request operation
    already in place.

    :param pytest.MonkeyPatch monkeypatch: Replaces ``request_operation`` for the test.
    :param str action: Request action to match, such as ``publish`` or ``resolve``.
    :param str | None name: Upload name to match, such as ``attempt-<id>``; any request of
        ``action`` when ``None``.
    :return list[str]: The operation id of the worker left unconfirmed, once one has been.
    """
    respond = session.request_operation
    leaked = []

    def leaky(request, *, operation_id, directory, environment, deadline):
        matches = request["action"] == action and name in {None, request.get("name")}
        if not matches or leaked:
            return respond(
                request,
                operation_id=operation_id,
                directory=directory,
                environment=environment,
                deadline=deadline,
            )
        leaked.append(operation_id)
        with Path(os.devnull).open("w") as sink:
            process.run_supervised(
                "true",
                env=dict(os.environ),
                stdout=sink,
                stderr=sink,
                timeout=10,
                attempt_dir=directory,
                attempt_id=operation_id,
            )
        process.write_attempt_lifecycle(
            directory,
            attempt_id=operation_id,
            state="terminating",
            ended_by="cancelled",
            cleanup_confirmed=False,
        )
        raise UnsafeProcessCleanupError(
            f"mlflow worker cleanup is uncertain for attempt {operation_id!r}; recover {directory}."
        )

    monkeypatch.setattr(session, "request_operation", leaky)
    monkeypatch.setattr(worker, "request_operation", leaky)
    return leaked


def signal_when_uploads_start(
    started: list[str], signum: int, *, count: int = 1
) -> threading.Thread:
    """Signal the test process once its upload workers register, or after 30 seconds to unblock it."""

    def send() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if len(started) >= count and len(shutdown._active_children) >= count:
                break
            time.sleep(0.02)
        os.kill(os.getpid(), signum)

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    return sender


def hang_uploads_until_killed(monkeypatch: pytest.MonkeyPatch, started: list[str]) -> None:
    """Make every MLflow upload run a real supervised worker that only a signal ends.

    The worker registers as ``runtime.remote_worker.run_worker`` registers a real one, so the
    shutdown handler names and kills it as an MLflow request. Every other request reaches the
    request operation already in place.

    :param pytest.MonkeyPatch monkeypatch: Replaces the request function for the test.
    :param list[str] started: Receives the record name of each upload as it starts.
    """
    respond = session.request_operation

    def upload_until_killed(
        request: Mapping[str, object],
        *,
        operation_id: str,
        directory: Path,
        environment: Mapping[str, str],
        deadline: float,
    ) -> JsonObject:
        if request["action"] != "publish":
            return respond(
                request,
                operation_id=operation_id,
                directory=directory,
                environment=environment,
                deadline=deadline,
            )
        started.append(str(request["name"]))
        with Path(os.devnull).open("w") as sink:
            result = process.run_supervised(
                "sleep 30",
                env=dict(os.environ),
                stdout=sink,
                stderr=sink,
                timeout=60,
                attempt_dir=directory,
                attempt_id=operation_id,
                label="MLflow request",
            )
        raise OperationOutcomeUnknownError(f"MLflow worker failed (exit {result.return_code}).")

    monkeypatch.setattr(session, "request_operation", upload_until_killed)
    monkeypatch.setattr(worker, "request_operation", upload_until_killed)


class RunnerKilled(BaseException):
    """Stands in for the orchestrator dying, which no handler of the run may observe."""


def killed_sweep(
    tmp_path, monkeypatch, *, n_jobs, generation_id=None, in_flight=True, persistent=True
):
    """Kill a sweep of ``n_jobs`` managed attempts outright, by default with all of them in flight.

    Each attempt's managed run and the generation's summary run exist. With the attempts in
    flight, the trainers never start, no outcome is recorded and every trial stays RUNNING.
    None of the run's failure handling or reconciliation runs, so it records no failed state
    and finishes no MLflow run of its own failure.

    :param str | None generation_id: The killed run's generation, as an MCP runner names it.
    :param bool in_flight: Kill the run with its trials in flight; ``False`` lets every trial
        finish, its accepted outcome sent, and kills the run before it publishes, so no trial is
        left for recovery to settle.
    :param bool persistent: Record the trials in a journal; ``False`` keeps them in memory, so
        they are gone with the killed run and only their bound attempt directories remain.
    :return tuple[Experiment, list, list[str]]: The experiment, the upload log, and the attempt ids.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    if persistent:
        config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(
        n_trials=n_jobs,
        n_jobs=n_jobs,
        allow_no_gpu_isolation=True,
        sampler={"type": "random", "seed": 1},
        search_space={"model.depth": {"type": "categorical", "choices": [1, 2, 3]}},
    )
    experiment = Experiment.model_validate(config)
    launched = threading.Barrier(n_jobs)

    def dies(*args, **kwargs):
        launched.wait(10)
        raise RunnerKilled()

    def dies_unpublished(*args, **kwargs):
        raise RunnerKilled()

    with monkeypatch.context() as patch:
        if in_flight:
            patch.setattr(phase_module, "launch_trial", dies)
            patch.setattr(phase_module, "_record_outcome", lambda *args, **kwargs: None)
        else:
            patch.setattr(generation_module, "_publish_generation", dies_unpublished)
        patch.setattr(guards_module, "_reconcile_existing_studies", lambda *args, **kwargs: None)
        patch.setattr(
            generation_module, "_persist_failed_state_unless_recorded", lambda *args: None
        )
        patch.setattr(publication, "deliver_closeout", lambda *args, **kwargs: None)
        with pytest.raises(RunnerKilled):
            run_experiment(experiment, generation_id=generation_id)
    assert not any(session.tracking_root(experiment).glob("*/delivery/generation.json"))
    if not persistent:
        bindings = session.attempt_tracking_records(experiment, session.BINDING_FILE, generation_id)
        assert len(bindings) == n_jobs
        return experiment, uploads, [session.read_binding(b.parent)["attempt_id"] for b in bindings]
    trials = read_phase_trials(validate_ledger(experiment), study_name(experiment, "depth")).trials
    state = TrialState.RUNNING if in_flight else TrialState.COMPLETE
    assert [trial.state for trial in trials] == [state] * n_jobs
    return experiment, uploads, [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in trials]


def attach_wandb_transport(api: object, transport: object) -> None:
    """Initialize the real SDK client's instance fields for a test transport without authenticating."""
    vars(api).update(_runs={}, _service_api=transport, api_key=None)


@dataclasses.dataclass(frozen=True)
class SwitchableWandb:
    """A fake W&B server the real supervised reader polls, switched by a mode file.

    ``up`` answers a finished run whose summary holds ``x=1.0``, ``crashed`` answers a run
    that crashed, ``denied`` rejects the credentials with HTTP 401, and ``hang`` touches
    ``hung`` and blocks. Every run lookup appends the mode it saw to ``polls``.
    """

    mode: Path
    polls: Path
    hung: Path

    def switch(self, mode: str) -> None:
        """Answer every later run lookup in ``mode``.

        :param str mode: ``up``, ``empty`` (summary still unpublished), ``crashed``, ``denied`` or ``hang``.
        """
        self.mode.write_text(mode)

    def lookups(self, mode: str) -> int:
        """Count the run lookups answered in ``mode`` so far.

        :param str mode: Mode to count.
        :return int: Lookups that saw ``mode``.
        """
        if not self.polls.exists():
            return 0
        return self.polls.read_text().splitlines().count(mode)


def switchable_wandb(install_sdk, tmp_path: Path) -> SwitchableWandb:
    """Install a fake ``wandb.apis.public`` for the real reader worker, starting ``up``.

    :param install_sdk: The ``wandb_worker_sdk`` fixture's installer.
    :param Path tmp_path: Directory holding the mode, lookup and hang files.
    :return SwitchableWandb: The switch and its lookup log.
    """
    tracker = SwitchableWandb(
        tmp_path / "wandb-mode", tmp_path / "wandb-lookups", tmp_path / "wandb-hung"
    )
    tracker.switch("up")
    install_sdk(f"""
        import time
        from pathlib import Path

        class _Denied(Exception):
            response = type("Response", (), {{"status_code": 401}})()

        class Api:
            def __init__(self, **kwargs):
                pass

            def run(self, path):
                mode = Path({str(tracker.mode)!r}).read_text()
                with Path({str(tracker.polls)!r}).open("a") as polls:
                    polls.write(mode + "\\n")
                if mode == "hang":
                    Path({str(tracker.hung)!r}).touch()
                    time.sleep(60)
                if mode == "denied":
                    raise _Denied("credentials rejected")
                state = "crashed" if mode == "crashed" else "finished"
                summary = {{}} if mode == "empty" else {{"x": 1.0}}
                return type("Run", (), {{"state": state, "summary_metrics": summary}})()
    """)
    return tracker


def wandb_objective_experiment(
    tmp_path: Path,
    *,
    n_trials: int,
    n_jobs: int = 1,
    switch_at_launch: tuple[SwitchableWandb, int, str] | None = None,
    max_consecutive_failures: int = 5,
    metric_max: float | None = None,
    timeout_seconds: float = 30,
) -> tuple[Experiment, Path]:
    """Build a persistent sweep whose objective W&B reports once its trainer exits cleanly.

    Each trainer launch appends a line to the returned log and waits until ``n_jobs``
    trainers have launched, so parallel trials finish training together.

    :param Path tmp_path: Directory holding the ledger, workdir, trainer and launch log.
    :param int n_trials: The phase's trial target.
    :param int n_jobs: The phase's parallel workers.
    :param tuple[SwitchableWandb, int, str] | None switch_at_launch: The fake W&B server, a
        launch number and a mode: that launch's trainer switches the server to the mode, so
        its own evidence and every later lookup see it.
    :param int max_consecutive_failures: The phase's failure limit.
    :param float | None metric_max: Upper bound above which the objective is infeasible.
    :param float timeout_seconds: One W&B capture's polling budget, defaults to 30.
    :return tuple[Experiment, Path]: The experiment and its trainer launch log.
    """
    from phasesweep.config.models import Metric
    from phasesweep.evidence.models import WandbExtractor
    from tests.conftest import make_experiment, write_trainer

    launches = tmp_path / "launches"
    switch = (
        None
        if switch_at_launch is None
        else (switch_at_launch[1], str(switch_at_launch[0].mode), switch_at_launch[2])
    )
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import time
        from pathlib import Path
        launches = Path({str(launches)!r})
        with launches.open("a") as log:
            log.write("launch\\n")
        switch = {switch!r}
        if switch is not None and len(launches.read_text().splitlines()) == switch[0]:
            Path(switch[1]).write_text(switch[2])
        deadline = time.monotonic() + 15
        while len(launches.read_text().splitlines()) < {n_jobs} and time.monotonic() < deadline:
            time.sleep(0.02)
        """,
    )
    remote = WandbExtractor(
        type="wandb",
        base_url="https://example.test",
        entity="e",
        project="p",
        metric_key="x",
        poll_seconds=0.01,
        timeout_seconds=timeout_seconds,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        metric=Metric(extractor=remote, max=metric_max),
        n_trials=n_trials,
        n_jobs=n_jobs,
        max_consecutive_failures=max_consecutive_failures,
    )
    return experiment, launches
