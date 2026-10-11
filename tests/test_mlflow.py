"""Native MLflow configuration, history, binding, and retry contracts."""

from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from phasesweep.config import Experiment, MlflowSettings
from phasesweep.evidence.mlflow import compile_query, observation_id, select_observation
from phasesweep.evidence.models import MlflowExtractor
from phasesweep.tracking.mlflow import _capture, _create


def test_cpu_example_import_and_help_without_sdk(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    path = Path(__file__).parents[1] / "src/phasesweep/examples/mlflow_train.py"
    namespace = runpy.run_path(str(path))
    with pytest.raises(SystemExit) as stopped:
        namespace["main"](["--help"])
    assert stopped.value.code == 0
    assert "--config" in capsys.readouterr().out


def spec(selection="last", **kwargs):
    """Build an explicit native history selector."""
    return MlflowExtractor(type="mlflow", metric_key="loss", select=selection, **kwargs)


@pytest.mark.parametrize(
    "selection,kwargs,expected",
    [("last", {}, 7), ("min", {}, 2), ("max", {}, 9), ("step", {"step": 1}, 2)],
)
def test_ordering(selection, kwargs, expected):
    history = [
        {"value": 7, "step": 3, "timestamp": 2},
        {"value": 2, "step": 1, "timestamp": 4},
        {"value": 9, "step": 0, "timestamp": 5},
        {"value": 3, "step": 1, "timestamp": 2},
    ]
    assert select_observation(history, spec(selection, **kwargs))["value"] == expected


@pytest.mark.parametrize("selection", ["min", "max"])
def test_tie_earliest_position(selection):
    history = [
        {"value": 2, "step": 4, "timestamp": 1},
        {"value": 2, "step": 1, "timestamp": 3},
        {"value": 2, "step": 1, "timestamp": 2},
    ]
    assert select_observation(history, spec(selection)) == {"value": 2.0, "step": 1, "timestamp": 2}


@pytest.mark.parametrize(
    "history",
    [
        [],
        [{"value": True, "step": 0, "timestamp": 1}],
        [{"value": float("nan"), "step": 0, "timestamp": 1}],
        [{"value": 1, "step": 0, "timestamp": 1}, {"value": 2, "step": 0, "timestamp": 1}],
        [{"value": 1, "step": -1, "timestamp": 1}],
    ],
)
def test_invalid_history(history):
    with pytest.raises(ValueError):
        select_observation(history, spec())


@pytest.mark.parametrize(
    "uri,accepted",
    [
        ("databricks", True),
        ("databricks://WorkProfile", True),
        ("http://localhost:5000", True),
        ("databricks://Work Profile", False),
    ],
)
def test_native_target_spelling(uri, accepted):
    if not accepted:
        with pytest.raises(ValidationError, match="no spaces in the profile name"):
            MlflowSettings(tracking_uri=uri, experiment_name="existing")
        return
    assert MlflowSettings(tracking_uri=uri, experiment_name="existing").tracking_uri == uri


@pytest.mark.parametrize("extra", [{}, {"experiment_id": "1", "experiment_name": "x"}])
def test_exactly_one_experiment(extra):
    with pytest.raises(ValidationError):
        MlflowSettings(tracking_uri="databricks", **extra)


def test_selector_requires_explicit_step():
    with pytest.raises(ValidationError):
        spec("step")
    with pytest.raises(ValidationError):
        spec("last", step=2)


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
    from phasesweep.tracking.mlflow import identity_tags

    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=binding["run_id"],
            experiment_id=binding["experiment_id"],
            lifecycle_stage="active",
            status=status,
        ),
        data=SimpleNamespace(
            tags=identity_tags(binding), params={}, metrics=metrics or {"loss": 2.0}
        ),
    )


def test_shared_history_once():
    binding = _binding()
    selectors = [spec("last"), spec("min"), spec("step", step=1)]
    query = {
        "revision": 1,
        "selections": {observation_id(s): s.model_dump() for s in selectors},
        "gates": [],
    }
    calls = []

    def history(run_id, key):
        calls.append((run_id, key))
        return [
            SimpleNamespace(value=2.0, step=1, timestamp=2),
            SimpleNamespace(value=3.0, step=2, timestamp=1),
        ]

    client = SimpleNamespace(get_run=lambda _: _run(binding), get_metric_history=history)
    capture = _capture(
        client,
        {
            "binding": binding,
            "query": query,
            "settings": {"timeout_seconds": 1, "poll_seconds": 0.01},
            "timeout_at": time.monotonic() + 1,
        },
    )
    assert calls == [(binding["run_id"], "loss")]
    assert [capture["observations"][observation_id(s)]["value"] for s in selectors] == [3, 2, 2]


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_unsuccessful_remote_cannot_be_finished(status):
    binding = _binding()
    client = SimpleNamespace(get_run=lambda _: _run(binding, status=status))
    with pytest.raises(ValueError, match="unsuccessfully"):
        _capture(
            client,
            {"binding": binding, "query": {}, "settings": {"timeout_seconds": 1}, "timeout_at": 0},
        )


def test_worker_keeps_sdk_retries_for_idempotent_operations(tmp_path, monkeypatch):
    """Reads such as evidence capture ride out transient HTTP failures."""
    import json

    from phasesweep.tracking import mlflow as worker

    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    seen = []
    monkeypatch.setattr(
        worker,
        "_dispatch",
        lambda request: seen.append(os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES")) or {},
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"action": "capture"}))
    assert worker.main([str(request), str(tmp_path / "response.json")]) == 0
    assert seen == [None]


def test_creation_binds_an_existing_identity_and_creates_only_when_absent(monkeypatch):
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    binding = _binding()
    identity = {k: v for k, v in binding.items() if k != "run_id"}
    created = []

    class Page(list):
        token = None

    def create_run(*args, **kwargs):
        # Only the non-idempotent create is never replayed by the SDK.
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
        created.append(kwargs["tags"])
        return _run(binding)

    client = SimpleNamespace(
        get_experiment=lambda _: SimpleNamespace(experiment_id="1", lifecycle_stage="active"),
        search_runs=lambda *a, **k: Page([_run(binding)]),
        create_run=create_run,
    )
    request = {"binding": identity, "settings": {"experiment_id": "1"}, "run_name": "r"}
    assert _create(client, request) == binding
    assert created == []
    client.search_runs = lambda *a, **k: Page()
    assert _create(client, request) == binding
    assert len(created) == 1
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ
    client.search_runs = lambda *a, **k: Page([_run(binding), _run(binding)])
    with pytest.raises(ValueError, match="ambiguous"):
        _create(client, request)


@pytest.mark.parametrize("target", ["experiment_id", "experiment_name"])
def test_unknown_experiment_is_reported_as_missing(target):
    from phasesweep.tracking.mlflow import _resolve

    def unknown_id(_):
        raise _server_error("RESOURCE_DOES_NOT_EXIST")()

    client = SimpleNamespace(get_experiment=unknown_id, get_experiment_by_name=lambda _: None)
    with pytest.raises(ValueError, match="missing or deleted; provision an existing experiment"):
        _resolve(client, {target: "404"})


def test_no_cross_backend_publishing(tmp_path):
    config = {
        "experiment": "test",
        "workdir": str(tmp_path),
        "trial_command": "true {overrides}",
        "mlflow": {"tracking_uri": "databricks", "experiment_id": "1"},
        "phases": [
            {
                "name": "p",
                "n_trials": 1,
                "search_space": {"x": {"type": "categorical", "choices": [1]}},
            }
        ],
    }
    config["override_format"] = "hydra"
    config["metric"] = {
        "name": "loss",
        "goal": "minimize",
        "extractor": {"type": "json", "path": "result.json", "key": "loss"},
    }
    with pytest.raises(ValidationError, match="MLflow objective"):
        Experiment.model_validate(config)


def native_experiment(tmp_path):
    """Build the complete native example in an isolated artifact namespace."""
    from pathlib import Path

    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "examples/mlflow.yaml").read_text())
    config.update(workdir=str(tmp_path), storage=None)
    return Experiment.model_validate(config)


def test_uncertain_creation_retries_the_same_identity(tmp_path, monkeypatch):
    """A failed create is retried by identity, and never blocks a later creation."""
    from phasesweep.tracking import session

    binding = _binding()
    identity = {k: v for k, v in binding.items() if k != "run_id"}
    path = tmp_path / "mlflow-binding.json"
    requests = []
    outcomes = [RuntimeError("response lost"), binding]

    def operation(request, **kwargs):
        requests.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(session, "request_operation", operation)
    experiment = native_experiment(tmp_path)
    actual = session._create_bound(
        experiment, identity, path, {}, run_name="r", deadline=None, attempt_directory=tmp_path
    )
    assert actual == binding
    assert len(requests) == 2 and requests[0] == requests[1]
    assert session.read_binding(tmp_path) == binding

    path.unlink()
    outcomes[:] = [RuntimeError("server down"), RuntimeError("server down"), binding]
    with pytest.raises(RuntimeError, match="server down"):
        session._create_bound(
            experiment, identity, path, {}, run_name="r", deadline=None, attempt_directory=tmp_path
        )
    assert not path.exists()
    assert (
        session._create_bound(
            experiment, identity, path, {}, run_name="r", deadline=None, attempt_directory=tmp_path
        )
        == binding
    )


@pytest.mark.parametrize("field", ["run_id", "experiment_id", "attempt_id", "unbound"])
def test_wrong_binding_is_rejected_before_remote_access(tmp_path, monkeypatch, field):
    from phasesweep.evidence.evaluation import ExtractorError, TrialContext, _capture_mlflow
    from phasesweep.tracking import session

    binding = _binding()
    expected = {k: v for k, v in binding.items() if k != "run_id"}
    if field == "run_id":
        expected["run_id"] = "another-run"
    elif field != "unbound":
        binding[field] = "another-identity"
    session.write_record(tmp_path / "mlflow-expected.json", expected)
    if field != "unbound":
        session.write_record(tmp_path / "mlflow-binding.json", binding)
    experiment = native_experiment(tmp_path)
    ctx = TrialContext(
        experiment="test",
        phase="p",
        trial_id=0,
        generation_id="a" * 32,
        attempt_id="b" * 32,
        overrides_sha256="c" * 64,
        trial_dir=tmp_path,
        run_name="test",
        return_code=0,
        duration_seconds=1,
        mlflow_settings=experiment.mlflow.model_dump(),
        mlflow_query=compile_query(experiment, []),
    )

    def remote(*args, **kwargs):
        pytest.fail("A foreign binding must not contact MLflow")

    monkeypatch.setattr("phasesweep.tracking.mlflow.request_operation", remote)
    match = "bind_mlflow_run" if field == "unbound" else "different attempt"
    with pytest.raises(ExtractorError, match=match):
        _capture_mlflow(ctx, deadline=None)


def test_trainer_run_is_never_finished_by_controller():
    binding = {**_binding(), "run_mode": "trainer"}
    client = SimpleNamespace(get_run=lambda _: _run(binding, status="RUNNING"))
    with pytest.raises(TimeoutError, match="was still RUNNING when timeout_seconds ran out"):
        _capture(
            client,
            {"binding": binding, "query": {}, "settings": {"timeout_seconds": 0}, "timeout_at": 0},
        )


def test_missing_metric_is_not_replaced_by_result_metric():
    binding = _binding()
    query = {"selections": {observation_id(spec()): spec().model_dump()}, "gates": []}
    client = SimpleNamespace(get_run=lambda _: _run(binding, metrics={"phasesweep.objective": 1}))
    timeout_at = time.monotonic() + 0.4
    request = {
        "binding": binding,
        "query": query,
        "settings": {"timeout_seconds": 1, "poll_seconds": 0.01},
        "timeout_at": timeout_at,
    }
    with pytest.raises(TimeoutError, match="finished without metric 'loss' when timeout_seconds"):
        _capture(client, request)
    # The worker reports before the parent stops it at timeout_at.
    assert time.monotonic() < timeout_at


def test_failed_trainer_never_reads_remote_metrics(tmp_path, monkeypatch):
    from phasesweep.engine.trial import ExecutedTrial, extract_trial_result
    from phasesweep.evidence.evaluation import TrialContext
    from phasesweep.runtime.process import ProcessResult

    experiment = native_experiment(tmp_path)
    ctx = TrialContext(
        experiment="test",
        phase="p",
        trial_id=0,
        generation_id="a" * 32,
        attempt_id="b" * 32,
        overrides_sha256="c" * 64,
        trial_dir=tmp_path,
        run_name="test",
        return_code=1,
        duration_seconds=1,
    )

    def remote(*args, **kwargs):
        pytest.fail("Failed trainer evidence is never read")

    monkeypatch.setattr("phasesweep.engine.trial.run_extractor", remote)
    result = extract_trial_result(
        experiment=experiment,
        executed=ExecutedTrial(
            ctx, ProcessResult(return_code=1, timed_out=False, pid=0, duration_seconds=1)
        ),
    )
    assert result.metric is None and result.failure_reason


def test_pending_sync_uses_retained_projection(tmp_path, monkeypatch):
    from phasesweep.tracking import publication
    from phasesweep.tracking.session import read_record, tracking_root

    experiment = native_experiment(tmp_path)
    binding = _binding()
    projection = publication._projection(
        metrics={"phasesweep.objective": 2.0}, params={}, tags={}, summary={"metric": 2.0}
    )
    from phasesweep.engine.attempts import _register_active_attempt
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.runtime.process import write_attempt_lifecycle

    trial_dir = tmp_path / "unfinished-training"
    trial_dir.mkdir()
    write_attempt_lifecycle(trial_dir, attempt_id="d" * 32, state="allocated")
    _register_active_attempt(
        experiment,
        attempt_id="d" * 32,
        phase_name="depth",
        study_name="training",
        trial_number=0,
        trial_dir=trial_dir,
        generation_id=binding["generation_id"],
    )

    def outage(*args, **kwargs):
        raise RuntimeError("Disconnected")

    monkeypatch.setattr(publication, "call_tracking", outage)
    publication._enqueue(experiment, binding["generation_id"], "test", binding, projection, {})
    assert publication.delivery_status(experiment) == {"pending": 1, "delivered": 0}
    requests = []

    def deliver(experiment, generation_id, request, environment):
        requests.append(request)
        return {}

    monkeypatch.setattr(publication, "call_tracking", deliver)
    assert publication.sync_tracking(experiment, binding["generation_id"]) == {
        "pending": 0,
        "delivered": 1,
    }
    assert requests[0]["action"] == "publish"
    assert requests[0]["projection"] == projection
    record = read_record(
        tracking_root(experiment) / binding["generation_id"] / "delivery/test.json"
    )
    assert record["projection"] == projection
    assert publication.sync_tracking(experiment)["pending"] == 0
    assert len(requests) == 1
    assert (_attempts_dir(experiment) / f"{'d' * 32}.json").exists()


@pytest.mark.parametrize("terminal", [True, False])
def test_delivery_counts_are_scoped_to_their_reader(tmp_path, terminal):
    """CLI status counts what sync retries; MCP status counts its run; results ignore both."""
    import yaml

    from phasesweep.engine.read import read_status
    from phasesweep.engine.run import experiment_status
    from phasesweep.mcp.snapshots import capture_pre_generation_result_snapshot
    from phasesweep.tracking.session import tracking_root, write_record
    from tests.conftest import mark_current_format
    from tests.mcp_helpers import make_mcp_app, make_run_handle, stage_dead_run, write_run_status

    config = yaml.safe_load((Path(__file__).parents[1] / "examples/mlflow.yaml").read_text())
    config.update(workdir=str(tmp_path / "work"), storage=f"journal:///{tmp_path / 'j.journal'}")
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(yaml.safe_dump(config))
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "state_dir": str(tmp_path / "state"),
                "experiments": [{"id": "mlf", "config": str(config_path)}],
            }
        )
    )
    app, registry, store = make_mcp_app(catalog)
    experiment = registry.get("mlf").experiment
    mark_current_format(experiment)
    run_id = "mlf-" + "a" * 32
    if terminal:
        stage_dead_run(store, run_id, config_path, "mlf", cleanup_uncertain=False)
        write_run_status(
            store,
            run_id,
            returncode=1,
            result_snapshot_state="complete",
            result_snapshot=capture_pre_generation_result_snapshot(experiment),
        )
    else:
        store.create(
            make_run_handle(
                run_id=run_id, experiment_id="mlf", config_sha256=registry.get("mlf").config_sha256
            )
        )
        store.config_snapshot_path(run_id).write_bytes(config_path.read_bytes())
    for generation, name in [(run_id, "own"), ("other", "a"), ("other", "b")]:
        write_record(
            tracking_root(experiment) / generation / "delivery" / f"{name}.json",
            {"schema_version": 1, "generation_id": generation, "state": "pending"},
        )

    assert app.status(run_id)["tracking_delivery"] == {"pending": 1, "delivered": 0}
    assert experiment_status(experiment)["tracking_delivery"] == {"pending": 3, "delivered": 0}
    assert "tracking_delivery" not in read_status(experiment)
    (tracking_root(experiment) / run_id / "delivery" / "own.json").write_text("{")
    assert app.winners(run_id)["run_id"] == run_id


def test_tracking_registry_recovery_cannot_fail_optuna_trial(tmp_path):
    import optuna

    from phasesweep.engine.attempts import (
        _preflight_active_attempts,
        _PreflightCleanupReport,
        _register_tracking_operation,
    )
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.runtime.process import write_attempt_lifecycle

    experiment = native_experiment(tmp_path)
    study = optuna.create_study()
    study.ask()
    directory = tmp_path / "operation"
    directory.mkdir()
    write_attempt_lifecycle(directory, attempt_id="a" * 32, state="allocated")
    _register_tracking_operation(
        experiment, operation_id="a" * 32, generation_id="b" * 32, directory=directory
    )
    _preflight_active_attempts(experiment, _PreflightCleanupReport(), confirm=True)
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING
    assert not list(_attempts_dir(experiment).glob("*.json"))


def test_cache_uses_home_and_xdg(tmp_path, monkeypatch):
    from pathlib import Path

    from phasesweep.runtime.files import cache_dir

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache_dir() == Path.home() / ".cache/phasesweep"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_dir() == tmp_path / "phasesweep"
    for ignored in ("", "relative/cache"):
        monkeypatch.setenv("XDG_CACHE_HOME", ignored)
        assert cache_dir() == Path.home() / ".cache/phasesweep"


def test_missing_sdk_is_refused_before_any_trial(tmp_path, monkeypatch):
    from phasesweep.engine import run_experiment
    from phasesweep.errors import PhaseSweepError
    from phasesweep.tracking.publication import sync_tracking

    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.setattr(
        "phasesweep.tracking.session.prepare_attempt",
        lambda *a, **k: pytest.fail("No trial may be allocated without the SDK"),
    )
    experiment = native_experiment(tmp_path)
    for operation in (run_experiment, sync_tracking):
        with pytest.raises(PhaseSweepError, match=r"phasesweep\[mlflow\]"):
            operation(experiment)


def _server_error(code):
    def error():
        from mlflow.exceptions import MlflowException
        from mlflow.protos import databricks_pb2

        return MlflowException("private server detail", error_code=getattr(databricks_pb2, code))

    return error


def _upload_forbidden():
    from requests import HTTPError, Response

    response = Response()
    response.status_code = 403
    return HTTPError("private upload detail", response=response)


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            lambda: ModuleNotFoundError("No module named 'mlflow'", name="mlflow"),
            "requires phasesweep[mlflow]",
        ),
        (
            lambda: ModuleNotFoundError("No module named 'boto3'", name="boto3"),
            "needs Python module 'boto3'",
        ),
        (
            _server_error("PERMISSION_DENIED"),
            "(MlflowException: PERMISSION_DENIED); check tracking access and credentials.",
        ),
        (
            _server_error("RESOURCE_DOES_NOT_EXIST"),
            "(MlflowException: RESOURCE_DOES_NOT_EXIST); the experiment or run no longer exists.",
        ),
        (
            _server_error("INVALID_PARAMETER_VALUE"),
            "(MlflowException: INVALID_PARAMETER_VALUE); MLflow rejected a logged name or value.",
        ),
        (_upload_forbidden, "(HTTPError: 403); check tracking access and credentials."),
    ],
    ids=["sdk", "dependency", "denied", "deleted", "rejected", "upload"],
)
def test_worker_names_the_failure_cause(tmp_path, monkeypatch, error, expected):
    import json

    from phasesweep.tracking import mlflow as worker

    def dispatch(request):
        raise error()

    monkeypatch.setattr(worker, "_dispatch", dispatch)
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"action": "publish"}))
    assert worker.main([str(request), str(response)]) == 0
    message = json.loads(response.read_text())["message"]
    assert expected in message
    assert "private" not in message


def test_worker_crash_names_its_preserved_diagnostic(tmp_path, monkeypatch):
    from phasesweep.runtime.process import ProcessResult
    from phasesweep.tracking import mlflow as worker

    diagnostic = tmp_path / "mlflow-worker.stderr.log"
    crashed = ProcessResult(return_code=1, timed_out=False, pid=0, duration_seconds=0)
    monkeypatch.setattr(worker, "run_worker", lambda *args, **kwargs: (crashed, {}, diagnostic))
    with pytest.raises(RuntimeError, match=f"exit 1.*{diagnostic}"):
        worker.request_operation(
            {}, operation_id="op", directory=tmp_path, environment={}, deadline=0.0
        )


@pytest.mark.parametrize("failure", ["target", "attempt"])
def test_run_preparation_failure_never_acquires_gpu(tmp_path, monkeypatch, failure):
    """An unready target stops once before any trial; an attempt failure fails only its trial."""
    from phasesweep.engine import NoFeasibleTrialError, run_experiment
    from phasesweep.errors import PhaseSweepError
    from phasesweep.runtime.gpu import GpuPool

    def unavailable(*args, **kwargs):
        raise RuntimeError("MLflow experiment is missing or deleted.")

    def no_lease(*args, **kwargs):
        pytest.fail("Run binding must precede GPU acquisition")

    monkeypatch.setattr(GpuPool, "acquire", no_lease)
    if failure == "target":
        monkeypatch.setattr("phasesweep.tracking.session.call_tracking", unavailable)
        monkeypatch.setattr(
            "phasesweep.tracking.session.prepare_attempt",
            lambda *a, **k: pytest.fail("No trial may start against an unready target"),
        )
        with pytest.raises(PhaseSweepError, match=r"missing or deleted\. No trial was started"):
            run_experiment(native_experiment(tmp_path))
    else:
        monkeypatch.setattr(
            "phasesweep.tracking.session.require_tracking_target", lambda *a, **k: None
        )
        monkeypatch.setattr("phasesweep.tracking.session.prepare_attempt", unavailable)
        experiment = native_experiment(tmp_path)
        with pytest.raises(NoFeasibleTrialError, match="trials.csv records each failed"):
            run_experiment(experiment)
        from phasesweep.engine.paths import _experiment_dir

        # The run log and the trials.csv the error points to both name the cause.
        root = _experiment_dir(experiment)
        for record in (root / "run.log", next(root.glob("*/trials.csv"))):
            assert "MLflow run preparation: MLflow experiment is missing" in record.read_text()


@pytest.mark.parametrize(
    "error,expected",
    [
        (ValueError("MLflow experiment is missing or deleted."), "missing or deleted"),
        (TimeoutError("MLflow operation exceeded its deadline."), "check that the server is"),
    ],
    ids=["deleted", "unreachable"],
)
def test_later_phase_rechecks_a_bound_target(tmp_path, monkeypatch, error, expected):
    """A summary run bound by an earlier phase does not excuse a vanished target."""
    from phasesweep.engine.errors import TrackingTargetUnavailableError
    from phasesweep.tracking import session

    experiment = native_experiment(tmp_path)
    bound = session.tracking_root(experiment) / "generation" / "generation-binding.json"
    bound.parent.mkdir(parents=True)
    session.write_record(bound, _binding())
    actions = []

    def unavailable(experiment, generation_id, request, *args):
        actions.append(request["action"])
        raise error

    monkeypatch.setattr(session, "call_tracking", unavailable)
    with pytest.raises(TrackingTargetUnavailableError, match=expected):
        session.require_tracking_target(
            experiment, generation_id="generation", environment={}, deadline=None
        )
    assert actions == ["resolve"]


@pytest.mark.parametrize("problem", [None, "experiment", "attempt", "finished"])
def test_binding_helper_checks_target_and_preserves_parent(tmp_path, monkeypatch, problem):
    import sys

    from phasesweep.tracking import bind_mlflow_run
    from phasesweep.tracking.session import read_binding, write_record

    binding = {**_binding(), "run_mode": "trainer"}
    write_record(
        tmp_path / "mlflow-expected.json", {k: v for k, v in binding.items() if k != "run_id"}
    )
    for key in ("attempt_id", "generation_id", "overrides_sha256"):
        monkeypatch.setenv("PHASESWEEP_" + key.upper(), binding[key])
    monkeypatch.setenv("PHASESWEEP_TRIAL_DIR", str(tmp_path))
    run = _run(binding, status="FINISHED" if problem == "finished" else "RUNNING")
    run.data.tags = {"mlflow.parentRunId": "trainer-parent"}
    if problem == "experiment":
        run.info.experiment_id = "foreign"
    elif problem == "attempt":
        run.data.tags["phasesweep.attempt_id"] = "foreign"
    updates = []

    def set_tag(run_id, key, value):
        updates.append(key)
        run.data.tags[key] = value

    client = SimpleNamespace(get_run=lambda _: run, set_tag=set_tag)
    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(MlflowClient=lambda **kwargs: client)
    )
    if problem is not None:
        with pytest.raises(ValueError, match="different experiment|another attempt|RUNNING"):
            bind_mlflow_run(binding["run_id"])
        assert not updates
        assert not (tmp_path / "mlflow-binding.json").exists()
    else:
        bind_mlflow_run(binding["run_id"])
        assert read_binding(tmp_path) == binding
        assert run.data.tags["mlflow.parentRunId"] == "trainer-parent"
