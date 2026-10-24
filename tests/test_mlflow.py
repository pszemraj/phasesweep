"""Native MLflow configuration, history, binding, and retry contracts."""

from __future__ import annotations

import errno
import json
import os
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from phasesweep.config.models import Experiment
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.evidence.mlflow import compile_query, observation_id, select_observation
from phasesweep.evidence.models import MlflowExtractor, MlflowRequiredGate
from phasesweep.tracking.mlflow import (
    OperationOutcomeUnknownError,
    OperationRejectedError,
    _capture,
    _create,
    _find,
)
from phasesweep.tracking.models import MlflowSettings
from tests.tracking_helpers import (
    FakeWorkerProcess,
    ManualClock,
    attempt_identity,
    mlflow_binder,
    mlflow_binding,
    mlflow_run,
    native_experiment,
    scripted_response,
)


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


@pytest.mark.parametrize("field", ["poll_seconds", "timeout_seconds"])
@pytest.mark.parametrize("value", [2.5, float("inf"), float("nan")])
def test_timing_values_must_be_finite(tmp_path, field, value):
    """The process supervisor needs a finite wait, so YAML loading refuses .inf and .nan."""
    import math

    import yaml

    from phasesweep.config.io import load_experiment

    config = yaml.safe_load((Path(__file__).parents[1] / "examples/mlflow.yaml").read_text())
    config.update(workdir=str(tmp_path / "runs"))
    config["mlflow"][field] = value
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    if math.isfinite(value):
        assert getattr(load_experiment(path).mlflow, field) == value
    else:
        with pytest.raises(ValidationError, match="finite number"):
            load_experiment(path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("extra", [{}, {"experiment_id": "1", "experiment_name": "x"}])
def test_exactly_one_experiment(extra):
    with pytest.raises(ValidationError):
        MlflowSettings(tracking_uri="databricks", **extra)


def test_selector_requires_explicit_step():
    with pytest.raises(ValidationError):
        spec("step")
    with pytest.raises(ValidationError):
        spec("last", step=2)


def test_shared_history_once():
    binding = mlflow_binding()
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

    client = SimpleNamespace(get_run=lambda _: mlflow_run(binding), get_metric_history=history)
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


@pytest.mark.parametrize(
    "problem,message", [("deleted", "run is deleted; restore it"), ("moved", "does not match")]
)
def test_remote_identity_failures_name_their_cause(problem, message):
    from phasesweep.tracking.mlflow import verify_remote

    binding = mlflow_binding()
    run = mlflow_run(binding)
    if problem == "deleted":
        run.info.lifecycle_stage = "deleted"
    else:
        run.info.experiment_id = "foreign"
    with pytest.raises(ValueError, match=message):
        verify_remote(run, binding)


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_unsuccessful_remote_cannot_be_finished(status):
    binding = mlflow_binding()
    client = SimpleNamespace(get_run=lambda _: mlflow_run(binding, status=status))
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


class _Page(list):
    """One page of run-search results with no next page."""

    token = None


def test_creation_binds_an_existing_identity_and_creates_only_when_absent(monkeypatch):
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    binding = mlflow_binding()
    identity = {k: v for k, v in binding.items() if k != "run_id"}
    created = []

    def create_run(*args, **kwargs):
        # Only the non-idempotent create is never replayed by the SDK.
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
        created.append(kwargs["tags"])
        return mlflow_run(binding)

    client = SimpleNamespace(
        get_experiment=lambda _: SimpleNamespace(experiment_id="1", lifecycle_stage="active"),
        search_runs=lambda *a, **k: _Page([mlflow_run(binding)]),
        create_run=create_run,
    )
    request = {"binding": identity, "settings": {"experiment_id": "1"}, "run_name": "r"}
    assert _create(client, request) == binding
    assert created == []
    client.search_runs = lambda *a, **k: _Page()
    assert _create(client, request) == binding
    assert len(created) == 1
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ
    client.search_runs = lambda *a, **k: _Page([mlflow_run(binding), mlflow_run(binding)])
    with pytest.raises(ValueError, match="ambiguous"):
        _create(client, request)


def _creation_client(*, search=None, create_run=None):
    """Build a fake SDK client over one experiment whose creation never succeeds unprompted."""

    def never_created(*args, **kwargs):
        pytest.fail("This test never creates a run.")

    return SimpleNamespace(
        get_experiment=lambda _: SimpleNamespace(experiment_id="1", lifecycle_stage="active"),
        search_runs=search or (lambda *a, **k: _Page()),
        create_run=create_run or never_created,
    )


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


def _connection_lost():
    from requests.exceptions import ConnectionError as RequestsConnectionError

    return RequestsConnectionError("private connection detail")


def _timed_out():
    from mlflow.exceptions import MlflowException

    return MlflowException("API request to private-host failed with timeout exception.")


_REJECTIONS = (
    "INVALID_PARAMETER_VALUE",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "RESOURCE_DOES_NOT_EXIST",
    "BAD_REQUEST",
    "RESOURCE_ALREADY_EXISTS",
)


@pytest.mark.parametrize(
    "stage,failure,definite",
    [
        *[("create_run", _server_error(code), True) for code in _REJECTIONS],
        ("create_run", _server_error("INTERNAL_ERROR"), False),
        ("create_run", _server_error("TEMPORARILY_UNAVAILABLE"), False),
        ("create_run", _timed_out, False),
        ("create_run", _connection_lost, False),
        ("create_run", lambda: OSError("private socket detail"), False),
        ("search_runs", _server_error("INTERNAL_ERROR"), True),
        ("search_runs", _timed_out, True),
    ],
    ids=[
        *[f"rejected-{code.lower()}" for code in _REJECTIONS],
        "server-error",
        "unavailable",
        "timeout",
        "connection",
        "socket",
        "search-server-error",
        "search-timeout",
    ],
)
def test_worker_separates_definite_failures_from_unknown_outcomes(stage, failure, definite):
    """Only a server rejection, or a failure before the request is sent, proves no run exists."""
    error = failure()

    def fail(*args, **kwargs):
        raise error

    client = _creation_client(
        search=fail if stage == "search_runs" else None,
        create_run=fail if stage == "create_run" else None,
    )
    request = {"binding": attempt_identity(), "settings": {"experiment_id": "1"}, "run_name": "r"}
    if definite:
        with pytest.raises(type(error)) as caught:
            _create(client, request)
        assert caught.value is error
    else:
        with pytest.raises(OperationOutcomeUnknownError) as caught:
            _create(client, request)
        assert caught.value.__cause__ is error
        assert "private" not in str(caught.value)
        assert type(error).__name__ in str(caught.value)


def test_verification_failure_is_unknown_only_for_a_run_this_call_created():
    """A run this call created exists whatever verification says; only its binding is in doubt."""
    binding = mlflow_binding()
    deleted = mlflow_run(binding)
    deleted.info.lifecycle_stage = "deleted"
    created = []

    def create_run(*args, **kwargs):
        created.append(kwargs["tags"])
        return deleted

    client = _creation_client(create_run=create_run)
    request = {"binding": attempt_identity(), "settings": {"experiment_id": "1"}, "run_name": "r"}
    with pytest.raises(OperationOutcomeUnknownError, match="not bound") as caught:
        _create(client, request)
    assert isinstance(caught.value.__cause__, ValueError)
    assert len(created) == 1
    # A run found by search was not created by this call, so contradicting the identity is definite.
    client.search_runs = lambda *a, **k: _Page([deleted])
    with pytest.raises(ValueError, match="run is deleted") as caught:
        _create(client, request)
    assert not isinstance(caught.value, OperationOutcomeUnknownError)
    assert len(created) == 1


def test_worker_reports_an_unknown_outcome_with_its_own_status(tmp_path, monkeypatch):
    import json

    from phasesweep.tracking import mlflow as worker

    def dispatch(request):
        raise OperationOutcomeUnknownError("MLflow create may have taken effect.")

    monkeypatch.setattr(worker, "_dispatch", dispatch)
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"action": "create"}))
    assert worker.main([str(request), str(response)]) == 0
    assert json.loads(response.read_text()) == {
        "status": "unknown",
        "message": "MLflow create may have taken effect.",
    }


@pytest.mark.parametrize(
    "pages,timeout_after,expected_searches,expected",
    [
        ([[mlflow_run(mlflow_binding())]], 10, 1, {"found": True, "binding": mlflow_binding()}),
        (
            [[], [], [mlflow_run(mlflow_binding())]],
            10,
            3,
            {"found": True, "binding": mlflow_binding()},
        ),
        ([[]], 0, 1, {"found": False}),
        ([[mlflow_run(mlflow_binding()), mlflow_run(mlflow_binding())]], 10, 1, ValueError),
    ],
    ids=["found", "found-late", "absent", "ambiguous"],
)
def test_find_polls_for_the_identity_and_never_creates(
    pages, timeout_after, expected_searches, expected
):
    """A find searches at least once, polls until its bound, and has no way to create."""
    searches = []
    remaining = list(pages)

    def search(*args, **kwargs):
        searches.append(kwargs["filter_string"])
        return _Page(remaining.pop(0))

    client = _creation_client(search=search)
    request = {
        "action": "find",
        "binding": attempt_identity(),
        "settings": {"experiment_id": "1", "poll_seconds": 0.01, "timeout_seconds": 10},
        "timeout_at": time.monotonic() + timeout_after,
    }
    if expected is ValueError:
        with pytest.raises(ValueError, match="ambiguous"):
            _find(client, request)
    else:
        assert _find(client, request) == expected
    assert len(searches) == expected_searches
    assert all("phasesweep.attempt_id" in query for query in searches)


def test_find_action_runs_through_the_worker_entry_point(tmp_path, monkeypatch):
    """The worker routes a find request to a read-only lookup and reports its result."""
    import json

    from phasesweep.tracking import mlflow as worker

    binding = mlflow_binding()
    client = _creation_client(search=lambda *a, **k: _Page([mlflow_run(binding)]))
    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(MlflowClient=lambda **kwargs: client)
    )
    settings = {
        **native_experiment(tmp_path).mlflow.target(),
        "experiment_id": "1",
        "experiment_name": None,
    }
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "action": "find",
                "settings": settings,
                "binding": attempt_identity(),
                "timeout_at": time.monotonic() + 10,
            }
        )
    )
    assert worker.main([str(request), str(response)]) == 0
    assert json.loads(response.read_text()) == {
        "status": "ok",
        "result": {"found": True, "binding": binding},
    }


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


@pytest.mark.parametrize("location", ["metric", "constraints", "observe"])
def test_result_names_are_nonempty(tmp_path, location):
    config = native_experiment(tmp_path).model_dump()
    if location == "metric":
        config[location]["name"] = ""
    elif location == "constraints":
        config[location][0]["name"] = ""
    else:
        config[location] = [
            {"name": "", "extractor": {"type": "json", "path": "result.json", "key": "x"}}
        ]
    with pytest.raises(ValidationError) as caught:
        Experiment.model_validate(config)
    message = str(caught.value)
    assert location in message
    assert "name" in message
    assert "at least 1 character" in message or "nonempty" in message


@pytest.mark.parametrize(
    "source,name,prefix,rule",
    [
        ("constraint", "acc@1", "constraint", "characters"),
        ("observation", "a//b", "observed", "path"),
        ("observation", "a/./b", "observed", "path"),
        ("constraint", "a/../b", "constraint", "path"),
        ("constraint", "x" * 230, "constraint", "length"),
        ("observation", "x" * 231, "observed", "length"),
        ("fixed_overrides", "x" * 231, "override", "length"),
        ("search_space", "x" * 231, "override", "length"),
        ("repeats", "x" * 231, "override", "length"),
    ],
)
@pytest.mark.parametrize("publishing", [True, False])
def test_mlflow_published_keys_validated_only_with_target(
    tmp_path, source, name, prefix, rule, publishing
):
    config = native_experiment(tmp_path).model_dump()
    local_extractor = {"type": "json", "path": "result.json", "key": "x"}
    if source == "constraint":
        config["constraints"][0]["name"] = name
    elif source == "observation":
        config["observe"] = [{"name": name, "extractor": local_extractor}]
    elif source == "repeats":
        config["phases"][0]["repeats"] = {"key": name, "values": [1, 2]}
        config["phases"][0]["n_trials"] = 4
    else:
        config["phases"][0][source] = {
            name: 1 if source == "fixed_overrides" else {"type": "categorical", "choices": [1, 2]}
        }
    if not publishing:
        config["mlflow"] = None
        config["metric"]["extractor"] = local_extractor
        config["constraints"][0]["extractor"] = local_extractor
        for phase in config["phases"]:
            phase["gates"] = []
        Experiment.model_validate(config)
        return
    with pytest.raises(ValidationError) as caught:
        Experiment.model_validate(config)
    message = str(caught.value)
    assert repr(name) in message
    assert source in message
    assert f"phasesweep.{prefix}.{name}" in message
    assert rule in message
    assert "rename" in message.lower()
    if source in {"fixed_overrides", "search_space", "repeats"}:
        assert "depth" in message


def test_mlflow_published_keys_accept_boundaries_and_native_characters(tmp_path):
    config = native_experiment(tmp_path).model_dump()
    config["constraints"][0]["name"] = "x" * (250 - len("phasesweep.constraint."))
    config["observe"] = [
        {
            "name": "résultat: score/a-b.c_1",
            "extractor": {"type": "json", "path": "result.json", "key": "x"},
        }
    ]
    config["phases"][0]["fixed_overrides"] = {"x" * 230: 1}
    Experiment.model_validate(config)


class _ScriptedWorker:
    """A stand-in for the supervised worker that answers each action from a script."""

    def __init__(self, monkeypatch, **script):
        from phasesweep.tracking import session

        self.script = {action: list(outcomes) for action, outcomes in script.items()}
        self.requests = []
        monkeypatch.setattr(session, "request_operation", self)

    @property
    def actions(self):
        return [request["action"] for request in self.requests]

    def __call__(self, request, **kwargs):
        self.requests.append(request)
        outcome = scripted_response(self.script, request)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


_ABSENT = {"found": False}


@pytest.mark.parametrize("slot", ["attempt", "generation"])
@pytest.mark.parametrize(
    "failure",
    [TimeoutError("exceeded its deadline"), OperationOutcomeUnknownError("response lost")],
    ids=["timeout", "unknown"],
)
def test_unconfirmed_creation_is_reconciled_and_never_resent(tmp_path, monkeypatch, slot, failure):
    """A request that may have created its run is only searched for afterwards."""
    import json

    from phasesweep.tracking import session

    bind, path = mlflow_binder(tmp_path, slot)
    pending = path.with_name(f"{path.stem}.pending.json")
    binding = mlflow_binding()

    def marked_before_sending():
        record = json.loads(pending.read_text())
        assert (record["schema_version"], record["identity"], record["run_name"]) == (
            1,
            attempt_identity(),
            "r",
        )
        return failure

    worker = _ScriptedWorker(
        monkeypatch, create=[marked_before_sending], find=[{"found": True, "binding": binding}]
    )
    assert bind() == binding
    assert worker.actions == ["create", "find"]
    assert worker.requests[1]["binding"] == attempt_identity()
    assert worker.requests[1]["timeout_at"] > time.monotonic()
    assert session.read_record(path) == binding
    assert not pending.exists()


def test_unresolved_creation_survives_restarts_until_its_run_appears(tmp_path, monkeypatch):
    bind, path = mlflow_binder(tmp_path, "attempt")
    pending = path.with_name(f"{path.stem}.pending.json")
    binding = mlflow_binding()
    worker = _ScriptedWorker(
        monkeypatch,
        create=[OperationOutcomeUnknownError("response lost")],
        find=[_ABSENT, _ABSENT, {"found": True, "binding": binding}],
    )
    unresolved = f"attempt {'b' * 32} of generation {'a' * 32} is unresolved"
    with pytest.raises(RuntimeError, match=unresolved) as first:
        bind()
    # The failure that left the request unconfirmed stays the diagnostic cause.
    assert isinstance(first.value.__cause__, OperationOutcomeUnknownError)
    assert pending.exists() and not path.exists()
    # Only the record survives a restart; the retry reconciles and does not create.
    with pytest.raises(RuntimeError, match=unresolved) as retry:
        bind()
    assert retry.value.__cause__ is None
    assert worker.actions == ["create", "find", "find"]
    assert bind() == binding
    assert worker.actions == ["create", "find", "find", "find"]
    assert not pending.exists()
    assert bind() == binding
    assert len(worker.requests) == 4


def test_rejected_creation_leaves_no_record_and_is_not_reconciled(tmp_path, monkeypatch):
    bind, path = mlflow_binder(tmp_path, "attempt")
    rejection = "MLflow create failed (MlflowException: PERMISSION_DENIED); check access."
    worker = _ScriptedWorker(
        monkeypatch, create=[OperationRejectedError(rejection), mlflow_binding()]
    )
    with pytest.raises(OperationRejectedError, match="PERMISSION_DENIED"):
        bind()
    assert worker.actions == ["create"]
    assert not path.exists() and not path.with_name(f"{path.stem}.pending.json").exists()
    # Nothing was created, so a later call sends the request afresh.
    assert bind() == mlflow_binding()
    assert worker.actions == ["create", "create"]


@pytest.mark.parametrize(
    "failure,expired,error,later_actions",
    [
        (
            UnsafeProcessCleanupError("worker cleanup is uncertain"),
            False,
            UnsafeProcessCleanupError,
            ["create", "find"],
        ),
        (TimeoutError("exceeded its deadline"), True, RuntimeError, ["create"]),
    ],
    ids=["cleanup-uncertain", "deadline-passed"],
)
def test_creation_that_cannot_be_reconciled_keeps_its_record(
    tmp_path, monkeypatch, failure, expired, error, later_actions
):
    bind, path = mlflow_binder(tmp_path, "attempt")
    deadline = time.monotonic() - 1 if expired else None
    worker = _ScriptedWorker(monkeypatch, create=[failure], find=[_ABSENT])
    with pytest.raises(error):
        bind(deadline)
    assert worker.actions == ["create"]
    assert path.with_name(f"{path.stem}.pending.json").exists()
    # Later calls never create; an expired deadline does not even search.
    with pytest.raises(RuntimeError, match="unresolved"):
        bind(deadline)
    assert worker.actions == later_actions
    assert path.with_name(f"{path.stem}.pending.json").exists()


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.EIO, "response unreadable"),
        ValueError("response malformed"),
        KeyError("result"),
        RuntimeError("unclassified"),
    ],
    ids=["os-error", "value-error", "key-error", "runtime-error"],
)
def test_unclassified_creation_failure_keeps_its_record_and_is_not_resent(
    tmp_path, monkeypatch, failure
):
    """Only a worker's rejection proves nothing was created; any other failure keeps the record."""
    bind, path = mlflow_binder(tmp_path, "attempt")
    pending = path.with_name(f"{path.stem}.pending.json")
    binding = mlflow_binding()
    worker = _ScriptedWorker(
        monkeypatch, create=[failure], find=[_ABSENT, {"found": True, "binding": binding}]
    )
    with pytest.raises(type(failure)) as caught:
        bind()
    assert caught.value is failure
    assert worker.actions == ["create"]
    assert pending.exists() and not path.exists()
    # A later call only searches: it never sends a second create.
    with pytest.raises(RuntimeError, match="unresolved"):
        bind()
    assert worker.actions == ["create", "find"]
    assert pending.exists()
    assert bind() == binding
    assert worker.actions == ["create", "find", "find"]
    assert not pending.exists()


@pytest.mark.parametrize("recovery", ["bind", "sync"])
@pytest.mark.parametrize("reply", ["ok", "unknown"])
@pytest.mark.parametrize("slot", ["attempt", "generation"])
def test_unreadable_creation_response_leaves_the_run_recoverable(
    tmp_path, monkeypatch, slot, reply, recovery
):
    """A response the controller cannot read says nothing about a creation, whatever it held."""
    from phasesweep.engine.paths import _trial_dir_for
    from phasesweep.tracking import session

    experiment = native_experiment(tmp_path)
    identity = attempt_identity()
    directory = _trial_dir_for(
        experiment,
        "p",
        3,
        generation_id=identity["generation_id"],
        attempt_id=identity["attempt_id"],
    )
    directory.mkdir(parents=True)
    bind, path = mlflow_binder(
        tmp_path, slot, experiment=experiment, identity=identity, directory=directory
    )
    pending = path.with_name(f"{path.stem}.pending.json")
    binding = mlflow_binding()
    created = (
        {"status": "ok", "result": binding}
        if reply == "ok"
        else {"status": "unknown", "message": "Creation may still commit."}
    )
    worker = FakeWorkerProcess(
        monkeypatch,
        unreadable=["create"],
        create=[created],
        find=[
            {"status": "ok", "result": _ABSENT},
            {"status": "ok", "result": {"found": True, "binding": binding}},
        ],
    )
    with pytest.raises(RuntimeError, match="unresolved") as caught:
        bind()
    # The unreadable acknowledgement stays the diagnostic cause; the request is only searched for.
    unconfirmed = caught.value.__cause__
    assert isinstance(unconfirmed, OperationOutcomeUnknownError)
    assert isinstance(unconfirmed.__cause__, OSError) and unconfirmed.__cause__.errno == errno.EIO
    assert worker.actions == ["create", "find"]
    assert pending.exists() and not path.exists()

    # The run appears and reads work again: the identity binds it without a second create.
    if recovery == "bind":
        assert bind() == binding
    else:
        session.resolve_pending_creations(experiment, {}, identity["generation_id"])
    assert worker.actions == ["create", "find", "find"]
    assert session.read_record(path) == binding
    assert not pending.exists()


def _write_pending(path, identity):
    from phasesweep.tracking import session

    path.parent.mkdir(parents=True, exist_ok=True)
    session.write_record(path, {"schema_version": 1, "identity": identity, "run_name": "r"})
    return path


def test_sync_binds_late_runs_and_leaves_absent_ones_pending(tmp_path, monkeypatch):
    """Leftover records get one read-only search each; a found run is bound, an absent one waits."""
    from phasesweep.engine.paths import _trial_dir_for
    from phasesweep.tracking import session

    experiment = native_experiment(tmp_path)
    first, second = "1" * 32, "2" * 32
    root = session.tracking_root(experiment)

    def attempt_dir(generation, attempt):
        return _trial_dir_for(experiment, "p", 3, generation_id=generation, attempt_id=attempt)

    pending = {
        "generation": _write_pending(
            root / first / "generation-binding.pending.json", attempt_identity(first, first)
        ),
        "absent": _write_pending(
            attempt_dir(first, "absent") / "mlflow-binding.pending.json",
            attempt_identity(first, "absent"),
        ),
        "bound": _write_pending(
            attempt_dir(first, "bound") / "mlflow-binding.pending.json",
            attempt_identity(first, "bound"),
        ),
        "other": _write_pending(
            attempt_dir(second, "late") / "mlflow-binding.pending.json",
            attempt_identity(second, "late"),
        ),
    }
    session.write_record(pending["bound"].with_name("mlflow-binding.json"), mlflow_binding())
    searched = []

    def find(experiment, generation_id, request, environment, deadline=None):
        searched.append((generation_id, request["action"], request["binding"]["attempt_id"]))
        assert request["timeout_at"] <= time.monotonic()
        if request["binding"]["attempt_id"] == "absent":
            return _ABSENT
        return {"found": True, "binding": {**request["binding"], "run_id": "late-run"}}

    monkeypatch.setattr(session, "call_tracking", find)
    session.resolve_pending_creations(experiment, {}, first)
    assert sorted(searched) == sorted([(first, "find", first), (first, "find", "absent")])
    assert session.read_binding(root / first, "generation-binding.json")["run_id"] == "late-run"
    assert not pending["generation"].exists()
    assert not pending["bound"].exists()
    # A record whose search finds nothing, and another generation's, stay pending.
    assert pending["absent"].exists() and pending["other"].exists()
    assert not pending["absent"].with_name("mlflow-binding.json").exists()

    session.resolve_pending_creations(experiment, {}, None)
    assert not pending["other"].exists()
    assert session.read_binding(pending["other"].parent)["run_id"] == "late-run"
    assert pending["absent"].exists()
    assert {action for _, action, _ in searched} == {"find"}


@pytest.mark.parametrize(
    "error,raises",
    [
        (RuntimeError("MLflow find failed (MlflowException: PERMISSION_DENIED)."), False),
        (ValueError("Multiple MLflow runs carry this attempt identity."), False),
        (UnsafeProcessCleanupError("worker cleanup is uncertain"), True),
    ],
    ids=["failed", "ambiguous", "cleanup-uncertain"],
)
def test_sync_keeps_a_record_whose_search_fails(tmp_path, monkeypatch, caplog, error, raises):
    from phasesweep.tracking import session

    experiment = native_experiment(tmp_path)
    generation = "1" * 32
    path = _write_pending(
        session.tracking_root(experiment) / generation / "generation-binding.pending.json",
        attempt_identity(generation, generation),
    )

    def find(*args, **kwargs):
        raise error

    monkeypatch.setattr(session, "call_tracking", find)
    if raises:
        with pytest.raises(type(error)):
            session.resolve_pending_creations(experiment, {}, generation)
    else:
        session.resolve_pending_creations(experiment, {}, generation)
        assert "still unresolved" in caplog.text
    assert path.exists()
    assert not path.with_name("generation-binding.json").exists()


@pytest.mark.parametrize("field", ["run_id", "experiment_id", "attempt_id", "unbound"])
def test_wrong_binding_is_rejected_before_remote_access(tmp_path, monkeypatch, field):
    from phasesweep.evidence.evaluation import ExtractorError, TrialContext, _capture_mlflow
    from phasesweep.tracking import session

    binding = mlflow_binding()
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
    binding = {**mlflow_binding(), "run_mode": "trainer"}
    client = SimpleNamespace(get_run=lambda _: mlflow_run(binding, status="RUNNING"))
    with pytest.raises(TimeoutError, match="was still RUNNING when timeout_seconds ran out"):
        _capture(
            client,
            {"binding": binding, "query": {}, "settings": {"timeout_seconds": 0}, "timeout_at": 0},
        )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    """Replace the capture worker's clock without waiting for polling deadlines.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces the worker's clock.
    :return ManualClock: Fake time and recorded poll sleeps.
    """
    from phasesweep.tracking import mlflow as worker

    fake = ManualClock()
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=fake.monotonic, sleep=fake.sleep))
    return fake


def _capture_request(binding, *, gates=(), timeout_seconds=120, timeout_at=100.0):
    """Build a capture of metric ``loss`` plus the given presence gates, on the fake clock."""
    return {
        "binding": binding,
        "query": {
            "selections": {observation_id(spec()): spec().model_dump()},
            "gates": [gate.model_dump() for gate in gates],
        },
        "settings": {"timeout_seconds": timeout_seconds, "poll_seconds": 2},
        "timeout_at": timeout_at,
    }


def _states_client(binding, *states, metrics=None, params=None, tags=None):
    """Serve one run state per ``get_run`` call, repeating the last, over the given run data."""
    runs = []
    for state in states:
        run = mlflow_run(binding, status=state, metrics=metrics)
        run.data.params = params or {}
        run.data.tags = {**run.data.tags, **(tags or {})}
        runs.append(run)
    client = SimpleNamespace(
        reads=0,
        get_metric_history=lambda run_id, key: [SimpleNamespace(value=2.0, step=1, timestamp=2)],
    )

    def get_run(run_id):
        client.reads += 1
        return runs[min(client.reads, len(runs)) - 1]

    client.get_run = get_run
    return client


def _finished_failure(clock, *, gates=(), metrics=None, params=None, tags=None):
    """Capture a finished run that lacks something; return the failure after proving it was at once."""
    binding = mlflow_binding()
    client = _states_client(binding, "FINISHED", metrics=metrics, params=params, tags=tags)
    with pytest.raises(ValueError) as failure:
        _capture(client, _capture_request(binding, gates=gates))
    # One read and no sleep: the finished run was not given time to gain the item.
    assert (client.reads, clock.sleeps, clock.now) == (1, [], 0.0)
    return str(failure.value)


def test_missing_metric_is_not_replaced_by_result_metric(clock):
    message = _finished_failure(clock, metrics={"phasesweep.objective": 1, "accuracy": 0.9})
    assert message == (
        "MLflow run finished without metric 'loss'; "
        "available metric keys: ['accuracy', 'phasesweep.objective']."
    )


def test_finished_run_missing_a_param_fails_at_once(clock):
    gate = MlflowRequiredGate(type="mlflow_required", params=["lr"])
    message = _finished_failure(clock, gates=[gate], params={"seed": "1", "batch": "8"})
    assert message == (
        "MLflow run finished without param 'lr'; available param keys: ['batch', 'seed']."
    )


def test_finished_run_missing_a_tag_fails_at_once(clock):
    gate = MlflowRequiredGate(type="mlflow_required", tags=["dataset"])
    message = _finished_failure(clock, gates=[gate], tags={"stage": "dev"})
    available = sorted({*mlflow_run(mlflow_binding()).data.tags, "stage"})
    assert message == f"MLflow run finished without tag 'dataset'; available tag keys: {available}."


def test_finished_run_missing_every_kind_names_each_once(clock):
    gates = [
        MlflowRequiredGate(type="mlflow_required", params=["lr", "warmup"], tags=["dataset"]),
        MlflowRequiredGate(type="mlflow_required", params=["lr"]),
    ]
    message = _finished_failure(
        clock, gates=gates, metrics={"accuracy": 0.9}, params={"warmup": "1", "seed": "1"}
    )
    tags = sorted(mlflow_run(mlflow_binding()).data.tags)
    assert message == (
        "MLflow run finished without metric 'loss', param 'lr', tag 'dataset'; "
        f"available metric keys: ['accuracy']; param keys: ['seed', 'warmup']; tag keys: {tags}."
    )


def test_finished_run_with_an_empty_metric_history_fails_at_once(clock):
    binding = mlflow_binding()
    client = _states_client(binding, "FINISHED")
    client.get_metric_history = lambda run_id, key: []
    with pytest.raises(ValueError) as failure:
        _capture(client, _capture_request(binding))
    assert str(failure.value) == "MLflow run finished with metric 'loss' but no history for it."
    assert clock.sleeps == []


def test_item_logged_before_finishing_is_accepted_without_polling(clock):
    binding = mlflow_binding()
    gate = MlflowRequiredGate(type="mlflow_required", params=["lr"], tags=["dataset"])
    client = _states_client(binding, "FINISHED", params={"lr": "0.1"}, tags={"dataset": "v2"})
    capture = _capture(client, _capture_request(binding, gates=[gate]))
    assert (capture["params"], capture["tags"]) == ({"lr": "0.1"}, {"dataset": "v2"})
    assert clock.sleeps == []


def test_finished_without_a_required_metric_reaches_the_trial_as_the_extractor_error(
    tmp_path, monkeypatch
):
    """The worker's report crosses the real entry point and response reader unchanged."""
    import shlex

    import mlflow

    from phasesweep.evidence.evaluation import ExtractorError, TrialContext, _capture_mlflow
    from phasesweep.runtime import process
    from phasesweep.tracking import mlflow as worker
    from phasesweep.tracking import session

    binding = mlflow_binding()
    session.write_record(
        tmp_path / "mlflow-expected.json", {k: v for k, v in binding.items() if k != "run_id"}
    )
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
    run = mlflow_run(binding, metrics={"cost": 1.0, "accuracy": 0.9})
    monkeypatch.setattr(
        mlflow, "MlflowClient", lambda tracking_uri: SimpleNamespace(get_run=lambda _: run)
    )

    def run_in_process(command, **kwargs):
        request_path, response_path = shlex.split(command)[-2:]
        assert worker.main([request_path, response_path]) == 0
        return process.ProcessResult(return_code=0, timed_out=False, pid=0, duration_seconds=0)

    monkeypatch.setattr(process, "run_supervised", run_in_process)
    with pytest.raises(ExtractorError) as failure:
        _capture_mlflow(ctx, deadline=None)
    assert str(failure.value) == (
        "MLflow run finished without metric 'eval/loss'; "
        "available metric keys: ['accuracy', 'cost']."
    )


def test_running_run_is_polled_until_it_finishes_and_then_judged_at_once(clock):
    binding = {**mlflow_binding(), "run_mode": "trainer"}
    accepted = _states_client(binding, "RUNNING", "RUNNING", "FINISHED")
    assert _capture(accepted, _capture_request(binding))["run_state"] == "FINISHED"
    assert clock.sleeps == [2, 2]
    clock.sleeps.clear()
    lacking = _states_client(binding, "RUNNING", "RUNNING", "FINISHED", metrics={"accuracy": 0.9})
    with pytest.raises(ValueError, match="finished without metric 'loss'"):
        _capture(lacking, _capture_request(binding))
    # The first FINISHED read decides; the 120 s budget is not waited out.
    assert lacking.reads == 3
    assert clock.sleeps == [2, 2]


def test_running_run_times_out_at_the_deadline(clock):
    binding = {**mlflow_binding(), "run_mode": "trainer"}
    client = _states_client(binding, "RUNNING")
    request = _capture_request(binding, timeout_seconds=120, timeout_at=100.0)
    with pytest.raises(TimeoutError, match="was still RUNNING when timeout_seconds ran out"):
        _capture(client, request)
    # Polling stops min(5, timeout_seconds / 2) seconds before timeout_at.
    assert clock.now == 95.0
    assert set(clock.sleeps) <= {1.0, 2.0}


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
    binding = mlflow_binding()
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
    path = publication._save_intent(
        experiment, binding["generation_id"], "test", binding, projection
    )
    assert publication._deliver(experiment, path, {}) is False
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

    from phasesweep.engine.read import experiment_status, read_status
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
            result_snapshot=capture_pre_generation_result_snapshot(experiment).model_dump(
                mode="json"
            ),
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


@pytest.mark.parametrize(
    "state,cleanup_confirmed,retired",
    [
        ("allocated", None, True),
        ("exited", True, True),
        ("exited", False, False),
        ("launching", None, False),
    ],
)
def test_tracking_operation_retires_its_entry_only_on_proven_cleanup(
    tmp_path, state, cleanup_confirmed, retired
):
    """The registry entry outlives a worker whose cleanup the lifecycle record does not prove."""
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.runtime.process import write_attempt_lifecycle
    from phasesweep.tracking.session import tracking_operation

    experiment = native_experiment(tmp_path)
    with tracking_operation(experiment, "a" * 32) as (operation_id, directory):
        assert len(list(_attempts_dir(experiment).glob("*.json"))) == 1
        write_attempt_lifecycle(
            directory,
            attempt_id=operation_id,
            state=state,
            cleanup_confirmed=cleanup_confirmed,
        )
    assert bool(list(_attempts_dir(experiment).glob("*.json"))) is not retired


def test_tracking_operation_keeps_the_body_cleanup_error_over_an_unreadable_record(tmp_path):
    """An unresolved worker the body reported is not replaced by the record read failing."""
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.runtime.process import ATTEMPT_LIFECYCLE_FILE
    from phasesweep.tracking.session import tracking_operation

    experiment = native_experiment(tmp_path)
    error = UnsafeProcessCleanupError("worker still unresolved")
    with (
        pytest.raises(UnsafeProcessCleanupError) as caught,
        tracking_operation(experiment, "a" * 32) as (_, directory),
    ):
        (directory / ATTEMPT_LIFECYCLE_FILE).write_text("{")
        raise error
    assert caught.value is error
    assert len(list(_attempts_dir(experiment).glob("*.json"))) == 1


def test_tracking_operation_treats_an_unreadable_record_as_cleanup_uncertainty(tmp_path):
    """A record that cannot be read never reads as proof the worker is gone."""
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
    from phasesweep.runtime.process import ATTEMPT_LIFECYCLE_FILE
    from phasesweep.tracking.session import tracking_operation

    experiment = native_experiment(tmp_path)
    with (
        pytest.raises(ProcessCleanupUncertainError, match="may still be running") as caught,
        tracking_operation(experiment, "a" * 32) as (_, directory),
    ):
        (directory / ATTEMPT_LIFECYCLE_FILE).write_text("{")
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.action is OperatorAction.RESTORE_TREE
    assert len(list(_attempts_dir(experiment).glob("*.json"))) == 1


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
    from phasesweep.engine.run import run_experiment
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


@pytest.mark.parametrize(
    "action,error,expected",
    [
        (
            "publish",
            lambda: ModuleNotFoundError("No module named 'mlflow'", name="mlflow"),
            "requires phasesweep[mlflow]",
        ),
        (
            "publish",
            lambda: ModuleNotFoundError("No module named 'boto3'", name="boto3"),
            "needs Python module 'boto3'",
        ),
        (
            "publish",
            _server_error("PERMISSION_DENIED"),
            "(MlflowException: PERMISSION_DENIED); check tracking access and credentials.",
        ),
        (
            "publish",
            _server_error("RESOURCE_DOES_NOT_EXIST"),
            "(MlflowException: RESOURCE_DOES_NOT_EXIST); the experiment or run no longer exists.",
        ),
        (
            "publish",
            _server_error("INVALID_PARAMETER_VALUE"),
            "(MlflowException: INVALID_PARAMETER_VALUE); MLflow rejected a logged name or value.",
        ),
        (
            "resolve",
            _server_error("INVALID_PARAMETER_VALUE"),
            "(MlflowException: INVALID_PARAMETER_VALUE); MLflow rejected the configured experiment.",
        ),
        ("publish", _upload_forbidden, "(HTTPError: 403); check tracking access and credentials."),
    ],
    ids=["sdk", "dependency", "denied", "deleted", "rejected", "bad-target", "upload"],
)
def test_worker_names_the_failure_cause(tmp_path, monkeypatch, action, error, expected):
    import json

    from phasesweep.tracking import mlflow as worker

    def dispatch(request):
        raise error()

    monkeypatch.setattr(worker, "_dispatch", dispatch)
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"action": action}))
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
    with pytest.raises(OperationOutcomeUnknownError, match=f"exit 1.*{diagnostic}"):
        worker.request_operation(
            {}, operation_id="op", directory=tmp_path, environment={}, deadline=0.0
        )


@pytest.mark.parametrize(
    "return_code,timed_out,response,expected",
    [
        (0, False, {"status": "ok", "result": {"x": 1}}, {"x": 1}),
        (0, False, {"status": "error", "message": "rejected"}, OperationRejectedError),
        (
            0,
            False,
            {"status": "unknown", "message": "may have taken effect"},
            OperationOutcomeUnknownError,
        ),
        (1, False, {}, OperationOutcomeUnknownError),
        (0, False, {"status": "other"}, OperationOutcomeUnknownError),
        (0, True, {}, TimeoutError),
        (0, False, {"status": "ok", "result": [1]}, OperationOutcomeUnknownError),
        (0, False, {"status": "ok"}, OperationOutcomeUnknownError),
        (0, False, {"status": "error", "message": 5}, OperationOutcomeUnknownError),
        (0, False, {"status": "error"}, OperationOutcomeUnknownError),
        (0, False, {"status": "unknown", "message": ["lost"]}, OperationOutcomeUnknownError),
        (
            0,
            False,
            OSError(errno.EIO, "private path", "response.json"),
            OperationOutcomeUnknownError,
        ),
        (0, False, json.JSONDecodeError("Expecting value", "", 0), OperationOutcomeUnknownError),
        (0, False, ValueError("Expected a JSON object, not list."), OperationOutcomeUnknownError),
        (
            0,
            False,
            UnsafeProcessCleanupError("worker cleanup is uncertain"),
            UnsafeProcessCleanupError,
        ),
    ],
    ids=[
        "ok",
        "error",
        "unknown",
        "crashed",
        "invalid-status",
        "timed-out",
        "ok-result-not-an-object",
        "ok-result-missing",
        "error-message-not-text",
        "error-message-missing",
        "unknown-message-not-text",
        "response-unreadable",
        "response-not-json",
        "response-not-an-object",
        "cleanup-uncertain",
    ],
)
def test_worker_outcomes_reach_the_controller_as_distinct_exceptions(
    tmp_path, monkeypatch, return_code, timed_out, response, expected
):
    """Only a well-formed error proves the request failed; an untrusted response proves nothing."""
    from phasesweep.runtime.process import ProcessResult
    from phasesweep.tracking import mlflow as worker

    result = ProcessResult(return_code=return_code, timed_out=timed_out, pid=0, duration_seconds=0)
    seen = []

    def run_worker(*args, **kwargs):
        seen.append(kwargs["statuses"])
        if isinstance(response, Exception):
            raise response
        return result, response, None

    monkeypatch.setattr(worker, "run_worker", run_worker)
    call = {"operation_id": "op", "directory": tmp_path, "environment": {}, "deadline": 0.0}
    if isinstance(expected, dict):
        assert worker.request_operation({}, **call) == expected
    else:
        with pytest.raises(expected) as caught:
            worker.request_operation({}, **call)
        # Each outcome is exactly its class, so a definite rejection is never the unknown subclass.
        assert type(caught.value) is expected
        if isinstance(response, Exception):
            # A response that cannot be read keeps its cause; uncertain cleanup passes through as is.
            assert (
                caught.value if expected is type(response) else caught.value.__cause__
            ) is response
            assert "private" not in str(caught.value)
    assert seen == [frozenset({"ok", "error", "unknown"})]


@pytest.mark.parametrize("failure", ["target", "attempt"])
def test_run_preparation_failure_never_acquires_gpu(tmp_path, monkeypatch, failure):
    """An unready target stops once before any trial; an attempt failure fails only its trial."""
    from phasesweep.engine.run import run_experiment
    from phasesweep.engine.selection import NoFeasibleTrialError
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
        with pytest.raises(NoFeasibleTrialError, match="trials.csv records why each failed"):
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
    session.write_record(bound, mlflow_binding())
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

    binding = {**mlflow_binding(), "run_mode": "trainer"}
    write_record(
        tmp_path / "mlflow-expected.json", {k: v for k, v in binding.items() if k != "run_id"}
    )
    for key in ("attempt_id", "generation_id", "overrides_sha256"):
        monkeypatch.setenv("PHASESWEEP_" + key.upper(), binding[key])
    monkeypatch.setenv("PHASESWEEP_TRIAL_DIR", str(tmp_path))
    run = mlflow_run(binding, status="FINISHED" if problem == "finished" else "RUNNING")
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


def test_accepted_projection_reports_observed_values() -> None:
    """Observed values are read from the trial's own attributes and mirrored as metrics."""
    import optuna

    from phasesweep.engine.state import (
        DURATION_ATTR,
        FEASIBLE_ATTR,
        OVERRIDES_ATTR,
        constraint_attr,
        observed_attr,
    )
    from phasesweep.tracking.publication import _accepted_projection

    def projection(**observed: float):
        return _accepted_projection(
            optuna.trial.create_trial(
                value=0.5,
                state=optuna.trial.TrialState.COMPLETE,
                user_attrs={
                    OVERRIDES_ATTR: json.dumps({"lr": 0.1}),
                    FEASIBLE_ATTR: True,
                    DURATION_ATTR: 1.5,
                    constraint_attr("memory"): 4.0,
                    **{observed_attr(name): value for name, value in observed.items()},
                },
            ),
            {},
        )

    reported = projection(peak=7.5, steps=3)
    assert reported["metrics"]["phasesweep.observed.peak"] == 7.5
    assert reported["metrics"]["phasesweep.observed.steps"] == 3
    assert reported["metrics"]["phasesweep.constraint.memory"] == 4.0
    assert reported["summary"]["observed"] == {"peak": 7.5, "steps": 3}

    bare = projection()
    assert not any(key.startswith("phasesweep.observed.") for key in bare["metrics"])
    assert bare["summary"]["observed"] == {}
