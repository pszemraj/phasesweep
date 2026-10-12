"""Supervised public-client operations for native MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Protocol

from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.json import (
    JsonObject,
    json_integer,
    json_number,
    json_number_map,
    json_object,
    json_optional_text,
    json_text,
    json_text_map,
)
from phasesweep.runtime.remote_worker import run_worker
from phasesweep.runtime.time import utc_now_iso

if TYPE_CHECKING:
    from mlflow.entities import Metric, Param, RunTag

    from phasesweep.tracking.records import (
        CaptureRequest,
        CreateRequest,
        MlflowCapture,
        MlflowTarget,
        Observation,
        Projection,
        PublishRequest,
        RunBinding,
        RunIdentity,
    )

_SDK_RETRIES = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
# Time a capture keeps for its last poll and its report before the parent stops it.
_CAPTURE_REPORT_SECONDS = 5.0


class _SdkExperiment(Protocol):
    """The experiment fields read from the public client."""

    @property
    def experiment_id(self) -> str: ...

    @property
    def lifecycle_stage(self) -> str: ...


class _SdkRunInfo(Protocol):
    """The run metadata read from the public client."""

    @property
    def run_id(self) -> str: ...

    @property
    def experiment_id(self) -> str: ...

    @property
    def lifecycle_stage(self) -> str: ...

    @property
    def status(self) -> str: ...


class _SdkRunData(Protocol):
    """The logged run data read from the public client."""

    @property
    def metrics(self) -> Mapping[str, float]: ...

    @property
    def params(self) -> Mapping[str, str]: ...

    @property
    def tags(self) -> Mapping[str, str]: ...


class _SdkRun(Protocol):
    """A public MLflow Run entity."""

    @property
    def info(self) -> _SdkRunInfo: ...

    @property
    def data(self) -> _SdkRunData: ...


class _SdkMetric(Protocol):
    """One logged point of a metric history."""

    @property
    def value(self) -> float: ...

    @property
    def step(self) -> int: ...

    @property
    def timestamp(self) -> int: ...


class _SdkRunPage(Protocol):
    """One page of run-search results, with the token of the next page."""

    @property
    def token(self) -> str | None: ...

    def __iter__(self) -> Iterator[_SdkRun]: ...


class _SdkClient(Protocol):
    """The public MLflow client methods PhaseSweep calls, and nothing else."""

    def get_experiment(self, experiment_id: str) -> _SdkExperiment: ...

    def get_experiment_by_name(self, name: str) -> _SdkExperiment | None: ...

    def search_runs(
        self, experiment_ids: list[str], *, filter_string: str, page_token: str | None
    ) -> _SdkRunPage: ...

    def create_run(self, experiment_id: str, *, tags: dict[str, str], run_name: str) -> _SdkRun: ...

    def get_run(self, run_id: str) -> _SdkRun: ...

    def get_metric_history(self, run_id: str, key: str) -> Sequence[_SdkMetric]: ...

    def set_terminated(self, run_id: str, *, status: str) -> None: ...

    def set_tag(self, run_id: str, key: str, value: str) -> object: ...

    def log_batch(
        self,
        run_id: str,
        *,
        metrics: Sequence[Metric],
        params: Sequence[Param],
        tags: Sequence[RunTag],
        synchronous: bool,
    ) -> object: ...

    def log_artifact(self, run_id: str, local_path: str, *, artifact_path: str) -> None: ...


def require_mlflow_sdk() -> None:
    """Check that the optional SDK is installed before allocating tracked work.

    The check locates the package without importing it: importing MLflow
    starts its client telemetry, which belongs in the supervised workers.

    :raises PhaseSweepError: The MLflow client package is not installed.
    """
    import importlib.util

    if importlib.util.find_spec("mlflow") is None:
        from phasesweep.errors import OperatorAction, PhaseSweepError

        raise PhaseSweepError(
            "MLflow tracking requires the optional SDK in the PhaseSweep environment. "
            'Install this distribution with its mlflow extra: pip install "phasesweep[mlflow]".',
            action=OperatorAction.FIX_CONFIG,
        )


def request_operation(
    request: Mapping[str, object],
    *,
    operation_id: str,
    directory: Path,
    environment: Mapping[str, str],
    deadline: float,
) -> JsonObject:
    """Execute a bounded SDK operation in a registered process slot.

    :param Mapping[str, object] request: Trusted operation and payload; it must serialize to JSON.
    :param str operation_id: PhaseSweep process identity, never the MLflow ID.
    :param Path directory: Registered operation or attempt directory.
    :param Mapping[str, str] environment: Composed authentication environment.
    :param float deadline: Absolute monotonic deadline.
    :raises RuntimeError: The worker failed, timed out, or rejected the request.
    :raises ValueError: The worker's response is malformed.
    :return JsonObject: Successful public-client response.
    """
    result, response, diagnostic_path = run_worker(
        request,
        Path(__file__).resolve(),
        "mlflow",
        statuses=frozenset({"ok", "error"}),
        run_id=operation_id,
        trial_dir=directory,
        environment=environment,
        deadline=deadline,
    )
    if result.timed_out:
        raise TimeoutError("MLflow operation exceeded its deadline.")
    if result.return_code != 0 or response.get("status") not in {"ok", "error"}:
        diagnostic = f" Diagnostic preserved at {diagnostic_path}." if diagnostic_path else ""
        raise RuntimeError(f"MLflow worker failed (exit {result.return_code}).{diagnostic}")
    if response["status"] == "error":
        raise RuntimeError(json_text(response, "message"))
    return json_object(response["result"])


def identity_tags(binding: Mapping[str, object]) -> dict[str, str]:
    """Return the reserved identity tags shared with trainer-created runs.

    :param Mapping[str, object] binding: Durable PhaseSweep run identity.
    :return dict[str, str]: Reserved remote identity, without credentials.
    """
    return {
        f"phasesweep.{key}": str(binding[key])
        for key in ("experiment", "phase", "generation_id", "attempt_id", "overrides_sha256")
    }


def verify_remote(run: _SdkRun, binding: RunBinding) -> None:
    """Require a remote run to retain its exact experiment and attempt tags.

    :param _SdkRun run: Public MLflow Run entity.
    :param RunBinding binding: Locally recorded identity.
    :raises ValueError: Remote identity contradicts the binding, or the run is deleted.
    """
    if run.info.run_id != binding["run_id"] or run.info.experiment_id != binding["experiment_id"]:
        raise ValueError("MLflow run does not match the bound experiment/run identity.")
    if run.info.lifecycle_stage != "active":
        # Deleting an experiment deletes its runs too; restoring it restores them.
        raise ValueError("MLflow run is deleted; restore it, or its experiment, in MLflow.")
    if any(run.data.tags.get(k) != v for k, v in identity_tags(binding).items()):
        raise ValueError("MLflow run has conflicting PhaseSweep identity tags.")


def _resolve(client: _SdkClient, settings: MlflowTarget) -> str:
    """Resolve an existing, active experiment through the public client.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param MlflowTarget settings: Trusted target or operation payload.
    :raises ValueError: The experiment does not exist or has been deleted.
    :return str: Existing experiment ID.
    """
    experiment_id = settings.get("experiment_id")
    experiment_name = settings.get("experiment_name")
    experiment: _SdkExperiment | None
    try:
        if experiment_id:
            experiment = client.get_experiment(experiment_id)
        elif experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
        else:
            raise ValueError("MLflow requires exactly one of experiment_id or experiment_name.")
    except Exception as exc:
        # An unknown ID raises where an unknown name returns None.
        if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise
        experiment = None
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(
            "MLflow experiment is missing or deleted; provision an existing experiment first."
        )
    return str(experiment.experiment_id)


@contextmanager
def _without_sdk_replay() -> Iterator[None]:
    """Disable the SDK's HTTP retries for one non-idempotent request.

    A timed-out POST can have succeeded, so the caller's next try reconciles
    by identity instead of replaying it. Idempotent reads keep SDK retries.
    Databricks targets go through the Databricks SDK, whose own retry layer
    this setting does not reach.

    :return Iterator[None]: Scope in which the SDK sends each request once.
    """
    previous = os.environ.get(_SDK_RETRIES)
    os.environ[_SDK_RETRIES] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_SDK_RETRIES, None)
        else:
            os.environ[_SDK_RETRIES] = previous


def _create(client: _SdkClient, request: CreateRequest) -> RunBinding:
    """Bind the run carrying this exact identity, creating it only when none exists.

    A lost creation response is indistinguishable from a request that never
    arrived, so every attempt searches the identity tags before creating.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param CreateRequest request: Trusted target and identity.
    :raises ValueError: More than one run carries the identity.
    :return RunBinding: Verified remote run binding.
    """
    experiment_id = _resolve(client, request["settings"])
    identity: RunIdentity = {**request["binding"], "experiment_id": experiment_id}
    tags = identity_tags(identity)
    # Double quotes are the identifier quoting both OSS and Databricks search
    # document; Databricks does not document backticks.
    query = " AND ".join(
        f"tags.\"{key}\" = '{tags[key]}'"
        for key in ("phasesweep.attempt_id", "phasesweep.generation_id")
    )
    matches: list[_SdkRun] = []
    token: str | None = None
    while True:
        page = client.search_runs([experiment_id], filter_string=query, page_token=token)
        matches.extend(page)
        token = page.token
        if not token:
            break
    if len(matches) > 1:
        raise ValueError("Multiple MLflow runs carry this attempt identity; binding is ambiguous.")
    if matches:
        run = matches[0]
    else:
        parent_run_id = identity.get("parent_run_id")
        if parent_run_id:
            tags["mlflow.parentRunId"] = parent_run_id
        with _without_sdk_replay():
            run = client.create_run(experiment_id, tags=tags, run_name=request["run_name"])
    binding: RunBinding = {**identity, "run_id": run.info.run_id}
    verify_remote(run, binding)
    return binding


def _capture(client: _SdkClient, request: CaptureRequest) -> MlflowCapture:
    """Fetch finished evidence, sharing complete histories across selectors.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param CaptureRequest request: Trusted target, bound run, and shared query.
    :raises ValueError: The run no longer matches its binding, ended FAILED or KILLED,
        or restarted during capture.
    :raises TimeoutError: The run is unfinished, or its evidence is still missing, when
        ``timeout_seconds`` runs out; the message names what is missing.
    :return MlflowCapture: Frozen shared native evidence.
    """
    from phasesweep.evidence.mlflow import select_observation
    from phasesweep.evidence.models import MlflowExtractor, MlflowRequiredGate

    binding, query = request["binding"], request["query"]
    # The parent stops this worker when timeout_seconds runs out (``timeout_at``,
    # a shared monotonic instant). Ending the wait just before lets the worker
    # report what is missing; a nearer phase/run deadline still stops it first.
    reserve = min(_CAPTURE_REPORT_SECONDS, request["settings"]["timeout_seconds"] / 2)
    stop = request["timeout_at"] - reserve
    while True:
        run = client.get_run(binding["run_id"])
        verify_remote(run, binding)
        if run.info.status in {"FAILED", "KILLED"}:
            raise ValueError("MLflow run ended unsuccessfully; remote metrics cannot rescue it.")
        if run.info.status == "RUNNING" and binding["run_mode"] == "managed":
            client.set_terminated(binding["run_id"], status="FINISHED")
            run = client.get_run(binding["run_id"])
        missing: list[str] = []
        if run.info.status == "FINISHED":
            selections = {
                identity: MlflowExtractor.model_validate(specification)
                for identity, specification in query["selections"].items()
            }
            gates = [MlflowRequiredGate.model_validate(gate) for gate in query["gates"]]
            keys = sorted({spec.metric_key for spec in selections.values()})
            missing = [f"metric {key!r}" for key in keys if key not in run.data.metrics] + [
                f"{namespace[:-1]} {key!r}"
                for gate in gates
                for namespace, wanted, present in (
                    ("params", gate.params, run.data.params),
                    ("tags", gate.tags, run.data.tags),
                )
                for key in wanted
                if key not in present
            ]
            if not missing:
                observations: dict[str, Observation] = {}
                for key in keys:
                    history = client.get_metric_history(binding["run_id"], key)
                    if not history:
                        missing.append(f"metric {key!r} history")
                        break
                    records: list[Mapping[str, object]] = [
                        {
                            "value": m.value,
                            "step": m.step,
                            "timestamp": m.timestamp,
                            "model_id": getattr(m, "model_id", None),
                            "dataset_name": getattr(m, "dataset_name", None),
                            "dataset_digest": getattr(m, "dataset_digest", None),
                        }
                        for m in history
                    ]
                    for identity, spec in selections.items():
                        if spec.metric_key == key:
                            observations[identity] = select_observation(records, spec)
                else:
                    final = client.get_run(binding["run_id"])
                    verify_remote(final, binding)
                    if final.info.status != "FINISHED":
                        raise ValueError("MLflow run restarted during evidence capture.")
                    return {
                        "kind": "mlflow",
                        "binding": binding,
                        "run_state": "FINISHED",
                        "query": query,
                        "observations": observations,
                        "retrieved_at": utc_now_iso(),
                        "params": {
                            key: final.data.params[key] for gate in gates for key in gate.params
                        },
                        "tags": {key: final.data.tags[key] for gate in gates for key in gate.tags},
                    }
        if time.monotonic() >= stop:
            state = (
                f"finished without {', '.join(missing)}"
                if missing
                else f"was still {run.info.status}"
            )
            raise TimeoutError(f"MLflow run {state} when timeout_seconds ran out.")
        time.sleep(min(request["settings"]["poll_seconds"], max(0, stop - time.monotonic())))


def _publish(client: _SdkClient, request: PublishRequest) -> JsonObject:
    """Mirror one retained projection, then finish a still-running owned run.

    A run that is already terminal keeps its status, so a remote outcome is
    never overwritten; the projection still records PhaseSweep's result. The
    summary artifact is named after the projection, so a winner projection
    never replaces its run's trial summary.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param PublishRequest request: Trusted target, bound run, and projection.
    :raises ValueError: Termination was requested for a trainer-owned run.
    :return JsonObject: Empty success response.
    """
    from mlflow.entities import Metric, Param, RunTag

    binding, projection = request["binding"], request["projection"]
    run = client.get_run(binding["run_id"])
    verify_remote(run, binding)
    finish = request.get("finish")
    if finish and binding["run_mode"] != "managed":
        raise ValueError("Only runs PhaseSweep created may be terminated.")
    client.log_batch(
        binding["run_id"],
        metrics=[
            Metric(k, v, projection["timestamp"], 0) for k, v in projection["metrics"].items()
        ],
        params=[Param(k, v) for k, v in projection["params"].items()],
        tags=[RunTag(k, v) for k, v in projection["tags"].items()],
        synchronous=True,
    )
    with TemporaryDirectory(prefix="phasesweep-mlflow-summary-") as directory:
        path = Path(directory) / f"{request['name']}.json"
        path.write_text(
            json.dumps(projection["summary"], sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        client.log_artifact(binding["run_id"], str(path), artifact_path="phasesweep")
    if finish and run.info.status not in {"FINISHED", "FAILED", "KILLED"}:
        client.set_terminated(binding["run_id"], status=finish)
    return {}


def _projection(value: object) -> Projection:
    """Narrow a decoded retry projection to the shape the upload reads.

    :param object value: Decoded projection from a delivery record.
    :raises ValueError: A field is absent or has the wrong type.
    :return Projection: The narrowed projection.
    """
    projection = json_object(value)
    metrics = json_number_map(projection["metrics"])
    params = json_text_map(projection["params"])
    tags = json_text_map(projection["tags"])
    if metrics is None or params is None or tags is None:
        raise ValueError("MLflow projection needs numeric metrics and string params and tags.")
    return {
        "timestamp": json_integer(projection, "timestamp"),
        "metrics": metrics,
        "params": params,
        "tags": tags,
        "summary": json_object(projection["summary"]),
    }


def _dispatch(request: JsonObject) -> JsonObject:
    """Dispatch trusted controller requests using public MLflow APIs.

    :param JsonObject request: Trusted controller operation.
    :raises ValueError: The request names an action other than ``resolve``, ``create``,
        ``capture``, or ``publish``, or its payload is malformed.
    :return JsonObject: Successful operation payload.
    """
    from mlflow import MlflowClient

    from phasesweep.evidence.mlflow import parse_query
    from phasesweep.tracking.models import MlflowSettings
    from phasesweep.tracking.records import validate_binding, validate_identity

    settings = MlflowSettings.model_validate(request["settings"]).target()
    client: _SdkClient = MlflowClient(tracking_uri=settings["tracking_uri"])
    action = request["action"]
    if action == "resolve":
        return {"experiment_id": _resolve(client, settings)}
    if action == "create":
        created = _create(
            client,
            {
                "action": "create",
                "settings": settings,
                "binding": validate_identity(request["binding"]),
                "run_name": json_text(request, "run_name"),
            },
        )
        return json_object(created)
    if action == "capture":
        captured = _capture(
            client,
            {
                "action": "capture",
                "settings": settings,
                "binding": validate_binding(request["binding"]),
                "query": parse_query(request["query"]),
                "timeout_at": json_number(request, "timeout_at"),
            },
        )
        return json_object(captured)
    if action == "publish":
        return _publish(
            client,
            {
                "action": "publish",
                "settings": settings,
                "name": json_text(request, "name"),
                "binding": validate_binding(request["binding"]),
                "projection": _projection(request["projection"]),
                "finish": json_optional_text(request, "finish"),
            },
        )
    raise ValueError("Unknown internal MLflow operation.")


class _Arguments(argparse.Namespace):
    """Private request and response paths."""

    request: Path
    response: Path


def main(argv: list[str] | None = None) -> int:
    """Run an internal request without leaking SDK error details.

    :param list[str] | None argv: Private request/response paths.
    :return int: Zero after a structured response, 130 on interruption.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv, namespace=_Arguments())
    request = json_object(json.loads(args.request.read_text(encoding="utf-8")))
    response: JsonObject
    try:
        response = {"status": "ok", "result": _dispatch(request)}
    except ImportError as exc:
        missing = exc.name or "an optional dependency"
        response = {
            "status": "error",
            "message": "MLflow requires phasesweep[mlflow] in the controller environment."
            if missing.partition(".")[0] == "mlflow"
            else f"MLflow {request['action']} needs Python module {missing!r}, which is not installed in the controller environment.",
        }
    except (ValueError, TimeoutError) as exc:
        response = {"status": "error", "message": str(exc)}
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from mlflow.exceptions import MlflowException
        from requests.exceptions import RequestException

        if not isinstance(exc, MlflowException | RequestException | OSError):
            raise
        # The server's error code, e.g. PERMISSION_DENIED, or the HTTP status of an
        # artifact upload carries no secrets; the exception text may.
        code: object = getattr(exc, "error_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        cause = f"{type(exc).__name__}: {code}" if code else type(exc).__name__
        if code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND", 404}:
            advice = "the experiment or run no longer exists"
        elif code in {"INVALID_PARAMETER_VALUE", 400}:
            advice = (
                "MLflow rejected the configured experiment"
                if request["action"] == "resolve"
                else "MLflow rejected a logged name or value"
            )
        else:
            advice = "check tracking access and credentials"
        response = {
            "status": "error",
            "message": f"MLflow {request['action']} failed ({cause}); {advice}.",
        }
    private_atomic_write_text(args.response, json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
