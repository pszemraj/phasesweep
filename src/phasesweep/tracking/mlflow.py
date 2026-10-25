"""Supervised public-client operations for native MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
        FindRequest,
        FindResult,
        MlflowCapture,
        MlflowTarget,
        Observation,
        Projection,
        PublishRequest,
        RunBinding,
        RunIdentity,
    )

_SDK_RETRIES = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
# Time a capture or find keeps for its last poll and its report before the parent stops it.
_REPORT_SECONDS = 5.0
# Worker response statuses; "unknown" reports a creation that may have taken effect.
_STATUSES = frozenset({"ok", "error", "unknown"})
# Server verdicts that reject a creation request outright, so no run was created.
_REJECTED_CREATION = frozenset(
    {
        "INVALID_PARAMETER_VALUE",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "RESOURCE_DOES_NOT_EXIST",
        "BAD_REQUEST",
        "RESOURCE_ALREADY_EXISTS",
    }
)


class OperationOutcomeUnknownError(RuntimeError):
    """An operation failed without proving that its request took no effect."""


class OperationRejectedError(RuntimeError):
    """A confirmed request rejection, including creation refused before sending."""


class _SdkExperiment(Protocol):
    """The experiment fields read from the public client."""

    @property
    def experiment_id(self) -> str:
        """Return the MLflow experiment identifier."""
        ...

    @property
    def lifecycle_stage(self) -> str:
        """Return whether the experiment is active or deleted."""
        ...


class _SdkRunInfo(Protocol):
    """The run metadata read from the public client."""

    @property
    def run_id(self) -> str:
        """Return the MLflow run identifier."""
        ...

    @property
    def experiment_id(self) -> str:
        """Return the identifier of the experiment that owns the run."""
        ...

    @property
    def lifecycle_stage(self) -> str:
        """Return whether the run is active or deleted."""
        ...

    @property
    def status(self) -> str:
        """Return the run's execution status, independently of its lifecycle stage."""
        ...


class _SdkRunData(Protocol):
    """The logged run data read from the public client."""

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return the run's metric names and latest scalar values."""
        ...

    @property
    def params(self) -> Mapping[str, str]:
        """Return the run's logged parameter names and string values."""
        ...

    @property
    def tags(self) -> Mapping[str, str]:
        """Return the run's tag names and string values."""
        ...


class _SdkRun(Protocol):
    """A public MLflow Run entity."""

    @property
    def info(self) -> _SdkRunInfo:
        """Return the run's identity, lifecycle stage and execution status."""
        ...

    @property
    def data(self) -> _SdkRunData:
        """Return the run's logged metrics, parameters and tags."""
        ...


class _SdkMetric(Protocol):
    """One logged point of a metric history."""

    @property
    def value(self) -> float:
        """Return the scalar value of this metric history point."""
        ...

    @property
    def step(self) -> int:
        """Return the training step recorded with this metric point."""
        ...

    @property
    def timestamp(self) -> int:
        """Return the metric point's Unix timestamp in milliseconds."""
        ...


class _SdkRunPage(Protocol):
    """One page of run-search results, with the token of the next page."""

    @property
    def token(self) -> str | None:
        """Return the continuation token, or no token when search results are exhausted."""
        ...

    def __iter__(self) -> Iterator[_SdkRun]: ...


class _SdkClient(Protocol):
    """The public MLflow client methods PhaseSweep calls, and nothing else."""

    def get_experiment(self, experiment_id: str) -> _SdkExperiment:
        """Look up an MLflow experiment by its identifier.

        :param str experiment_id: Remote experiment identifier.
        :return _SdkExperiment: Experiment identity and lifecycle stage.
        """
        ...

    def get_experiment_by_name(self, name: str) -> _SdkExperiment | None:
        """Look up an MLflow experiment by its exact name.

        :param str name: Remote experiment name.
        :return _SdkExperiment | None: Matching experiment, or None when absent.
        """
        ...

    def search_runs(
        self, experiment_ids: list[str], *, filter_string: str, page_token: str | None
    ) -> _SdkRunPage:
        """Search one page of runs matching a remote filter.

        :param list[str] experiment_ids: Experiments whose runs may match.
        :param str filter_string: MLflow run-search filter expression.
        :param str | None page_token: Previous page's continuation token, or None for the first page.
        :return _SdkRunPage: Matching runs and the next page's token.
        """
        ...

    def create_run(self, experiment_id: str, *, tags: dict[str, str], run_name: str) -> _SdkRun:
        """Create a run with its identity tags and display name.

        :param str experiment_id: Experiment that will own the run.
        :param dict[str, str] tags: Initial tags, including the PhaseSweep attempt identity.
        :param str run_name: Human-readable run name.
        :return _SdkRun: Newly created remote run.
        """
        ...

    def get_run(self, run_id: str) -> _SdkRun:
        """Read a run's current metadata and logged data.

        :param str run_id: Remote run identifier.
        :return _SdkRun: Run identity, status, metrics, parameters and tags.
        """
        ...

    def get_metric_history(self, run_id: str, key: str) -> Sequence[_SdkMetric]:
        """Read the logged history of one metric.

        :param str run_id: Run whose metric history is read.
        :param str key: Logged metric name.
        :return Sequence[_SdkMetric]: Metric points with their values, steps and timestamps.
        """
        ...

    def set_terminated(self, run_id: str, *, status: str) -> None:
        """Finish a managed run with the requested terminal status.

        :param str run_id: Remote run identifier.
        :param str status: Terminal execution status to record.
        """
        ...

    def set_tag(self, run_id: str, key: str, value: str) -> object:
        """Write one tag on an existing run.

        :param str run_id: Remote run identifier.
        :param str key: Tag name.
        :param str value: Tag value.
        :return object: SDK response, unused by PhaseSweep.
        """
        ...

    def log_batch(
        self,
        run_id: str,
        *,
        metrics: Sequence[Metric],
        params: Sequence[Param],
        tags: Sequence[RunTag],
        synchronous: bool,
    ) -> object:
        """Log a batch of metrics, parameters and tags to an existing run.

        :param str run_id: Remote run identifier.
        :param Sequence[Metric] metrics: Timestamped metric points to log.
        :param Sequence[Param] params: Parameters to log.
        :param Sequence[RunTag] tags: Tags to log.
        :param bool synchronous: Whether to wait for the SDK to complete the batch.
        :return object: SDK response, unused by PhaseSweep.
        """
        ...

    def log_artifact(self, run_id: str, local_path: str, *, artifact_path: str) -> None:
        """Upload one local file to a run's artifact directory.

        :param str run_id: Remote run identifier.
        :param str local_path: Local file to upload.
        :param str artifact_path: Destination directory relative to the run's artifact root.
        """
        ...


def require_mlflow_sdk() -> None:
    """Locate the optional SDK without importing it or starting client telemetry.

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
    :raises TimeoutError: The worker did not finish by the deadline.
    :raises UnsafeProcessCleanupError: The worker's cleanup is uncertain, so it may still run.
    :raises OperationOutcomeUnknownError: The worker crashed, its response could not be read
        or trusted, or it reported that a creation may have taken effect.
    :raises OperationRejectedError: The worker reported that the request failed.
    :return JsonObject: Successful public-client response.
    """
    try:
        result, response, diagnostic_path = run_worker(
            request,
            Path(__file__).resolve(),
            "mlflow",
            statuses=_STATUSES,
            run_id=operation_id,
            trial_dir=directory,
            environment=environment,
            deadline=deadline,
        )
    except (OSError, ValueError) as exc:
        # The worker has usually run to completion by now, so an unreadable response says nothing.
        raise OperationOutcomeUnknownError(
            f"MLflow worker response could not be read ({_failure_cause(exc)}); "
            "its outcome is unconfirmed."
        ) from exc
    if result.timed_out:
        raise TimeoutError("MLflow operation exceeded its deadline.")
    if result.return_code != 0 or response.get("status") not in _STATUSES:
        diagnostic = f" Diagnostic preserved at {diagnostic_path}." if diagnostic_path else ""
        raise OperationOutcomeUnknownError(
            f"MLflow worker failed (exit {result.return_code}).{diagnostic}"
        )
    status = response["status"]
    try:
        if status == "ok":
            return json_object(response["result"])
        message = json_text(response, "message")
    except (KeyError, ValueError) as exc:
        raise OperationOutcomeUnknownError(
            f"MLflow worker response is malformed ({_failure_cause(exc)}); "
            "its outcome is unconfirmed."
        ) from exc
    if status == "error":
        raise OperationRejectedError(message)
    raise OperationOutcomeUnknownError(message)


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

    A timed-out POST can have succeeded, so the caller reconciles by identity
    instead of replaying it. Idempotent reads keep SDK retries.
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


def _identity_run(client: _SdkClient, identity: RunIdentity) -> _SdkRun | None:
    """Search the run carrying this exact attempt and generation identity.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param RunIdentity identity: Identity with its resolved experiment ID.
    :raises ValueError: More than one run carries the identity.
    :return _SdkRun | None: The one matching run, or ``None`` when no run carries the identity.
    """
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
        page = client.search_runs(
            [identity["experiment_id"]], filter_string=query, page_token=token
        )
        matches.extend(page)
        token = page.token
        if not token:
            break
    if len(matches) > 1:
        raise ValueError("Multiple MLflow runs carry this attempt identity; binding is ambiguous.")
    return matches[0] if matches else None


def _error_code(exc: BaseException) -> object:
    """Return the secret-free code a failure carries: its MLflow error code or HTTP status.

    :param BaseException exc: SDK or transport failure.
    :return object: The code, or ``None`` when the failure carries none.
    """
    return getattr(exc, "error_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )


def _failure_cause(exc: BaseException) -> str:
    """Name a failure by its type and code, never by its text, which may hold secrets.

    :param BaseException exc: SDK or transport failure.
    :return str: Type name, with the code when there is one.
    """
    code = _error_code(exc)
    return f"{type(exc).__name__}: {code}" if code else type(exc).__name__


def _create(client: _SdkClient, request: CreateRequest) -> RunBinding:
    """Find or create the run carrying an exact identity, without SDK creation retries.

    Unconfirmed creation failures and failed verification of a newly created
    run report an unknown outcome rather than replaying the request.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param CreateRequest request: Trusted target and identity.
    :raises ValueError: More than one run carries the identity, or a run found by
        search contradicts it.
    :raises OperationOutcomeUnknownError: The creation request failed without a server
        rejection, or the run it created fails verification.
    :return RunBinding: Verified remote run binding.
    """
    experiment_id = _resolve(client, request["settings"])
    identity: RunIdentity = {**request["binding"], "experiment_id": experiment_id}
    run = _identity_run(client, identity)
    created = run is None
    if run is None:
        tags = identity_tags(identity)
        parent_run_id = identity.get("parent_run_id")
        if parent_run_id:
            tags["mlflow.parentRunId"] = parent_run_id
        try:
            with _without_sdk_replay():
                run = client.create_run(experiment_id, tags=tags, run_name=request["run_name"])
        except Exception as exc:
            from mlflow.exceptions import MlflowException

            if isinstance(exc, MlflowException) and exc.error_code in _REJECTED_CREATION:
                raise
            raise OperationOutcomeUnknownError(
                f"MLflow create may have taken effect ({_failure_cause(exc)}); "
                "its outcome is unconfirmed."
            ) from exc
    binding: RunBinding = {**identity, "run_id": run.info.run_id}
    try:
        verify_remote(run, binding)
    except ValueError as exc:
        if not created:
            raise
        raise OperationOutcomeUnknownError(
            f"MLflow created a run that failed verification ({exc}); it is not bound."
        ) from exc
    return binding


def _find(client: _SdkClient, request: FindRequest) -> FindResult:
    """Poll for an identity's existing run without creating one.

    Always search once, then poll until just before timeout_at to report absence
    before the parent deadline stops the worker.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param FindRequest request: Trusted target, identity, and polling bound.
    :raises ValueError: More than one run carries the identity, or the run contradicts it.
    :return FindResult: The verified binding, or that no run carries the identity.
    """
    experiment_id = _resolve(client, request["settings"])
    identity: RunIdentity = {**request["binding"], "experiment_id": experiment_id}
    reserve = min(_REPORT_SECONDS, request["settings"]["timeout_seconds"] / 2)
    stop = request["timeout_at"] - reserve
    while True:
        run = _identity_run(client, identity)
        if run is not None:
            binding: RunBinding = {**identity, "run_id": run.info.run_id}
            verify_remote(run, binding)
            return {"found": True, "binding": binding}
        if time.monotonic() >= stop:
            return {"found": False}
        time.sleep(min(request["settings"]["poll_seconds"], max(0, stop - time.monotonic())))


def _finished_without(
    data: _SdkRunData, *, metrics: Iterable[str], params: Iterable[str], tags: Iterable[str]
) -> str | None:
    """Name the required metrics, params and tags a finished run lacks, and the keys it has.

    :param _SdkRunData data: Logged data of the finished run.
    :param Iterable[str] metrics: Required metric names.
    :param Iterable[str] params: Required param names.
    :param Iterable[str] tags: Required tag names.
    :return str | None: The failure text, or ``None`` when the run has everything required.
    """
    lacking: list[str] = []
    available: list[str] = []
    for kind, wanted, present in (
        ("metric", metrics, data.metrics),
        ("param", params, data.params),
        ("tag", tags, data.tags),
    ):
        absent = sorted({key for key in wanted if key not in present})
        if absent:
            lacking += [f"{kind} {key!r}" for key in absent]
            available.append(f"{kind} keys: {sorted(present)!r}")
    if not lacking:
        return None
    return f"MLflow run finished without {', '.join(lacking)}; available {'; '.join(available)}."


def _capture(client: _SdkClient, request: CaptureRequest) -> MlflowCapture:
    """Fetch finished evidence, sharing complete histories across selectors.

    FINISHED runs fail immediately when required evidence is missing. Unfinished
    runs are polled until just before ``timeout_at``.

    :param _SdkClient client: Public MLflow client in the supervised worker.
    :param CaptureRequest request: Trusted target, bound run, and shared query.
    :raises ValueError: The run no longer matches its binding, ended FAILED or KILLED,
        restarted during capture, or finished without a required metric, param or tag
        (the message names it and the keys the run has) or without a metric's history.
    :raises TimeoutError: The run is still not finished when ``timeout_seconds`` runs out.
    :return MlflowCapture: Frozen shared native evidence.
    """
    from phasesweep.evidence.mlflow import select_observation
    from phasesweep.evidence.models import MlflowExtractor, MlflowRequiredGate

    binding, query = request["binding"], request["query"]
    # Stop just before timeout_at, a shared monotonic instant, so the worker can report
    # the run's state; a nearer phase or run deadline still stops it first.
    reserve = min(_REPORT_SECONDS, request["settings"]["timeout_seconds"] / 2)
    stop = request["timeout_at"] - reserve
    while True:
        run = client.get_run(binding["run_id"])
        verify_remote(run, binding)
        if run.info.status in {"FAILED", "KILLED"}:
            raise ValueError("MLflow run ended unsuccessfully; remote metrics cannot rescue it.")
        if run.info.status == "RUNNING" and binding["run_mode"] == "managed":
            client.set_terminated(binding["run_id"], status="FINISHED")
            run = client.get_run(binding["run_id"])
        if run.info.status == "FINISHED":
            selections = {
                identity: MlflowExtractor.model_validate(specification)
                for identity, specification in query["selections"].items()
            }
            gates = [MlflowRequiredGate.model_validate(gate) for gate in query["gates"]]
            keys = sorted({spec.metric_key for spec in selections.values()})
            lacking = _finished_without(
                run.data,
                metrics=keys,
                params=[key for gate in gates for key in gate.params],
                tags=[key for gate in gates for key in gate.tags],
            )
            if lacking is not None:
                if (
                    binding["run_mode"] == "managed"
                    and MlflowExtractor.model_validate(query["objective"]).metric_key
                    not in run.data.metrics
                ):
                    lacking += (
                        " Check whether the objective was logged under another key or logging "
                        "went to another run. Log the configured metric key into the supplied "
                        "run; after end_run() or another start_run(), resume its saved ID. "
                        "Give a logger the supplied run ID: read MLFLOW_RUN_ID before any "
                        "MLflow call, or use mlflow.active_run().info.run_id while the run is active."
                    )
                raise ValueError(lacking)
            observations: dict[str, Observation] = {}
            for key in keys:
                history = client.get_metric_history(binding["run_id"], key)
                if not history:
                    raise ValueError(
                        f"MLflow run finished with metric {key!r} but no history for it."
                    )
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
                "params": {key: final.data.params[key] for gate in gates for key in gate.params},
                "tags": {key: final.data.tags[key] for gate in gates for key in gate.tags},
            }
        if time.monotonic() >= stop:
            raise TimeoutError(
                f"MLflow run was still {run.info.status} when timeout_seconds ran out."
            )
        time.sleep(min(request["settings"]["poll_seconds"], max(0, stop - time.monotonic())))


def _publish(client: _SdkClient, request: PublishRequest) -> JsonObject:
    """Upload a retained projection and apply its managed run's terminal outcome.

    The retained outcome takes precedence over the status left by the trainer.
    Projection-specific artifact names keep winner uploads distinct from trial summaries.

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
    if finish and run.info.status != finish:
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
        ``find``, ``capture``, or ``publish``, or its payload is malformed.
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
    if action == "find":
        found = _find(
            client,
            {
                "action": "find",
                "settings": settings,
                "binding": validate_identity(request["binding"]),
                "timeout_at": json_number(request, "timeout_at"),
            },
        )
        return json_object(found)
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
    except OperationOutcomeUnknownError as exc:
        response = {"status": "unknown", "message": str(exc)}
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
        code = _error_code(exc)
        cause = _failure_cause(exc)
        failure: BaseException | None = exc
        while failure is not None and not isinstance(failure, ConnectionRefusedError):
            failure = failure.__cause__ or failure.__context__
        if isinstance(failure, ConnectionRefusedError):
            cause = _failure_cause(failure)
            advice = (
                "the tracking server refused the connection; check tracking_uri "
                "and that the server is running"
            )
        elif code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND", 404}:
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
