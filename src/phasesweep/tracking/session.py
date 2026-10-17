"""Durable MLflow run bindings and engine-facing supervised operations."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.json import (
    JsonObject,
    json_object,
    json_text,
    json_text_map,
    strict_json_loads,
)
from phasesweep.tracking.mlflow import (
    OperationOutcomeUnknownError,
    OperationRejectedError,
    identity_tags,
    request_operation,
    verify_remote,
)
from phasesweep.tracking.records import validate_binding, validate_identity

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment
    from phasesweep.engine.state import Winner
    from phasesweep.tracking.mlflow import _SdkClient
    from phasesweep.tracking.records import (
        CreateAction,
        FindAction,
        PendingCreation,
        RunBinding,
        RunIdentity,
        TrackingAction,
    )

log = logging.getLogger(__name__)
BINDING_FILE = "mlflow-binding.json"
INHERITS_FILE = "mlflow-inherits.json"
_PENDING_SUFFIX = ".pending.json"
_PARENT_LOCK = threading.RLock()


def tracking_root(experiment: Experiment) -> Path:
    """Return mutable tracking state outside scientific generation manifests.

    :param Experiment experiment: Local experiment namespace.
    :return Path: Tracking state root.
    """
    from phasesweep.engine.paths import _experiment_dir

    return _experiment_dir(experiment) / "tracking"


def write_record(path: Path, record: Mapping[str, object]) -> None:
    """Atomically persist private tracking state.

    :param Path path: Record destination.
    :param Mapping[str, object] record: JSON-compatible payload.
    """
    private_atomic_write_text(
        path,
        json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n",
        require_private_dir=False,
    )


def read_record(path: Path) -> JsonObject:
    """Read a current-format tracking record.

    :param Path path: Record to read.
    :raises ValueError: The record does not use the current format.
    :return JsonObject: Decoded current-format record.
    """
    data: object = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported MLflow tracking record format.")
    return json_object(data)


def read_binding(directory: Path, name: str = BINDING_FILE) -> RunBinding:
    """Load a local run binding.

    :param Path directory: Directory holding the binding.
    :param str name: Binding file name; an attempt's by default.
    :return RunBinding: Validated binding.
    """
    return validate_binding(read_record(directory / name))


def bind_mlflow_run(run_id: str) -> None:
    """Bind a trainer-created MLflow run to the current PhaseSweep attempt.

    Call once inside the trainer's active run, before logging evaluations.

    :param str run_id: Real run ID returned by the trainer's MLflow logger.
    :raises ValueError: The run conflicts with the current attempt or experiment.
    """
    from mlflow import MlflowClient

    directory = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
    expected = read_record(directory / "mlflow-expected.json")
    if (
        expected["run_mode"] != "trainer"
        or expected["attempt_id"] != os.environ["PHASESWEEP_ATTEMPT_ID"]
        or expected["generation_id"] != os.environ["PHASESWEEP_GENERATION_ID"]
        or expected["overrides_sha256"] != os.environ["PHASESWEEP_OVERRIDES_SHA256"]
    ):
        raise ValueError("MLflow binding helper requires this trainer-owned attempt.")
    binding = validate_binding({**expected, "run_id": run_id})
    if (directory / BINDING_FILE).exists() and read_binding(directory) != binding:
        raise ValueError("This attempt is already bound to another MLflow run.")
    client: _SdkClient = MlflowClient(tracking_uri=binding["tracking_uri"])
    run = client.get_run(run_id)
    if run.info.experiment_id != expected["experiment_id"] or run.info.lifecycle_stage != "active":
        raise ValueError("Trainer MLflow run belongs to a different experiment.")
    # An ended run predates this attempt, so its metrics cannot be its evidence.
    if run.info.status != "RUNNING":
        raise ValueError("Bind the trainer's MLflow run while it is still RUNNING.")
    tags = identity_tags(binding)
    if any(key in run.data.tags and run.data.tags[key] != value for key, value in tags.items()):
        raise ValueError("Trainer MLflow run already belongs to another attempt.")
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    verify_remote(client.get_run(run_id), binding)
    write_record(directory / BINDING_FILE, binding)


def read_inherits(directory: Path) -> dict[str, str]:
    """Load the native run of each parent winner an attempt inherited.

    :param Path directory: Attempt directory.
    :raises ValueError: The record is not a mapping of phase names to run IDs.
    :return dict[str, str]: Run ID by parent phase name.
    """
    inherits = json_text_map(read_record(directory / INHERITS_FILE).get("inherits"))
    if inherits is None:
        raise ValueError("MLflow inherited-run record is malformed.")
    return inherits


def operation_deadline(experiment: Experiment, deadline: float | None = None) -> float:
    """Cap tracking work by its timeout and any remaining execution budget.

    :param Experiment experiment: Experiment with MLflow settings.
    :param float | None deadline: Existing absolute execution deadline.
    :return float: Absolute bounded operation deadline.
    """
    assert experiment.mlflow is not None
    stop = time.monotonic() + experiment.mlflow.timeout_seconds
    return min(stop, deadline) if deadline is not None else stop


@contextmanager
def tracking_operation(experiment: Experiment, generation_id: str) -> Iterator[tuple[str, Path]]:
    """Register a tracking worker and retain ownership until its cleanup is confirmed.

    Unreadable lifecycle records retain the registry entry. Entries are retired
    when the worker never launched or cleanup is confirmed.

    :param Experiment experiment: Registry owner.
    :param str generation_id: Producing generation.
    :raises ProcessCleanupUncertainError: The lifecycle record is unreadable and
        the body raised nothing; an exception the body raised propagates instead.
    :return Iterator[tuple[str, Path]]: Operation identity and process directory.
    """
    from phasesweep.engine.attempts import _register_tracking_operation, _retire_active_attempt
    from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle

    operation_id = uuid4().hex
    directory = tracking_root(experiment) / generation_id / "operations" / operation_id
    directory.mkdir(parents=True, exist_ok=True)
    write_attempt_lifecycle(directory, attempt_id=operation_id, state="allocated")
    _register_tracking_operation(
        experiment, operation_id=operation_id, generation_id=generation_id, directory=directory
    )
    body_raised = False
    try:
        yield operation_id, directory
    except BaseException:
        body_raised = True
        raise
    finally:
        try:
            lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=operation_id)
        except ValueError as exc:
            # The body's own failure may already say the worker is unresolved, so it wins.
            if not body_raised:
                raise ProcessCleanupUncertainError(
                    f"Tracking operation {operation_id} has an unreadable lifecycle record in "
                    f"{directory} ({exc}), so its worker may still be running. Its attempt "
                    "registry entry is kept. Delete the entry file only if you are certain "
                    "nothing is running.",
                    action=OperatorAction.RESTORE_TREE,
                ) from exc
            log.warning(
                "Tracking operation %s has an unreadable lifecycle record in %s (%s); "
                "its attempt registry entry is kept.",
                operation_id,
                directory,
                exc,
            )
        else:
            if lifecycle is not None and (
                lifecycle.state == "allocated"
                or (lifecycle.state == "exited" and lifecycle.cleanup_confirmed)
            ):
                _retire_active_attempt(experiment, operation_id)


def call_tracking(
    experiment: Experiment,
    generation_id: str,
    request: TrackingAction,
    environment: Mapping[str, str],
    deadline: float | None = None,
) -> JsonObject:
    """Perform a registered tracking-only operation.

    :param Experiment experiment: Experiment with configured tracking.
    :param str generation_id: Operation's generation.
    :param TrackingAction request: Trusted SDK action.
    :param Mapping[str, str] environment: Composed credentials/transport.
    :param float | None deadline: Optional absolute run deadline.
    :return JsonObject: Successful worker response.
    """
    assert experiment.mlflow is not None
    with tracking_operation(experiment, generation_id) as (operation_id, directory):
        return request_operation(
            {"settings": experiment.mlflow.target(), **request},
            operation_id=operation_id,
            directory=directory,
            environment=environment,
            deadline=operation_deadline(experiment, deadline),
        )


def _create_bound(
    experiment: Experiment,
    identity: RunIdentity,
    path: Path,
    environment: Mapping[str, str],
    *,
    run_name: str,
    deadline: float | None,
    attempt_directory: Path | None = None,
) -> RunBinding:
    """Bind or create an identity's run, persisting intent before sending creation.

    A confirmed rejection removes pending intent. Unknown outcomes, timeouts,
    and other failures retain it; subsequent calls search rather than create.
    An empty search leaves the creation unresolved across restarts.

    :param Experiment experiment: Configured experiment.
    :param RunIdentity identity: Exact creation identity.
    :param Path path: Durable binding destination.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str run_name: Display name for a newly created run.
    :param float | None deadline: Existing execution deadline.
    :param Path | None attempt_directory: Existing process slot for managed attempt creation.
    :raises OperationRejectedError: The worker reported that the request failed.
    :raises RuntimeError: The creation outcome is unresolved.
    :return RunBinding: Bound native run identity.
    """
    if path.exists():
        return read_binding(path.parent, path.name)
    pending = path.with_suffix(_PENDING_SUFFIX)

    def reconcile(cause: BaseException | None = None) -> RunBinding:
        """Look for the run a sent creation request may have created; never create one.

        :param BaseException | None cause: The failure that left the request unconfirmed, if any.
        :raises RuntimeError: No run carries the identity before the deadline.
        :return RunBinding: The run that request created.
        """
        if deadline is None or time.monotonic() < deadline:
            found = _find_bound(
                experiment,
                identity,
                environment,
                timeout_at=operation_deadline(experiment, deadline),
                deadline=deadline,
                attempt_directory=attempt_directory,
            )
            if found is not None:
                return found
        raise RuntimeError(
            f"MLflow run creation for attempt {identity['attempt_id']} of generation "
            f"{identity['generation_id']} is unresolved: an earlier request may still create "
            "the run, so it is not sent again. sync-tracking binds the run if it appears."
        ) from cause

    if pending.exists():
        binding = reconcile()
    else:
        record: PendingCreation = {"schema_version": 1, "identity": identity, "run_name": run_name}
        write_record(pending, record)
        request: CreateAction = {"action": "create", "binding": identity, "run_name": run_name}
        try:
            created = _send(experiment, identity, request, environment, deadline, attempt_directory)
        except OperationRejectedError:
            # A refused or unsent request created nothing, so a later call may send it afresh.
            pending.unlink(missing_ok=True)
            raise
        except (OperationOutcomeUnknownError, TimeoutError) as exc:
            binding = reconcile(exc)
        else:
            binding = validate_binding(created)
    write_record(path, binding)
    pending.unlink(missing_ok=True)
    return binding


def _send(
    experiment: Experiment,
    identity: RunIdentity,
    request: TrackingAction,
    environment: Mapping[str, str],
    deadline: float | None,
    attempt_directory: Path | None,
) -> JsonObject:
    """Send one request through the process slot the identity's creation uses.

    :param Experiment experiment: Configured experiment.
    :param RunIdentity identity: Creation identity owning the slot.
    :param TrackingAction request: Trusted SDK action.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param float | None deadline: Existing execution deadline.
    :param Path | None attempt_directory: A managed attempt's directory; ``None`` for the
        generation run, which takes a registered tracking operation.
    :return JsonObject: Successful worker response.
    """
    assert experiment.mlflow is not None
    if attempt_directory is None:
        return call_tracking(experiment, identity["generation_id"], request, environment, deadline)
    return request_operation(
        {"settings": experiment.mlflow.target(), **request},
        operation_id=identity["attempt_id"],
        directory=attempt_directory,
        environment=environment,
        deadline=operation_deadline(experiment, deadline),
    )


def _find_bound(
    experiment: Experiment,
    identity: RunIdentity,
    environment: Mapping[str, str],
    *,
    timeout_at: float,
    deadline: float | None,
    attempt_directory: Path | None,
) -> RunBinding | None:
    """Search an identity's run read-only, polling until ``timeout_at``.

    :param Experiment experiment: Configured experiment.
    :param RunIdentity identity: Exact creation identity.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param float timeout_at: Absolute monotonic time the search stops polling at.
    :param float | None deadline: Existing execution deadline.
    :param Path | None attempt_directory: A managed attempt's directory, if it owns the slot.
    :return RunBinding | None: The verified binding, or ``None`` when no run carries the identity.
    """
    request: FindAction = {"action": "find", "binding": identity, "timeout_at": timeout_at}
    found = _send(experiment, identity, request, environment, deadline, attempt_directory)
    return validate_binding(found["binding"]) if found.get("found") is True else None


def resolve_pending_creations(
    experiment: Experiment, environment: Mapping[str, str], generation_id: str | None
) -> None:
    """Reconcile pending creations with one read-only search per record, without polling.

    A found run is bound and its pending record removed. Missing runs and search
    failures retain their records; failures also emit a warning.

    :param Experiment experiment: Configured experiment.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None generation_id: Generation to resolve, or ``None`` for every generation.
    :raises UnsafeProcessCleanupError: A search worker's cleanup is uncertain.
    """
    from phasesweep.engine.paths import _experiment_dir
    from phasesweep.errors import UnsafeProcessCleanupError

    scope = generation_id or "*"
    attempt_pending = Path(BINDING_FILE).with_suffix(_PENDING_SUFFIX).name
    pending_records = sorted(
        tracking_root(experiment).glob(f"{scope}/generation-binding{_PENDING_SUFFIX}")
    ) + sorted(
        _experiment_dir(experiment).glob(
            f"*/trial_*__generation_{scope}__attempt_*/{attempt_pending}"
        )
    )
    for pending in pending_records:
        path = pending.with_suffix("").with_suffix(".json")
        if not path.exists():
            identity = validate_identity(read_record(pending)["identity"])
            try:
                binding = _find_bound(
                    experiment,
                    identity,
                    environment,
                    timeout_at=time.monotonic(),
                    deadline=None,
                    attempt_directory=None,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                if isinstance(exc, UnsafeProcessCleanupError):
                    raise
                log.warning(
                    "MLflow run creation for attempt %s of generation %s is still unresolved (%s).",
                    identity["attempt_id"],
                    identity["generation_id"],
                    exc,
                )
                continue
            if binding is None:
                continue
            write_record(path, binding)
        pending.unlink(missing_ok=True)


def _resolve_source(
    experiment: Experiment,
    *,
    generation_id: str,
    environment: Mapping[str, str],
    deadline: float | None,
) -> str:
    """Resolve the source experiment and hold it to the cohort's first resolution.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Current invocation identity.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :raises ValueError: The experiment resolves differently within this cohort.
    :return str: The resolved experiment ID.
    """
    assert experiment.mlflow is not None
    with _PARENT_LOCK:
        resolved = call_tracking(
            experiment, generation_id, {"action": "resolve"}, environment, deadline
        )
        experiment_id = json_text(resolved, "experiment_id")
        source_path = tracking_root(experiment) / "source.json"
        source = {
            "schema_version": 1,
            "source": experiment.mlflow.evidence_identity(),
            "experiment_id": experiment_id,
        }
        if source_path.exists() and read_record(source_path) != source:
            raise ValueError("MLflow experiment resolution changed within this scientific cohort.")
        write_record(source_path, source)
        return experiment_id


def prepare_generation(
    experiment: Experiment,
    *,
    generation_id: str,
    environment: Mapping[str, str],
    deadline: float | None,
) -> RunBinding:
    """Resolve the source experiment and bind this generation's summary run once.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Current invocation identity.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :return RunBinding: The generation summary run's binding.
    """
    assert experiment.mlflow is not None
    with _PARENT_LOCK:
        parent_path = tracking_root(experiment) / generation_id / "generation-binding.json"
        if parent_path.exists():
            return read_binding(parent_path.parent, parent_path.name)
        experiment_id = _resolve_source(
            experiment, generation_id=generation_id, environment=environment, deadline=deadline
        )
        from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint

        parent_identity: RunIdentity = {
            "schema_version": 1,
            "tracking_uri": experiment.mlflow.tracking_uri,
            "experiment": experiment.experiment,
            "generation_id": generation_id,
            "overrides_sha256": _experiment_semantic_fingerprint(experiment),
            "run_mode": "managed",
            "experiment_id": experiment_id,
            "phase": "generation",
            "attempt_id": generation_id,
        }
        return _create_bound(
            experiment,
            parent_identity,
            parent_path,
            environment,
            run_name=f"{experiment.experiment}-generation-{generation_id}",
            deadline=deadline,
        )


def require_tracking_target(
    experiment: Experiment,
    *,
    generation_id: str,
    environment: Mapping[str, str],
    deadline: float | None,
) -> None:
    """Resolve the MLflow target before allocating a phase's trials.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Current invocation identity.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Phase/run deadline; reaching it stays a timeout.
    :raises TrackingTargetUnavailableError: The target cannot be resolved or reached.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError
    from phasesweep.errors import PhaseSweepError

    try:
        if (tracking_root(experiment) / generation_id / "generation-binding.json").exists():
            # An earlier phase bound the summary run; the target must still resolve.
            _resolve_source(
                experiment, generation_id=generation_id, environment=environment, deadline=deadline
            )
        else:
            prepare_generation(
                experiment, generation_id=generation_id, environment=environment, deadline=deadline
            )
    except PhaseSweepError:
        raise
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        cause = str(exc).rstrip(".")
        if isinstance(exc, TimeoutError):
            if deadline is not None and time.monotonic() >= deadline:
                raise
            # The SDK retries a refused connection until the operation times out.
            cause = "it did not answer within timeout_seconds; check that the server is reachable"
        raise TrackingTargetUnavailableError(
            f"MLflow tracking target is not ready: {cause}. No trial was started."
        ) from exc


def prepare_attempt(
    experiment: Experiment,
    *,
    phase: str,
    generation_id: str,
    attempt_id: str,
    run_name: str,
    overrides_sha256: str,
    inherited: Mapping[str, Winner],
    directory: Path,
    environment: Mapping[str, str],
    deadline: float | None,
) -> RunIdentity:
    """Bind the source experiment and optionally create the trainer's native run.

    Persist inherited winner run IDs beside the attempt identity for later projection replay.

    :param Experiment experiment: MLflow-backed experiment.
    :param str phase: Current phase name.
    :param str generation_id: Current invocation identity.
    :param str attempt_id: Allocated trainer identity.
    :param str run_name: The trainer's ``PHASESWEEP_RUN_NAME``, reused for a managed run.
    :param str overrides_sha256: Existing prepared-input identity.
    :param Mapping[str, Winner] inherited: Parent-phase winners fixed into this attempt.
    :param Path directory: Registered attempt directory.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :return RunIdentity: Expected identity, a full binding with a run ID in managed mode.
    """
    assert experiment.mlflow is not None
    parent = prepare_generation(
        experiment, generation_id=generation_id, environment=environment, deadline=deadline
    )
    identity: RunIdentity = {
        "schema_version": 1,
        "tracking_uri": experiment.mlflow.tracking_uri,
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "overrides_sha256": overrides_sha256,
        "experiment_id": parent["experiment_id"],
        "phase": phase,
        "attempt_id": attempt_id,
        "run_mode": experiment.mlflow.run_mode,
        "parent_run_id": parent["run_id"],
    }
    write_record(
        directory / INHERITS_FILE,
        {
            "schema_version": 1,
            "inherits": {
                name: validate_binding(winner.objective_provenance["remote_capture"]["binding"])[
                    "run_id"
                ]
                for name, winner in inherited.items()
                if winner.objective_provenance is not None
            },
        },
    )
    write_record(directory / "mlflow-expected.json", identity)
    if experiment.mlflow.run_mode == "managed":
        return _create_bound(
            experiment,
            identity,
            directory / BINDING_FILE,
            environment,
            run_name=run_name,
            deadline=deadline,
            attempt_directory=directory,
        )
    return identity
