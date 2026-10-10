"""Durable MLflow run bindings and engine-facing supervised operations."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from phasesweep.config.common import is_sha256_hex
from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.json import strict_json_loads
from phasesweep.tracking.mlflow import identity_tags, request_operation, verify_remote

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment

BINDING_FILE = "mlflow-binding.json"
_PARENT_LOCK = threading.RLock()


def tracking_root(experiment: Experiment) -> Path:
    """Return mutable tracking state outside scientific generation manifests.

    :param Experiment experiment: Local experiment namespace.
    :return Path: Tracking state root.
    """
    from phasesweep.engine.paths import _experiment_dir

    return _experiment_dir(experiment) / "tracking"


def write_record(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically persist private tracking state using existing file primitives.

    :param Path path: Record destination.
    :param Mapping[str, Any] record: JSON-compatible payload.
    """
    private_atomic_write_text(
        path,
        json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n",
        require_private_dir=False,
    )


def read_record(path: Path) -> dict[str, Any]:
    """Read a current-format tracking record.

    :param Path path: Record to read.
    :raises ValueError: The record does not use the current format.
    :return dict[str, Any]: Decoded current-format record.
    """
    data = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported MLflow tracking record format.")
    return data


def validate_binding(binding: Any) -> None:
    """Require the current binding's source and immutable attempt identities.

    :param Any binding: Decoded local run binding.
    :raises ValueError: A required identity is absent or invalid.
    """
    if not isinstance(binding, dict) or binding.get("schema_version") != 1:
        raise ValueError("Invalid MLflow run binding format.")
    fields = (
        "tracking_uri",
        "experiment_id",
        "run_id",
        "experiment",
        "phase",
        "generation_id",
        "attempt_id",
    )
    if (
        any(not isinstance(binding.get(k), str) or not binding[k] for k in fields)
        or binding.get("run_mode") not in {"managed", "trainer"}
        or not is_sha256_hex(binding.get("overrides_sha256"))
    ):
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")


def read_binding(directory: Path) -> dict[str, Any]:
    """Load the authoritative local run binding.

    :param Path directory: Attempt directory.
    :return dict[str, Any]: Validated binding.
    """
    binding = read_record(directory / BINDING_FILE)
    validate_binding(binding)
    return binding


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
    """Register a controller worker that owns no Optuna trial.

    :param Experiment experiment: Registry owner.
    :param str generation_id: Producing generation.
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
    try:
        yield operation_id, directory
    finally:
        lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=operation_id)
        if lifecycle is not None and (
            lifecycle.state == "allocated"
            or (lifecycle.state == "exited" and lifecycle.cleanup_confirmed)
        ):
            _retire_active_attempt(experiment, operation_id)


def call_tracking(
    experiment: Experiment,
    generation_id: str,
    request: dict[str, Any],
    environment: Mapping[str, str],
    deadline: float | None = None,
) -> dict[str, Any]:
    """Perform a registered tracking-only operation.

    :param Experiment experiment: Experiment with configured tracking.
    :param str generation_id: Operation's generation.
    :param dict[str, Any] request: Trusted SDK action.
    :param Mapping[str, str] environment: Composed credentials/transport.
    :param float | None deadline: Optional absolute run deadline.
    :return dict[str, Any]: Successful worker response.
    """
    assert experiment.mlflow is not None
    with tracking_operation(experiment, generation_id) as (operation_id, directory):
        return request_operation(
            {"settings": experiment.mlflow.model_dump(mode="json"), **request},
            operation_id=operation_id,
            directory=directory,
            environment=environment,
            deadline=operation_deadline(experiment, deadline),
        )


def _create_bound(
    experiment: Experiment,
    identity: dict[str, Any],
    path: Path,
    environment: Mapping[str, str],
    *,
    deadline: float | None,
    attempt_directory: Path | None = None,
) -> dict[str, Any]:
    """Bind or create the identity's run, retrying once after an uncertain failure.

    Each try searches the identity tags before creating, so a retry after a
    lost response binds the run that request created instead of duplicating it.

    :param Experiment experiment: Configured experiment.
    :param dict[str, Any] identity: Exact creation identity.
    :param Path path: Durable binding destination.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param float | None deadline: Existing execution deadline.
    :param Path | None attempt_directory: Existing process slot for managed attempt creation.
    :return dict[str, Any]: Bound native run identity.
    """
    assert experiment.mlflow is not None
    if path.exists():
        return read_binding(path.parent) if path.name == BINDING_FILE else read_record(path)
    request = {
        "action": "create",
        "binding": identity,
        "run_name": f"{identity['experiment']}-{identity['phase']}-{identity['attempt_id']}",
    }

    def execute() -> dict[str, Any]:
        """Run creation or reconciliation in the registered process slot.

        :return dict[str, Any]: Verified remote binding.
        """
        assert experiment.mlflow is not None
        if attempt_directory is None:
            return call_tracking(
                experiment, identity["generation_id"], request, environment, deadline
            )
        return request_operation(
            {"settings": experiment.mlflow.model_dump(mode="json"), **request},
            operation_id=identity["attempt_id"],
            directory=attempt_directory,
            environment=environment,
            deadline=operation_deadline(experiment, deadline),
        )

    try:
        binding = execute()
    except (OSError, RuntimeError) as exc:
        from phasesweep.errors import UnsafeProcessCleanupError

        if isinstance(exc, UnsafeProcessCleanupError) or (
            deadline is not None and time.monotonic() >= deadline
        ):
            raise
        binding = execute()
    write_record(path, binding)
    return binding


def prepare_attempt(
    experiment: Experiment,
    *,
    phase: str,
    generation_id: str,
    attempt_id: str,
    overrides_sha256: str,
    directory: Path,
    environment: Mapping[str, str],
    deadline: float | None,
) -> dict[str, Any]:
    """Bind the source experiment and optionally create the trainer's run.

    :param Experiment experiment: MLflow-backed experiment.
    :param str phase: Current phase name.
    :param str generation_id: Current invocation identity.
    :param str attempt_id: Allocated trainer identity.
    :param str overrides_sha256: Existing prepared-input identity.
    :param Path directory: Registered attempt directory.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :return dict[str, Any]: Expected binding, with a run ID in managed mode.
    """
    assert experiment.mlflow is not None
    base = {
        "schema_version": 1,
        "tracking_uri": experiment.mlflow.tracking_uri,
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "overrides_sha256": overrides_sha256,
        "run_mode": "managed",
    }
    with _PARENT_LOCK:
        parent_path = tracking_root(experiment) / generation_id / "generation-binding.json"
        if parent_path.exists():
            parent = read_record(parent_path)
        else:
            resolved = call_tracking(
                experiment, generation_id, {"action": "resolve"}, environment, deadline
            )
            source_path = tracking_root(experiment) / "source.json"
            source = {
                "schema_version": 1,
                "source": experiment.mlflow.evidence_identity(),
                **resolved,
            }
            if source_path.exists() and read_record(source_path) != source:
                raise ValueError(
                    "MLflow experiment resolution changed within this scientific cohort."
                )
            write_record(source_path, source)
            from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint

            parent_identity = {
                **base,
                **resolved,
                "phase": "generation",
                "attempt_id": generation_id,
                "overrides_sha256": _experiment_semantic_fingerprint(experiment),
            }
            parent = _create_bound(
                experiment, parent_identity, parent_path, environment, deadline=deadline
            )
    identity = {
        **base,
        "experiment_id": parent["experiment_id"],
        "phase": phase,
        "attempt_id": attempt_id,
        "run_mode": experiment.mlflow.run_mode,
        "parent_run_id": parent["run_id"],
    }
    write_record(directory / "mlflow-expected.json", identity)
    if experiment.mlflow.run_mode == "managed":
        return _create_bound(
            experiment,
            identity,
            directory / BINDING_FILE,
            environment,
            deadline=deadline,
            attempt_directory=directory,
        )
    return identity


def bind_trainer_run(run_id: str) -> None:
    """Attach native trainer logging to its prepared PhaseSweep identity.

    :param str run_id: Trainer-created remote run identity.
    :raises ValueError: Run ownership, experiment, or identity conflicts.
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
    binding = {**expected, "run_id": run_id}
    validate_binding(binding)
    if (directory / BINDING_FILE).exists() and read_binding(directory) != binding:
        raise ValueError("This attempt is already bound to another MLflow run.")
    client = MlflowClient(tracking_uri=expected["tracking_uri"])
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
