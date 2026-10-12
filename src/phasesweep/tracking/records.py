"""Import-safe record shapes for native MLflow tracking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NotRequired, TypeAlias, TypedDict

from phasesweep.config.common import is_sha256_hex
from phasesweep.runtime.json import JsonObject

RunMode: TypeAlias = Literal["managed", "trainer"]
"""Whether PhaseSweep creates the attempt's MLflow run or the trainer does."""


class SourceIdentity(TypedDict):
    """Fields that determine the scientific source, excluding polling budgets."""

    tracking_uri: str
    experiment_id: str | None
    experiment_name: str | None
    run_mode: RunMode


class MlflowTarget(SourceIdentity):
    """The configured target as a supervised worker receives it."""

    poll_seconds: float
    timeout_seconds: float


class RunIdentity(TypedDict):
    """The exact attempt a native run must carry, known before the run exists."""

    schema_version: Literal[1]
    tracking_uri: str
    experiment_id: str
    experiment: str
    phase: str
    generation_id: str
    attempt_id: str
    overrides_sha256: str
    run_mode: RunMode
    parent_run_id: NotRequired[str]


class RunBinding(RunIdentity):
    """A run identity bound to its native MLflow run."""

    run_id: str


class Observation(TypedDict):
    """One selected metric point and its native position."""

    value: float
    step: int
    timestamp: int


class MlflowQuery(TypedDict):
    """Shared native selections and gates of one phase, as extractor and gate dumps."""

    revision: int
    objective: JsonObject
    constraints: dict[str, JsonObject]
    gates: list[JsonObject]
    selections: dict[str, JsonObject]


class MlflowCapture(TypedDict):
    """Frozen native evidence captured from one finished run."""

    kind: Literal["mlflow"]
    binding: RunBinding
    run_state: Literal["FINISHED"]
    query: MlflowQuery
    observations: dict[str, Observation]
    retrieved_at: str
    params: dict[str, str]
    tags: dict[str, str]


class Projection(TypedDict):
    """A retryable result mirror with a stable metric timestamp."""

    timestamp: int
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]
    summary: JsonObject


class DeliveryRecord(TypedDict):
    """A projection retained until its upload is acknowledged."""

    schema_version: Literal[1]
    generation_id: str
    source: SourceIdentity
    binding: RunBinding
    projection: Projection
    finish: str | None
    state: Literal["pending", "delivered"]


class ResolveAction(TypedDict):
    """Controller request to resolve the configured experiment."""

    action: Literal["resolve"]


class CreateAction(TypedDict):
    """Controller request to bind or create the run carrying an identity."""

    action: Literal["create"]
    binding: RunIdentity
    run_name: str


class PublishAction(TypedDict):
    """Controller request to mirror a retained delivery record, as persisted."""

    action: Literal["publish"]
    name: str
    binding: JsonObject
    projection: JsonObject
    finish: str | None


TrackingAction: TypeAlias = ResolveAction | CreateAction | PublishAction
"""A controller request before the configured target is attached."""


class ResolveRequest(ResolveAction):
    """A resolve action with its target."""

    settings: MlflowTarget


class CreateRequest(CreateAction):
    """A create action with its target."""

    settings: MlflowTarget


class CaptureRequest(TypedDict):
    """Worker request to fetch finished evidence for a bound run."""

    action: Literal["capture"]
    settings: MlflowTarget
    binding: RunBinding
    query: MlflowQuery
    timeout_at: float


class PublishRequest(TypedDict):
    """Worker request to mirror a validated projection."""

    action: Literal["publish"]
    settings: MlflowTarget
    name: str
    binding: RunBinding
    projection: Projection
    finish: str | None


def _identity_text(record: Mapping[str, object], key: str) -> str:
    """Read one required, non-empty identity string.

    :param Mapping[str, object] record: Decoded identity or binding.
    :param str key: Identity field.
    :raises ValueError: The field is absent, not a string, or empty.
    :return str: The field's value.
    """
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")
    return value


def _parse_identity(value: object) -> tuple[RunIdentity, Mapping[str, object]]:
    """Require the current format's source and immutable attempt identities.

    :param object value: Decoded local identity or binding.
    :raises ValueError: A required identity is absent or invalid.
    :return tuple[RunIdentity, Mapping[str, object]]: The narrowed identity and the record it
        was read from.
    """
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Invalid MLflow run binding format.")
    record: Mapping[str, object] = value
    run_mode: RunMode
    if record.get("run_mode") == "managed":
        run_mode = "managed"
    elif record.get("run_mode") == "trainer":
        run_mode = "trainer"
    else:
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")
    overrides_sha256 = record.get("overrides_sha256")
    if not is_sha256_hex(overrides_sha256):
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")
    identity: RunIdentity = {
        "schema_version": 1,
        "tracking_uri": _identity_text(record, "tracking_uri"),
        "experiment_id": _identity_text(record, "experiment_id"),
        "experiment": _identity_text(record, "experiment"),
        "phase": _identity_text(record, "phase"),
        "generation_id": _identity_text(record, "generation_id"),
        "attempt_id": _identity_text(record, "attempt_id"),
        "overrides_sha256": overrides_sha256,
        "run_mode": run_mode,
    }
    parent_run_id = record.get("parent_run_id")
    if isinstance(parent_run_id, str):
        identity["parent_run_id"] = parent_run_id
    elif parent_run_id is not None:
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")
    return identity, record


def validate_identity(identity: object) -> RunIdentity:
    """Require an attempt identity that names no native run yet.

    :param object identity: Decoded identity a run will be bound to.
    :raises ValueError: A required identity is absent or invalid.
    :return RunIdentity: The narrowed identity.
    """
    return _parse_identity(identity)[0]


def validate_binding(binding: object) -> RunBinding:
    """Require the current binding's source, native run, and immutable attempt identities.

    :param object binding: Decoded local run binding.
    :raises ValueError: A required identity is absent or invalid.
    :return RunBinding: The narrowed binding.
    """
    identity, record = _parse_identity(binding)
    return {**identity, "run_id": _identity_text(record, "run_id")}
