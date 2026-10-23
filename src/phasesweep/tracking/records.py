"""Import-safe record shapes for native MLflow tracking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, NotRequired, TypeAlias

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from phasesweep.config.common import is_sha256_hex
from phasesweep.runtime.json import JsonObject

RunMode: TypeAlias = Literal["managed", "trainer"]
"""Whether PhaseSweep creates the attempt's MLflow run or the trainer does."""
IdentityText: TypeAlias = Annotated[str, Field(min_length=1)]


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
    tracking_uri: IdentityText
    experiment_id: IdentityText
    experiment: IdentityText
    phase: IdentityText
    generation_id: IdentityText
    attempt_id: IdentityText
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


class FindAction(TypedDict):
    """Controller request to look up, and never create, the run carrying an identity."""

    action: Literal["find"]
    binding: RunIdentity
    timeout_at: float


class PublishAction(TypedDict):
    """Controller request to mirror a retained delivery record, as persisted."""

    action: Literal["publish"]
    name: str
    binding: JsonObject
    projection: JsonObject
    finish: str | None


TrackingAction: TypeAlias = ResolveAction | CreateAction | FindAction | PublishAction
"""A controller request before the configured target is attached."""


class CreateRequest(CreateAction):
    """A create action with its target."""

    settings: MlflowTarget


class FindRequest(FindAction):
    """A find action with its target."""

    settings: MlflowTarget


class FindResult(TypedDict):
    """The outcome of a find: the verified binding when exactly one run carries the identity."""

    found: bool
    binding: NotRequired[RunBinding]


class PendingCreation(TypedDict):
    """A creation request that may have been sent, kept until its run is bound."""

    schema_version: Literal[1]
    identity: RunIdentity
    run_name: str


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


_IDENTITY_ADAPTER: TypeAdapter[RunIdentity] = TypeAdapter(RunIdentity)


def _parse_identity(value: object) -> tuple[RunIdentity, Mapping[str, object]]:
    """Require the current format's source and immutable attempt identities.

    :param object value: Decoded local identity or binding.
    :raises ValueError: A required identity is absent or invalid.
    :return tuple[RunIdentity, Mapping[str, object]]: The narrowed identity and the record it
        was read from.
    """
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Invalid MLflow run binding format.")
    if not is_sha256_hex(value.get("overrides_sha256")):
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")
    try:
        identity = _IDENTITY_ADAPTER.validate_python(
            {
                key: item
                for key, item in value.items()
                if key != "parent_run_id" or item is not None
            },
            strict=True,
        )
    except ValidationError:
        raise ValueError("MLflow binding lacks a valid source/attempt identity.") from None
    return identity, value


def validate_identity(identity: object) -> RunIdentity:
    """Narrow a decoded attempt identity; extra keys such as ``run_id`` are dropped, not refused.

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
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("MLflow binding lacks a valid source/attempt identity.")
    return {**identity, "run_id": run_id}
