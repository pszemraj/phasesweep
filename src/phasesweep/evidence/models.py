"""Evidence extractor and gate config models."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, get_args
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from phasesweep.config.common import (
    ConfigFloat,
    ConfigInt,
    StrictJsonScalar,
    _Frozen,
    _normalize_http_endpoint,
    _require_finite,
    _validate_optional_bounds,
)


def _validate_trial_path(value: str) -> str:
    """Require a non-empty path inside the trial directory.

    :param str value: Candidate trial-relative path.
    :raises ValueError: If ``value`` is empty, absolute, escapes upward, or
        contains a NUL byte that no filesystem call can accept.
    :return str: Validated trial-relative path.
    """
    path = Path(value)
    if not value or "\0" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"trial-relative path required; got {value!r}.")
    return value


def _validate_trial_file_path(value: str) -> str:
    """Require a descendant-file path relative to the trial directory.

    Unlike an artifact-size directory source, a file source cannot name the root.

    :param str value: Candidate trial-relative file path.
    :raises ValueError: If ``value`` violates the shared path contract or
        resolves to the trial directory itself.
    :return str: Validated descendant file path.
    """
    validated = _validate_trial_path(value)
    if not Path(validated).parts:
        raise ValueError(
            "trial-relative path required; a file path must name a descendant "
            f"of trial_dir, not {value!r}."
        )
    return validated


def _validate_json_key(value: str | None) -> str | None:
    """Require dotted JSON keys with non-empty segments.

    :param str | None value: Candidate dotted JSON key, or ``None``.
    :raises ValueError: If ``value`` has empty key segments.
    :return str | None: Validated JSON key, or ``None``.
    """
    if value is None:
        return None
    if not value or any(not part for part in value.split(".")):
        raise ValueError(f"JSON key must be a non-empty dotted path; got {value!r}.")
    return value


_TrialFilePath: TypeAlias = Annotated[str, AfterValidator(_validate_trial_file_path)]
"""A validated descendant-file path relative to the trial directory."""
_JsonKey: TypeAlias = Annotated[str, AfterValidator(_validate_json_key)]
"""A dotted JSON lookup key with nonempty segments."""


class JsonExtractor(_Frozen):
    """Extract a scalar from a JSON file via a dot-separated key path."""

    type: Literal["json"]
    path: _TrialFilePath = Field(description="Path relative to trial_dir, e.g. 'result.json'.")
    key: _JsonKey = Field(description="Dot-separated key into the JSON, e.g. 'eval.loss'.")


class JsonEnvelopeExtractor(_Frozen):
    """Extract a scalar from a versioned, attempt-bound result envelope."""

    type: Literal["json_envelope"]
    path: _TrialFilePath = Field(default="result.json", description="Path relative to trial_dir.")
    objective_name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    checkpoint: str | None = Field(default=None, min_length=1)
    expected_step: ConfigInt | None = Field(default=None, ge=0)
    evaluation_axis: str = Field(
        default="step",
        min_length=1,
        description=(
            "Training-progress axis the sweep holds fixed, which selection's "
            "evaluation_point check compares ranked objectives on. 'step' is the "
            "envelope's evaluation step; any other name must appear in the "
            "envelope's evaluation.progress, e.g. 'tokens' for a fixed-token sweep."
        ),
    )


class LogRegexExtractor(_Frozen):
    """Extract a scalar from a log file via regex with a named 'value' group."""

    type: Literal["log_regex"]
    file: _TrialFilePath = Field(
        default="stdout.log",
        description=(
            "File relative to trial_dir. 'stdout.log' and 'stderr.log' are written "
            "automatically; supply a custom path if your trainer logs elsewhere."
        ),
    )
    pattern: str = Field(
        description=(
            "Python regex with a named group 'value' that captures the metric. "
            r"Example: r'eval_loss=(?P<value>[0-9.eE+-]+)'."
        )
    )
    select: Literal["last", "first", "min", "max"] = "last"

    @field_validator("pattern")
    @classmethod
    def _valid_pattern(cls, value: str) -> str:
        """Require a compilable regex with the metric's named capture group.

        :param str value: Configured Python regex.
        :raises ValueError: The regex is invalid or lacks a named ``value`` group.
        :return str: Validated pattern, unchanged.
        """
        try:
            pattern = re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid metric regex: {exc}") from exc
        if "value" not in pattern.groupindex:
            raise ValueError("Metric regex requires a named (?P<value>...) group.")
        return value


class _WandbSummarySource(_Frozen):
    """One normalized remote target and a bounded finished-summary query."""

    base_url: str = "https://api.wandb.ai"
    entity: str = Field(min_length=1, pattern=r"^[^/\s]+$")
    project: str = Field(min_length=1, pattern=r"^[^/\s]+$")
    poll_seconds: ConfigFloat = Field(default=2.0, gt=0, allow_inf_nan=False)
    timeout_seconds: ConfigFloat = Field(default=120.0, ge=1, allow_inf_nan=False)

    @field_validator("base_url")
    @classmethod
    def _normalized_endpoint(cls, value: str) -> str:
        """Normalize the endpoint without admitting credentials or query strings.

        :param str value: Configured W&B API base URL.
        :raises ValueError: The URL is not HTTP(S) or carries credentials, a
            query, or a fragment.
        :return str: URL with lowercase scheme and host and no trailing slash.
        """
        parsed = urlsplit(value)
        endpoint = _normalize_http_endpoint(parsed)
        if endpoint is None or parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "W&B base_url requires an HTTP(S) endpoint without credentials, query, or fragment."
            )
        return endpoint


class WandbExtractor(_WandbSummarySource):
    """Select one exact scalar key from the attempt's finished W&B summary."""

    type: Literal["wandb"]
    metric_key: str = Field(min_length=1)
    evaluation_axis: str = Field(
        default="_step",
        min_length=1,
        description=(
            "History key logged alongside metric_key that marks where it was "
            "measured, e.g. 'eval/step' under define_metric('eval/*', "
            "step_metric='eval/step'); selection's evaluation_point check compares "
            "ranked objectives on it. '_step' is W&B's own history step."
        ),
    )

    @model_validator(mode="after")
    def _validate_axis_is_not_the_metric(self) -> WandbExtractor:
        """Reject an evaluation axis naming the metric's own history series.

        :raises ValueError: ``evaluation_axis`` is ``metric_key`` or, for an
            aggregate key such as ``eval/loss.min``, its ``eval/loss`` series.
        :return WandbExtractor: Self, unchanged.
        """
        if self.evaluation_axis in {self.metric_key, self.metric_key.rpartition(".")[0]}:
            raise ValueError(
                f"evaluation_axis {self.evaluation_axis!r} names the metric's own series; "
                "set it to the key that marks where the metric was measured, such as eval/step."
            )
        return self


class MlflowExtractor(_Frozen):
    """Select a native metric-history observation from the attempt's MLflow run."""

    type: Literal["mlflow"]
    metric_key: str = Field(min_length=1)
    select: Literal["last", "min", "max", "step"]
    step: ConfigInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _exact_step(self) -> MlflowExtractor:
        """Require step exclusively for exact-step selection.

        :raises ValueError: Selector and step disagree.
        :return MlflowExtractor: Validated extractor.
        """
        if (self.select == "step") != (self.step is not None):
            raise ValueError("MLflow step is required exactly when select=step.")
        return self


class MlflowRequiredGate(_Frozen):
    """Require finite metrics and named metadata on the bound MLflow run."""

    type: Literal["mlflow_required"]
    metrics: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    params: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    tags: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _requirements(self) -> MlflowRequiredGate:
        """Require at least one presence declaration.

        :raises ValueError: The gate declares no requirements.
        :return MlflowRequiredGate: Validated gate.
        """
        if not (self.metrics or self.params or self.tags):
            raise ValueError("mlflow_required needs metrics, params, or tags.")
        return self


ObjectiveExtractor = (
    JsonExtractor | JsonEnvelopeExtractor | LogRegexExtractor | WandbExtractor | MlflowExtractor
)
LocalExtractor = JsonExtractor | JsonEnvelopeExtractor | LogRegexExtractor
"""The extractors that read a trial's own files. The remote kinds wait for every requested key."""

ExtractorKind = Literal["json", "json_envelope", "log_regex", "wandb", "mlflow"]
"""The ``type`` of every objective extractor, as recorded evidence names it."""

EXTRACTOR_KINDS: frozenset[ExtractorKind] = frozenset(get_args(ExtractorKind))
""":data:`ExtractorKind` as a set, for checking a ``kind`` read back from disk."""

POSITIONED_EXTRACTOR_KINDS: frozenset[ExtractorKind] = frozenset(
    {"json_envelope", "wandb", "mlflow"}
)
"""The kinds whose recorded evidence names where on the evaluation axis the objective was
measured. ``json`` and ``log_regex`` objectives record no position, so selection's
``evaluation_point`` check cannot run on them."""


def objective_evidence_assurance(extractor: ObjectiveExtractor) -> dict[str, str | bool]:
    """Describe the identity checks enforced by an objective extractor.

    Local extractors are location-scoped. Only json_envelope checks the reported
    attempt, generation, and override digest, plus the objective name, split,
    and policy. It requires checkpoint and step metadata, but binds their values
    only when the corresponding optional configuration fields are declared.
    Remote extractors key evidence by target and run ID; W&B does not enforce
    envelope metadata. Other kinds leave envelope assurance flags false.

    :param ObjectiveExtractor extractor: Objective extractor to describe.
    :return dict[str, str | bool]: Extractor kind and assurance flags.
        *_declared marks an optional configured value; *_value_bound marks its
        enforced equality with the envelope.
    """
    if isinstance(extractor, JsonEnvelopeExtractor):
        checkpoint_declared = extractor.checkpoint is not None
        expected_step_declared = extractor.expected_step is not None
        return {
            "kind": extractor.type,
            "attempt_location_scoped": True,
            "attempt_identity_bound": True,
            "source_identity_keyed": False,
            "objective_name_bound": True,
            "split_bound": True,
            "evaluation_policy_bound": True,
            "checkpoint_declared": checkpoint_declared,
            "checkpoint_value_bound": checkpoint_declared,
            "expected_step_declared": expected_step_declared,
            "expected_step_value_bound": expected_step_declared,
        }
    return {
        "kind": extractor.type,
        "attempt_location_scoped": not isinstance(extractor, WandbExtractor | MlflowExtractor),
        "attempt_identity_bound": False,
        "source_identity_keyed": isinstance(extractor, WandbExtractor | MlflowExtractor),
        "objective_name_bound": False,
        "split_bound": False,
        "evaluation_policy_bound": False,
        "checkpoint_declared": False,
        "checkpoint_value_bound": False,
        "expected_step_declared": False,
        "expected_step_value_bound": False,
    }


class _ObjectiveEvidenceFields(BaseModel):
    """Assurance flags for the shared MCP objective-evidence model.

    See :func:`objective_evidence_assurance` for each flag's meaning.
    """

    kind: ExtractorKind
    attempt_location_scoped: bool
    attempt_identity_bound: bool
    source_identity_keyed: bool
    objective_name_bound: bool
    split_bound: bool
    evaluation_policy_bound: bool
    checkpoint_declared: bool
    checkpoint_value_bound: bool
    expected_step_declared: bool
    expected_step_value_bound: bool


class RequiredFileGate(_Frozen):
    """Require a file to exist under the trial directory."""

    type: Literal["required_file"]
    path: _TrialFilePath


class JsonEqualsGate(_Frozen):
    """Require exact JSON scalar type and value equality.

    Booleans, integers, and floats are distinct. Non-scalars and non-finite
    floats are rejected at config load.
    """

    type: Literal["json_equals"]
    path: _TrialFilePath
    key: _JsonKey
    value: StrictJsonScalar

    @field_validator("value", mode="before")
    @classmethod
    def _value_is_json_scalar(cls, value: object) -> object:
        """Reject unsupported scalar values with a diagnostic naming their type.

        :param object value: Raw expected value straight from the config.
        :raises ValueError: If ``value`` is not a JSON scalar, or is a
            non-finite float.
        :return object: The unchanged value, for the strict union to validate.
        """
        if not isinstance(value, (bool, int, float, str, type(None))):
            raise ValueError(
                "json_equals gate value must be a JSON scalar (bool, int, float, str, or "
                f"null); got {type(value).__name__}. Quote YAML dates ('2024-01-01') "
                "to compare against a JSON string."
            )
        if isinstance(value, float):
            _require_finite("json_equals gate value", value)
        return value


class JsonScalarBoundGate(_Frozen):
    """Require a JSON key to be a finite scalar within optional bounds."""

    type: Literal["json_scalar_bound"]
    path: _TrialFilePath
    key: _JsonKey
    min: ConfigFloat | None = None
    max: ConfigFloat | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> JsonScalarBoundGate:
        """Reject empty/non-finite bounds and ``min > max``.

        :return JsonScalarBoundGate: Validated gate config.
        """
        _validate_optional_bounds(
            label="json_scalar_bound gate",
            min_value=self.min,
            max_value=self.max,
        )
        return self


class ArtifactSizeGate(_Frozen):
    """Require artifact bytes to fall inside optional bounds.

    Directory sources recursively count regular files and file-symlink targets,
    but do not traverse directory symlinks.
    """

    type: Literal["artifact_size"]
    source: Literal["file", "directory", "json"]
    path: str
    key: _JsonKey | None = None
    min_bytes: ConfigInt | None = Field(default=None, ge=0)
    max_bytes: ConfigInt | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def _trial_path_is_relative(cls, value: str) -> str:
        """Validate the trial-relative artifact path.

        :param str value: Candidate trial-relative path.
        :raises ValueError: The path is empty, absolute, escapes upward, or contains NUL.
        :return str: Validated trial-relative path.
        """
        return _validate_trial_path(value)

    @model_validator(mode="after")
    def _validate_source_and_bounds(self) -> ArtifactSizeGate:
        """Reject ambiguous source specs and invalid byte bounds.

        :raises ValueError: If source/key pairing or byte bounds are invalid.
        :return ArtifactSizeGate: Validated gate config.
        """
        if self.source != "directory" and not Path(self.path).parts:
            raise ValueError(
                "artifact_size gate path='.' is valid only with source=directory; "
                f"source={self.source!r} requires a descendant file path."
            )
        if self.source == "json" and self.key is None:
            raise ValueError("artifact_size gate with source=json must define key.")
        if self.source != "json" and self.key is not None:
            raise ValueError("artifact_size gate key is only valid with source=json.")
        _validate_optional_bounds(
            label="artifact_size gate",
            min_value=self.min_bytes,
            max_value=self.max_bytes,
        )
        return self


class Sha256Gate(_Frozen):
    """Require a file's SHA-256 digest to match an expected hex string."""

    type: Literal["sha256"]
    path: _TrialFilePath
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        """Require a full 64-character lowercase/uppercase hex digest.

        :param str value: Candidate SHA-256 hex digest.
        :raises ValueError: If ``value`` is not a full hex digest.
        :return str: Lowercase SHA-256 hex digest.
        """
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("sha256 gate requires a full 64-character hex digest.")
        return value.lower()


class WandbSummaryRequiredGate(_WandbSummarySource):
    """Require keys in the same finished capture used by other remote consumers."""

    type: Literal["wandb_summary_required"]
    keys: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


def wandb_gate_identity(gate: WandbSummaryRequiredGate) -> str:
    """Return a canonical identity for one W&B summary-presence gate.

    :param WandbSummaryRequiredGate gate: Gate whose declaration is frozen into a capture.
    :return str: SHA-256 identity stable across enclosing gate-list reordering.
    """
    identity = {
        "base_url": gate.base_url,
        "entity": gate.entity,
        "project": gate.project,
        "keys": sorted(set(gate.keys)),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class WandbQuery:
    """An attempt's shared query compiled from existing evidence declarations."""

    source: _WandbSummarySource
    numeric_keys: tuple[str, ...]
    presence_keys: tuple[str, ...]
    constraint_keys: tuple[tuple[str, str], ...] = ()
    gate_keys: tuple[tuple[str, tuple[str, ...]], ...] = ()


def compose_wandb_environment(
    query: WandbQuery,
    configured: Mapping[str, str],
    environment: Mapping[str, str],
    *,
    require_online: bool = True,
) -> dict[str, str]:
    """Bind remote evidence before environment identity, preserving other settings.

    :param WandbQuery query: Phase's agreed target and evidence requirements.
    :param Mapping[str, str] configured: Explicit experiment environment entries.
    :param Mapping[str, str] environment: Already composed trainer environment.
    :param bool require_online: Whether disabled/offline ambient logging is invalid for this use.
    :return dict[str, str]: Bound base environment, before the generated attempt ID.
    :raises ValueError: Explicit managed settings conflict, or remote logging is disabled.
    """
    source = query.source
    bindings = {
        "WANDB_BASE_URL": source.base_url,
        "WANDB_ENTITY": source.entity,
        "WANDB_PROJECT": source.project,
        "WANDB_RESUME": "never",
    }
    for key, expected in bindings.items():
        if key in configured:
            actual = configured[key]
            if key == "WANDB_BASE_URL":
                actual = _WandbSummarySource._normalized_endpoint(actual)
            if actual != expected:
                raise ValueError(
                    f"env.{key} conflicts with managed W&B evidence; remove it or match the evidence target."
                )
    if "WANDB_RUN_ID" in configured:
        raise ValueError("env.WANDB_RUN_ID conflicts with the generated W&B attempt ID; remove it.")
    if require_online and (
        environment.get("WANDB_MODE", "online").lower() in {"offline", "dryrun", "disabled"}
        or environment.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes", "on"}
    ):
        raise ValueError(
            "W&B evidence requires online logging; remove offline/disabled W&B settings."
        )
    result = dict(environment)
    result.update(bindings)
    result.pop("WANDB_RUN_ID", None)
    return result


Gate = Annotated[
    RequiredFileGate
    | JsonEqualsGate
    | JsonScalarBoundGate
    | ArtifactSizeGate
    | Sha256Gate
    | WandbSummaryRequiredGate
    | MlflowRequiredGate,
    Field(discriminator="type"),
]
