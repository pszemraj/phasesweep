"""Evidence extractor and gate config models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from phasesweep.config.common import _Frozen, _validate_optional_bounds


def _validate_trial_path(value: str) -> str:
    """Require a non-empty path inside the trial directory.

    :param str value: Candidate trial-relative path.
    :raises ValueError: If ``value`` is empty, absolute, or escapes upward.
    :return str: Validated trial-relative path.
    """
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"trial-relative path required; got {value!r}.")
    return value


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


class _TrialPathModel(_Frozen):
    """Mixin for config models containing trial-relative path fields."""

    @field_validator("path", "file", check_fields=False)
    @classmethod
    def _trial_path_is_relative(cls, value: str) -> str:
        """Validate trial-relative path fields.

        :param str value: Candidate trial-relative path.
        :return str: Validated trial-relative path.
        """
        return _validate_trial_path(value)


class _JsonKeyModel(_Frozen):
    """Mixin for config models containing dotted JSON key fields."""

    @field_validator("key", check_fields=False)
    @classmethod
    def _json_key_is_valid(cls, value: str | None) -> str | None:
        """Validate dotted JSON key fields.

        :param str | None value: Candidate dotted JSON key, or ``None``.
        :return str | None: Validated JSON key, or ``None``.
        """
        return _validate_json_key(value)


class JsonExtractor(_TrialPathModel, _JsonKeyModel):
    """Extract a scalar from a JSON file via a dot-separated key path."""

    type: Literal["json"]
    path: str = Field(description="Path relative to trial_dir, e.g. 'result.json'.")
    key: str = Field(description="Dot-separated key into the JSON, e.g. 'eval.loss'.")


class JsonEnvelopeExtractor(_TrialPathModel):
    """Extract a scalar from a versioned, attempt-bound result envelope."""

    type: Literal["json_envelope"]
    path: str = Field(default="result.json", description="Path relative to trial_dir.")
    objective_name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    checkpoint: str | None = Field(default=None, min_length=1)
    expected_step: int | None = Field(default=None, ge=0)


class LogRegexExtractor(_TrialPathModel):
    """Extract a scalar from a log file via regex with a named 'value' group."""

    type: Literal["log_regex"]
    file: str = Field(
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


class WandbExtractor(_Frozen):
    """Extract a scalar from this attempt's finished W&B run summary."""

    type: Literal["wandb"]
    entity: str
    project: str
    metric_key: str = Field(description="Key on wandb.run.summary, e.g. 'eval/loss'.")
    poll_seconds: float = Field(default=2.0, gt=0.0)
    timeout_seconds: float = Field(default=120.0, ge=1.0)


ObjectiveExtractor = JsonEnvelopeExtractor | LogRegexExtractor | WandbExtractor
Extractor = JsonExtractor | ObjectiveExtractor


def objective_evidence_assurance(extractor: ObjectiveExtractor) -> dict[str, str | bool]:
    """Describe which objective-evidence identities the extractor genuinely enforces.

    Grounded in what ``phasesweep.evidence.evaluation._extract_json_envelope``
    actually checks, not in the extractor's coarse type alone:

    - ``objective_name``, ``split``, ``policy`` are required
      :class:`JsonEnvelopeExtractor` fields (``min_length=1``) and are
      unconditionally checked against the envelope's own reported
      ``objective.name`` / ``objective.split`` / ``evaluation.policy``. They
      are bound whenever the extractor kind is ``json_envelope`` at all.
    - ``checkpoint`` and ``expected_step`` are *optional* extractor fields.
      The envelope must always structurally report a non-empty checkpoint and
      a non-negative step (or extraction fails), but that value is compared
      against the configured value only when the config declares one
      (``cfg.checkpoint`` / ``cfg.expected_step`` is not ``None``). Reporting
      a coarse ``True`` regardless of whether either was declared overstates
      what is actually enforced when they are left unset.
    - ``log_regex`` and ``wandb`` extractors have no objective_name/split/
      policy/checkpoint/expected_step concept at all, so every one of those
      flags is ``False`` for them.
    - A single coarse ``attempt_bound`` claim overstated weak extractors, so
      it is split into three precise flags (review v0.5.15 / item C):

      - ``attempt_location_scoped`` is ``True`` for every extractor kind:
        each reads evidence from a location — a trial directory for
        ``json_envelope``/``log_regex``, or a W&B run id for ``wandb`` —
        that is uniquely scoped to this generation+attempt. Scoping alone is
        weak: nothing in a ``log_regex`` file's *contents* identifies the
        attempt that produced it, so a file misplaced or symlinked into the
        wrong trial directory would be read as gospel.
      - ``attempt_identity_bound`` is ``True`` only for ``json_envelope``:
        the envelope structurally echoes ``generation_id``/``attempt_id``/
        ``overrides_sha256`` in its own body, and
        ``_extract_json_envelope`` cross-checks those reported values
        against the runtime's own identity before accepting the result.
        ``log_regex`` has no identity fields to check at all, and ``wandb``
        is keyed by run id rather than by any self-reported identity inside
        the run summary, so both are ``False``.
      - ``source_identity_keyed`` is ``True`` only for ``wandb``: the
        evidence source itself — the W&B run — is addressed by the
        immutable attempt identity (``WANDB_RUN_ID=attempt_id``) rather than
        by filesystem location, so a wrong-attempt run cannot silently
        appear at the right path the way a misplaced log file could.
        ``json_envelope`` and ``log_regex`` read location-addressed files,
        so this is ``False`` for both; the envelope's stronger guarantee is
        already captured by ``attempt_identity_bound``.

    :param ObjectiveExtractor extractor: Configured objective extractor to describe.
    :return dict[str, str | bool]: Assurance payload with the extractor ``kind``
        plus per-field boolean flags describing exactly what the runtime
        enforces. ``checkpoint_declared``/``expected_step_declared`` report
        whether the config pinned a value; ``checkpoint_value_bound``/
        ``expected_step_value_bound`` report whether the runtime actually
        validates the envelope against that declared value (``True`` only
        when the corresponding ``*_declared`` flag is also ``True``).
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
        "attempt_location_scoped": True,
        "attempt_identity_bound": False,
        "source_identity_keyed": isinstance(extractor, WandbExtractor),
        "objective_name_bound": False,
        "split_bound": False,
        "evaluation_policy_bound": False,
        "checkpoint_declared": False,
        "checkpoint_value_bound": False,
        "expected_step_declared": False,
        "expected_step_value_bound": False,
    }


class _ObjectiveEvidenceFields(BaseModel):
    """Assurance-flag field set shared by MCP result payloads and persisted snapshots.

    ``phasesweep.mcp.server.ObjectiveEvidencePayload`` and
    ``phasesweep.mcp.snapshots.ObjectiveEvidenceSnapshot`` each subclass this
    alongside their own strict base (``extra="forbid"``, and for snapshots also
    ``allow_inf_nan=False``, inert here since every field below is a bool or
    ``Literal``) so the identical field set is declared exactly once while each
    site keeps its own class name and JSON schema entry. See
    :func:`objective_evidence_assurance` for exactly what each flag means.
    """

    kind: Literal["json_envelope", "log_regex", "wandb"]
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


class RequiredFileGate(_TrialPathModel):
    """Require a file to exist under the trial directory."""

    type: Literal["required_file"]
    path: str


class JsonEqualsGate(_TrialPathModel, _JsonKeyModel):
    """Require a JSON key to equal an expected scalar value."""

    type: Literal["json_equals"]
    path: str
    key: str
    value: Any


class JsonScalarBoundGate(_TrialPathModel, _JsonKeyModel):
    """Require a JSON key to be a finite scalar within optional bounds."""

    type: Literal["json_scalar_bound"]
    path: str
    key: str
    min: float | None = None
    max: float | None = None

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


class ArtifactSizeGate(_TrialPathModel, _JsonKeyModel):
    """Require artifact bytes to fall inside optional bounds."""

    type: Literal["artifact_size"]
    source: Literal["file", "directory", "json"]
    path: str
    key: str | None = None
    min_bytes: int | None = Field(default=None, ge=0)
    max_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_source_and_bounds(self) -> ArtifactSizeGate:
        """Reject ambiguous source specs and invalid byte bounds.

        :raises ValueError: If source/key pairing or byte bounds are invalid.
        :return ArtifactSizeGate: Validated gate config.
        """
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


class Sha256Gate(_TrialPathModel):
    """Require a file's SHA-256 digest to match an expected hex string."""

    type: Literal["sha256"]
    path: str
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


class WandbSummaryRequiredGate(_Frozen):
    """Require keys in this attempt's finished W&B run summary."""

    type: Literal["wandb_summary_required"]
    entity: str
    project: str
    keys: list[str] = Field(min_length=1)
    poll_seconds: float = Field(default=2.0, gt=0.0)
    timeout_seconds: float = Field(default=120.0, ge=1.0)


Gate = Annotated[
    RequiredFileGate
    | JsonEqualsGate
    | JsonScalarBoundGate
    | ArtifactSizeGate
    | Sha256Gate
    | WandbSummaryRequiredGate,
    Field(discriminator="type"),
]
