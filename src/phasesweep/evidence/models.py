"""Evidence extractor and gate config models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from phasesweep.config.common import _Frozen, _require_finite, _validate_optional_bounds

StrictJsonScalar = StrictBool | StrictInt | StrictFloat | StrictStr | None
"""The exact set of values a JSON document can hold at a leaf position.

Strict (non-coercing) members on purpose: ``StrictInt`` rejects ``bool`` and
``StrictStr`` rejects a YAML ``date``, so nothing is silently widened into a
neighbouring type. Anything outside this union — a mapping, a sequence, a
``datetime.date`` — is not representable in parsed JSON and is rejected at
config load rather than normalized later.
"""


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
    """Require a trial-relative path that names a descendant file location.

    ``.`` legitimately names the whole trial directory for an
    ``artifact_size`` directory gate, but every extractor and file-backed gate
    requires a file below that root. Rejecting the root at config load avoids
    deferring an impossible file open to the first trial.

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


class _TrialFilePathModel(_Frozen):
    """Mixin for config models containing trial-relative file path fields."""

    @field_validator("path", "file", check_fields=False)
    @classmethod
    def _trial_file_path_is_relative(cls, value: str) -> str:
        """Validate trial-relative file path fields.

        :param str value: Candidate trial-relative file path.
        :raises ValueError: If the path is unsafe or names the trial root.
        :return str: Validated descendant file path.
        """
        return _validate_trial_file_path(value)


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


class JsonExtractor(_TrialFilePathModel, _JsonKeyModel):
    """Extract a scalar from a JSON file via a dot-separated key path."""

    type: Literal["json"]
    path: str = Field(description="Path relative to trial_dir, e.g. 'result.json'.")
    key: str = Field(description="Dot-separated key into the JSON, e.g. 'eval.loss'.")


class JsonEnvelopeExtractor(_TrialFilePathModel):
    """Extract a scalar from a versioned, attempt-bound result envelope."""

    type: Literal["json_envelope"]
    path: str = Field(default="result.json", description="Path relative to trial_dir.")
    objective_name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    checkpoint: str | None = Field(default=None, min_length=1)
    expected_step: int | None = Field(default=None, ge=0)


class LogRegexExtractor(_TrialFilePathModel):
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
    """Shared location and polling contract for one W&B run summary."""

    base_url: str = Field(default="https://api.wandb.ai", min_length=1)
    entity: str = Field(min_length=1, pattern=r"^[^/]+$")
    project: str = Field(min_length=1, pattern=r"^[^/]+$")
    poll_seconds: float = Field(default=2.0, gt=0.0, allow_inf_nan=False)
    timeout_seconds: float = Field(default=120.0, ge=1.0, allow_inf_nan=False)

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        """Normalize the endpoint spelling used by the W&B public API.

        :param str value: Configured W&B API base URL.
        :raises ValueError: The value consists only of slashes.
        :return str: Base URL without trailing slashes.
        """
        normalized = value.rstrip("/")
        if not normalized:
            raise ValueError("W&B base_url must contain a non-slash endpoint.")
        return normalized


class WandbExtractor(_WandbSummarySource):
    """Extract a scalar from this attempt's finished W&B run summary."""

    type: Literal["wandb"]
    metric_key: str = Field(description="Key on wandb.run.summary, e.g. 'eval/loss'.")


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


class RequiredFileGate(_TrialFilePathModel):
    """Require a file to exist under the trial directory."""

    type: Literal["required_file"]
    path: str


class JsonEqualsGate(_TrialFilePathModel, _JsonKeyModel):
    """Require a JSON key to equal an expected JSON scalar.

    ``value`` must be a JSON scalar (``bool``, ``int``, ``float``, ``str``, or
    ``null``) and is validated strictly, because *type identity is part of this
    gate's semantics*: :func:`phasesweep.evidence.evaluation._json_equals`
    compares with ``type(actual) is type(gate.value)``, so ``true``, ``1``, and
    ``1.0`` are three different gates. Coercion would silently rewrite one into
    another, so no member of the union coerces.

    Non-scalars (mappings, sequences, YAML dates) are rejected outright rather
    than merely never matching (PR #5 review / reviewer 2, blocker 4). The
    study fingerprint hashes ``model_dump(mode="json")`` through
    ``json.dumps(..., default=str)``, which normalizes exactly the values that
    strict JSON cannot hold: ``{1: "x"}`` and ``{"1": "x"}`` collapse to one
    digest, as do ``date(2024, 1, 1)`` and ``"2024-01-01"``. Since parsed JSON
    never yields an int-keyed mapping or a ``date``, those variants could never
    pass while their surviving twins could — two configs that judge feasibility
    in mutually exclusive ways would share one phase fingerprint and be allowed
    to reuse each other's study. Rejecting them at load keeps the fingerprint
    faithful to runtime behaviour. Non-finite floats (``.nan``, ``.inf``) are
    rejected for the same reason: JSON has no encoding for them.
    """

    type: Literal["json_equals"]
    path: str
    key: str
    value: StrictJsonScalar

    @field_validator("value", mode="before")
    @classmethod
    def _value_is_json_scalar(cls, value: object) -> object:
        """Reject values strict JSON cannot hold, naming the offending type.

        Runs before the ``StrictJsonScalar`` union purely for the message: the
        union alone reports one "input should be a valid <member>" error per
        member, which never says *why* a mapping or a YAML date is wrong. The
        union remains the authority on type identity (it is what refuses to
        coerce ``1`` into ``1.0``); this validator only front-runs the cases a
        config author actually hits.

        :param object value: Raw expected value straight from the config.
        :raises ValueError: If ``value`` is not a JSON scalar, or is a
            non-finite float.
        :return object: The unchanged value, for the strict union to validate.
        """
        if not isinstance(value, (bool, int, float, str, type(None))):
            raise ValueError(
                "json_equals gate value must be a JSON scalar (bool, int, float, str, or "
                f"null); got {type(value).__name__}. Parsed JSON never produces that type, so "
                "such a gate could never pass, and the study fingerprint would render it "
                "identically to a value that can — silently sharing one study between two "
                "configs that disagree on feasibility. YAML dates are the common case: quote "
                "them ('2024-01-01') to compare against the JSON string."
            )
        if isinstance(value, float):
            _require_finite("json_equals gate value", value)
        return value


class JsonScalarBoundGate(_TrialFilePathModel, _JsonKeyModel):
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


class Sha256Gate(_TrialFilePathModel):
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


class WandbSummaryRequiredGate(_WandbSummarySource):
    """Require keys in this attempt's finished W&B run summary."""

    type: Literal["wandb_summary_required"]
    keys: list[str] = Field(min_length=1)


Gate = Annotated[
    RequiredFileGate
    | JsonEqualsGate
    | JsonScalarBoundGate
    | ArtifactSizeGate
    | Sha256Gate
    | WandbSummaryRequiredGate,
    Field(discriminator="type"),
]
