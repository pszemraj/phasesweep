"""YAML config schema for phasesweep experiments and suites.

The full schema is validated up-front before any phase runs, so a typo in the
last phase fails immediately rather than three hours into the sweep.
"""

from __future__ import annotations

import contextlib
import copy
import math
import re
import string
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phasesweep.storage_urls import storage_backend


class _Frozen(BaseModel):
    """Base for all config models: frozen + reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_finite(label: str, value: float) -> None:
    """Raise ``ValueError`` if ``value`` is NaN or +/-inf.

    Used at config-load to keep silently-broken bounds out of the runtime
    (review v0.5.2 / blocker 3): a constraint with ``max: .nan`` would otherwise
    be vacuous because ``x > nan`` is always ``False``.

    Args:
        label: Human-readable name of the field (e.g. ``"constraint.max"``);
            included verbatim in the error message.
        value: The numeric value to validate.

    Raises:
        ValueError: If ``value`` is not finite (``math.isfinite`` returns False).

    """
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite; got {value!r}")


# --------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------


class JsonExtractor(_Frozen):
    """Extract a scalar from a JSON file via a dot-separated key path."""

    type: Literal["json"]
    path: str = Field(description="Path relative to trial_dir, e.g. 'result.json'.")
    key: str = Field(description="Dot-separated key into the JSON, e.g. 'eval.loss'.")


class LogRegexExtractor(_Frozen):
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
    """Extract a scalar from a completed W&B run's summary."""

    type: Literal["wandb"]
    entity: str
    project: str
    run_name_template: str = Field(
        default="{experiment}-{phase}-{trial_id}",
        description=(
            "Template the trial uses to name its W&B run. Available substitutions: "
            "{experiment}, {phase}, {trial_id}, {run_name}."
        ),
    )
    metric_key: str = Field(description="Key on wandb.run.summary, e.g. 'eval/loss'.")
    poll_seconds: float = Field(default=2.0, gt=0.0)
    timeout_seconds: float = Field(default=120.0, ge=0.0)


Extractor = JsonExtractor | LogRegexExtractor | WandbExtractor


# --------------------------------------------------------------------------------------
# Evidence gates
# --------------------------------------------------------------------------------------


class RequiredFileGate(_Frozen):
    """Require a file to exist under the trial directory."""

    type: Literal["required_file"]
    path: str


class JsonEqualsGate(_Frozen):
    """Require a JSON key to equal an expected scalar value."""

    type: Literal["json_equals"]
    path: str
    key: str
    value: Any


class JsonScalarBoundGate(_Frozen):
    """Require a JSON key to be a finite scalar within optional bounds."""

    type: Literal["json_scalar_bound"]
    path: str
    key: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> JsonScalarBoundGate:
        """Reject empty/non-finite bounds and ``min > max``."""
        if self.min is None and self.max is None:
            raise ValueError("json_scalar_bound gate must define at least one of min/max.")
        if self.min is not None:
            _require_finite("json_scalar_bound.min", self.min)
        if self.max is not None:
            _require_finite("json_scalar_bound.max", self.max)
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"json_scalar_bound gate min ({self.min}) must be <= max ({self.max})."
            )
        return self


class ArtifactSizeGate(_Frozen):
    """Require an artifact file size to fall inside optional byte bounds."""

    type: Literal["artifact_size"]
    path: str
    min_bytes: int | None = Field(default=None, ge=0)
    max_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ArtifactSizeGate:
        """Reject empty bounds and ``min_bytes > max_bytes``."""
        if self.min_bytes is None and self.max_bytes is None:
            raise ValueError("artifact_size gate must define min_bytes and/or max_bytes.")
        if (
            self.min_bytes is not None
            and self.max_bytes is not None
            and self.min_bytes > self.max_bytes
        ):
            raise ValueError(
                f"artifact_size gate min_bytes ({self.min_bytes}) must be <= "
                f"max_bytes ({self.max_bytes})."
            )
        return self


class Sha256Gate(_Frozen):
    """Require a file's SHA-256 digest to match an expected hex string."""

    type: Literal["sha256"]
    path: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        """Require a full 64-character lowercase/uppercase hex digest."""
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("sha256 gate requires a full 64-character hex digest.")
        return value.lower()


class WandbSummaryRequiredGate(_Frozen):
    """Require keys to be present in a completed W&B run summary."""

    type: Literal["wandb_summary_required"]
    entity: str
    project: str
    keys: list[str] = Field(min_length=1)
    run_name_template: str = "{experiment}-{phase}-{trial_id}"
    poll_seconds: float = Field(default=2.0, gt=0.0)
    timeout_seconds: float = Field(default=120.0, ge=0.0)


Gate = Annotated[
    RequiredFileGate
    | JsonEqualsGate
    | JsonScalarBoundGate
    | ArtifactSizeGate
    | Sha256Gate
    | WandbSummaryRequiredGate,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------------------
# Metric / constraints
# --------------------------------------------------------------------------------------


class Metric(_Frozen):
    """Primary optimization objective: name, direction, and how to extract it."""

    name: str = "objective"
    goal: Literal["minimize", "maximize"] = "minimize"
    extractor: Extractor = Field(discriminator="type")


class Constraint(_Frozen):
    """A scalar bound that trials must satisfy to be considered feasible."""

    name: str
    extractor: Extractor = Field(discriminator="type")
    max: float | None = None
    min: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> Constraint:
        """Reject empty/non-finite bounds and ``min > max`` (Pydantic post-init hook).

        Returns:
            Self, unchanged. Pydantic ``mode='after'`` validator protocol.

        """
        if self.max is None and self.min is None:
            raise ValueError(f"Constraint {self.name!r} must define at least one of min/max.")
        if self.max is not None:
            _require_finite(f"Constraint {self.name!r} max", self.max)
        if self.min is not None:
            _require_finite(f"Constraint {self.name!r} min", self.min)
        if self.max is not None and self.min is not None and self.min > self.max:
            raise ValueError(f"Constraint {self.name!r}: min ({self.min}) > max ({self.max}).")
        return self


# --------------------------------------------------------------------------------------
# Search space (with bounds validation)
# --------------------------------------------------------------------------------------


class FloatParam(_Frozen):
    """Continuous float search parameter with optional log-scale and step."""

    type: Literal["float"]
    low: float
    high: float
    log: bool = False
    step: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> FloatParam:
        """Reject non-finite bounds, ``low > high``, log+nonpositive, log+step combos.

        Returns:
            Self, unchanged. Pydantic ``mode='after'`` validator protocol.

        """
        _require_finite("float param low", self.low)
        _require_finite("float param high", self.high)
        if self.low > self.high:
            raise ValueError(f"float param: low ({self.low}) > high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError("log-scale float param requires low > 0")
        if self.step is not None:
            _require_finite("float param step", self.step)
            if self.step <= 0:
                raise ValueError("float param step must be > 0")
        if self.log and self.step is not None:
            raise ValueError("float param cannot use both log=true and step")
        return self


class IntParam(_Frozen):
    """Integer search parameter with optional log-scale and step."""

    type: Literal["int"]
    low: int
    high: int
    log: bool = False
    step: int = 1

    @model_validator(mode="after")
    def _validate(self) -> IntParam:
        """Reject ``low > high``, log+nonpositive, non-positive step, log+step!=1.

        Returns:
            Self, unchanged. Pydantic ``mode='after'`` validator protocol.

        """
        if self.low > self.high:
            raise ValueError(f"int param: low ({self.low}) > high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError("log-scale int param requires low > 0")
        if self.step <= 0:
            raise ValueError("int param step must be > 0")
        if self.log and self.step != 1:
            # Optuna's IntDistribution rejects this at construction time.
            # Catch it here so config-load fails instead of trial-launch.
            raise ValueError("int param cannot use log=true with step != 1")
        return self


class CategoricalParam(_Frozen):
    """Categorical search parameter with an explicit list of choices."""

    type: Literal["categorical"]
    choices: list[Any] = Field(min_length=1)

    @field_validator("choices")
    @classmethod
    def _choices_are_optuna_scalars(cls, choices: list[Any]) -> list[Any]:
        """Reject categorical choices Optuna can't store (lists, dicts, NaN, ...).

        Args:
            choices: The candidate choices list pre-validation.

        Returns:
            The same list, unchanged. Raises ``ValueError`` if any element is
            not an Optuna-compatible scalar.

        """
        # Optuna only accepts None|bool|int|float|str as categorical choices.
        # Anything else (lists, dicts, custom objects) fails at suggest time.
        allowed = (str, int, float, bool, type(None))
        for c in choices:
            if not isinstance(c, allowed):
                raise ValueError(
                    "categorical choices must be Optuna-compatible scalars "
                    "(None, bool, int, float, or str); "
                    f"got {type(c).__name__}: {c!r}"
                )
            if isinstance(c, float) and not math.isfinite(c):
                raise ValueError(f"categorical float choices must be finite; got {c!r}")
        return choices


SearchParam = FloatParam | IntParam | CategoricalParam


# --------------------------------------------------------------------------------------
# Sampler (pruner removed — see TODO.md)
# --------------------------------------------------------------------------------------


class Sampler(_Frozen):
    """Optuna sampler configuration."""

    type: Literal["tpe", "random", "grid", "cmaes"] = "tpe"
    seed: int | None = None
    n_startup_trials: int = Field(default=10, ge=0)  # tpe only


class Contract(_Frozen):
    """Named fixed-comparison contract shared across phases or studies."""

    fixed_overrides: dict[str, Any] = Field(default_factory=dict)
    gates: list[Gate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_override_key_syntax(self) -> Contract:
        """Reject malformed contract override keys."""
        for key in self.fixed_overrides:
            _validate_override_key(key, label="contract fixed_overrides")
        return self


class Promotion(_Frozen):
    """Conditional phase promotion against a previously-computed baseline winner."""

    min_delta_vs: str
    min_delta: float = 0.0
    requires_gates: bool = True
    on_fail: Literal["stop", "skip", "continue_baseline"] = "stop"

    @model_validator(mode="after")
    def _validate_delta(self) -> Promotion:
        """Require a finite promotion delta."""
        _require_finite("promotion.min_delta", self.min_delta)
        return self


# --------------------------------------------------------------------------------------
# Phase / experiment
# --------------------------------------------------------------------------------------


class Phase(_Frozen):
    """One stage in a sequential hyperparameter sweep."""

    name: str
    comment: str | None = Field(
        default=None,
        description=(
            "Free-text note describing the design intent of this phase: why this "
            "search space, why this sampler, what hypothesis is being tested. "
            "Surfaced by `phasesweep validate` and `phasesweep show-winners`. "
            "Excluded from the semantic fingerprint — editing the comment never "
            "invalidates the study."
        ),
    )
    inherits: list[str] = Field(
        default_factory=list,
        description="Phase names whose winners become fixed overrides for this phase.",
    )
    fixed_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Hard-coded overrides applied to every trial in this phase.",
    )
    contracts: list[str] = Field(
        default_factory=list,
        description="Named fixed-comparison contracts applied to every trial in this phase.",
    )
    search_space: dict[str, SearchParam] = Field(
        default_factory=dict,
        description="Map of override-key -> sampling spec. Supports dotted keys.",
    )
    n_trials: int = Field(ge=1)
    n_jobs: int = Field(
        default=1,
        ge=1,
        description=(
            "Parallel trials within this phase. When gpu_ids is also set, each "
            "trial gets exclusive access to one GPU via CUDA_VISIBLE_DEVICES."
        ),
    )
    gpu_ids: list[int] | None = Field(
        default=None,
        description=(
            "Explicit list of CUDA device indices to partition across parallel trials. "
            "When None and n_jobs > 1, phasesweep auto-detects via CUDA_VISIBLE_DEVICES "
            "or nvidia-smi."
        ),
    )

    @field_validator("gpu_ids")
    @classmethod
    def _gpu_ids_non_negative(cls, value: list[int] | None) -> list[int] | None:
        """Reject negative GPU indices, which would silently disable isolation.

        Args:
            value: The candidate ``gpu_ids`` list, or ``None``.

        Returns:
            The same value, unchanged. Raises ``ValueError`` if any element
            is negative.

        """
        if value is None:
            return None
        bad = [v for v in value if v < 0]
        if bad:
            raise ValueError(
                f"gpu_ids must be non-negative CUDA device indices; got {bad}. "
                "(CUDA_VISIBLE_DEVICES=-1 hides all devices and would silently "
                "disable GPU isolation.)"
            )
        return value

    max_consecutive_failures: int = Field(
        default=5,
        ge=1,
        description=(
            "Abort the phase after this many consecutive failed/infeasible trials. "
            "Set high if your trainer legitimately fails a lot; low to fail fast on "
            "broken configs."
        ),
    )
    allow_no_gpu_isolation: bool = Field(
        default=False,
        description=(
            "When n_jobs > 1 and no GPUs are detected, phasesweep fails by default "
            "to prevent parallel trials from stampeding the same device. Set True "
            "for intentional CPU-only parallel sweeps."
        ),
    )
    sampler: Sampler = Field(default_factory=Sampler)
    timeout_seconds_per_trial: float | None = Field(default=86400.0, ge=0)
    allow_unbounded_trials: bool = Field(
        default=False,
        description=(
            "Set true only when an intentionally unbounded trial is acceptable. "
            "Otherwise timeout_seconds_per_trial must be finite."
        ),
    )
    timeout_seconds_per_phase: float | None = Field(default=None, ge=0)
    allow_partial_grid: bool = Field(
        default=False,
        description=(
            "Grid phases must run the full matrix by default. Set true to permit "
            "n_trials smaller than the grid cardinality."
        ),
    )
    allow_seed_search: bool = Field(
        default=False,
        description=(
            "By default, search-space keys named seed or ending in .seed are rejected "
            "so stochastic variance is not mistaken for a model/config improvement."
        ),
    )
    gates: list[Gate] = Field(default_factory=list)
    promotion: Promotion | None = None

    @field_validator("name")
    @classmethod
    def _name_is_safe(cls, v: str) -> str:
        """Reject empty names and any character outside ``[A-Za-z0-9_-]``.

        Args:
            v: The candidate phase name.

        Returns:
            The same name, unchanged. Raises ``ValueError`` if any character
            is disallowed (the name is used as a filesystem path component).

        """
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError(f"Phase name {v!r} must be non-empty and [A-Za-z0-9_-] only.")
        return v

    @model_validator(mode="after")
    def _validate_override_key_syntax(self) -> Phase:
        r"""Reject malformed override keys before they hit the override renderer.

        Hydra/argparse rendering quotes values shell-safely, but a malformed
        *key* like ``""``, ``"."``, ``"a..b"``, or ``" lr"`` would either
        produce broken hydra commands (``=value``, ``..a=value``) or silently
        treat surface noise (whitespace) as part of the key (review v0.5.6 /
        non-blocking hardening item).

        Permissible keys: dotted paths whose every segment is non-empty and
        matches ``[A-Za-z0-9_\-]+``. This covers ``lr``, ``model.depth``,
        ``hydra.run.dir``, ``data.train_path``, ``optim.weight-decay``.

        Returns:
            Self, unchanged. Pydantic post-init validator protocol.

        """
        for key in self.fixed_overrides:
            _validate_override_key(key, label=f"phase {self.name!r} fixed_overrides")
        for key in self.search_space:
            _validate_override_key(key, label=f"phase {self.name!r} search_space")
        return self

    @model_validator(mode="after")
    def _validate_timeouts_and_seed_policy(self) -> Phase:
        """Require bounded trials unless explicitly waived and reject seed sweeps."""
        if self.timeout_seconds_per_trial is None:
            if not self.allow_unbounded_trials:
                raise ValueError(
                    f"Phase {self.name!r}: timeout_seconds_per_trial is required unless "
                    "allow_unbounded_trials: true is set."
                )
        elif not math.isfinite(self.timeout_seconds_per_trial):
            raise ValueError(
                f"Phase {self.name!r}: timeout_seconds_per_trial must be finite, "
                f"got {self.timeout_seconds_per_trial!r}."
            )
        if self.timeout_seconds_per_phase is not None and not math.isfinite(
            self.timeout_seconds_per_phase
        ):
            raise ValueError(
                f"Phase {self.name!r}: timeout_seconds_per_phase must be finite, "
                f"got {self.timeout_seconds_per_phase!r}."
            )
        if not self.allow_seed_search:
            seed_keys = [key for key in self.search_space if key == "seed" or key.endswith(".seed")]
            if seed_keys:
                raise ValueError(
                    f"Phase {self.name!r}: trainer seed keys cannot be in search_space "
                    f"by default: {seed_keys}. Move seeds to fixed_overrides or set "
                    "allow_seed_search: true for an explicit variance audit."
                )
        return self


_OVERRIDE_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_override_key(key: object, *, label: str) -> None:
    """Reject override keys that would render to ambiguous shell arguments.

    Args:
        key: The candidate key from ``fixed_overrides`` or ``search_space``.
            Must be a non-empty string of dotted ``[A-Za-z0-9_-]`` segments.
        label: Human-readable context prepended to any error message
            (e.g. ``"phase 'lr' search_space"``).

    Raises:
        ValueError: ``key`` is not a string; empty; has leading/trailing or
            embedded whitespace; or contains an empty dotted segment or a
            segment with disallowed characters.

    """
    if not isinstance(key, str):
        raise ValueError(
            f"{label}: override key must be a string, got {type(key).__name__}: {key!r}."
        )
    if not key:
        raise ValueError(f"{label}: override key cannot be empty.")
    if key != key.strip():
        raise ValueError(f"{label}: override key {key!r} has leading or trailing whitespace.")
    if any(c.isspace() for c in key):
        raise ValueError(f"{label}: override key {key!r} contains whitespace.")
    parts = key.split(".")
    for i, part in enumerate(parts):
        if not part:
            raise ValueError(
                f"{label}: override key {key!r} has an empty dotted segment at "
                f"position {i}. Use simple keys like 'lr' or dotted keys like "
                f"'model.depth'; do not use leading/trailing dots or '..'."
            )
        if not _OVERRIDE_KEY_SEGMENT.match(part):
            raise ValueError(
                f"{label}: override key {key!r} segment {part!r} contains "
                "invalid characters (allowed: alphanumerics, underscore, dash, "
                "and '.' as the segment separator)."
            )


class Experiment(_Frozen):
    """Top-level experiment: trial command, metric, constraints, and ordered phases."""

    experiment: str
    storage: str | None = Field(
        default=None,
        description=(
            "Optuna storage URL. Use sqlite:///path.db for resumable single-job studies, "
            "journal:///path.journal for parallel studies, or any RDB URL Optuna accepts. "
            "Null for non-resumable in-memory runs (not recommended). "
            "phasesweep does NOT silently rewrite SQLite to JournalStorage; choose the "
            "scheme intentionally so study identity stays stable across n_jobs changes."
        ),
    )
    workdir: str = Field(default="./runs", description="Where per-trial directories are created.")
    trial_command: str = Field(
        description=(
            "Shell command template. Placeholders: {overrides}, {trial_dir}, {trial_id}, "
            "{phase}, {run_name}, {overrides_path}."
        )
    )
    override_format: Literal["hydra", "argparse", "json_file"] = "hydra"
    metric: Metric
    constraints: list[Constraint] = Field(default_factory=list)
    contracts: dict[str, Contract] = Field(default_factory=dict)
    phases: list[Phase] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds_per_run: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_run_timeout(self) -> Experiment:
        """Reject non-finite run wallclock guards."""
        if self.timeout_seconds_per_run is not None and not math.isfinite(
            self.timeout_seconds_per_run
        ):
            raise ValueError(
                f"timeout_seconds_per_run must be finite, got {self.timeout_seconds_per_run!r}."
            )
        return self

    @field_validator("experiment")
    @classmethod
    def _experiment_name_is_safe(cls, v: str) -> str:
        """Experiment name is used as a filesystem path component (lock files, study names).

        Same constraint as phase names: ``[A-Za-z0-9_-]`` only. Without this,
        names like ``../../etc/evil`` could escape the lock-file path under
        ``$TMPDIR/phasesweep-locks/`` and the experiment::phase study name.

        Returns:
            The validated experiment name, unchanged. Raises ``ValueError``
            on any disallowed character.

        """
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError(f"Experiment name {v!r} must be non-empty and [A-Za-z0-9_-] only.")
        return v

    @model_validator(mode="after")
    def _validate_phase_graph(self) -> Experiment:
        """Validate phase composition + per-phase semantic consistency.

        Each phase is checked for:
          * local fixed/sampled key collision (review v0.5.2 / blocker 5)
          * transitive inherited locked-key collisions (v0.5 review item #4)
          * unresolved multi-parent locked-key collisions (v0.5 review item #5)
          * sampler / search-space compatibility (v0.5.2 / blocker 2)
          * grid divisibility for float params (v0.5.2 / blocker 4)
          * SQLite + parallel n_jobs (v0.5.2 / blocker 6)

        Returns:
            Self, unchanged. Pydantic post-init validator protocol; raises
            ``ValueError`` on any inconsistency listed above.

        """
        seen: dict[str, Phase] = {}
        locked_keys_by_phase: dict[str, set[str]] = {}

        for phase in self.phases:
            if phase.name in seen:
                raise ValueError(f"Duplicate phase name {phase.name!r}.")
            for contract_name in phase.contracts:
                if contract_name not in self.contracts:
                    raise ValueError(
                        f"Phase {phase.name!r} references unknown contract {contract_name!r}."
                    )
            for parent in phase.inherits:
                if parent not in seen:
                    raise ValueError(
                        f"Phase {phase.name!r} inherits from {parent!r}, "
                        f"which is not a prior phase."
                    )

            # Local same-phase collision (blocker 5): sampling a key that the
            # same phase also lists as fixed silently lets sampled win — that's
            # not "fixed" in any meaningful sense.
            local_collisions = set(phase.fixed_overrides) & set(phase.search_space)
            if local_collisions:
                raise ValueError(
                    f"Phase {phase.name!r} declares key(s) {sorted(local_collisions)} "
                    "in both fixed_overrides and search_space. A key must be either "
                    "fixed or sampled, not both."
                )

            contract_keys: set[str] = set()
            contract_key_owner: dict[str, str] = {}
            for contract_name in phase.contracts:
                contract = self.contracts[contract_name]
                for key in contract.fixed_overrides:
                    if key in contract_key_owner:
                        raise ValueError(
                            f"Phase {phase.name!r} applies contracts with conflicting "
                            f"fixed key {key!r}: {contract_key_owner[key]!r} and "
                            f"{contract_name!r}."
                        )
                    contract_keys.add(key)
                    contract_key_owner[key] = contract_name

            contract_local_collisions = contract_keys & (
                set(phase.fixed_overrides) | set(phase.search_space)
            )
            if contract_local_collisions:
                raise ValueError(
                    f"Phase {phase.name!r} tries to override contract-locked key(s) "
                    f"{sorted(contract_local_collisions)}. Contract keys are immutable "
                    "inside the applying phase."
                )

            # Local dotted-key namespace collision (review v0.5.3 / blocker 5):
            # `{model: llama, model.depth: 16}` is silently corrupting in every
            # render format we support. Reject at config-load.
            local_keys = set(phase.fixed_overrides) | set(phase.search_space.keys()) | contract_keys
            prefix_collisions = _find_prefix_collisions(local_keys)
            if prefix_collisions:
                pairs = ", ".join(f"{a!r} ⊏ {b!r}" for a, b in prefix_collisions)
                raise ValueError(
                    f"Phase {phase.name!r} has dotted-key namespace collision(s): "
                    f"{pairs}. A key and a sub-key cannot both be overridden — the "
                    "rendered command would be contradictory (Hydra/argparse) or the "
                    "json_file would have to be both a scalar and a nested object."
                )

            # Transitive inherited locked keys + multi-parent collision detection.
            inherited_keys: set[str] = set()
            parent_owners: dict[str, list[str]] = {}
            for parent in phase.inherits:
                for key in locked_keys_by_phase[parent]:
                    inherited_keys.add(key)
                    parent_owners.setdefault(key, []).append(parent)

            unresolved = {
                k: owners
                for k, owners in parent_owners.items()
                if len(owners) > 1 and k not in phase.fixed_overrides
            }
            if unresolved:
                details = ", ".join(
                    f"{k!r} from {sorted(set(o))}" for k, o in sorted(unresolved.items())
                )
                raise ValueError(
                    f"Phase {phase.name!r} inherits conflicting locked key(s) from multiple "
                    f"parents: {details}. Resolve explicitly with phase.fixed_overrides "
                    f"or remove one inherit."
                )

            collisions = inherited_keys & set(phase.search_space.keys())
            if collisions:
                raise ValueError(
                    f"Phase {phase.name!r} re-samples key(s) {sorted(collisions)} "
                    f"that are locked by inherited phase(s) {phase.inherits!r} "
                    f"(possibly transitively). Either remove the key from search_space, "
                    f"or drop the inherit."
                )

            # Inherited / local dotted-key prefix collision (review v0.5.3 /
            # blocker 5). E.g. parent locks `model` and child samples
            # `model.depth`, or vice-versa. Same render-time corruption hazard
            # as the local-only case but caught across the inheritance graph.
            combined_keys = (
                inherited_keys
                | contract_keys
                | set(phase.fixed_overrides)
                | set(phase.search_space)
            )
            inh_prefix_collisions = _find_prefix_collisions(combined_keys)
            if inh_prefix_collisions:
                pairs = ", ".join(f"{a!r} ⊏ {b!r}" for a, b in inh_prefix_collisions)
                raise ValueError(
                    f"Phase {phase.name!r} has dotted-key namespace collision(s) "
                    f"across inherited and local overrides: {pairs}. "
                    "A key and a sub-key cannot both be overridden across the "
                    "inheritance chain."
                )

            # Sampler/search-space compatibility (blocker 2): catch at config-load
            # so `phasesweep validate` is meaningful, not at first trial launch.
            _validate_sampler_search_space(phase)

            # Storage policy (blocker 6): SQLite + parallel writes deadlocks under
            # contention; we no longer auto-remap to JournalStorage. Tell the user.
            _validate_storage_policy(self.storage, phase)

            # Trial command template (v0.5.3 follow-up): render once with
            # placeholder overrides per phase. Catches typos like `{trail_dir}`,
            # unknown placeholders, and unbalanced braces at config-load instead
            # of three minutes into a sweep.
            _validate_trial_command_template(self, phase, inherited_keys)

            locked_keys_by_phase[phase.name] = (
                inherited_keys
                | contract_keys
                | set(phase.fixed_overrides)
                | set(phase.search_space)
            )
            seen[phase.name] = phase

        names = {self.metric.name} | {c.name for c in self.constraints}
        if len(names) != 1 + len(self.constraints):
            raise ValueError("Metric and constraint names must all be distinct.")
        return self

    def phase_by_name(self, name: str) -> Phase:
        """Look up a phase by name.

        Args:
            name: Phase name to search for.

        Returns:
            The matching :class:`Phase`.

        Raises:
            KeyError: No phase in ``self.phases`` has that name.

        """
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(name)


class SuiteDefaults(_Frozen):
    """Shared defaults applied to every study in a suite."""

    storage: str | None = None
    workdir: str = "./runs"
    trial_command: str | None = None
    override_format: Literal["hydra", "argparse", "json_file"] = "hydra"
    metric: Metric | None = None
    constraints: list[Constraint] = Field(default_factory=list)
    contracts: dict[str, Contract] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds_per_run: float | None = Field(default=None, ge=0)


class StudySpec(_Frozen):
    """One experiment-like study inside a suite run plan."""

    name: str
    depends_on: list[str] = Field(default_factory=list)
    storage: str | None = None
    workdir: str | None = None
    trial_command: str | None = None
    override_format: Literal["hydra", "argparse", "json_file"] | None = None
    metric: Metric | None = None
    constraints: list[Constraint] | None = None
    contracts: dict[str, Contract] = Field(default_factory=dict)
    phases: list[Phase] = Field(min_length=1)
    env: dict[str, str] | None = None
    timeout_seconds_per_run: float | None = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def _study_name_is_safe(cls, value: str) -> str:
        """Study names are used as experiment-name suffixes and path components."""
        if not value or not all(c.isalnum() or c in "_-" for c in value):
            raise ValueError(f"Study name {value!r} must be non-empty and [A-Za-z0-9_-] only.")
        return value


class Suite(_Frozen):
    """Suite of independent or dependency-ordered phase-chain studies."""

    suite: str
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    studies: list[StudySpec] = Field(min_length=1)

    @field_validator("suite")
    @classmethod
    def _suite_name_is_safe(cls, value: str) -> str:
        """Suite names are used as output path and experiment-name prefixes."""
        if not value or not all(c.isalnum() or c in "_-" for c in value):
            raise ValueError(f"Suite name {value!r} must be non-empty and [A-Za-z0-9_-] only.")
        return value

    @model_validator(mode="after")
    def _validate_study_graph(self) -> Suite:
        """Require unique, prior-only study dependencies."""
        seen: set[str] = set()
        for study in self.studies:
            if study.name in seen:
                raise ValueError(f"Duplicate study name {study.name!r}.")
            for dep in study.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"Study {study.name!r} depends_on {dep!r}, which is not a prior study."
                    )
            seen.add(study.name)
        return self

    def experiment_for_study(self, study: StudySpec) -> Experiment:
        """Compile a suite study into a normal :class:`Experiment`."""
        defaults = self.defaults

        def value(name: str, *, required: bool = False) -> Any:
            if name in study.model_fields_set:
                selected = getattr(study, name)
            else:
                selected = getattr(defaults, name)
            if required and selected is None:
                raise ValueError(
                    f"Suite {self.suite!r} study {study.name!r} must define {name!r} "
                    "or inherit it from suite.defaults."
                )
            return copy.deepcopy(selected)

        env = copy.deepcopy(defaults.env)
        if "env" in study.model_fields_set and study.env is not None:
            env.update(study.env)

        contracts = copy.deepcopy(defaults.contracts)
        contracts.update(copy.deepcopy(study.contracts))

        return Experiment(
            experiment=f"{self.suite}__{study.name}",
            storage=value("storage"),
            workdir=value("workdir", required=True),
            trial_command=value("trial_command", required=True),
            override_format=value("override_format", required=True),
            metric=value("metric", required=True),
            constraints=value("constraints") or [],
            contracts=contracts,
            phases=copy.deepcopy(study.phases),
            env=env,
            timeout_seconds_per_run=value("timeout_seconds_per_run"),
        )


Config = Experiment | Suite


# --------------------------------------------------------------------------------------
# Cross-cutting validation helpers (called from Experiment._validate_phase_graph)
# --------------------------------------------------------------------------------------


def _key_parts(key: str) -> tuple[str, ...]:
    """Split a dotted override key into its non-empty path components.

    Args:
        key: A dotted override key like ``"model.depth"``.

    Returns:
        The non-empty segments as a tuple (e.g. ``("model", "depth")``).
        Empty segments are dropped to match how the override renderer
        traverses dotted paths.

    """
    return tuple(part for part in key.split(".") if part)


def _find_prefix_collisions(keys: set[str]) -> list[tuple[str, str]]:
    """Return pairs ``(short, long)`` where ``short`` is a strict path-prefix of ``long``.

    Two keys collide when one's dot-path is a strict prefix of the other's.

    Examples:
        * ``model`` and ``model.depth`` collide — Hydra/argparse renders
          ``model=llama model.depth=16`` (contradictory), and ``json_file``
          cannot represent both a scalar and a nested object at the same key.
        * ``model.depth`` and ``model.depths`` do **not** collide (different
          siblings, same depth).
        * ``a.b.c`` and ``a.b.c.d`` do collide.

    Used in ``Experiment._validate_phase_graph`` (review v0.5.3 / blocker 5).

    Args:
        keys: The combined fixed-overrides / search-space / inherited key set
            for one phase.

    Returns:
        Lexically-sorted ``(short, long)`` pairs where ``short`` is a strict
        path-prefix of ``long``. Empty list when no collisions are present.

    """
    parts_by_key = {key: _key_parts(key) for key in keys}
    seen: set[tuple[str, str]] = set()

    items = list(parts_by_key.items())
    for i, (a, a_parts) in enumerate(items):
        for b, b_parts in items[i + 1 :]:
            if a_parts == b_parts:
                continue
            if len(a_parts) < len(b_parts):
                shorter_parts, shorter_key, longer_key = a_parts, a, b
                longer_parts = b_parts
            else:
                shorter_parts, shorter_key, longer_key = b_parts, b, a
                longer_parts = a_parts
            if longer_parts[: len(shorter_parts)] == shorter_parts:
                seen.add((shorter_key, longer_key))

    return sorted(seen)


def _validate_sampler_search_space(phase: Phase) -> None:
    """Reject sampler/search-space combinations Optuna will not accept at runtime.

    Run at config-load (review v0.5.2 / blocker 2). Catches:

    * CMA-ES with categorical parameters — Optuna's ``CmaEsSampler`` is float-only;
      categorical params silently fail every trial trying to cast 'b' to float.
    * Grid sampler with log-scale floats or ints — Optuna's ``GridSampler`` does
      not enumerate log-spaced values.
    * Grid sampler with float param missing ``step``.
    * Grid sampler with float ``(high - low)`` not an integer multiple of ``step`` —
      naive enumeration emits values above ``high`` (review v0.5.2 / blocker 4).
    """
    sampler_type = phase.sampler.type
    space = phase.search_space

    if sampler_type == "cmaes":
        cats = [name for name, p in space.items() if isinstance(p, CategoricalParam)]
        if cats:
            raise ValueError(
                f"Phase {phase.name!r}: sampler.type='cmaes' does not support "
                f"categorical parameters: {cats}. Use sampler.type='tpe' or "
                f"remove the categorical params from this phase."
            )
        # Optional dependency check at config-load (review v0.5.6 / non-blocking
        # hardening item). Without this, the import error fires from
        # ``_build_sampler`` mid-run, *after* ``phasesweep validate`` already
        # said the config is fine.
        try:
            import cmaes  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"Phase {phase.name!r}: sampler.type='cmaes' requires the "
                "'cmaes' package, which is not installed. Reinstall phasesweep "
                "or install it directly with `pip install cmaes`."
            ) from exc

    if sampler_type == "grid":
        cardinality = 1
        for name, param in space.items():
            if isinstance(param, FloatParam):
                if param.log:
                    raise ValueError(
                        f"Phase {phase.name!r}: grid sampler does not support "
                        f"log-scale float param {name!r}."
                    )
                if param.step is None:
                    raise ValueError(
                        f"Phase {phase.name!r}: grid sampler requires 'step' "
                        f"for float param {name!r}."
                    )
                _validate_float_grid_divides(phase.name, name, param)
                assert param.step is not None
                cardinality *= int(round((param.high - param.low) / param.step)) + 1
            elif isinstance(param, IntParam) and param.log:
                raise ValueError(
                    f"Phase {phase.name!r}: grid sampler does not support "
                    f"log-scale int param {name!r}."
                )
            elif isinstance(param, IntParam):
                cardinality *= ((param.high - param.low) // param.step) + 1
            elif isinstance(param, CategoricalParam):
                cardinality *= len(param.choices)
        if not phase.allow_partial_grid and phase.n_trials < cardinality:
            raise ValueError(
                f"Phase {phase.name!r}: grid sampler has {cardinality} combinations "
                f"but n_trials={phase.n_trials}. Grid phases run the full matrix by "
                "default; increase n_trials or set allow_partial_grid: true."
            )


def _validate_float_grid_divides(phase_name: str, param_name: str, param: FloatParam) -> None:
    """Require ``(high - low) / step`` to be (very nearly) an integer.

    Without this check, naive grid enumeration ``[low + i*step for i in range(n+1)]``
    emits values above ``high`` whenever the interval isn't an exact multiple of step
    (review v0.5.2 / blocker 4). Example: ``low=0, high=1, step=0.6`` -> ``[0, 0.6, 1.2]``.

    Args:
        phase_name: Phase containing the offending parameter; quoted in the error.
        param_name: Parameter name; quoted in the error.
        param: The :class:`FloatParam`; ``param.step`` must be non-``None`` (caller guarded).

    Raises:
        ValueError: ``(high - low) / step`` is not within ``1e-9`` of an integer.

    """
    assert param.step is not None  # guarded by caller
    span = param.high - param.low
    ratio = span / param.step
    nearest = round(ratio)
    if not math.isclose(ratio, nearest, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Phase {phase_name!r}: grid float param {param_name!r}: "
            f"(high - low) / step must be an integer. "
            f"Got low={param.low}, high={param.high}, step={param.step} "
            f"(ratio={ratio}). Pick a step that evenly divides the interval."
        )


def _validate_storage_policy(storage: str | None, phase: Phase) -> None:
    """Reject SQLite storage with parallel ``n_jobs`` (review v0.5.2 / blocker 6).

    SQLite serializes writers; concurrent Optuna trials cause ``database is locked``
    errors. Earlier versions auto-rewrote ``sqlite:///x.db`` to a JournalStorage path,
    but that fragmented study identity behind a single URL — the same config could
    point at two different studies depending on ``n_jobs``. Now we require the user
    to pick the scheme explicitly.

    The check uses :func:`phasesweep.storage_urls.storage_backend` so all
    SQLAlchemy SQLite dialects (``sqlite:///``, ``sqlite+pysqlite:///``, ...)
    are rejected. Earlier versions only matched the bare ``sqlite:///`` prefix
    and let driver-qualified URLs through unsafely (review v0.5.7 / blocker 1).

    Args:
        storage: The experiment-level storage URL, or ``None`` (in-memory).
        phase: The phase being validated; its ``n_jobs`` decides whether the
            SQLite restriction applies.

    Raises:
        ValueError: ``phase.n_jobs > 1`` AND ``storage`` resolves to SQLite.

    """
    if storage is None:
        return
    if phase.n_jobs > 1 and storage_backend(storage) == "sqlite":
        raise ValueError(
            f"Phase {phase.name!r} has n_jobs={phase.n_jobs} with SQLite storage "
            f"({storage!r}). SQLite serializes writers and will deadlock under "
            "parallel Optuna access. Use storage: journal:///path.journal for a "
            "single-host parallel sweep, or an RDB URL such as "
            "postgresql://... for durable storage and dashboard access from a "
            "single phasesweep orchestrator."
        )


def _placeholder_value_for(param: SearchParam) -> Any:
    """Synthesize one valid value for a search-space param (used in template render preflight).

    Args:
        param: A concrete search parameter from a phase's ``search_space``.

    Returns:
        For ``FloatParam`` the interval midpoint; for ``IntParam`` the
        integer midpoint; for ``CategoricalParam`` the first listed choice.

    Raises:
        ValueError: Unrecognised parameter subclass (defensive; the union is
            closed in practice).

    """
    if isinstance(param, FloatParam):
        return (param.low + param.high) / 2
    if isinstance(param, IntParam):
        return (param.low + param.high) // 2
    if isinstance(param, CategoricalParam):
        return param.choices[0]
    raise ValueError(f"Unhandled param: {param!r}")  # pragma: no cover


def _format_field_names(template: str) -> set[str]:
    """Return the *real* ``str.format`` field names referenced by ``template``.

    Substring matching on ``"{overrides_path}"`` is unsafe: an escaped
    ``{{overrides_path}}`` in the template renders as the literal string
    ``{overrides_path}`` (see Python docs on PEP 3101 format strings) and would
    fool a substring check into thinking the placeholder is used (review v0.5.6
    / blocker 2). ``string.Formatter().parse()`` walks the template the same
    way ``str.format`` does and reports only true field references, with
    escaped braces handled correctly.

    Field expressions like ``{trial_dir!s}``, ``{m.name}``, and ``{a[0]}`` all
    have the *root* name extracted (``trial_dir``, ``m``, ``a``) so a check
    like ``"overrides" in fields`` does the right thing.

    Args:
        template: A ``str.format``-style template string.

    Returns:
        The set of root field names actually referenced (escaped ``{{...}}``
        excluded; attribute and item accessors collapsed to the root name).

    """
    fields: set[str] = set()
    for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(template):
        if field_name is None:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        fields.add(root)
    return fields


def _validate_trial_command_template(
    experiment: Experiment, phase: Phase, inherited_keys: set[str]
) -> None:
    """Render ``trial_command`` once per phase with placeholder overrides.

    Catches at config-load (not at trial-launch, three minutes into a sweep):

    * Typos like ``{trail_dir}`` or any other unknown ``{placeholder}``.
    * Unbalanced braces (``{trial_dir`` -> ``str.format`` raises ``ValueError``).
    * Phases declaring ``override_format: json_file`` but a template missing
      ``{overrides_path}`` (rendered fine, but the trainer never sees the JSON
      and silently runs with defaults).
    * Phases with ``override_format: hydra`` or ``argparse`` and any inherited,
      fixed, or sampled overrides but a template missing ``{overrides}`` —
      same silent-no-op failure mode (review v0.5.6 / blocker 2).

    Both placeholder checks parse real ``str.format`` field names so that an
    escaped ``{{overrides}}`` (rendered as the literal string ``{overrides}``)
    correctly does *not* count as referencing the placeholder.

    The rendered command is then discarded — this is preflight only.

    Args:
        experiment: The :class:`Experiment` being validated.
        phase: The specific phase whose ``trial_command`` is being rendered.
        inherited_keys: Locked keys inherited from parents — used to decide
            whether ``{overrides}`` is required for hydra/argparse formats.

    Raises:
        ValueError: Any of the failure modes listed above (typo, unbalanced
            braces, missing required placeholder for the chosen
            ``override_format``).

    """
    # Lazy import to avoid a circular config <-> overrides cycle.
    from phasesweep.overrides import render_command

    # Build a synthetic override dict: one value per locked or sampled key.
    # Inherited keys are present in the real call too (they come from parent
    # winners), so the placeholder set must include them or render_command
    # could miss a key the trainer expects.
    overrides: dict[str, Any] = {k: "<inherited>" for k in inherited_keys}
    for contract_name in phase.contracts:
        overrides.update(experiment.contracts[contract_name].fixed_overrides)
    overrides.update(phase.fixed_overrides)
    for name, param in phase.search_space.items():
        overrides[name] = _placeholder_value_for(param)

    has_overrides = bool(overrides)

    placeholder_dir = Path(tempfile.mkdtemp(prefix="phasesweep_validate_"))
    try:
        # Parse the template once so we can both (a) preflight-render below and
        # (b) check that the documented placeholders are actually referenced.
        # Both arms surface the same "failed to render" wrapping for unbalanced
        # braces so the user sees one consistent error message regardless of
        # which check happens to detect the problem first.
        try:
            fields = _format_field_names(experiment.trial_command)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Phase {phase.name!r}: trial_command failed to render — "
                f"{type(exc).__name__}: {exc}. Check for unbalanced braces."
            ) from exc

        try:
            rendered = render_command(
                experiment.trial_command,
                overrides,
                experiment.override_format,
                trial_dir=placeholder_dir,
                trial_id=0,
                phase=phase.name,
                run_name=f"{experiment.experiment}-{phase.name}-validate",
            )
        except KeyError as exc:
            # str.format raises KeyError(name) for unknown placeholders.
            bad = exc.args[0] if exc.args else "<unknown>"
            raise ValueError(
                f"Phase {phase.name!r}: trial_command references unknown placeholder "
                f"{{{bad}}}. Supported: {{overrides}}, {{overrides_path}} (json_file "
                f"only), {{trial_dir}}, {{trial_id}}, {{phase}}, {{run_name}}."
            ) from exc
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Phase {phase.name!r}: trial_command failed to render — "
                f"{type(exc).__name__}: {exc}. Check for unbalanced braces."
            ) from exc

        # When a phase has no overrides at all (no inherited, fixed, or sampled
        # keys), a constant trial_command is legitimate — the user is sweeping
        # the same configuration repeatedly, e.g. for variance estimation.
        if not has_overrides:
            return

        if experiment.override_format == "json_file" and "overrides_path" not in fields:
            raise ValueError(
                f"override_format='json_file' but phase {phase.name!r} has "
                "inherited, fixed, or sampled overrides and trial_command "
                "does not reference {overrides_path}. The trainer would "
                "never see the override JSON. Either add {overrides_path} "
                "to trial_command, or switch to override_format='hydra' / "
                "'argparse' (which use the {overrides} placeholder)."
            )
        if experiment.override_format in ("hydra", "argparse") and "overrides" not in fields:
            raise ValueError(
                f"override_format={experiment.override_format!r} but phase "
                f"{phase.name!r} has inherited, fixed, or sampled overrides "
                "and trial_command does not reference {overrides}. All "
                "sampled parameters would be ignored — the trainer would "
                "run with the same hard-coded configuration every trial. "
                f"Add {{overrides}} to trial_command, or switch to "
                "override_format='json_file' (which uses {overrides_path})."
            )

        del rendered  # preflight-only; we just wanted to know it didn't blow up
    finally:
        # json_file mode wrote an overrides.json; clean it up.
        with contextlib.suppress(OSError):
            for child in placeholder_dir.iterdir():
                child.unlink()
            placeholder_dir.rmdir()


class _StrictMappingLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` subclass that rejects duplicate mapping keys.

    Default ``yaml.safe_load`` silently keeps the last value for a duplicated
    key. For experiment specs that's a footgun: a YAML like::

        search_space:
          lr: {type: float, low: 1e-5, high: 1e-3, log: true}
          lr: {type: float, low: 1e-4, high: 1e-2, log: true}  # silently wins

    would run a sweep against the *second* range with no warning. We override
    the default mapping constructor to raise on collisions, mirroring how
    ``Phase``'s collision validators behave for cross-phase keys.
    """


def _construct_mapping_strict(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct a YAML mapping while rejecting duplicate keys.

    Used as :class:`_StrictMappingLoader`'s mapping constructor: replaces the
    default constructor (which silently keeps the last value for duplicate
    keys) with one that raises ``ConstructorError`` so misspelled or
    copy-pasted duplicate keys surface at load time rather than after the
    sweep finishes.

    Args:
        loader: The active YAML loader; used to construct nested objects.
        node: The mapping node to construct.
        deep: Whether to construct nested objects deeply (PyYAML contract).

    Returns:
        The constructed mapping.

    Raises:
        yaml.constructor.ConstructorError: ``node`` is not a mapping, has an
            unhashable key, or has a duplicate key.

    """
    if not isinstance(node, yaml.MappingNode):  # pragma: no cover - safety net
        raise yaml.constructor.ConstructorError(
            None, None, f"expected a mapping node, found {node.id}", node.start_mark
        )

    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key: {exc}",
                key_node.start_mark,
            ) from None
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictMappingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_strict,
)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a strict mapping."""
    text = Path(path).read_text()
    try:
        data = yaml.load(text, Loader=_StrictMappingLoader)  # noqa: S506 — strict SafeLoader subclass
    except yaml.constructor.ConstructorError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping.")
    return data


def load_config(path: str | Path) -> Config:
    """Parse and validate either a single experiment YAML or a suite YAML.

    Args:
        path: Filesystem path to a phasesweep YAML file.

    Returns:
        :class:`Experiment` for legacy/current single-study configs, or
        :class:`Suite` for configs with a top-level ``suite`` key.

    """
    data = _load_yaml_mapping(path)
    if "suite" in data:
        return Suite.model_validate(data)
    return Experiment.model_validate(data)


def load_experiment(path: str | Path) -> Experiment:
    """Parse and validate a single experiment YAML.

    Uses a strict loader that rejects duplicate mapping keys. PyYAML's default
    ``safe_load`` silently keeps the last value for a duplicate key, which can
    cause an experiment to silently use the wrong search range or fixed
    override (review v0.5.6 / non-blocking hardening item).

    Args:
        path: Filesystem path to the experiment YAML file.

    Returns:
        A fully-validated :class:`Experiment` instance with all Pydantic and
        cross-phase consistency checks applied.

    Raises:
        ValueError: YAML parse error, top-level is not a mapping, duplicate
            mapping keys, or any Pydantic / cross-phase validation failure.

    """
    data = _load_yaml_mapping(path)
    if "suite" in data:
        raise ValueError(f"{path}: expected a single experiment config, got a suite config.")
    return Experiment.model_validate(data)
