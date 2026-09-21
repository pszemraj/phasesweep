"""Pydantic config models for experiments, phases, and protocols."""

from __future__ import annotations

import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from phasesweep.config.common import (
    ConfigFloat,
    ConfigInt,
    _find_prefix_collisions,
    _Frozen,
    _require_finite,
    _validate_optional_bounds,
    _validate_override_key,
    _validate_safe_name,
)
from phasesweep.config.search import (
    NON_RESUMABLE_SAMPLERS,
    STOCHASTIC_SAMPLERS,
    CategoricalParam,
    Sampler,
    SearchParam,
    _placeholder_values_for,
    _validate_sampler_search_space,
)
from phasesweep.evidence.models import (
    Extractor,
    Gate,
    ObjectiveExtractor,
    WandbExtractor,
    WandbQuery,
    WandbSummaryRequiredGate,
    _WandbSummarySource,
    compose_wandb_environment,
    objective_evidence_assurance,
    wandb_gate_identity,
)
from phasesweep.runtime.files import (
    canonical_storage_identity,
    local_storage_url,
    storage_backend,
    storage_is_in_memory,
)

OverrideFormat = Literal["yaml_file", "argparse", "hydra", "json_file"]


class Metric(_Frozen):
    """Primary optimization objective: name, direction, and how to extract it."""

    name: str = "objective"
    goal: Literal["minimize", "maximize"] = "minimize"
    extractor: ObjectiveExtractor = Field(discriminator="type")

    @field_validator("name")
    @classmethod
    def _reject_reserved_name(cls, value: str) -> str:
        """Reject the winner direction metadata key as a metric name.

        :param str value: Configured metric name.
        :raises ValueError: The name is reserved by the winner artifact schema.
        :return str: The validated metric name.
        """
        if value == "goal":
            raise ValueError("Metric name 'goal' is reserved for winner direction metadata.")
        return value


def _metric_semantics_payload(metric: Metric) -> dict[str, Any]:
    """Return the persisted and agent-visible semantics of one metric.

    :param Metric metric: Configured optimization metric.
    :return dict[str, Any]: Metric name, goal, and objective-evidence assurance.
    """
    return {
        "name": metric.name,
        "goal": metric.goal,
        "objective_evidence": objective_evidence_assurance(metric.extractor),
    }


def _metric_scoring_line(metric: Metric) -> str:
    """Describe the scalar source, within-trial selection, and between-trial goal."""
    return (
        f"metric={metric.name!r} goal={metric.goal} "
        f"extractor={metric.extractor.model_dump(mode='json', exclude_none=True)}"
    )


class Constraint(_Frozen):
    """A scalar bound that trials must satisfy to be considered feasible."""

    name: str
    extractor: Extractor = Field(discriminator="type")
    max: ConfigFloat | None = None
    min: ConfigFloat | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> Constraint:
        """Reject empty/non-finite bounds and ``min > max`` (Pydantic post-init hook).

        Returns:
            Self, unchanged. Pydantic ``mode='after'`` validator protocol.

        """
        _validate_optional_bounds(
            label=f"Constraint {self.name!r}",
            min_value=self.min,
            max_value=self.max,
        )
        return self


GpuPolicy = Literal["single_per_trial", "whole_node", "none"]


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
    search_space: dict[str, SearchParam] = Field(
        default_factory=dict,
        description="Map of override-key -> sampling spec. Supports dotted keys.",
    )
    n_trials: ConfigInt = Field(ge=1)
    n_jobs: ConfigInt = Field(
        default=1,
        ge=1,
        description=(
            "Parallel trials within this phase. When gpu_ids or gpu_devices is "
            "also set, each trial gets exclusive access to one CUDA-visible device."
        ),
    )
    gpu_policy: GpuPolicy = Field(
        default="single_per_trial",
        description=(
            "CUDA visibility policy for trial subprocesses. single_per_trial leases "
            "one visible CUDA token per trial. whole_node requires n_jobs=1 plus an "
            "explicit gpu_ids or gpu_devices list and leases every configured token "
            "for the trial. none disables phasesweep CUDA isolation and GPU host "
            "locks."
        ),
    )
    gpu_ids: list[ConfigInt] | None = Field(
        default=None,
        description=(
            "Explicit list of CUDA device indices to partition across parallel trials. "
            "When None, phasesweep auto-detects numeric CUDA_VISIBLE_DEVICES or "
            "nvidia-smi output, including for n_jobs == 1."
        ),
    )
    gpu_devices: list[str] | None = Field(
        default=None,
        description=(
            "Explicit CUDA_VISIBLE_DEVICES tokens to partition across parallel trials. "
            "Use this for GPU UUIDs or MIG instance IDs; mutually exclusive with gpu_ids."
        ),
    )

    @field_validator("gpu_ids")
    @classmethod
    def _gpu_ids_non_negative(cls, value: list[int] | None) -> list[int] | None:
        """Reject negative GPU indices, which would silently disable isolation.

        Args:
            value: The candidate ``gpu_ids`` list, or ``None``.

        Returns:
            The same value, unchanged.

        Raises:
            ValueError: ``value`` is an empty list, or any element is negative
                (``CUDA_VISIBLE_DEVICES=-1`` hides all devices and would silently
                disable GPU isolation).

        """
        if value is None:
            return None
        if not value:
            raise ValueError("gpu_ids must be omitted or contain at least one CUDA device index.")
        bad = [v for v in value if v < 0]
        if bad:
            raise ValueError(
                f"gpu_ids must be non-negative CUDA device indices; got {bad}. "
                "(CUDA_VISIBLE_DEVICES=-1 hides all devices and would silently "
                "disable GPU isolation.)"
            )
        return value

    @field_validator("gpu_devices")
    @classmethod
    def _gpu_devices_non_empty_tokens(cls, value: list[str] | None) -> list[str] | None:
        """Normalize and validate explicit CUDA device tokens.

        :param list[str] | None value: Candidate CUDA device tokens, or ``None``.
        :raises ValueError: If ``value`` is an empty list, or any stripped token is
            empty, contains a comma, or is ``-1``.
        :return list[str] | None: Stripped CUDA device tokens, or ``None``.
        """
        if value is None:
            return None
        normalized = [token.strip() for token in value]
        if not normalized:
            raise ValueError(
                "gpu_devices must be omitted or contain at least one CUDA device token."
            )
        bad = [token for token in normalized if not token or "," in token or token == "-1"]
        if bad:
            raise ValueError(
                "gpu_devices entries must be non-empty CUDA_VISIBLE_DEVICES tokens "
                f"without commas or -1; got {bad}."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_gpu_isolation_config(self) -> Phase:
        """Reject ambiguous explicit GPU isolation settings.

        :raises ValueError: If ``gpu_ids`` and ``gpu_devices`` are both set, if
            ``gpu_policy='whole_node'`` is combined with ``n_jobs != 1``, lacks an
            explicit device list, or repeats a device token, or if
            ``gpu_policy='none'`` is combined with a device list or with
            ``n_jobs > 1`` without ``allow_no_gpu_isolation``.
        :return Phase: Self, unchanged.
        """
        if self.gpu_ids is not None and self.gpu_devices is not None:
            raise ValueError("gpu_ids and gpu_devices are mutually exclusive.")
        if self.gpu_policy == "whole_node" and self.n_jobs != 1:
            raise ValueError(
                "gpu_policy='whole_node' requires n_jobs=1 because each trial receives "
                "the full configured CUDA-visible device set."
            )
        if self.gpu_policy == "whole_node" and self.gpu_ids is None and self.gpu_devices is None:
            raise ValueError(
                "gpu_policy='whole_node' requires an explicit gpu_ids or gpu_devices "
                "list: the whole-node device set is the trainer's world size — a "
                "semantic input, not a throughput knob — so it cannot be left to "
                "ambient CUDA_VISIBLE_DEVICES or nvidia-smi detection."
            )
        if self.gpu_policy == "whole_node":
            # The phase fingerprint records the DECLARED token count as the
            # trainer's world size, but the runtime pool normalizes and dedupes
            # tokens before leasing. A repeated token therefore promises a
            # 2-GPU run and delivers a 1-GPU run under a 2-GPU study identity
            # (PR #5 review / reviewer 2, blocker 2). Tokens are stripped
            # defensively: ``_gpu_devices_non_empty_tokens`` already strips
            # ``gpu_devices``, but this validator must not depend on that to
            # see ``[' GPU-a ', 'GPU-a']`` as the duplicate pair it is.
            declared = self.gpu_ids if self.gpu_ids is not None else self.gpu_devices
            tokens = [str(token).strip() for token in declared or []]
            duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
            if duplicates:
                field = "gpu_ids" if self.gpu_ids is not None else "gpu_devices"
                raise ValueError(
                    "gpu_policy='whole_node' requires unique device tokens because the "
                    "effective device count is part of experiment semantics; "
                    f"{field} repeats {duplicates}. The phase fingerprint records the "
                    "declared device count as the trainer's world size, while the GPU "
                    "pool leases each token once — so a duplicate would run a smaller "
                    "world than the study identity claims. List each device once, or "
                    "use gpu_policy='single_per_trial' if you meant a pool rather than "
                    "a world size."
                )
        if self.gpu_policy == "none":
            if self.gpu_ids is not None or self.gpu_devices is not None:
                raise ValueError(
                    "gpu_policy='none' cannot be combined with gpu_ids or gpu_devices "
                    "because phasesweep CUDA isolation and GPU host locks are disabled."
                )
            if self.n_jobs > 1 and not self.allow_no_gpu_isolation:
                raise ValueError(
                    "gpu_policy='none' with n_jobs > 1 can oversubscribe the host. "
                    "Set allow_no_gpu_isolation=true only when CPU-only or external "
                    "isolation is intentional."
                )
        return self

    max_consecutive_failures: ConfigInt = Field(
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
            "When GPU isolation cannot be established, phasesweep fails by default "
            "for parallel sweeps. Set True for intentional CPU-only or "
            "externally-isolated runs."
        ),
    )
    sampler: Sampler = Field(default_factory=Sampler)
    timeout_seconds_per_trial: ConfigFloat | None = Field(default=86400.0, ge=0)
    allow_unbounded_trials: bool = Field(
        default=False,
        description=(
            "Set true only when an intentionally unbounded trial is acceptable. "
            "Otherwise timeout_seconds_per_trial must be finite."
        ),
    )
    timeout_seconds_per_phase: ConfigFloat | None = Field(default=None, ge=0)
    allow_incomplete_on_timeout: bool = Field(
        default=False,
        description=(
            "By default, phase/run wallclock timeouts fail closed before winner "
            "selection if fewer than n_trials finished. Set true to allow a "
            "partial phase winner and persist completion metadata."
        ),
    )
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

    @field_validator("name")
    @classmethod
    def _name_is_safe(cls, v: str) -> str:
        """Reject unsafe names and reserved experiment-root directory names.

        Args:
            v: The candidate phase name.

        Returns:
            The same name, unchanged. Raises ``ValueError`` if any character
            is disallowed or the name collides with engine-managed records.

        """
        v = _validate_safe_name("Phase", v)
        if v.casefold() == "attempts":
            raise ValueError("Phase name 'attempts' is reserved for the runtime recovery registry.")
        if v.casefold() == "generations":
            raise ValueError(
                "Phase name 'generations' is reserved for immutable generation records."
            )
        return v

    @model_validator(mode="after")
    def _validate_override_key_syntax(self) -> Phase:
        r"""Reject malformed override keys before they hit the override renderer.

        CLI override rendering quotes values shell-safely, but a malformed
        *key* like ``""``, ``"."``, ``"a..b"``, or ``" lr"`` would either
        produce broken commands (``-- 1``, ``=value``, ``..a=value``) or silently
        treat surface noise (whitespace) as part of the key (review v0.5.6 /
        non-blocking hardening item).

        Permissible keys: dotted paths whose every segment is non-empty and
        matches ``[A-Za-z0-9_\-]+``. This covers ``lr``, ``model.depth``,
        ``trainer.run.dir``, ``data.train_path``, ``optim.weight-decay``.

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
        """Require bounded trials unless explicitly waived and reject seed sweeps.

        :raises ValueError: If timeout or seed-search policy validation fails.
        :return Phase: Self, unchanged.
        """
        if self.timeout_seconds_per_trial is None:
            if not self.allow_unbounded_trials:
                raise ValueError(
                    f"Phase {self.name!r}: timeout_seconds_per_trial is required unless "
                    "allow_unbounded_trials: true is set."
                )
        else:
            _require_finite(
                f"Phase {self.name!r}: timeout_seconds_per_trial",
                self.timeout_seconds_per_trial,
            )
        if self.timeout_seconds_per_phase is not None:
            _require_finite(
                f"Phase {self.name!r}: timeout_seconds_per_phase",
                self.timeout_seconds_per_phase,
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


class ExecutionContext(_Frozen):
    """Explicit trainer execution-context contract (review v0.5.17 / blocker 4).

    Without this block, the trainer inherits the orchestrator's *entire*
    ambient environment and current working directory — two unbounded
    implicit semantic inputs that never contribute to study identity, so one
    persistent study can silently mix evaluations from different trainers,
    datasets, or credentials. Declaring the contract makes both explicit,
    passes them to the trainer deterministically, and binds them into the
    semantic fingerprint.
    """

    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for every trainer subprocess. Relative paths "
            "resolve against the invocation cwd at launch (same rule as "
            "workdir); prefer an absolute path so CLI and MCP invocations "
            "agree. When set, the RESOLVED path joins the semantic "
            "fingerprint, so the same study can never mix trainers reached "
            "through different working directories. When unset, trainers run "
            "in the invocation cwd and that effective resolved directory joins "
            "the fingerprint."
        ),
    )
    inherit_env: Literal["all", "none"] | list[str] = Field(
        default="all",
        description=(
            "Which ambient environment variables the trainer inherits. 'all' "
            "(default) preserves the historical full-inheritance behavior. "
            "'none' starts from a minimal documented base (PATH, HOME, LANG, "
            "LC_ALL, TMPDIR, USER, LOGNAME, TZ). A list inherits the base "
            "plus exactly the named variables. Configured `env` values are "
            "always applied on top and are always semantic. In persistent "
            "studies, inherited values outside `passthrough_env` must match "
            "the existing trial cohort before another trial is allocated."
        ),
    )
    passthrough_env: list[str] = Field(
        default_factory=list,
        description=(
            "Ambient credential or transport variables that may rotate without "
            "changing a persistent study's semantic environment cohort. These "
            "names are inherited in addition to a narrowed `inherit_env` contract, "
            "but their values are excluded from the cohort digest. Their sorted "
            "names and classification remain part of the config fingerprint."
        ),
    )
    record_env: bool = Field(
        default=False,
        description=(
            "Write the trainer environment into each trial directory as "
            "`environment.json` (digest, inherit_env contract, and every "
            "name-value pair). Off by default because those values are "
            "secrets; the file is created owner-only (0600), unlike the rest "
            "of the trial directory. Every trial always records the "
            "environment's SHA-256 digest and its variable NAMES as study "
            "attributes regardless of this flag. The digest covers semantic "
            "values only; pass-through values are never persisted in study "
            "attributes."
        ),
    )

    @model_validator(mode="after")
    def _validate_inherit_names(self) -> ExecutionContext:
        """Reject empty or whitespace-only inherited-variable names.

        :raises ValueError: A listed variable name is empty or padded.
        :return ExecutionContext: Self, unchanged.
        """
        named_contracts = {"passthrough_env": self.passthrough_env}
        if isinstance(self.inherit_env, list):
            named_contracts["inherit_env"] = self.inherit_env
        for label, names in named_contracts.items():
            bad = [name for name in names if not name or name != name.strip()]
            if bad:
                raise ValueError(f"{label} names must be nonempty and unpadded: {bad!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"{label} names must be unique.")
        if isinstance(self.inherit_env, list):
            overlap = sorted(set(self.inherit_env) & set(self.passthrough_env))
            if overlap:
                raise ValueError(
                    "inherit_env and passthrough_env must not overlap; classify each "
                    f"ambient variable once: {overlap!r}"
                )
        return self


class Experiment(_Frozen):
    """Top-level experiment: trial command, metric, constraints, and ordered phases."""

    experiment: str
    storage: str | None = Field(
        default=None,
        description=(
            "Local Optuna storage URL, or auto for study.db inside the experiment artifact "
            "namespace (study.journal when any phase has n_jobs > 1). "
            "Use sqlite:///path.db for resumable single-job studies, "
            "journal:///path.journal for parallel studies, or null for non-resumable "
            "in-memory runs. "
            "Explicit SQLite URLs are never rewritten to JournalStorage. Changing auto's "
            "backend changes storage identity and cannot resume an existing tree."
        ),
    )
    workdir: str = Field(default="./runs", description="Where per-trial directories are created.")
    trial_command: str = Field(
        description=(
            "Shell command template. Placeholders: {config_path}, {overrides}, {overrides_path}, "
            "{trial_dir}, {trial_id}, {phase}, {run_name}."
        )
    )
    trainer_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Base trainer configuration embedded in this PhaseSweep YAML. In the default "
            "yaml_file mode, inherited, fixed, and sampled dotted-path overrides are "
            "applied to this mapping and the complete result is written to "
            "{config_path} for every trial."
        ),
    )
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Operator-supplied trainer/data/dependency identity included in persistent-study "
            "fingerprints. Values must change whenever trial meaning changes outside this YAML."
        ),
    )
    override_format: OverrideFormat = "yaml_file"
    metric: Metric
    constraints: list[Constraint] = Field(default_factory=list)
    phases: list[Phase] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    execution: ExecutionContext = Field(
        default_factory=ExecutionContext,
        description=(
            "Explicit trainer execution context: working directory and "
            "environment-inheritance contract (review v0.5.17 / blocker 4)."
        ),
    )
    timeout_seconds_per_run: ConfigFloat | None = Field(default=None, ge=0)

    @property
    def resolved_storage(self) -> str | None:
        """Resolve auto storage within this experiment's artifact namespace.

        :return str | None: Absolute auto URL, or the explicit storage unchanged.
        """
        if self.storage != "auto":
            return self.storage
        root = Path(self.workdir).expanduser().resolve() / self.experiment
        parallel = any(phase.n_jobs > 1 for phase in self.phases)
        return local_storage_url(
            root / ("study.journal" if parallel else "study.db"),
            "journal" if parallel else "sqlite",
        )

    @field_validator("storage")
    @classmethod
    def _storage_is_local(cls, value: str | None) -> str | None:
        """Allow only in-memory, local SQLite, local Journal, and auto storage.

        :param str | None value: Configured Optuna storage locator.
        :raises ValueError: The locator selects an unsupported storage backend.
        :return str | None: The validated locator, unchanged.
        """
        if value == "auto" or storage_is_in_memory(value):
            return value
        backend = storage_backend(value)
        if backend not in {"sqlite", "journal"}:
            raise ValueError(
                "storage must be in-memory, sqlite:///, journal:///, or auto; "
                f"got backend {backend!r}."
            )
        canonical_storage_identity(value)
        return value

    @model_validator(mode="after")
    def _validate_persistent_provenance(self) -> Experiment:
        """Require meaningful external-input identity for persistent study reuse.

        :raises ValueError: If any provenance key or value is blank, or if
            persistent ``storage`` is set while ``provenance`` is empty.
        :return Experiment: Self, unchanged.
        """
        invalid = [
            key for key, value in self.provenance.items() if not key.strip() or not value.strip()
        ]
        if invalid:
            raise ValueError(f"provenance keys and values must be nonempty strings: {invalid}")
        if not storage_is_in_memory(self.resolved_storage) and not self.provenance:
            raise ValueError(
                "Persistent storage requires a nonempty provenance mapping that identifies "
                "the trainer, data, and dependency revision used by this experiment."
            )
        return self

    @model_validator(mode="after")
    def _validate_run_timeout(self) -> Experiment:
        """Reject non-finite run wallclock guards.

        :raises ValueError: If ``timeout_seconds_per_run`` is non-finite.
        :return Experiment: Self, unchanged.
        """
        if self.timeout_seconds_per_run is not None:
            _require_finite("timeout_seconds_per_run", self.timeout_seconds_per_run)
        return self

    @model_validator(mode="after")
    def _validate_trainer_config_mode(self) -> Experiment:
        """Reject an embedded trainer config that the selected mode would ignore.

        :raises ValueError: ``trainer_config`` is nonempty outside ``yaml_file`` mode.
        :return Experiment: Self, unchanged.
        """
        if self.override_format != "yaml_file" and self.trainer_config:
            raise ValueError(
                "trainer_config is consumed only by the default override_format='yaml_file'. "
                f"The selected compatibility format {self.override_format!r} would ignore it; "
                "remove trainer_config or use yaml_file with {config_path}."
            )
        return self

    @field_validator("experiment")
    @classmethod
    def _experiment_name_is_safe(cls, v: str) -> str:
        """Experiment name is used as a filesystem path component (lock files, study names).

        Same constraint as phase names: ``[A-Za-z0-9_-]`` only. Without this,
        names like ``../../etc/evil`` could escape the configured lock-file
        directory and the experiment::phase study name.

        Returns:
            The validated experiment name, unchanged. Raises ``ValueError``
            on any disallowed character.

        """
        return _validate_safe_name("Experiment", v)

    @model_validator(mode="after")
    def _validate_phase_graph(self) -> Experiment:
        """Validate ordered phase composition and per-phase semantics.

        Each exported override key carries its producing phase as its origin.
        Re-exporting preserves that origin, making a same-origin diamond valid.
        Different origins for one inherited key require a child fixed override;
        a fixed child key deliberately establishes a new origin.

        :raises ValueError: A phase graph, override composition, sampler,
            storage, value-wire contract, or trial command is invalid.
        :return Experiment: Self, unchanged.
        """
        seen: dict[str, Phase] = {}
        seen_casefolded: dict[str, str] = {}
        origins_by_phase: dict[str, dict[str, str]] = {}

        for phase in self.phases:
            if phase.name in seen:
                raise ValueError(f"Duplicate phase name {phase.name!r}.")
            casefolded_name = phase.name.casefold()
            if casefolded_name in seen_casefolded:
                raise ValueError(
                    f"Phase names {seen_casefolded[casefolded_name]!r} and "
                    f"{phase.name!r} must be unique case-insensitively."
                )
            for parent in phase.inherits:
                if parent not in seen:
                    raise ValueError(
                        f"Phase {phase.name!r} inherits from {parent!r}, "
                        "which is not a prior phase."
                    )

            local_collisions = set(phase.fixed_overrides) & set(phase.search_space)
            if local_collisions:
                raise ValueError(
                    f"Phase {phase.name!r} declares key(s) {sorted(local_collisions)} "
                    "in both fixed_overrides and search_space. A key must be either "
                    "fixed or sampled, not both."
                )

            inherited_origins: dict[str, set[str]] = {}
            for parent in phase.inherits:
                for key, origin in origins_by_phase[parent].items():
                    inherited_origins.setdefault(key, set()).add(origin)

            unresolved = {
                key: origins
                for key, origins in inherited_origins.items()
                if len(origins) > 1 and key not in phase.fixed_overrides
            }
            if unresolved:
                details = ", ".join(
                    f"{key!r} from {sorted(origins)}" for key, origins in sorted(unresolved.items())
                )
                raise ValueError(
                    f"Phase {phase.name!r} inherits conflicting key(s) from multiple parents: "
                    f"{details}. Resolve explicitly with phase.fixed_overrides or remove one inherit."
                )

            inherited_collisions = set(inherited_origins) & set(phase.search_space)
            if inherited_collisions:
                raise ValueError(
                    f"Phase {phase.name!r} re-samples key(s) {sorted(inherited_collisions)} "
                    f"that are locked by inherited phase(s) {phase.inherits!r} "
                    f"(possibly transitively). Either remove the key from search_space, "
                    "or drop the inherit."
                )

            resolved_origins = {
                key: next(iter(origins))
                for key, origins in inherited_origins.items()
                if key not in phase.fixed_overrides
            }
            resolved_origins.update({key: f"{phase.name}:fixed" for key in phase.fixed_overrides})
            resolved_origins.update({key: f"{phase.name}:sampled" for key in phase.search_space})
            prefix_collisions = _find_prefix_collisions(set(resolved_origins))
            if prefix_collisions:
                pairs = ", ".join(f"{a!r} ⊏ {b!r}" for a, b in prefix_collisions)
                raise ValueError(
                    f"Phase {phase.name!r} has dotted-key namespace collision(s) "
                    f"across inherited and local overrides: {pairs}. "
                    "A key and a sub-key cannot both be overridden across the inheritance chain."
                )

            _validate_sampler_search_space(phase)
            _validate_storage_policy(self.resolved_storage, phase)
            _validate_sampler_resumability(self.resolved_storage, phase)
            _validate_cli_override_values(self, phase)
            _validate_trial_command_template(self, phase, frozenset(inherited_origins))
            query = _wandb_query(self, phase.gates, gate_location=f"phases[{len(seen)}].gates")
            if query is not None:
                compose_wandb_environment(query, self.env, self.env)

            origins_by_phase[phase.name] = resolved_origins
            seen[phase.name] = phase
            seen_casefolded[casefolded_name] = phase.name

        names = {self.metric.name} | {c.name for c in self.constraints}
        if len(names) != 1 + len(self.constraints):
            raise ValueError("Metric and constraint names must all be distinct.")
        return self


def _validate_storage_policy(storage: str | None, phase: Phase) -> None:
    """Reject SQLite storage for parallel phases.

    :param str | None storage: Experiment-level storage URL, or ``None`` for memory.
    :param Phase phase: The phase whose parallelism is being validated.
    :raises ValueError: SQLite storage is used with ``n_jobs > 1``.
    """
    if storage is None:
        return
    backend = storage_backend(storage)
    if phase.n_jobs > 1 and backend == "sqlite":
        raise ValueError(
            f"Phase {phase.name!r} has n_jobs={phase.n_jobs} with SQLite storage. "
            "SQLite serializes writers and will deadlock under "
            "parallel Optuna access. Use storage: journal:///path.journal for a "
            "single-host parallel sweep."
        )


def _validate_sampler_resumability(storage: str | None, phase: Phase) -> None:
    """Require an explicit seed, and a non-resumable acknowledgement, on persistent storage.

    A persistent study outlives the process that created it, which makes two
    sampler properties operator decisions rather than defaults (review v0.5.18 /
    finding F7):

    1. Reproducibility. An unseeded ``tpe``/``random``/``cmaes`` phase draws a
       different sequence on every invocation, so the durable trials it
       accumulates cannot be reproduced or explained afterwards.
    2. Resumability. ``tpe`` and ``cmaes`` suggestions depend on process-local
       RNG/optimizer state Optuna storage does not persist, so
       :func:`phasesweep.engine.study_policy._validate_sampler_continuation` hard-rejects
       resuming such a phase mid-target. That guard fires only *after* the
       operator has been interrupted; requiring ``acknowledge_nonresumable``
       here puts the run-the-target-in-one-invocation contract in front of them
       at config load instead.

    ``grid`` is exempt from both: it enumerates a fixed matrix and resumes from
    stored grid assignments. In-memory storage (``None`` or the sentinels
    recognized by :func:`phasesweep.runtime.files.storage_is_in_memory`) is
    exempt entirely — there is no durable study to reproduce or resume, so the
    ``sampler`` block stays optional and the ``tpe`` default remains fine.

    :param str | None storage: Experiment-level storage URL, or ``None``.
    :param Phase phase: Phase whose sampler contract is checked.
    :raises ValueError: ``storage`` is persistent and the phase's sampler is
        stochastic without an explicit ``seed``, or is non-resumable without
        ``acknowledge_nonresumable: true``.
    """
    if storage is None or storage_is_in_memory(storage):
        return

    sampler = phase.sampler
    if sampler.type in STOCHASTIC_SAMPLERS and sampler.seed is None:
        raise ValueError(
            f"Phase {phase.name!r}: sampler.type={sampler.type!r} with persistent storage "
            "requires an explicit sampler.seed. A durable study outlives the "
            "process that created it, so an unseeded stochastic sampler leaves trials nobody "
            "can reproduce or explain. Add an integer 'seed' to this phase's sampler block, "
            "or use sampler.type: grid to enumerate a fixed matrix."
        )
    if sampler.type in NON_RESUMABLE_SAMPLERS and not sampler.acknowledge_nonresumable:
        raise ValueError(
            f"Phase {phase.name!r}: sampler.type={sampler.type!r} with persistent storage "
            "requires sampler.acknowledge_nonresumable: true. This sampler's "
            "suggestions depend on process-local state Optuna storage does not persist, so "
            "PhaseSweep refuses to resume the phase mid-target: an interrupted run cannot be "
            "continued and n_trials cannot be raised later — each target must run in one "
            "invocation. Add 'acknowledge_nonresumable: true' to this phase's sampler block "
            "to accept that contract, or use sampler.type: random with a seed (resumable and "
            "reproducible) or sampler.type: grid."
        )


def _validate_cli_override_values(experiment: Experiment, phase: Phase) -> None:
    """Reject scalar/list CLI override values with no faithful wire form.

    Config load delegates to the same recursive renderer used at trial launch,
    so the accepted values cannot drift between preflight and execution. The
    supported domain is ``None``, ``bool``, ``int``, finite ``float``, ``str``,
    and lists/tuples of those. Structured trainer configuration belongs in the
    default ``yaml_file`` mode.

    :param Experiment experiment: Experiment being validated; supplies
        ``override_format``.
    :param Phase phase: Phase whose composed fixed values are checked.
    :raises ValueError: A composed value is outside the selected CLI value contract.
    """
    if experiment.override_format == "json_file":
        from phasesweep.runtime.commands import dump_overrides_json

        for key, value in phase.fixed_overrides.items():
            try:
                dump_overrides_json(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Phase {phase.name!r}: fixed_overrides {key!r}: {exc}") from exc
        for param in phase.search_space.values():
            if isinstance(param, CategoricalParam):
                for choice in param.choices:
                    dump_overrides_json(choice)
        return
    if experiment.override_format not in {"argparse", "hydra"}:
        return

    from phasesweep.runtime.commands import _OverrideValueError, _render_override_value

    override_format: Literal["argparse", "hydra"] = (
        "hydra" if experiment.override_format == "hydra" else "argparse"
    )

    def validation_error(location: str, exc: _OverrideValueError) -> ValueError:
        offender = exc.value
        if isinstance(offender, Mapping):
            hint = (
                f"A mapping has no {override_format} wire form the fingerprint "
                "preserves faithfully. Use the default override_format='yaml_file' "
                "for structured trainer configuration."
            )
        elif isinstance(offender, float):
            hint = (
                "JSON has no representation for non-finite floats; use a finite "
                'value, or quote it in YAML (e.g. "inf") if the trial command '
                "should receive it as text."
            )
        elif isinstance(offender, (list, tuple)):
            hint = "Override lists cannot recursively contain themselves."
        elif isinstance(offender, str) and override_format == "hydra":
            hint = (
                "Hydra cannot pass literal strings containing '${' interpolation syntax or "
                "control characters. Use the default override_format='yaml_file' if the "
                "trainer needs that literal value."
            )
        else:
            hint = (
                "YAML resolves unquoted scalars such as 2024-01-01 or 12:30:00 "
                "into Python date/datetime objects; quote the value in YAML "
                '(e.g. "2024-01-01") to send it as text, or use '
                "the default override_format='yaml_file' if the trainer needs a "
                "structured value."
            )
        where = f" at position {exc.position}" if exc.position else ""
        return ValueError(
            f"Phase {phase.name!r}: override_format={override_format!r} but "
            f"{location} holds a value{where} that {override_format} cannot render "
            f"faithfully (type {type(offender).__name__}): {offender!r}. "
            f"{override_format} override values must have one canonical wire form "
            "the JSON-mode fingerprint preserves faithfully: null, booleans, "
            f"integers, finite floats, strings, and lists of those. {hint}"
        )

    for key, param in phase.search_space.items():
        if not isinstance(param, CategoricalParam):
            continue
        rendered: dict[str, tuple[int, Any]] = {}
        for index, choice in enumerate(param.choices):
            try:
                wire = _render_override_value(choice, override_format)
            except _OverrideValueError as exc:
                raise validation_error(
                    f"categorical search_space key {key!r} choice at index {index}", exc
                ) from exc
            earlier = rendered.get(wire)
            if earlier is not None:
                earlier_index, earlier_choice = earlier
                raise ValueError(
                    f"Phase {phase.name!r}: categorical search_space key {key!r} has "
                    f"choices {earlier_choice!r} at index {earlier_index} and {choice!r} "
                    f"at index {index}, which both render as {wire!r} under "
                    f"override_format={override_format!r}. Distinct search choices must "
                    "produce distinct trainer command values; use different choices or "
                    "the default override_format='yaml_file'."
                )
            rendered[wire] = (index, choice)
    for key, value in phase.fixed_overrides.items():
        try:
            _render_override_value(value, override_format)
        except _OverrideValueError as exc:
            raise validation_error(f"fixed_overrides key {key!r}", exc) from exc


def _format_field_names(template: str) -> set[str]:
    """Return the *real* ``str.format`` field names referenced by ``template``.

    ``string.Formatter().parse()`` walks the template the same way
    ``str.format`` does and reports only true field references, with escaped
    braces handled correctly.

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
    experiment: Experiment,
    phase: Phase,
    inherited_keys: frozenset[str],
) -> None:
    """Preflight the trial command with its inherited keys.

    :param Experiment experiment: Experiment containing the command template.
    :param Phase phase: Phase whose candidate execution is being checked.
    :param frozenset[str] inherited_keys: Keys inherited from direct parents.
    :raises ValueError: The trial command cannot render or reach the trainer.

    Render ``trial_command`` once per phase with placeholder overrides.

    Catches at config-load (not at trial-launch, three minutes into a sweep):

    * Typos like ``{trail_dir}`` or any other unknown ``{placeholder}``.
    * Unbalanced braces (``{trial_dir`` -> ``str.format`` raises ``ValueError``).
    * The primary ``yaml_file`` mode missing ``{config_path}``.
    * Phases using the explicit ``argparse`` compatibility format with
      overrides but a template missing
      ``{overrides}`` — the same silent-no-op failure mode (review v0.5.6 /
      blocker 2).

    Both placeholder checks parse real ``str.format`` field names so that an
    escaped ``{{overrides}}`` (rendered as the literal string ``{overrides}``)
    correctly does *not* count as referencing the placeholder.

    The rendered command is then discarded — this is preflight only.

    Args:
        experiment: The :class:`Experiment` being validated.
        phase: The specific phase whose ``trial_command`` is being rendered.
        inherited_keys: Locked keys inherited from parents — used to decide
            whether ``{overrides}`` is required for scalar/list CLI formats.

    Raises:
        ValueError: Any of the failure modes listed above (typo, unbalanced
            braces, missing required placeholder for the chosen
            ``override_format``).

    """
    # Lazy import to avoid a circular config <-> overrides cycle.
    from phasesweep.runtime.commands import (
        dump_json_file_overrides,
        dump_trial_trainer_config_yaml,
        render_command,
    )

    # Build a synthetic override dict: one value per locked or sampled key.
    # Inherited keys are present in the real call too (they come from parent
    # winners), so the placeholder set must include them or render_command
    # could miss a key the trainer expects.
    overrides: dict[str, Any] = {k: "<inherited>" for k in inherited_keys}
    overrides.update(phase.fixed_overrides)
    overrides.update(_placeholder_values_for(phase.search_space))

    has_overrides = bool(overrides)

    placeholder_dir = Path("__phasesweep_validate_trial_dir__")
    if experiment.override_format == "yaml_file":
        try:
            dump_trial_trainer_config_yaml(experiment.trainer_config, overrides)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Phase {phase.name!r}: trainer_config and composed overrides are invalid — "
                f"{type(exc).__name__}: {exc}."
            ) from exc
    elif experiment.override_format == "json_file":
        dump_json_file_overrides(overrides)
    # Parse the template once so we can both (a) preflight-render below and
    # (b) check that the documented placeholders are actually referenced.
    # Both arms surface the same "failed to render" wrapping for unbalanced
    # braces so the user sees one consistent error message regardless of
    # which check happens to detect the problem first.
    try:
        fields = _format_field_names(experiment.trial_command)
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Phase {phase.name!r}: trial_command failed to render — "
            f"{type(exc).__name__}: {exc}. Check for unbalanced braces."
        ) from exc

    try:
        render_command(
            experiment.trial_command,
            overrides,
            experiment.override_format,
            trial_dir=placeholder_dir,
            trial_id=0,
            phase=phase.name,
            run_name=f"{experiment.experiment}-{phase.name}-validate",
            trainer_config=experiment.trainer_config,
            write_files=False,
        )
    except KeyError as exc:
        # str.format raises KeyError(name) for unknown placeholders.
        bad = exc.args[0] if exc.args else "<unknown>"
        raise ValueError(
            f"Phase {phase.name!r}: trial_command references unknown placeholder "
            f"{{{bad}}}. Supported: {{config_path}} (yaml_file only), "
            f"{{overrides}}, {{overrides_path}} (json_file only), {{trial_dir}}, {{trial_id}}, {{phase}}, {{run_name}}."
        ) from exc
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Phase {phase.name!r}: trial_command failed to render — "
            f"{type(exc).__name__}: {exc}. Check for unbalanced braces."
        ) from exc

    if experiment.override_format == "yaml_file":
        if "config_path" not in fields:
            raise ValueError(
                f"override_format='yaml_file' but phase {phase.name!r} trial_command "
                "does not reference {config_path}. The trainer would never receive "
                "the complete per-trial YAML. Add {config_path} to trial_command, or "
                "select an explicit compatibility override_format."
            )
        if fields & {"overrides", "overrides_path"}:
            raise ValueError(
                f"override_format='yaml_file' but phase {phase.name!r} trial_command "
                "references an overrides-only placeholder. Use {config_path}; it names "
                "the complete trainer YAML after all overrides are composed."
            )
        return

    if "config_path" in fields:
        raise ValueError(
            f"override_format={experiment.override_format!r} but phase {phase.name!r} "
            "trial_command references {config_path}, which is only available in "
            "override_format='yaml_file'."
        )

    if experiment.override_format == "json_file":
        if "overrides" in fields:
            raise ValueError("json_file requires {overrides_path}, not {overrides}.")
        required_placeholder = "overrides_path"
    else:
        if "overrides_path" in fields:
            raise ValueError("{overrides_path} is only available with json_file.")
        required_placeholder = "overrides"

    # When a phase has no overrides at all (no inherited, fixed, or sampled
    # keys), a constant trial_command is legitimate — the user is sweeping
    # the same configuration repeatedly, e.g. for variance estimation.
    if not has_overrides:
        return

    if required_placeholder not in fields:
        raise ValueError(
            f"override_format={experiment.override_format!r} but phase "
            f"{phase.name!r} has inherited, fixed, or sampled overrides "
            f"and trial_command does not reference {{{required_placeholder}}}. All "
            "sampled parameters would be ignored — the trainer would "
            "run with the same hard-coded configuration every trial. "
            f"Add {{{required_placeholder}}} to trial_command, or use the default "
            "override_format='yaml_file' with trainer_config and {config_path}."
        )


def _wandb_query(
    experiment: Experiment, gates: list[Gate], *, gate_location: str = "gates"
) -> WandbQuery | None:
    """Compile existing declarations into one target and one capture per phase.

    :param Experiment experiment: Primary objective and scalar constraints.
    :param list[Gate] gates: Current phase's gates.
    :param str gate_location: Configuration path for actionable conflict diagnostics.
    :return WandbQuery | None: Agreed query, or no remote consumer.
    :raises ValueError: Remote consumers disagree on target or polling policy.
    """
    consumers: list[tuple[str, _WandbSummarySource]] = []
    if isinstance(experiment.metric.extractor, WandbExtractor):
        consumers.append(("metric.extractor", experiment.metric.extractor))
    consumers.extend(
        (f"constraints[{index}].extractor", constraint.extractor)
        for index, constraint in enumerate(experiment.constraints)
        if isinstance(constraint.extractor, WandbExtractor)
    )
    consumers.extend(
        (f"{gate_location}[{index}]", gate)
        for index, gate in enumerate(gates)
        if isinstance(gate, WandbSummaryRequiredGate)
    )
    if not consumers:
        return None
    first_location, first = consumers[0]
    for location, source in consumers[1:]:
        for field in ("base_url", "entity", "project", "poll_seconds", "timeout_seconds"):
            if getattr(source, field) != getattr(first, field):
                raise ValueError(
                    f"{location}.{field} conflicts with {first_location}.{field}; W&B consumers must share one target and polling policy."
                )
    numeric = {source.metric_key for _, source in consumers if isinstance(source, WandbExtractor)}
    presence = {
        key
        for _, source in consumers
        if isinstance(source, WandbSummaryRequiredGate)
        for key in source.keys
    }
    return WandbQuery(
        first,
        tuple(sorted(numeric)),
        tuple(sorted(presence)),
        tuple(
            (constraint.name, constraint.extractor.metric_key)
            for constraint in experiment.constraints
            if isinstance(constraint.extractor, WandbExtractor)
        ),
        tuple(
            (wandb_gate_identity(gate), tuple(gate.keys))
            for gate in gates
            if isinstance(gate, WandbSummaryRequiredGate)
        ),
    )


Config = Experiment
