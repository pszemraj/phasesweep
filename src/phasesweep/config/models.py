"""Pydantic config models for experiments, phases, suites, and protocols."""

from __future__ import annotations

import copy
import math
import string
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from phasesweep.config.common import (
    _find_prefix_collisions,
    _Frozen,
    _require_finite,
    _validate_optional_bounds,
    _validate_override_key,
    _validate_safe_name,
)
from phasesweep.config.search import (
    Sampler,
    SearchParam,
    _placeholder_values_for,
    _validate_sampler_search_space,
)
from phasesweep.evidence.models import Extractor, Gate, ObjectiveExtractor
from phasesweep.runtime.files import storage_backend


class Metric(_Frozen):
    """Primary optimization objective: name, direction, and how to extract it."""

    name: str = "objective"
    goal: Literal["minimize", "maximize"] = "minimize"
    extractor: ObjectiveExtractor = Field(discriminator="type")


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
        _validate_optional_bounds(
            label=f"Constraint {self.name!r}",
            min_value=self.min,
            max_value=self.max,
        )
        return self


class Contract(_Frozen):
    """Named fixed-comparison contract shared across phases or studies."""

    fixed_overrides: dict[str, Any] = Field(default_factory=dict)
    gates: list[Gate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_override_key_syntax(self) -> Contract:
        """Reject malformed contract override keys.

        :raises ValueError: If any fixed override key has invalid syntax.
        :return Contract: Self, unchanged.
        """
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
        """Require a finite promotion delta.

        :raises ValueError: If ``min_delta`` is not finite.
        :return Promotion: Self, unchanged.
        """
        _require_finite("promotion.min_delta", self.min_delta)
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
            "Parallel trials within this phase. When gpu_ids or gpu_devices is "
            "also set, each trial gets exclusive access to one CUDA-visible device."
        ),
    )
    gpu_policy: GpuPolicy = Field(
        default="single_per_trial",
        description=(
            "CUDA visibility policy for trial subprocesses. single_per_trial leases "
            "one visible CUDA token per trial. whole_node requires n_jobs=1 and "
            "leases every configured or detected visible token for the trial. none "
            "disables phasesweep CUDA isolation and GPU host locks."
        ),
    )
    gpu_ids: list[int] | None = Field(
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
            The same value, unchanged. Raises ``ValueError`` if any element
            is negative.

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
        """Reject ambiguous explicit GPU isolation settings."""
        if self.gpu_ids is not None and self.gpu_devices is not None:
            raise ValueError("gpu_ids and gpu_devices are mutually exclusive.")
        if self.gpu_policy == "whole_node" and self.n_jobs != 1:
            raise ValueError(
                "gpu_policy='whole_node' requires n_jobs=1 because each trial receives "
                "the full configured CUDA-visible device set."
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
            "When GPU isolation cannot be established, phasesweep fails by default "
            "for parallel sweeps. Set True for intentional CPU-only or "
            "externally-isolated runs."
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
        return _validate_safe_name("Phase", v)

    @model_validator(mode="after")
    def _validate_override_key_syntax(self) -> Phase:
        r"""Reject malformed override keys before they hit the override renderer.

        argparse/Hydra rendering quotes values shell-safely, but a malformed
        *key* like ``""``, ``"."``, ``"a..b"``, or ``" lr"`` would either
        produce broken commands (``-- 1``, ``=value``, ``..a=value``) or silently
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
    override_format: Literal["argparse", "hydra", "json_file"] = "argparse"
    metric: Metric
    constraints: list[Constraint] = Field(default_factory=list)
    contracts: dict[str, Contract] = Field(default_factory=dict)
    phases: list[Phase] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds_per_run: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_run_timeout(self) -> Experiment:
        """Reject non-finite run wallclock guards.

        :raises ValueError: If ``timeout_seconds_per_run`` is non-finite.
        :return Experiment: Self, unchanged.
        """
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
        names like ``../../etc/evil`` could escape the configured lock-file
        directory and the experiment::phase study name.

        Returns:
            The validated experiment name, unchanged. Raises ``ValueError``
            on any disallowed character.

        """
        return _validate_safe_name("Experiment", v)

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
            if phase.promotion is not None and phase.promotion.min_delta_vs not in seen:
                raise ValueError(
                    f"Phase {phase.name!r} promotion references "
                    f"{phase.promotion.min_delta_vs!r}, which is not a prior phase."
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

            # A scalar key and one of its dotted subkeys cannot coexist in any
            # supported render format, whether both are local or one is inherited.
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


def _validate_storage_policy(storage: str | None, phase: Phase) -> None:
    """Reject SQLite storage with parallel ``n_jobs`` (review v0.5.2 / blocker 6).

    SQLite serializes writers; concurrent Optuna trials cause ``database is locked``
    errors. Earlier versions auto-rewrote ``sqlite:///x.db`` to a JournalStorage path,
    but that fragmented study identity behind a single URL — the same config could
    point at two different studies depending on ``n_jobs``. Now we require the user
    to pick the scheme explicitly.

    The check uses :func:`phasesweep.runtime.files.storage_backend` so all
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
    * Phases with ``override_format: argparse`` or ``hydra`` and any inherited,
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
            whether ``{overrides}`` is required for argparse/hydra formats.

    Raises:
        ValueError: Any of the failure modes listed above (typo, unbalanced
            braces, missing required placeholder for the chosen
            ``override_format``).

    """
    # Lazy import to avoid a circular config <-> overrides cycle.
    from phasesweep.runtime.commands import render_command

    # Build a synthetic override dict: one value per locked or sampled key.
    # Inherited keys are present in the real call too (they come from parent
    # winners), so the placeholder set must include them or render_command
    # could miss a key the trainer expects.
    overrides: dict[str, Any] = {k: "<inherited>" for k in inherited_keys}
    for contract_name in phase.contracts:
        overrides.update(experiment.contracts[contract_name].fixed_overrides)
    overrides.update(phase.fixed_overrides)
    overrides.update(_placeholder_values_for(phase.search_space))

    has_overrides = bool(overrides)

    placeholder_dir = Path("__phasesweep_validate_trial_dir__")
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
            write_files=False,
        )
    except KeyError as exc:
        # str.format raises KeyError(name) for unknown placeholders.
        bad = exc.args[0] if exc.args else "<unknown>"
        raise ValueError(
            f"Phase {phase.name!r}: trial_command references unknown placeholder "
            f"{{{bad}}}. Supported: {{overrides}}, {{overrides_path}} (json_file "
            f"only), {{trial_dir}}, {{trial_id}}, {{phase}}, {{run_name}}."
        ) from exc
    except (ValueError, TypeError, IndexError) as exc:
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
            "to trial_command, or switch to override_format='argparse' / "
            "'hydra' (which use the {overrides} placeholder)."
        )
    if experiment.override_format in ("argparse", "hydra") and "overrides" not in fields:
        raise ValueError(
            f"override_format={experiment.override_format!r} but phase "
            f"{phase.name!r} has inherited, fixed, or sampled overrides "
            "and trial_command does not reference {overrides}. All "
            "sampled parameters would be ignored — the trainer would "
            "run with the same hard-coded configuration every trial. "
            f"Add {{overrides}} to trial_command, or switch to "
            "override_format='json_file' (which uses {overrides_path})."
        )


class SuiteDefaults(_Frozen):
    """Shared defaults applied to every study in a suite."""

    storage: str | None = None
    workdir: str = "./runs"
    trial_command: str | None = None
    override_format: Literal["argparse", "hydra", "json_file"] = "argparse"
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
    override_format: Literal["argparse", "hydra", "json_file"] | None = None
    metric: Metric | None = None
    constraints: list[Constraint] | None = None
    contracts: dict[str, Contract] = Field(default_factory=dict)
    phases: list[Phase] = Field(min_length=1)
    env: dict[str, str] | None = None
    timeout_seconds_per_run: float | None = Field(default=None, ge=0)
    promotion: Promotion | None = None

    @field_validator("name")
    @classmethod
    def _study_name_is_safe(cls, value: str) -> str:
        """Validate study names used as experiment-name suffixes and path components.

        :param str value: Candidate study name.
        :raises ValueError: If ``value`` is not a safe name.
        :return str: The validated study name, unchanged.
        """
        return _validate_safe_name("Study", value)


class Suite(_Frozen):
    """Suite of independent or dependency-ordered phase-chain studies."""

    suite: str
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    studies: list[StudySpec] = Field(min_length=1)

    @field_validator("suite")
    @classmethod
    def _suite_name_is_safe(cls, value: str) -> str:
        """Validate suite names used as output path and experiment-name prefixes.

        :param str value: Candidate suite name.
        :raises ValueError: If ``value`` is not a safe name.
        :return str: The validated suite name, unchanged.
        """
        return _validate_safe_name("Suite", value)

    @model_validator(mode="after")
    def _validate_study_graph(self) -> Suite:
        """Require unique, prior-only study dependencies.

        :raises ValueError: If study names duplicate or dependencies point forward.
        :return Suite: Self, unchanged.
        """
        seen: set[str] = set()
        phases_by_study: dict[str, set[str]] = {}
        for study in self.studies:
            if study.name in seen:
                raise ValueError(f"Duplicate study name {study.name!r}.")
            for dep in study.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"Study {study.name!r} depends_on {dep!r}, which is not a prior study."
                    )
            if study.promotion is not None:
                selector = study.promotion.min_delta_vs
                baseline_study, _, baseline_phase = selector.partition(".")
                if baseline_study not in seen:
                    raise ValueError(
                        f"Study {study.name!r} promotion references {baseline_study!r}, "
                        "which is not a prior study."
                    )
                if baseline_phase and baseline_phase not in phases_by_study[baseline_study]:
                    raise ValueError(
                        f"Study {study.name!r} promotion references missing baseline phase "
                        f"{selector!r}."
                    )
            seen.add(study.name)
            phases_by_study[study.name] = {phase.name for phase in study.phases}
        return self

    def experiment_for_study(self, study: StudySpec) -> Experiment:
        """Compile a suite study into a normal :class:`Experiment`.

        :param StudySpec study: Suite study to compile.
        :raises ValueError: If a required study field is missing from both the
            study and suite defaults.
        :return Experiment: Concrete experiment config for ``study``.
        """
        defaults = self.defaults

        def value(name: str, *, required: bool = False) -> Any:
            """Resolve a study field, falling back to suite defaults.

            :param str name: Field name to resolve.
            :param bool required: Whether ``None`` is invalid after fallback.
            :raises ValueError: If ``required`` is true and the resolved value is ``None``.
            :return Any: Deep-copied resolved value.
            """
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
