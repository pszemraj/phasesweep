"""Pydantic config models for experiments, phases, and protocols."""

from __future__ import annotations

import string
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from phasesweep.config.common import (
    ConfigFloat,
    ConfigInt,
    _find_prefix_collisions,
    _Frozen,
    _require_finite,
    _validate_optional_bounds,
    _validate_override_key,
    _validate_safe_name,
    anchor_relative_path,
)
from phasesweep.config.search import (
    NON_RESUMABLE_SAMPLERS,
    STOCHASTIC_SAMPLERS,
    CategoricalParam,
    Repeats,
    Sampler,
    SearchParam,
    _placeholder_values_for,
    _validate_sampler_search_space,
    fixed_configuration_target_refusal,
    phase_draws,
    sampler_search_space,
)
from phasesweep.evidence.models import (
    POSITIONED_EXTRACTOR_KINDS,
    Extractor,
    Gate,
    JsonEnvelopeExtractor,
    JsonExtractor,
    LocalExtractor,
    ObjectiveExtractor,
    WandbExtractor,
    WandbQuery,
    WandbSummaryRequiredGate,
    _WandbSummarySource,
    compose_wandb_environment,
    objective_evidence_assurance,
    wandb_gate_identity,
)
from phasesweep.runtime.commands import (
    _OverrideValueError,
    _render_override_value,
    dump_json_file_overrides,
    dump_overrides_json,
    dump_trial_trainer_config_yaml,
    flatten_trainer_config,
    merged_cli_overrides,
    render_command,
)
from phasesweep.runtime.files import (
    canonical_storage_identity,
    file_url_path,
    local_storage_url,
)
from phasesweep.tracking.models import MlflowSettings

OverrideFormat = Literal["yaml_file", "argparse", "hydra", "json_file"]


class Metric(_Frozen):
    """Primary optimization objective: name, direction, how to extract it, and viability bounds."""

    name: str = "objective"
    goal: Literal["minimize", "maximize"] = "minimize"
    extractor: ObjectiveExtractor = Field(discriminator="type")
    max: ConfigFloat | None = None
    min: ConfigFloat | None = None

    @model_validator(mode="after")
    def _validate_viability_bounds(self) -> Metric:
        """Reject non-finite or inverted viability bounds when any is set.

        :return Metric: Self, unchanged.
        """
        if self.min is not None or self.max is not None:
            _validate_optional_bounds(
                label=f"Metric {self.name!r}",
                min_value=self.min,
                max_value=self.max,
            )
        return self

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
    """Describe the scalar source, within-trial selection, and between-trial goal.

    :param Metric metric: Configured optimization metric.
    :return str: One line naming the metric, its goal, its viability bounds when set, and
        its extractor config.
    """
    bounds = metric.model_dump(mode="json", include={"min", "max"}, exclude_none=True)
    return (
        f"metric={metric.name!r} goal={metric.goal} "
        + (f"bounds={bounds} " if bounds else "")
        + f"extractor={metric.extractor.model_dump(mode='json', exclude_none=True)}"
    )


def experiment_contract_lines(experiment: Experiment) -> list[str]:
    """List the experiment-level contract lines that ``validate`` and ``run --dry-run`` print.

    :param Experiment experiment: Loaded experiment.
    :return list[str]: One line per experiment-wide contract, in print order.
    """
    return [
        _metric_scoring_line(experiment.metric),
        _evaluation_point_line(experiment.metric),
        *(_observation_line(observation) for observation in experiment.observe),
        _environment_contract_line(experiment),
        _trainer_cwd_line(experiment),
    ]


def experiment_warnings(experiment: Experiment) -> list[str]:
    """List the advisory warnings about the experiment as a whole.

    Shared by ``validate`` and ``run --dry-run`` beside
    :func:`phasesweep.config.search.sampler_warnings`, which covers one phase. A
    warning never blocks a run.

    :param Experiment experiment: Loaded experiment.
    :return list[str]: One sentence per warning, empty when nothing is amiss.
    """
    warnings: list[str] = []
    flags = _argparse_boolean_flags(experiment)
    if flags:
        warnings.append(
            "argparse renders booleans as --key=true/--key=false; a store_true flag rejects a "
            "value, so set such flags in trial_command or have the trainer accept a value. "
            f"Rendered that way: {', '.join(flags)}."
        )
    return warnings


def _argparse_boolean_flags(experiment: Experiment) -> list[str]:
    """Name each flag ``override_format: argparse`` would render as a boolean value.

    :param Experiment experiment: Loaded experiment.
    :return list[str]: One ``--key (where)`` entry per boolean ``trainer_config`` leaf, fixed
        override or categorical choice, empty under any other override format.
    """
    if experiment.override_format != "argparse":
        return []
    flags = [
        f"--{key} (trainer_config)"
        for key, value in flatten_trainer_config(experiment.trainer_config).items()
        if isinstance(value, bool)
    ]
    for phase in experiment.phases:
        flags.extend(
            f"--{key} (phase {phase.name!r} fixed_overrides)"
            for key, value in phase.fixed_overrides.items()
            if isinstance(value, bool)
        )
        flags.extend(
            f"--{key} (phase {phase.name!r} search_space)"
            for key, param in sampler_search_space(phase).items()
            if isinstance(param, CategoricalParam)
            and any(isinstance(choice, bool) for choice in param.choices)
        )
    return flags


def _observation_line(observation: Observation) -> str:
    """Name one report-only observation and what its extractor reads.

    The line lets a mistyped path or key show before a run, since a value the
    trainer never wrote is only logged as a warning once a trial finishes.

    :param Observation observation: One configured ``observe`` entry.
    :return str: One line with the observation's name, its source and a report-only marker.
    """
    extractor = observation.extractor
    if isinstance(extractor, JsonExtractor):
        source = f"json {extractor.path} key {extractor.key}"
    elif isinstance(extractor, JsonEnvelopeExtractor):
        source = (
            f"json_envelope {extractor.path} objective {extractor.objective_name} "
            f"split {extractor.split} policy {extractor.policy}"
        )
    else:
        source = (
            f"log_regex {extractor.file} pattern {extractor.pattern!r} select {extractor.select}"
        )
    return f"observe {observation.name}: {source} (report-only)"


def _evaluation_point_line(metric: Metric) -> str:
    """Say whether selection can check that ranked objectives share an evaluation position.

    The check runs only for extractors whose evidence names a position, so the
    other kinds would otherwise report ``n/a`` on every winner without saying why.

    :param Metric metric: Configured optimization metric.
    :return str: One line naming the support, the axis and what binds the position.
    """
    extractor = metric.extractor
    prefix = "evaluation_point check:"
    if extractor.type not in POSITIONED_EXTRACTOR_KINDS:
        return (
            f"{prefix} n/a ({extractor.type} objectives record no evaluation position; "
            "json_envelope via report_objective, wandb with evaluation_axis, or mlflow would)"
        )
    if isinstance(extractor, JsonEnvelopeExtractor):
        source = (
            "the envelope; any other axis from evaluation.progress"
            if extractor.evaluation_axis == "step"
            else "the envelope's evaluation.progress"
        )
        return f"{prefix} supported (axis {extractor.evaluation_axis!r} from {source})"
    if isinstance(extractor, WandbExtractor):
        return (
            f"{prefix} supported (axis {extractor.evaluation_axis!r} located in W&B run history; "
            "advisory, n/a when no position is found)"
        )
    return f"{prefix} supported (position is the step of the selected observation)"


def _environment_contract_line(experiment: Experiment) -> str:
    """Describe the trainer environment contract and which of its values pin a study.

    The line is static: it reads the config and never ``os.environ``, so ``validate``
    prints the same text from any shell.

    :param Experiment experiment: Loaded experiment.
    :return str: One line naming the digested inherited names, the ``env`` keys and the
        pass-through names, each name in exactly one group.
    """
    execution = experiment.execution
    contract = execution.inherit_env
    admitted = execution.inherited_env_names()
    configured = set(experiment.env)
    passed_through = set(execution.passthrough_env)
    # A configured value is never inherited and always digested, even under a passthrough name.
    excluded = passed_through - configured
    if admitted is None:
        inherited = "inherit_env=all (every ambient variable is inherited and digested" + (
            ", except the passthrough_env names)" if excluded else ")"
        )
    else:
        label = f"[{', '.join(contract)}]" if isinstance(contract, list) else contract
        digested = (
            ", ".join(name for name in admitted if name not in passed_through | configured)
            or "none"
        )
        inherited = f"inherit_env={label} (inherited when set, and digested: {digested})"
    passthrough = (
        f"passthrough_env: {', '.join(sorted(excluded))} (inherited when set, excluded from the digest)"
        if excluded
        else "passthrough_env: none"
    )
    shadowed = sorted(passed_through & configured)
    if shadowed:
        passthrough += (
            f" ({', '.join(shadowed)} set under env, so the configured value is digested)"
        )
    parts = [inherited, f"env keys: {', '.join(sorted(configured)) or 'none'}", passthrough]
    if experiment.storage is not None:
        parts.append(
            "a persistent study refuses to resume when any value outside passthrough_env changes"
        )
    return "environment: " + "; ".join(parts)


def _trainer_cwd_line(experiment: Experiment) -> str:
    """Describe the trainer's working directory, which is part of every study's identity.

    An unset ``execution.cwd`` is the one input the config file does not pin:
    the line says what it follows instead of printing a directory that
    depends on the shell ``validate`` happened to run from.

    :param Experiment experiment: Loaded experiment.
    :return str: One line naming the resolved ``execution.cwd``, or what an unset one follows.
    """
    configured = experiment.execution.cwd
    if configured is None:
        return (
            "trainer cwd: follows the directory phasesweep runs from (the catalog cwd under "
            "MCP) and is part of the study identity; set execution.cwd to pin it"
        )
    resolved = Path(configured).expanduser().resolve()
    return f"trainer cwd: {resolved} (execution.cwd; part of the study identity)"


class _NamedExtractor(_Frozen):
    """A named scalar read from a trial by an extractor, besides the metric."""

    _label: ClassVar[str]

    name: str
    extractor: Extractor = Field(discriminator="type")

    @model_validator(mode="after")
    def _validate_evaluation_axis_unused(self) -> Self:
        """Reject a non-default ``evaluation_axis`` that nothing would read.

        Only the metric extractor's axis feeds selection's ``evaluation_point``
        check, so a named extractor that sets one is a misplaced setting.

        :raises ValueError: The extractor sets ``evaluation_axis``.
        :return Self: Self, unchanged.
        """
        extractor = self.extractor
        if not isinstance(extractor, JsonEnvelopeExtractor | WandbExtractor):
            return self
        if extractor.evaluation_axis != type(extractor).model_fields["evaluation_axis"].default:
            raise ValueError(
                f"{self._label} {self.name!r} sets evaluation_axis, which only the metric "
                "extractor's evaluation_point check reads; remove it."
            )
        return self


class Constraint(_NamedExtractor):
    """A scalar bound that trials must satisfy to be considered feasible."""

    _label: ClassVar[str] = "Constraint"

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


class Observation(_NamedExtractor):
    """A scalar recorded beside the metric that never affects selection or identity."""

    _label: ClassVar[str] = "Observation"

    extractor: LocalExtractor = Field(discriminator="type")

    @field_validator("extractor", mode="before")
    @classmethod
    def _refuse_remote_extractor(cls, value: object, info: ValidationInfo) -> object:
        """Refuse a W&B or MLflow extractor with the reason and the local alternative.

        :param object value: Raw or already built extractor.
        :param ValidationInfo info: Validation info carrying the observation's name.
        :raises ValueError: The extractor is a remote kind.
        :return object: The value, unchanged.
        """
        kind = value.get("type") if isinstance(value, Mapping) else getattr(value, "type", None)
        if kind in ("wandb", "mlflow"):
            raise ValueError(
                f"Observation {info.data.get('name')!r} cannot use a {kind} extractor: "
                "remote extractors wait for every requested key, so a value the trainer "
                "never wrote would stall the trial until the poll timeout. Write the value "
                "with report_objective(..., extra=...) and read it with a local json extractor."
            )
        return value


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
    repeats: Repeats | None = Field(
        default=None,
        description=(
            "Run every configuration once per repeat value (typically seeds) and rank "
            "configurations by a reducer over their complete groups."
        ),
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
            # 2-GPU run and delivers a 1-GPU run under a 2-GPU study identity.
            # Tokens are stripped defensively: ``_gpu_devices_non_empty_tokens``
            # already strips ``gpu_devices``, but this validator must not depend
            # on that to see ``[' GPU-a ', 'GPU-a']`` as the duplicate pair it is.
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
    timeout_seconds_per_trial: ConfigFloat | None = Field(default=259200.0, ge=0)
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
            "partial phase winner and persist completion metadata. Reusing a "
            "stored partial result, by rerun or --from-phase, also requires it."
        ),
    )
    allow_partial_grid: bool = Field(
        default=False,
        description=(
            "Grid phases must run the full matrix by default. Set true to permit "
            "n_trials smaller than the grid cardinality. Valid only with sampler.type grid."
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
            The same name, unchanged.

        Raises:
            ValueError: Any character is disallowed, or the name collides with
                engine-managed records.

        """
        v = _validate_safe_name("Phase", v)
        if v.casefold() == "attempts":
            raise ValueError("Phase name 'attempts' is reserved for the runtime recovery registry.")
        if v.casefold() == "generations":
            raise ValueError(
                "Phase name 'generations' is reserved for immutable generation records."
            )
        if v.casefold() == "tracking":
            raise ValueError("Phase name 'tracking' is reserved for MLflow tracking records.")
        return v

    @model_validator(mode="after")
    def _validate_override_key_syntax(self) -> Phase:
        r"""Reject malformed override keys before they hit the override renderer.

        CLI override rendering quotes values shell-safely, but a malformed
        *key* like ``""``, ``"."``, ``"a..b"``, or ``" lr"`` would either
        produce broken commands (``-- 1``, ``=value``, ``..a=value``) or silently
        treat surface noise (whitespace) as part of the key.

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
        if self.repeats is not None:
            _validate_override_key(self.repeats.key, label=f"phase {self.name!r} repeats.key")
        return self

    @model_validator(mode="after")
    def _validate_repeat_key_is_its_own(self) -> Phase:
        """Keep the repeat key out of the keys the phase fixes or searches.

        :raises ValueError: ``repeats.key`` is also a ``search_space`` or ``fixed_overrides`` key.
        :return Phase: Self, unchanged.
        """
        if self.repeats is None:
            return self
        key = self.repeats.key
        for field, keys in (
            ("search_space", self.search_space),
            ("fixed_overrides", self.fixed_overrides),
        ):
            if key in keys:
                raise ValueError(
                    f"Phase {self.name!r}: repeats.key {key!r} is also a {field} key. The "
                    "repeat axis supplies that key's value on every trial; remove it from "
                    f"{field}."
                )
        return self

    @model_validator(mode="after")
    def _validate_timeouts_and_seed_keys(self) -> Phase:
        """Require bounded trials unless explicitly waived and refuse searched seeds.

        :raises ValueError: If a timeout is invalid, or a search-space key is a seed.
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
        seed_keys = [key for key in self.search_space if key == "seed" or key.endswith(".seed")]
        if seed_keys:
            example = seed_keys[0]
            raise ValueError(
                f"Phase {self.name!r}: seed keys cannot be searched: {seed_keys}. A searched "
                "seed lets seed noise win as if it were a configuration. Move the seed to "
                f"fixed_overrides, or measure seed noise with repeats: {{key: {example}, "
                "values: [0, 1, 2]}."
            )
        return self


def _is_environment_name(name: str) -> bool:
    """Return whether ``name`` can be the name of a process environment variable.

    ``os.execve`` raises ``ValueError`` for an empty name, a name containing
    ``=``, and any name or value containing NUL, and no ambient variable can
    carry such a name.

    :param str name: Candidate environment variable name.
    :return bool: ``True`` when the name is nonempty and holds no ``=`` or NUL.
    """
    return bool(name) and "=" not in name and "\x00" not in name


# Minimal environment base used when the execution contract narrows
# inheritance below "all": enough for a shell + interpreter to start and
# write temp files, nothing that can carry semantic configuration.
#
# CUDA_VISIBLE_DEVICES is deliberately NOT here. It decides whether a trial
# trains on CPU or GPU, so admitting the ambient value would let two operators
# running the identical config from differently-exported shells write
# CPU-trained and GPU-trained evaluations into one study under one semantic
# fingerprint — exactly the unfingerprinted ambient input the narrowed
# contracts exist to exclude (the fingerprint covers the contract's mode/names
# and configured ``env``, never ambient values). Bind trainer visibility with
# a top-level ``env.CUDA_VISIBLE_DEVICES`` entry, which is fingerprinted and
# also drives GPU pool discovery, or name the variable in an ``inherit_env``
# list to opt its ambient value in explicitly.
BASE_INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME", "TZ")

# Inherited names whose values stay out of the environment digest unless the config
# says otherwise. TMPDIR is a scratch location, not a scientific input, and a
# sandboxed shell sets its own. PATH stays digested: it decides which interpreter
# and tools run.
DEFAULT_PASSTHROUGH_ENV = ("TMPDIR",)


class ExecutionContext(_Frozen):
    """Explicit trainer execution-context contract.

    The ambient environment and the current working directory are two unbounded
    implicit semantic inputs that never contribute to study identity, so one
    persistent study could silently mix evaluations from different trainers,
    datasets, or credentials. This contract makes both explicit: the default
    passes the trainer only :data:`BASE_INHERITED_ENV` plus configured ``env``
    values, ``inherit_env`` widens that on purpose, and the choices bind into
    the semantic fingerprint.
    """

    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for every trainer subprocess. A relative path "
            "resolves against the config file's directory (same rule as "
            "workdir), or against the process cwd for an Experiment built in "
            "Python. When set, the RESOLVED path joins the semantic "
            "fingerprint, so the same study can never mix trainers reached "
            "through different working directories. When unset, trainers run "
            "in the invocation cwd (the catalog entry's cwd under MCP) and that "
            "effective resolved directory joins the fingerprint."
        ),
    )
    inherit_env: Literal["all", "none"] | list[str] = Field(
        default="none",
        description=(
            "Which ambient environment variables the trainer inherits. 'none' "
            "(default) passes a minimal documented base (PATH, HOME, LANG, "
            "LC_ALL, TMPDIR, USER, LOGNAME, TZ) and drops everything else, "
            "such as PYTHONPATH, virtualenv settings, cache paths and "
            "credentials. A list inherits the base plus exactly the named "
            "variables. 'all' opts in to the full ambient environment. "
            "Configured `env` values are always applied on top and are always "
            "semantic. In persistent studies, inherited values outside "
            "`passthrough_env` (by default only `TMPDIR`) must match the "
            "existing trial cohort before another trial is allocated, so a "
            "different `PATH` is refused."
        ),
    )
    passthrough_env: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PASSTHROUGH_ENV),
        description=(
            "Ambient variables that may change without changing a persistent "
            "study's semantic environment cohort: rotating credentials, proxies "
            "and scratch locations. These names are inherited in addition to a "
            "narrowed `inherit_env` contract, but their values are excluded from "
            "the cohort digest. Their sorted names and classification remain "
            "part of the config fingerprint. The default is `TMPDIR`, a scratch "
            "location that differs between a sandboxed and an ordinary shell "
            "without changing any result; a list written here replaces the "
            "default, and `[]` digests `TMPDIR` like every other inherited "
            "variable."
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

    @field_validator("cwd")
    @classmethod
    def _anchor_cwd(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Reject an empty trainer working directory and anchor a relative one.

        :param str | None value: Configured trainer working directory.
        :param ValidationInfo info: Validation info carrying the config directory, if any.
        :raises ValueError: The path is the empty string.
        :return str | None: The path, resolved against the config file's
            directory when it is relative and the config came from a file.
        """
        if value is None:
            return None
        if value == "":
            raise ValueError(
                "execution.cwd must be a nonempty path; write '.' for the config file's "
                "directory, or omit it to run trainers in the directory phasesweep is "
                "started from."
            )
        return anchor_relative_path(value, info)

    @model_validator(mode="after")
    def _validate_inherit_names(self) -> ExecutionContext:
        """Reject inherited-variable names that no environment variable can have.

        :raises ValueError: A listed variable name is empty, padded, or contains
            ``=`` or NUL, or a name list repeats a name or the two lists overlap.
        :return ExecutionContext: Self, unchanged.
        """
        named_contracts = {"passthrough_env": self.passthrough_env}
        if isinstance(self.inherit_env, list):
            named_contracts["inherit_env"] = self.inherit_env
        for label, names in named_contracts.items():
            bad = [name for name in names if name != name.strip() or not _is_environment_name(name)]
            if bad:
                raise ValueError(
                    f"{label} names must be nonempty and unpadded, with no '=' or NUL "
                    f"character: {bad!r}"
                )
            if len(set(names)) != len(names):
                raise ValueError(f"{label} names must be unique.")
        if isinstance(self.inherit_env, list):
            overlap = sorted(set(self.inherit_env) & set(self.passthrough_env))
            if overlap:
                hint = ""
                if "passthrough_env" not in self.model_fields_set and set(overlap) == set(
                    DEFAULT_PASSTHROUGH_ENV
                ):
                    hint = (
                        f" (passthrough_env defaults to {list(DEFAULT_PASSTHROUGH_ENV)!r}, "
                        "which keeps it out of the environment digest; to digest it, set "
                        "`passthrough_env: []`)"
                    )
                raise ValueError(
                    "inherit_env and passthrough_env must not overlap; classify each "
                    f"ambient variable once: {overlap!r}{hint}"
                )
        return self

    def inherited_env_names(self) -> tuple[str, ...] | None:
        """List the ambient variable names the ``inherit_env`` contract admits.

        :return tuple[str, ...] | None: The base set plus any listed names, or None
            when ``inherit_env`` is ``all`` and every ambient variable is admitted.
        """
        if isinstance(self.inherit_env, list):
            return (*BASE_INHERITED_ENV, *self.inherit_env)
        return None if self.inherit_env == "all" else BASE_INHERITED_ENV


class Experiment(_Frozen):
    """Top-level experiment: trial command, metric, constraints, and ordered phases."""

    experiment: str
    storage: str | None = Field(
        default=None,
        description=(
            "Local Optuna journal storage URL, auto for study.journal inside the "
            "experiment artifact namespace, or null for non-resumable in-memory runs. "
            "Use journal:///path.journal for an explicit, resumable, parallel-safe study; "
            "a relative journal path resolves like workdir."
        ),
    )
    workdir: str = Field(
        default="./runs",
        # The default is relative too, so it must be anchored like a written value.
        validate_default=True,
        description=(
            "Where per-trial directories are created. A relative path resolves against "
            "the config file's directory, or against the process cwd for an Experiment "
            "built in Python."
        ),
    )
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
    mlflow: MlflowSettings | None = None
    metric: Metric
    constraints: list[Constraint] = Field(default_factory=list)
    observe: list[Observation] = Field(
        default_factory=list,
        description=(
            "Report-only scalars read from each accepted trial with local extractors. They "
            "are recorded and published, never ranked, never fingerprinted, and never fail a trial."
        ),
    )
    phases: list[Phase] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    execution: ExecutionContext = Field(
        default_factory=ExecutionContext,
        description=(
            "Explicit trainer execution context: working directory and "
            "environment-inheritance contract."
        ),
    )
    timeout_seconds_per_run: ConfigFloat | None = Field(default=None, ge=0)

    @property
    def resolved_storage(self) -> str | None:
        """Resolve auto storage within this experiment's artifact namespace.

        :return str | None: Absolute auto journal URL, or the explicit storage
            unchanged.
        """
        if self.storage != "auto":
            return self.storage
        root = Path(self.workdir).expanduser().resolve() / self.experiment
        return local_storage_url(root / "study.journal")

    def phases_from(self, from_phase: str | None) -> Iterator[tuple[int, Phase]]:
        """Yield the phases a run resuming at ``from_phase`` reaches, in order.

        :param str | None from_phase: Optional resume point; phases before it
            are skipped. ``None`` reaches every phase from the start.
        :return Iterator[tuple[int, Phase]]: Each reached phase paired with
            its declaration index. Empty if ``from_phase`` names no phase.
        """
        reached = from_phase is None
        for index, phase in enumerate(self.phases):
            reached = reached or phase.name == from_phase
            if reached:
                yield index, phase

    def sampled_locked_keys(self, phase_name: str) -> frozenset[str]:
        """Return the inherited keys a phase may not search.

        A key is locked when any inherited origin sampled it; a key every
        inherited origin only fixed may be searched.

        :param str phase_name: Name of a phase of this experiment.
        :raises KeyError: ``phase_name`` names no phase.
        :return frozenset[str]: Override keys that an inherited phase sampled.
        """
        origins_by_phase: dict[str, dict[str, str]] = {}
        for phase in self.phases:
            inherited = _inherited_origins(phase, origins_by_phase)
            if phase.name == phase_name:
                return frozenset(_sampled_locked_origins(inherited))
            origins_by_phase[phase.name] = _resolved_origins(phase, inherited)
        raise KeyError(phase_name)

    @field_validator("storage")
    @classmethod
    def _storage_is_local(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Allow only null, auto, and local journal storage that names a plain file.

        :param str | None value: Configured Optuna storage locator.
        :param ValidationInfo info: Validation info carrying the config directory, if any.
        :raises ValueError: The locator is not null, ``"auto"``, or a
            ``journal:///`` URL with the exact lowercase scheme, or the URL
            carries a query or fragment, or its path is empty, or a relative
            path anchored to the config directory would contain ``?`` or ``#``.
        :return str | None: The locator; a relative journal path in a config
            loaded from a file becomes an absolute four-slash URL under the
            config file's directory.
        """
        if value is None or value == "auto":
            return value
        if value.split(":", 1)[0] != "journal":
            raise ValueError(
                "storage must be null, auto, or a journal:/// URL with the lowercase "
                f"scheme journal; got {value!r}. PhaseSweep keeps its persistent ledger "
                "in an Optuna journal file: use `storage: auto` for "
                "<workdir>/<experiment>/study.journal, "
                "`storage: journal:///path/to/study.journal` for an explicit file, or "
                "`storage: null` for an in-memory dry run."
            )
        if "?" in value or "#" in value:
            raise ValueError(
                "storage journal URLs cannot carry a query or fragment, because everything "
                f"from the first ? or # on is dropped from the file path; got {value!r}. "
                "Remove it. To keep the journal under a directory whose name contains ? or #, "
                "use `storage: auto` with that workdir."
            )
        raw_path = file_url_path(value)
        if raw_path == "":
            raise ValueError(
                f"storage journal URL names no file; got {value!r}. Write "
                "`storage: journal:///path/to/study.journal`."
            )
        anchored_path = anchor_relative_path(raw_path, info)
        if anchored_path != raw_path:
            # The anchored path is absolute, so the URL takes the four-slash form.
            value = f"journal:///{anchored_path}"
            if "?" in value or "#" in value:
                raise ValueError(
                    "storage journal path is relative, and the config file's directory it "
                    "resolves against contains ? or #, which a journal URL cannot carry; got "
                    f"{anchored_path!r}. Use `storage: auto`, which keeps the journal under "
                    "workdir whatever its directory names contain."
                )
        canonical_storage_identity(value)
        return value

    @field_validator("workdir")
    @classmethod
    def _anchor_workdir(cls, value: str, info: ValidationInfo) -> str:
        """Reject an empty artifact-root directory and anchor a relative one.

        :param str value: Configured per-trial directory root.
        :param ValidationInfo info: Validation info carrying the config directory, if any.
        :raises ValueError: The path is the empty string.
        :return str: The path, resolved against the config file's directory when
            it is relative and the config came from a file.
        """
        if value == "":
            raise ValueError(
                "workdir must be a nonempty path; write '.' for the config file's "
                "directory, or omit workdir for ./runs."
            )
        return anchor_relative_path(value, info)

    @field_validator("env")
    @classmethod
    def _env_is_launchable(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject trainer environment entries that ``os.execve`` cannot pass.

        :param dict[str, str] value: Configured trainer environment overrides.
        :raises ValueError: A name is empty or contains ``=`` or NUL, or a value
            contains NUL.
        :return dict[str, str]: The validated mapping, unchanged.
        """
        bad_names = [name for name in value if not _is_environment_name(name)]
        if bad_names:
            raise ValueError(
                "env names must be nonempty and contain no '=' or NUL character, because "
                f"the trainer launch cannot pass them: {bad_names!r}. Rename or remove them."
            )
        bad_values = [name for name, text in value.items() if "\x00" in text]
        if bad_values:
            raise ValueError(
                "env values cannot contain NUL, because the trainer launch cannot pass "
                f"them; fix the value of {bad_values!r}."
            )
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
        if self.resolved_storage is not None and not self.provenance:
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
    def _validate_trainer_config_wire(self) -> Experiment:
        """Require the embedded trainer config to have a wire form in the selected format.

        ``yaml_file`` validates the base together with each phase's overrides.
        The other formats carry the base too, so its keys and leaves must
        render on their wire before any phase composes into it.

        :raises ValueError: ``trainer_config`` holds a key or value that the
            selected ``override_format`` cannot carry.
        :return Experiment: Self, unchanged.
        """
        fmt = self.override_format
        if fmt == "yaml_file" or not self.trainer_config:
            return self
        if fmt == "json_file":
            try:
                dump_overrides_json(self.trainer_config)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"trainer_config cannot be written as JSON for override_format='json_file': {exc}"
                ) from exc
            return self
        cli_format: Literal["argparse", "hydra"] = "hydra" if fmt == "hydra" else "argparse"
        try:
            flat = flatten_trainer_config(self.trainer_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trainer_config cannot be flattened for override_format={fmt!r}: {exc}"
            ) from exc
        for key, value in flat.items():
            try:
                _render_override_value(value, cli_format)
            except _OverrideValueError as exc:
                raise ValueError(f"trainer_config key {key!r}: {exc}") from exc
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
        a fixed child key deliberately establishes a new origin. A key is locked
        for a child only when some inherited origin sampled it; keys the
        inherited phases only fixed may be searched.

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

            inherited_origins = _inherited_origins(phase, origins_by_phase)
            repeat_keys = set() if phase.repeats is None else {phase.repeats.key}

            # A repeat value replaces whatever the parents agreed or disagreed on.
            unresolved = {
                key: origins
                for key, origins in inherited_origins.items()
                if len(origins) > 1 and key not in phase.fixed_overrides and key not in repeat_keys
            }
            if unresolved:
                details = ", ".join(
                    f"{key!r} from {sorted(origins)}" for key, origins in sorted(unresolved.items())
                )
                raise ValueError(
                    f"Phase {phase.name!r} inherits conflicting key(s) from multiple parents: "
                    f"{details}. Resolve explicitly with phase.fixed_overrides or remove one inherit."
                )

            locked = _sampled_locked_origins(inherited_origins)
            inherited_collisions = set(locked) & set(phase.search_space)
            if inherited_collisions:
                details = ", ".join(
                    f"{key} from {', '.join(repr(name) for name in sorted(locked[key]))}"
                    for key in sorted(inherited_collisions)
                )
                raise ValueError(
                    f"Phase {phase.name!r} re-samples key(s) {sorted(inherited_collisions)} "
                    f"that an inherited phase already sampled ({details}, possibly "
                    "transitively). A winner's sampled value cannot be searched again; "
                    "remove the key from search_space or drop the inherit. Keys a parent "
                    "only fixed may be searched."
                )

            resolved_origins = _resolved_origins(phase, inherited_origins)
            prefix_collisions = _find_prefix_collisions(set(resolved_origins) | repeat_keys)
            if prefix_collisions:
                pairs = ", ".join(f"{a!r} ⊏ {b!r}" for a, b in prefix_collisions)
                raise ValueError(
                    f"Phase {phase.name!r} has dotted-key namespace collision(s) "
                    f"across inherited and local overrides: {pairs}. "
                    "A key and a sub-key cannot both be overridden across the inheritance chain."
                )

            _validate_sampler_search_space(phase)
            _validate_sampler_resumability(self.resolved_storage, phase)
            _validate_in_memory_fixed_target(self.resolved_storage, phase)
            _validate_cli_override_values(self, phase)
            _validate_trial_command_template(self, phase, frozenset(inherited_origins))
            from phasesweep.evidence.mlflow import compile_query

            compile_query(self, phase.gates)
            query = _wandb_query(self, phase.gates, gate_location=f"phases[{len(seen)}].gates")
            if query is not None:
                compose_wandb_environment(query, self.env, self.env)

            origins_by_phase[phase.name] = resolved_origins
            seen[phase.name] = phase
            seen_casefolded[casefolded_name] = phase.name

        names = [
            self.metric.name,
            *(c.name for c in self.constraints),
            *(o.name for o in self.observe),
        ]
        if len(set(names)) != len(names):
            raise ValueError("Metric, constraint and observe names must all be distinct.")
        return self


def _inherited_origins(
    phase: Phase, origins_by_phase: Mapping[str, Mapping[str, str]]
) -> dict[str, set[str]]:
    """Collect the origins of every override key a phase inherits.

    :param Phase phase: Phase whose ``inherits`` list names the parents.
    :param Mapping[str, Mapping[str, str]] origins_by_phase: Resolved key origins
        (``<phase>:fixed`` or ``<phase>:sampled``) of each earlier phase.
    :return dict[str, set[str]]: Origins per inherited key, one entry per
        distinct producing phase.
    """
    inherited: dict[str, set[str]] = {}
    for parent in phase.inherits:
        for key, origin in origins_by_phase[parent].items():
            inherited.setdefault(key, set()).add(origin)
    return inherited


def _resolved_origins(phase: Phase, inherited: Mapping[str, set[str]]) -> dict[str, str]:
    """Resolve the origin of every override key a phase exports.

    A key the phase fixes or searches takes the phase as its new origin; any
    other inherited key keeps its single origin. The repeat key is never
    exported: its value differs across the members a winner stands on, so the
    winner's overrides drop it, together with any inherited value it replaced.

    :param Phase phase: Phase whose fixed, sampled and repeat keys apply.
    :param Mapping[str, set[str]] inherited: Origins per inherited key.
    :return dict[str, str]: Origin per exported key.
    """
    repeat_key = None if phase.repeats is None else phase.repeats.key
    resolved = {
        key: next(iter(origins))
        for key, origins in inherited.items()
        if key not in phase.fixed_overrides and key != repeat_key
    }
    resolved.update({key: f"{phase.name}:fixed" for key in phase.fixed_overrides})
    resolved.update({key: f"{phase.name}:sampled" for key in phase.search_space})
    return resolved


def _sampled_locked_origins(inherited: Mapping[str, set[str]]) -> dict[str, set[str]]:
    """Pick the inherited keys that some origin sampled.

    :param Mapping[str, set[str]] inherited: Origins per inherited key.
    :return dict[str, set[str]]: For each locked key, the names of the phases
        that sampled it.
    """
    locked: dict[str, set[str]] = {}
    for key, origins in inherited.items():
        sampled_by = {
            origin.rpartition(":")[0] for origin in origins if origin.endswith(":sampled")
        }
        if sampled_by:
            locked[key] = sampled_by
    return locked


def _validate_sampler_resumability(storage: str | None, phase: Phase) -> None:
    """Require an explicit seed, and a non-resumable acknowledgement, on persistent storage.

    A persistent study outlives the process that created it, which makes two
    sampler properties operator decisions rather than defaults:

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
    stored grid assignments. A phase with an empty ``search_space`` is exempt
    too, whatever its sampler: it draws nothing, so there is no sequence to
    reproduce and no sampler state to lose. In-memory storage (``storage: null``,
    the only in-memory spelling) is exempt entirely — there is no durable study
    to reproduce or resume, so the ``sampler`` block stays optional and the
    ``tpe`` default remains fine.

    :param str | None storage: Experiment-level storage URL, or ``None``.
    :param Phase phase: Phase whose sampler contract is checked.
    :raises ValueError: ``storage`` is persistent and the phase draws from a
        stochastic sampler without an explicit ``seed``, or from a
        non-resumable one without ``acknowledge_nonresumable: true``.
    """
    if storage is None or not phase_draws(phase):
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


def _validate_in_memory_fixed_target(storage: str | None, phase: Phase) -> None:
    """Require ``n_trials: 1`` for a fixed configuration that has no durable study.

    Every invocation on in-memory storage (``storage: null``) starts a fresh
    study, so a fixed configuration there has nothing to retry and
    :func:`phasesweep.config.search.fixed_configuration_target_refusal` would
    demand one trial at every launch; the config says so at load. A persistent
    study may hold failed attempts for a larger ``n_trials`` to replace, which
    only a launch can see, so persistent storage is checked there.

    :param str | None storage: Experiment-level storage URL, or ``None``.
    :param Phase phase: Phase whose trial target is checked.
    :raises ValueError: ``storage`` is in-memory and the phase, a fixed
        configuration, asks for more than one trial.
    """
    if storage is not None:
        return
    refusal = fixed_configuration_target_refusal(phase, finished=0, completed=0)
    if refusal is not None:
        raise ValueError(refusal)


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
        for key, value in phase.fixed_overrides.items():
            try:
                dump_overrides_json(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Phase {phase.name!r}: fixed_overrides {key!r}: {exc}") from exc
        for param in sampler_search_space(phase).values():
            if isinstance(param, CategoricalParam):
                for choice in param.choices:
                    dump_overrides_json(choice)
        return
    if experiment.override_format not in {"argparse", "hydra"}:
        return

    override_format: Literal["argparse", "hydra"] = (
        "hydra" if experiment.override_format == "hydra" else "argparse"
    )

    def validation_error(location: str, exc: _OverrideValueError) -> ValueError:
        """Build the load-time error for one value the CLI renderer rejected.

        :param str location: Where the offending value sits in the phase config.
        :param _OverrideValueError exc: Renderer rejection carrying the offending
            value and its nested position.
        :return ValueError: Error naming the phase, location, and value, with a
            hint for the offending value's kind.
        """
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

    repeat_key = None if phase.repeats is None else phase.repeats.key
    for key, param in sampler_search_space(phase).items():
        if not isinstance(param, CategoricalParam):
            continue
        axis = (
            f"repeats.key {key!r}" if key == repeat_key else f"categorical search_space key {key!r}"
        )
        rendered: dict[str, tuple[int, Any]] = {}
        for index, choice in enumerate(param.choices):
            try:
                wire = _render_override_value(choice, override_format)
            except _OverrideValueError as exc:
                raise validation_error(f"{axis} choice at index {index}", exc) from exc
            earlier = rendered.get(wire)
            if earlier is not None:
                earlier_index, earlier_choice = earlier
                raise ValueError(
                    f"Phase {phase.name!r}: {axis} has "
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
    braces handled correctly. Fields nested inside a format spec, as in
    ``{trial_dir:{phase}}``, are included. Names are returned as written, so a
    field with an attribute or index accessor (``{m.name}``, ``{a[0]}``) keeps
    its accessor and the caller can refuse it.

    :param str template: A ``str.format``-style template string.
    :raises ValueError: The template has unbalanced braces.
    :return set[str]: Field names actually referenced (escaped ``{{...}}`` excluded).
    """
    fields: set[str] = set()
    for _literal, field_name, format_spec, _conversion in string.Formatter().parse(template):
        if field_name is None:
            continue
        fields.add(field_name)
        if format_spec:
            fields |= _format_field_names(format_spec)
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
    * Attribute or index access on a placeholder (``{trial_dir.name}``,
      ``{trial_dir[0]}``), which ``str.format`` would apply to the rendered string.
    * Unbalanced braces (``{trial_dir`` -> ``str.format`` raises ``ValueError``).
    * The primary ``yaml_file`` mode missing ``{config_path}``.
    * Phases using a non-YAML format with overrides or a ``trainer_config`` but
      a template missing the format's placeholder, the same silent-no-op
      failure mode.
    * A ``trainer_config`` and composed overrides that cannot be merged for
      the selected format.

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
        ValueError: Any of the failure modes listed above (typo, attribute or
            index access, unbalanced braces, missing required placeholder for
            the chosen ``override_format``).

    """
    # Build a synthetic override dict: one value per locked or sampled key.
    # Inherited keys are present in the real call too (they come from parent
    # winners), so the placeholder set must include them or render_command
    # could miss a key the trainer expects.
    overrides: dict[str, Any] = {k: "<inherited>" for k in inherited_keys}
    overrides.update(phase.fixed_overrides)
    overrides.update(_placeholder_values_for(sampler_search_space(phase)))

    # Every format delivers the base trainer_config, so a nonempty one needs the
    # format's placeholder just as an override does.
    has_overrides = bool(overrides) or (
        experiment.override_format != "yaml_file" and bool(experiment.trainer_config)
    )

    placeholder_dir = Path("__phasesweep_validate_trial_dir__")
    try:
        if experiment.override_format == "yaml_file":
            dump_trial_trainer_config_yaml(experiment.trainer_config, overrides)
        elif experiment.override_format == "json_file":
            dump_json_file_overrides(overrides, trainer_config=experiment.trainer_config)
        else:
            merged_cli_overrides(experiment.trainer_config, overrides)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Phase {phase.name!r}: trainer_config and composed overrides are invalid — "
            f"{type(exc).__name__}: {exc}."
        ) from exc
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

    accessed = sorted(name for name in fields if "." in name or "[" in name)
    if accessed:
        shown = ", ".join(f"{{{name}}}" for name in accessed)
        raise ValueError(
            f"Phase {phase.name!r}: trial_command placeholders cannot use attribute or "
            f"index access; found {shown}. Write the bare placeholder, such as "
            "{trial_dir}, and do any path or string manipulation in the command itself."
        )

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

    # When a phase has no overrides at all (no inherited, fixed, sampled, or
    # repeat keys) and no trainer_config to deliver, a constant trial_command
    # is legitimate: the phase is one fixed run of a trainer that takes no
    # input. Only a failed attempt is run again; repeated runs that all count
    # carry a repeat key.
    if not has_overrides:
        return

    if required_placeholder not in fields:
        raise ValueError(
            f"override_format={experiment.override_format!r} but phase "
            f"{phase.name!r} has inherited, fixed, or sampled overrides or a "
            f"trainer_config and trial_command does not reference "
            f"{{{required_placeholder}}}. All sampled parameters and the "
            "trainer_config would be ignored — the trainer would "
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
            (wandb_gate_identity(gate), tuple(sorted(set(gate.keys))))
            for gate in gates
            if isinstance(gate, WandbSummaryRequiredGate)
        ),
    )
