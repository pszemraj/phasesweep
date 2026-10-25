"""Pydantic config models for experiments, phases, and protocols."""

from __future__ import annotations

import posixpath
import re
import shlex
import string
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    SkipValidation,
    ValidationInfo,
    field_validator,
    model_validator,
)

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
    STOCHASTIC_SAMPLERS,
    CategoricalParam,
    Repeats,
    Sampler,
    SearchParam,
    _placeholder_values_for,
    _validate_sampler_search_space,
    fixed_configuration_target_refusal,
    phase_draws,
    sampler_capability_line,
    sampler_search_space,
    sampler_warnings,
    search_space_warnings,
)
from phasesweep.evidence.models import (
    POSITIONED_EXTRACTOR_KINDS,
    Gate,
    JsonEnvelopeExtractor,
    JsonExtractor,
    LocalExtractor,
    MlflowExtractor,
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
    compose_trainer_input,
    dump_overrides_json,
    flatten_trainer_config,
    render_command,
)
from phasesweep.runtime.files import (
    artifact_root_path,
    canonical_storage_identity,
    file_url_path,
    local_storage_url,
)
from phasesweep.tracking.models import MlflowSettings

OverrideFormat = Literal["yaml_file", "argparse", "hydra", "json_file"]

TRIAL_BOUND_ENV = (
    "PHASESWEEP_TRIAL_DIR",
    "PHASESWEEP_TRIAL_ID",
    "PHASESWEEP_PHASE",
    "PHASESWEEP_RUN_NAME",
    "PHASESWEEP_GENERATION_ID",
    "PHASESWEEP_ATTEMPT_ID",
    "PHASESWEEP_OVERRIDES_SHA256",
    "PHASESWEEP_OBJECTIVE_PATH",
)


class Metric(_Frozen):
    """Primary optimization objective: name, direction, how to extract it, and viability bounds."""

    name: str = Field(default="objective", min_length=1)
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
    """Describe the metric's name, goal, viability bounds, and extractor config.

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
        *(
            [
                "MLflow target: unchecked by validate and dry-run; credentials and experiment access are checked before the phase's first trial."
            ]
            if experiment.mlflow is not None
            else []
        ),
    ]


def experiment_warnings(experiment: Experiment) -> list[str]:
    """List experiment-wide advisory warnings; none blocks a run.

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
    credentials = _credential_like_inherited_names(experiment)
    if credentials:
        warnings.append(
            f"execution.inherit_env lists names that look like credentials: {', '.join(credentials)}. "
            "Each trial records the value of every listed name in plain text in the study "
            "ledger, and rotating one refuses a resume. Credentials belong in "
            "execution.passthrough_env, which is inherited without being recorded or compared."
        )
    if isinstance(experiment.metric.extractor, WandbExtractor | MlflowExtractor):
        reload_sources = _best_model_reload_sources(experiment)
        if reload_sources:
            warnings.append(
                f"load_best_model_at_end=True (set in {', '.join(reload_sources)}): scored "
                "best-checkpoint weights may differ from the terminal training step reported "
                "in remote history. The evaluation_point check compares logged progress and "
                "does not verify checkpoint identity. Use json_envelope to report policy, "
                "checkpoint and the evaluated step; report total training progress in "
                "evaluation.progress and select it with evaluation_axis."
            )
    return warnings


_BEST_MODEL_RELOAD_FLAG = "load_best_model_at_end"
_TRUTHY_WORDS = frozenset({"true", "1", "yes"})


def _best_model_reload_sources(experiment: Experiment) -> list[str]:
    """Name each place that explicitly enables ``load_best_model_at_end``.

    :param Experiment experiment: Loaded experiment.
    :return list[str]: ``trainer_config``, ``fixed_overrides`` and ``trial_command``, in
        that order, for each that enables it; empty when none does.
    """

    def enabled(settings: Mapping[str, JsonValue]) -> bool:
        """Return whether a flat dotted mapping sets the flag to ``True``.

        :param Mapping[str, JsonValue] settings: Flat dotted keys and their values.
        :return bool: Whether any key ending in the flag name holds ``True``.
        """
        return any(
            key.rsplit(".", 1)[-1] == _BEST_MODEL_RELOAD_FLAG and value is True
            for key, value in settings.items()
        )

    sources: list[str] = []
    if enabled(flatten_trainer_config(experiment.trainer_config)):
        sources.append("trainer_config")
    if any(enabled(phase.fixed_overrides) for phase in experiment.phases):
        sources.append("fixed_overrides")
    if _command_enables_flag(experiment.trial_command, _BEST_MODEL_RELOAD_FLAG):
        sources.append("trial_command")
    return sources


def _command_enables_flag(command: str, name: str) -> bool:
    """Whether a trial command switches the boolean option ``name`` on.

    Matches argparse's bare ``--name``, ``--name true`` and ``--name=true``, and Hydra's
    ``[+|++|~]dotted.path.name=true``. A bare flag counts when nothing, another option or a
    ``{placeholder}`` follows it, since a placeholder renders to options rather than a value;
    a falsy value or any other word after it leaves the option off.

    :param str command: The ``trial_command`` template, split like a shell would.
    :param str name: The option's final dotted name, for example ``load_best_model_at_end``.
    :return bool: True when a token explicitly enables the option.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        key, equals, value = token.partition("=")
        if equals:
            if key.lstrip("-+~").rsplit(".", 1)[-1] == name and value.lower() in _TRUTHY_WORDS:
                return True
        elif token == f"--{name}":
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if (
                following is None
                or following.startswith(("-", "{"))
                or following.lower() in _TRUTHY_WORDS
            ):
                return True
    return False


def experiment_disclosures(
    experiment: Experiment,
) -> Iterator[tuple[Phase | None, list[tuple[bool, str]]]]:
    """Yield shared contract and warning lines in their display order.

    The experiment block comes first, followed by one block per configured
    phase. The phase context lets callers insert their own phase headings;
    each line's boolean marks advisory warnings without prescribing a sink.

    :param Experiment experiment: Loaded experiment to describe.
    :return Iterator[tuple[Phase | None, list[tuple[bool, str]]]]: Context and
        ordered warning/content pairs; ``None`` identifies the experiment block.
    """
    yield (
        None,
        [
            *((False, line) for line in experiment_contract_lines(experiment)),
            (
                False,
                f"artifact root: {artifact_root_path(experiment.workdir, experiment.experiment)}",
            ),
            *((True, line) for line in experiment_warnings(experiment)),
        ],
    )
    for phase in experiment.phases:
        yield (
            phase,
            [
                (False, sampler_capability_line(phase, experiment.resolved_storage)),
                *((True, line) for line in sampler_warnings(phase)),
                *((True, line) for line in search_space_warnings(phase)),
            ],
        )


_CREDENTIAL_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def _credential_like_inherited_names(experiment: Experiment) -> list[str]:
    """Name each ``inherit_env`` list entry whose name suggests a credential.

    :param Experiment experiment: Loaded experiment.
    :return list[str]: The listed names containing ``TOKEN``, ``KEY``, ``SECRET``, ``PASSWORD``
        or ``CREDENTIAL`` in any case, sorted, empty under ``none`` and ``all``.
    """
    contract = experiment.execution.inherit_env
    if not isinstance(contract, list):
        return []
    return sorted(name for name in contract if _CREDENTIAL_NAME.search(name))


def _argparse_boolean_flags(experiment: Experiment) -> list[str]:
    """Name each flag ``override_format: argparse`` would render as a boolean value.

    :param Experiment experiment: Loaded experiment.
    :return list[str]: One ``--key (where)`` entry per boolean ``trainer_config`` leaf, fixed
        override, categorical choice or repeat value, empty under any other override format.
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
            f"--{key} (phase {phase.name!r} "
            f"{'repeats' if phase.repeats is not None and key == phase.repeats.key else 'search_space'})"
            for key, param in sampler_search_space(phase).items()
            if isinstance(param, CategoricalParam)
            and any(isinstance(choice, bool) for choice in param.choices)
        )
    return flags


def _observation_line(observation: Observation) -> str:
    """Describe one report-only observation's name and scalar source.

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
    """Describe whether the objective source binds a comparable evaluation position.

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
        pass-through names. A name set under ``env`` that is also passed through appears
        in both groups, flagged as digested.
    """
    execution = experiment.execution
    contract = execution.inherit_env
    configured = set(experiment.env)
    passed_through = set(execution.effective_passthrough_env())
    # A configured value is never inherited and always digested, even under a passthrough name.
    excluded = passed_through - configured
    digested = (
        ", ".join(
            name
            for name in execution.tracked_env_names()
            if name not in passed_through | configured
        )
        or "none"
    )
    if contract == "all":
        inherited = (
            f"inherit_env=all (inherited whole, and digested: {digested}; "
            "other ambient variables are not compared)"
        )
    else:
        label = f"[{', '.join(contract)}]" if isinstance(contract, list) else contract
        inherited = f"inherit_env={label} (inherited when set, and digested: {digested})"
    built_in = sorted(excluded & set(BUILTIN_PASSTHROUGH_ENV))
    listed = sorted(excluded - set(BUILTIN_PASSTHROUGH_ENV))
    if not excluded:
        passthrough = "passthrough_env: none"
    elif not listed:
        passthrough = (
            f"passthrough_env: {', '.join(built_in)} (built in; inherited when set, "
            "excluded from the digest)"
        )
    else:
        names = ", ".join(listed) + (f" and built-in {', '.join(built_in)}" if built_in else "")
        passthrough = f"passthrough_env: {names} (inherited when set, excluded from the digest)"
    shadowed = sorted(passed_through & configured)
    if shadowed:
        passthrough += (
            f" ({', '.join(shadowed)} set under env, so the configured value is digested)"
        )
    parts = [inherited, f"env keys: {', '.join(sorted(configured)) or 'none'}", passthrough]
    if experiment.storage is not None:
        parts.append("a persistent study refuses to resume when any digested value changes")
    return "environment: " + "; ".join(parts)


def _trainer_cwd_line(experiment: Experiment) -> str:
    """Describe the trainer working directory included in study identity.

    An unset execution.cwd is described by its resolution rule, independent of
    the shell running validation.

    :param Experiment experiment: Loaded experiment.
    :return str: One line naming the resolved ``execution.cwd``, or what an unset one follows.
    """
    configured = experiment.execution.cwd
    if configured is None:
        return (
            "trainer cwd: follows the directory phasesweep runs from (the catalog cwd under "
            "MCP) and is part of the study identity; set execution.cwd to pin it"
        )
    return (
        f"trainer cwd: {experiment.execution.resolved_cwd()} "
        "(execution.cwd; part of the study identity)"
    )


class _NamedExtractor(_Frozen):
    """A named scalar read from a trial by an extractor, besides the metric."""

    _label: ClassVar[str]

    name: str = Field(min_length=1)
    extractor: ObjectiveExtractor = Field(discriminator="type")

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
        """Require finite, ordered constraint bounds with at least one endpoint.

        :raises ValueError: Bounds are absent, non-finite, or inverted.
        :return Constraint: Validated constraint.
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
                "remote extractors require every requested key, so a value the trainer "
                "never wrote would fail the trial. Write the value "
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
    # The override-format validators check these values and name the key and the fix;
    # pydantic's own JsonValue check would refuse first with a bare "not a valid JSON value".
    fixed_overrides: dict[str, SkipValidation[JsonValue]] = Field(
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
    cost_key: str | None = Field(
        default=None,
        description=(
            "Numeric sampled parameter whose smaller values cost less. With mean repeats, "
            "advise the cheapest configuration within one standard error of the winner's "
            "mean, preferring better objectives for cost ties. Selection and result reads "
            "report this without changing the winner. Excluded from study fingerprints."
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
            "GPU leasing policy for trial subprocesses. single_per_trial leases "
            "one discovered or configured device per trial. whole_node requires n_jobs=1 "
            "and an explicit CUDA device list on Linux; macOS discovers its MPS device. "
            "none disables phasesweep GPU host locks and CUDA isolation."
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
        """Require a nonempty list of non-negative GPU indices when configured.

        :param list[int] | None value: Configured GPU indices.
        :raises ValueError: The list is empty or contains a negative index.
        :return list[int] | None: Validated indices, unchanged.
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
            explicit device list on Linux, or repeats a device token, or if
            ``gpu_policy='none'`` is combined with a device list or with
            ``n_jobs > 1`` without ``allow_no_gpu_isolation``.
        :return Phase: Self, unchanged.
        """
        if self.gpu_ids is not None and self.gpu_devices is not None:
            raise ValueError("gpu_ids and gpu_devices are mutually exclusive.")
        if self.gpu_policy == "whole_node" and self.n_jobs != 1:
            raise ValueError(
                "gpu_policy='whole_node' requires n_jobs=1 because each trial receives "
                "the full configured CUDA device set or discovered MPS device."
            )
        if (
            self.gpu_policy == "whole_node"
            and self.gpu_ids is None
            and self.gpu_devices is None
            and sys.platform != "darwin"
        ):
            raise ValueError(
                "gpu_policy='whole_node' requires an explicit gpu_ids or gpu_devices "
                "list because its device count defines the trainer's world size."
            )
        if self.gpu_policy == "whole_node":
            # Declared device count must match the pool's distinct normalized tokens.
            declared = self.gpu_ids if self.gpu_ids is not None else self.gpu_devices
            tokens = [str(token).strip() for token in declared or []]
            duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
            if duplicates:
                field = "gpu_ids" if self.gpu_ids is not None else "gpu_devices"
                raise ValueError(
                    "gpu_policy='whole_node' requires unique device tokens; "
                    f"{field} repeats {duplicates}. List each device once, or "
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
            "stored partial result, by a later run or --from-phase, also requires it."
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
        """Require a safe phase name distinct from engine-managed directory names.

        :param str v: Configured phase name.
        :raises ValueError: The name contains disallowed characters or is reserved.
        :return str: Validated phase name.
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
        """Validate fixed and sampled override keys as nonempty dotted paths.

        Each segment must match ``[A-Za-z0-9_-]+`` without whitespace.

        :raises ValueError: An override key has an invalid segment.
        :return Phase: Validated phase.
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
    def _validate_cost_key(self) -> Phase:
        """Require mean repeats and a numeric sampled parameter for the cost advisory.

        :raises ValueError: The cost key is not a numeric search-space key or repeats do not use the mean reducer.
        :return Phase: Self, unchanged.
        """
        if self.cost_key is None:
            return self
        if self.repeats is None or self.repeats.reduce != "mean":
            raise ValueError("cost_key requires repeats with reduce: mean.")
        parameter = self.search_space.get(self.cost_key)
        if parameter is None or (
            isinstance(parameter, CategoricalParam)
            and not all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in parameter.choices
            )
        ):
            raise ValueError(
                f"cost_key {self.cost_key!r} must name a numeric search_space parameter "
                "whose smaller values cost less."
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
        seed_keys = [
            key
            for key in self.search_space
            if key.rsplit(".", 1)[-1] in {"seed", "data_seed", "random_state"}
        ]
        if seed_keys:
            example = seed_keys[0]
            raise ValueError(
                f"Phase {self.name!r}: seed keys cannot be searched: {seed_keys}. A searched "
                "seed lets seed noise win as if it were a configuration. Fix the seed in "
                "fixed_overrides or trainer_config. To compare finalists across seeds, use "
                f"repeats: {{key: {example}, values: [0, 1, 2]}}."
            )
        return self


def _is_environment_name(name: str) -> bool:
    """Check whether a name is nonempty and contains neither ``=`` nor NUL.

    :param str name: Candidate environment variable name.
    :return bool: ``True`` when the name is nonempty and holds no ``=`` or NUL.
    """
    return bool(name) and "=" not in name and "\x00" not in name


# Base inherited variables support shell/interpreter startup and temporary files.
# GPU visibility requires configured env.CUDA_VISIBLE_DEVICES or explicit inheritance.
BASE_INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME", "TZ")

# Built-in pass-through names are inherited when set and never digested.
# PATH remains tracked because it selects executable tools.
BUILTIN_PASSTHROUGH_ENV = ("TMPDIR",)


class ExecutionContext(_Frozen):
    """Configure trainer working-directory and environment inheritance.

    The default inherits BASE_INHERITED_ENV plus configured env entries.
    Inheritance and pass-through settings determine the environment cohort
    recorded with the study.
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
            "semantic. In persistent studies, the base variables, the listed "
            "names and `env`, except `passthrough_env` and the built-in "
            "`TMPDIR`, must match the existing trial cohort before another "
            "trial is allocated, so a different `PATH` is refused. Under 'all' "
            "the cohort covers only the base variables, and every other "
            "ambient variable is inherited without being compared. Each trial "
            "records the compared values in plain text, so a credential "
            "belongs in `passthrough_env`, not in a list."
        ),
    )
    passthrough_env: list[str] = Field(
        default_factory=list,
        description=(
            "Ambient variables that may change without changing a persistent "
            "study's semantic environment cohort, such as rotating credentials "
            "and proxies. These names are inherited in addition to a narrowed "
            "`inherit_env` contract, but their values are excluded from the "
            "cohort digest. They add to the built-in `TMPDIR`, a scratch "
            "location that is always inherited when set and never digested, so "
            "a sandboxed and an ordinary shell agree. The effective sorted "
            "names remain part of the config fingerprint."
        ),
    )
    record_env: bool = Field(
        default=False,
        description=(
            "Write the exact environment the trainer process receives, including "
            "per-attempt, GPU and run-binding variables, into each attempt directory "
            "as `environment.json`, with the cohort digest and the inherit_env and "
            "passthrough_env contracts. Off by default "
            "because it holds the whole environment, credentials "
            "among it; the file is created owner-only (0600), unlike the rest "
            "of the trial directory. Regardless of this flag, every trial "
            "records the environment's SHA-256 digest and the compared "
            "name-value pairs in the study ledger, which leave out the "
            "pass-through values."
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
            ``=`` or NUL, a name list repeats a name, the two lists overlap, or an
            ``inherit_env`` list names a built-in pass-through variable.
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
            built_in = sorted(set(self.inherit_env) & set(BUILTIN_PASSTHROUGH_ENV))
            if built_in:
                raise ValueError(
                    f"inherit_env must not name {built_in!r}: it is always inherited when set "
                    "and never digested, so remove it."
                )
            overlap = sorted(set(self.inherit_env) & set(self.passthrough_env))
            if overlap:
                raise ValueError(
                    "inherit_env and passthrough_env must not overlap; classify each "
                    f"ambient variable once: {overlap!r}"
                )
        return self

    def resolved_cwd(self) -> Path | None:
        """Resolve the configured trainer directory through home and symlink expansion.

        :return Path | None: Absolute directory, or ``None`` to inherit the invocation directory.
        """
        return None if self.cwd is None else Path(self.cwd).expanduser().resolve()

    def effective_passthrough_env(self) -> tuple[str, ...]:
        """List every pass-through name: the built-in ones plus ``passthrough_env``.

        :return tuple[str, ...]: Sorted unique names inherited when set and never digested.
        """
        return tuple(sorted({*BUILTIN_PASSTHROUGH_ENV, *self.passthrough_env}))

    def inherit_env_contract(self) -> str | list[str]:
        """Return the environment inheritance contract in canonical persisted order.

        :return str | list[str]: ``"all"``, ``"none"``, or the sorted name list.
        """
        return sorted(self.inherit_env) if isinstance(self.inherit_env, list) else self.inherit_env

    def tracked_env_names(self) -> tuple[str, ...]:
        """List the ambient variable names the environment cohort covers.

        :return tuple[str, ...]: The base set plus any listed names. ``inherit_env: all`` adds
            none, so an ambient variable outside this set is inherited without being compared.
        """
        if isinstance(self.inherit_env, list):
            return (*BASE_INHERITED_ENV, *self.inherit_env)
        return BASE_INHERITED_ENV

    def inherited_env_names(self) -> tuple[str, ...] | None:
        """List the ambient variable names the ``inherit_env`` contract admits.

        :return tuple[str, ...] | None: The base set plus any listed names, or None
            when ``inherit_env`` is ``all`` and every ambient variable is admitted.
        """
        return None if self.inherit_env == "all" else self.tracked_env_names()


def _validate_mlflow_published_key(name: str, *, prefix: str, source: str) -> None:
    """Mirror MLflow's metric and param key validation for a published config name.

    :param str name: Configured result name or override key.
    :param str prefix: PhaseSweep publication prefix.
    :param str source: Config location that supplies the name.
    :raises ValueError: The published key has invalid characters, length or path form.
    """
    key = prefix + name
    if not re.fullmatch(r"[/\w.\- :]*", key):
        rule = "characters outside alphanumerics, underscores, periods, dashes, slashes, colons and spaces"
    elif len(key) > 250:
        rule = f"length {len(key)}, exceeding the 250-character limit"
    elif posixpath.normpath(key) != key or key == "." or key.startswith(("..", "/")):
        rule = "path form that is not a normalized relative path"
    else:
        return
    raise ValueError(f"{source} {name!r} produces MLflow key {key!r} with {rule}; rename it.")


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
    # Values skip pydantic's check for the reason given at Phase.fixed_overrides.
    trainer_config: dict[str, SkipValidation[JsonValue]] = Field(
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
        return local_storage_url(
            artifact_root_path(self.workdir, self.experiment) / "study.journal"
        )

    def phase(self, name: str) -> Phase:
        """Look up a configured phase by its unique name.

        :return Phase: The matching configured phase.
        :raises StopIteration: No phase has that name.
        """
        return next(phase for phase in self.phases if phase.name == name)

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
        """Reject unlaunchable entries and variables PhaseSweep sets per attempt.

        :param dict[str, str] value: Configured trainer environment overrides.
        :raises ValueError: A name is empty or contains ``=`` or NUL, or a value
            contains NUL, or a name is reserved for per-attempt bindings.
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
        reserved = [name for name in value if name in TRIAL_BOUND_ENV]
        if reserved:
            raise ValueError(
                f"env cannot set {reserved!r}: PhaseSweep sets these variables for "
                "every attempt. Remove them from env."
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
                f"Persistent storage (storage: {self.storage}) requires a nonempty provenance mapping that identifies "
                "the data and dependency revisions used by this experiment; each generation "
                "records the trainer's git commit on its own."
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
        """Require the trainer base and phase overrides to render in the selected format.

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
        """Require an experiment name matching ``[A-Za-z0-9_-]+``.

        :param str v: Configured experiment name.
        :raises ValueError: The name is empty or contains disallowed characters.
        :return str: Validated experiment name.
        """
        return _validate_safe_name("Experiment", v)

    @model_validator(mode="after")
    def _validate_mlflow_published_keys(self) -> Experiment:
        """Require config-authored published keys to satisfy MLflow's client rules.

        :raises ValueError: A constraint, observation or override produces an invalid key.
        :return Experiment: Self, unchanged.
        """
        if self.mlflow is None:
            return self
        for constraint in self.constraints:
            _validate_mlflow_published_key(
                constraint.name, prefix="phasesweep.constraint.", source="constraint"
            )
        for observation in self.observe:
            _validate_mlflow_published_key(
                observation.name, prefix="phasesweep.observed.", source="observation"
            )
        for phase in self.phases:
            for source, keys in (
                ("fixed_overrides", phase.fixed_overrides),
                ("search_space", phase.search_space),
            ):
                for key in keys:
                    _validate_mlflow_published_key(
                        key, prefix="phasesweep.override.", source=f"phase {phase.name!r} {source}"
                    )
            if phase.repeats is not None:
                _validate_mlflow_published_key(
                    phase.repeats.key,
                    prefix="phasesweep.override.",
                    source=f"phase {phase.name!r} repeats",
                )
        return self

    @model_validator(mode="after")
    def _validate_phase_graph(self) -> Experiment:
        """Validate ordered phase composition and per-phase semantics.

        Each exported override key carries its producing phase as its origin.
        Re-exporting preserves that origin, making a same-origin diamond valid.
        Different origins for one inherited key require a child fixed override;
        a fixed child key establishes a new origin. A key is locked
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

            _validate_sampler_search_space(phase, persistent=self.resolved_storage is not None)
            _validate_persistent_sampler_seed(self.resolved_storage, phase)
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


def _validate_persistent_sampler_seed(storage: str | None, phase: Phase) -> None:
    """Require a seed for persistent phases that draw from stochastic samplers.

    Seeded random draws reproduce at any n_jobs; TPE and CMA-ES require n_jobs=1
    and the same trial outcomes. Grid, fixed phases, and in-memory storage are exempt.

    :param str | None storage: Experiment-level storage URL, or ``None``.
    :param Phase phase: Phase whose sampler contract is checked.
    :raises ValueError: ``storage`` is persistent and the phase draws from a
        stochastic sampler without an explicit ``seed``.
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


def _validate_in_memory_fixed_target(storage: str | None, phase: Phase) -> None:
    """Require one trial for a fixed configuration on in-memory storage.

    A fresh study has no failures to retry. Persistent retry targets are checked
    against the existing study at launch.

    :param str | None storage: Experiment-level storage URL, or ``None``.
    :param Phase phase: Phase whose trial target is checked.
    :raises ValueError: ``storage`` is in-memory and the phase, a fixed
        configuration, asks for more than one trial.
    """
    if storage is not None:
        return
    refusal = fixed_configuration_target_refusal(phase, finished=0, measured=0)
    if refusal is not None:
        raise ValueError(refusal)


def _validate_cli_override_values(experiment: Experiment, phase: Phase) -> None:
    """Validate CLI fixed values, categorical choices, and wire-form uniqueness.

    The launch renderer accepts null, booleans, integers, finite floats, strings,
    and nested lists/tuples. Structured configuration uses yaml_file.

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
                raise ValueError(
                    f"Phase {phase.name!r}: {axis} choice at index {index}: {exc}"
                ) from exc
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
            raise ValueError(f"Phase {phase.name!r}: fixed_overrides key {key!r}: {exc}") from exc


def _format_field_names(template: str) -> set[str]:
    """Collect bare str.format fields, rejecting format specs and conversions.

    Escaped braces are excluded. Attribute and index accessors remain intact
    for the caller to reject.

    :param str template: A ``str.format``-style template string.
    :raises ValueError: The template has unbalanced braces, format specs or conversions.
    :return set[str]: Field names actually referenced (escaped ``{{...}}`` excluded).
    """
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{exc}. Check for unbalanced braces.") from exc
    fields: set[str] = set()
    formatted: list[str] = []
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        fields.add(field_name)
        if format_spec or conversion:
            suffix = f"!{conversion}" if conversion else ""
            if format_spec:
                suffix += f":{format_spec}"
            formatted.append(f"{{{field_name}{suffix}}}")
    if formatted:
        raise ValueError(
            "trial_command placeholders take no format spec or conversion; found "
            f"{', '.join(formatted)}. Format values in the shell instead, for example "
            "$(printf '%05d' {trial_id})."
        )
    return fields


def _validate_trial_command_template(
    experiment: Experiment,
    phase: Phase,
    inherited_keys: frozenset[str],
) -> None:
    """Preflight command placeholders and composed trainer input for one phase.

    Inherited, fixed, sampled, and repeat keys supply placeholder values.
    Only actual format fields count; escaped braces do not deliver input.

    :param Experiment experiment: Experiment containing the command and format.
    :param Phase phase: Phase whose command is rendered.
    :param frozenset[str] inherited_keys: Override keys supplied by direct parents.
    :raises ValueError: The template has unknown fields, accessors, format specs, conversions,
        malformed braces, a missing input placeholder, or trainer configuration that cannot be composed.
    """
    overrides: dict[str, Any] = {k: "<inherited>" for k in inherited_keys}
    overrides.update(phase.fixed_overrides)
    overrides.update(_placeholder_values_for(sampler_search_space(phase)))

    has_overrides = bool(overrides) or (
        experiment.override_format != "yaml_file" and bool(experiment.trainer_config)
    )

    placeholder_dir = Path("__phasesweep_validate_trial_dir__")
    try:
        compose_trainer_input(experiment.trainer_config, overrides, experiment.override_format)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Phase {phase.name!r}: trainer_config and composed overrides are invalid — "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    try:
        fields = _format_field_names(experiment.trial_command)
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Phase {phase.name!r}: trial_command failed to render — {type(exc).__name__}: {exc}"
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

    # An input-free fixed phase may use a constant command.
    if not has_overrides:
        return

    if required_placeholder not in fields:
        raise ValueError(
            f"override_format={experiment.override_format!r} but phase "
            f"{phase.name!r} has inherited, fixed, sampled or repeat overrides, or a "
            f"trainer_config and trial_command does not reference "
            f"{{{required_placeholder}}}. All inherited, fixed, sampled and repeat parameters and the "
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
