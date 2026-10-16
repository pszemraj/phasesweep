"""Search-space and sampler config models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from phasesweep.config.common import (
    ConfigFloat,
    ConfigInt,
    StrictJsonScalar,
    _Frozen,
    _require_finite,
)

if TYPE_CHECKING:
    from phasesweep.config.models import Phase


_OPTUNA_EXACT_INT_LIMIT = 1 << 53
_MAX_GRID_CARDINALITY = 4_096


def _float_step_count(low: float, high: float, step: float) -> int:
    """Return the exact decimal-string step count, rejecting off-lattice bounds.

    :param float low: Inclusive lower bound.
    :param float high: Inclusive upper bound.
    :param float step: Positive finite spacing.
    :raises ValueError: The configured high is not on the step lattice.
    :return int: Number of steps between the bounds, without enumeration.
    """
    ratio = (Fraction(str(high)) - Fraction(str(low))) / Fraction(str(step))
    if ratio.denominator != 1:
        raise ValueError(
            "float param: (high - low) / step must be an integer. "
            f"Got low={low}, high={high}, step={step} (ratio={ratio}). "
            "Pick a step that evenly divides the interval."
        )
    return ratio.numerator


def _validate_optuna_integer_domain(low: int, high: int) -> None:
    """Reject integer bounds that cannot round-trip through Optuna numeric storage.

    :param int low: Inclusive lower bound.
    :param int high: Inclusive upper bound.
    :raises ValueError: If the bounds are reversed or exceed the exact integer
        range of Optuna's floating-point internal representation.
    """
    if low > high:
        raise ValueError(f"int param: low ({low}) > high ({high})")
    if low < -_OPTUNA_EXACT_INT_LIMIT or high > _OPTUNA_EXACT_INT_LIMIT:
        raise ValueError(
            "numeric integer search bounds must stay within "
            f"[{-_OPTUNA_EXACT_INT_LIMIT}, {_OPTUNA_EXACT_INT_LIMIT}] so every "
            "value can round-trip exactly through Optuna; use categorical choices "
            "for larger integers"
        )


class FloatParam(_Frozen):
    """Continuous float search parameter with optional log-scale and step."""

    type: Literal["float"]
    low: ConfigFloat
    high: ConfigFloat
    log: bool = False
    step: ConfigFloat | None = None

    @model_validator(mode="after")
    def _validate(self) -> FloatParam:
        """Validate finite, ordered float bounds and compatible log/step settings.

        :raises ValueError: Bounds are non-finite or reversed, log has a nonpositive low
            or a step, or the step is nonpositive, non-finite, or misses the high bound.
        :return FloatParam: Validated parameter.
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
        if self.step is not None:
            _float_step_count(self.low, self.high, self.step)
        return self


class IntParam(_Frozen):
    """Integer search parameter with optional log-scale and step."""

    type: Literal["int"]
    low: ConfigInt
    high: ConfigInt
    log: bool = False
    step: ConfigInt = 1

    @model_validator(mode="after")
    def _validate(self) -> IntParam:
        """Validate exact integer bounds and compatible log/step settings.

        :raises ValueError: Bounds cannot round-trip through Optuna, the step is nonpositive
            or misses the high bound, or log has a nonpositive low or a step other than one.
        :return IntParam: Validated parameter.
        """
        _validate_optuna_integer_domain(self.low, self.high)
        if self.log and self.low <= 0:
            raise ValueError("log-scale int param requires low > 0")
        if self.step <= 0:
            raise ValueError("int param step must be > 0")
        if self.log and self.step != 1:
            raise ValueError("int param cannot use log=true with step != 1")
        if (self.high - self.low) % self.step:
            raise ValueError(
                f"int param: step={self.step} must evenly divide "
                f"high-low={self.high - self.low} so the configured high={self.high} is included."
            )
        return self


class CategoricalParam(_Frozen):
    """Categorical search parameter with an explicit list of choices."""

    type: Literal["categorical"]
    choices: list[Any] = Field(min_length=1)

    @field_validator("choices")
    @classmethod
    def _choices_are_unique_optuna_scalars(cls, choices: list[Any]) -> list[Any]:
        """Require finite Optuna scalars distinct under Python equality.

        Optuna persists the first equal choice, so ``1``, ``1.0``, and ``True``
        cannot coexist as choices. Duplicates also alter random sampling weights.

        :param choices: Candidate choices in configured order.
        :raises ValueError: A choice is unsupported, non-finite, or equal to an earlier choice.
        :return: The choices, unchanged.
        """
        _validate_optuna_choices(choices, label="categorical choices")
        return choices


def _validate_optuna_choices(choices: Sequence[object], *, label: str) -> None:
    """Require Optuna-storable scalars that stay distinct after Optuna persistence.

    Shared by categorical choices and repeat values, which both become an
    Optuna ``CategoricalDistribution``.

    :param Sequence[object] choices: Candidate choices in configured order.
    :param str label: Field name that starts each diagnostic.
    :raises ValueError: An element is not an Optuna-compatible scalar
        (``None``/``bool``/``int``/``float``/``str``), is a non-finite float,
        or compares equal to an earlier element.
    """
    allowed = (str, int, float, bool, type(None))
    for index, c in enumerate(choices):
        if not isinstance(c, allowed):
            raise ValueError(
                f"{label} must be Optuna-compatible scalars "
                "(None, bool, int, float, or str); "
                f"got {type(c).__name__}: {c!r}"
            )
        if isinstance(c, float) and not math.isfinite(c):
            raise ValueError(f"{label} must be finite floats; got {c!r}")
        for earlier_index in range(index):
            earlier = choices[earlier_index]
            if c == earlier:
                raise ValueError(
                    f"{label} must remain distinguishable after "
                    f"Optuna persistence; {c!r} at index {index} compares equal "
                    f"to {earlier!r} at index {earlier_index}. Remove equal choices."
                )


SearchParam = Annotated[
    FloatParam | IntParam | CategoricalParam,
    Field(discriminator="type"),
]


class Repeats(_Frozen):
    """Run every configuration once per repeat value and rank it by a reducer over them."""

    key: str = Field(description="Override key that receives each repeat value, such as seed.")
    values: list[StrictJsonScalar] = Field(
        min_length=2,
        description="Repeat values, at least two; every configuration runs once per value.",
    )
    reduce: Literal["mean", "median"] = Field(
        default="mean",
        description="How a configuration's member metrics combine into the value it ranks by.",
    )

    @field_validator("values")
    @classmethod
    def _values_are_distinct_optuna_scalars(
        cls, values: list[StrictJsonScalar]
    ) -> list[StrictJsonScalar]:
        """Apply the categorical choice rules: the repeat axis is a categorical axis.

        :param list[StrictJsonScalar] values: Configured repeat values.
        :raises ValueError: A value is a non-finite float or equals an earlier value.
        :return list[StrictJsonScalar]: The values, unchanged.
        """
        _validate_optuna_choices(values, label="repeats.values")
        return values


def sampler_search_space(phase: Phase) -> dict[str, SearchParam]:
    """Return the search space with the repeat axis appended when configured.

    :param Phase phase: Phase whose axes are listed.
    :return dict[str, SearchParam]: ``search_space`` with ``repeats.key`` appended when set.
    """
    space: dict[str, SearchParam] = dict(phase.search_space)
    if phase.repeats is not None:
        space[phase.repeats.key] = CategoricalParam(
            type="categorical", choices=list(phase.repeats.values)
        )
    return space


# Samplers that draw randomly and therefore need an explicit seed to make a
# durable study reproducible. `grid` is excluded: it enumerates a fixed matrix.
STOCHASTIC_SAMPLERS = frozenset({"tpe", "random", "cmaes"})


class Sampler(_Frozen):
    """Optuna sampler configuration."""

    type: Literal["tpe", "random", "grid", "cmaes"] = "tpe"
    seed: ConfigInt | None = Field(default=None, ge=0, le=2**32 - 1)
    n_startup_trials: ConfigInt = Field(
        default=10,
        ge=0,
        description=(
            "Number of random trials TPE runs before it starts fitting its "
            "surrogate model. Applies only to sampler.type 'tpe'; other "
            "sampler types never read it and must leave it at the default."
        ),
    )

    @model_validator(mode="after")
    def _validate_n_startup_trials_applies(self) -> Sampler:
        """Reject non-default startup trials for samplers that ignore the setting.

        An explicitly written default remains valid for every sampler.

        :raises ValueError: ``n_startup_trials`` is not the default value for
            a sampler type other than ``"tpe"``.
        :return Sampler: Self, unchanged.
        """
        default = Sampler.model_fields["n_startup_trials"].default
        if self.type != "tpe" and self.n_startup_trials != default:
            raise ValueError(
                f"sampler.n_startup_trials={self.n_startup_trials} is set for "
                f"sampler.type={self.type!r}. Only the tpe sampler uses "
                f"n_startup_trials (default {default}); remove it or set "
                "sampler.type: tpe."
            )
        return self


def phase_draws(phase: Phase) -> bool:
    """Return whether the search space or repeats supplies any sampler axis.

    :param Phase phase: Phase to inspect.
    :return bool: ``True`` when the sampler has any axis to draw.
    """
    return bool(sampler_search_space(phase))


def sampler_capability_line(phase: Phase, storage: str | None) -> str:
    """Describe a phase's resume and reproduction contract.

    Resuming requires persistent storage. Grid and seeded draws are reproducible;
    parallel TPE and CMA-ES draws also depend on trial completion order.

    :param Phase phase: Phase whose sampler capability is described.
    :param str | None storage: The experiment's resolved storage, ``None`` for
        an in-memory study.
    :return str: One line naming the phase and either its sampler type, seed,
        capability and repeat axis, or that it is a fixed configuration with no
        sampler draws.
    """
    resume = "not resumable: in-memory storage" if storage is None else "resumable"
    if not phase_draws(phase):
        return f"phase {phase.name!r}: fixed configuration, no sampler draws ({resume})"
    sampler = phase.sampler
    seed = "" if sampler.seed is None else f" seed={sampler.seed}"
    reproducible = (sampler.type == "grid" or sampler.seed is not None) and not (
        sampler.type in ("tpe", "cmaes") and phase.n_jobs > 1
    )
    if not reproducible:
        capability = resume
    elif storage is None:
        capability = f"reproducible, {resume}"
    else:
        capability = f"{resume}, reproducible"
    repeats = phase.repeats
    axis = (
        ""
        if repeats is None
        else f", repeats={repeats.key}×{len(repeats.values)} ({repeats.reduce})"
    )
    return f"phase {phase.name!r}: sampler={sampler.type}{seed} ({capability}){axis}"


def sampler_warnings(phase: Phase) -> list[str]:
    """List advisory sampler and trial-budget warnings without blocking execution.

    Fixed configurations have no sampler warnings; multiple requested trials
    may represent retries, which only the existing study can establish.

    :param Phase phase: Phase whose sampler, trial target and failure budget are checked.
    :return list[str]: One sentence per warning, empty when nothing is amiss.
    """
    sampler = phase.sampler
    warnings: list[str] = []
    if not phase_draws(phase) and phase.n_trials > 1:
        warnings.append(
            f"phase {phase.name!r} is a fixed configuration: n_trials above 1 only retries "
            "failed attempts, and a launch refuses more than one past the failed ones."
        )
    if sampler.type == "tpe" and phase_draws(phase) and phase.n_trials <= sampler.n_startup_trials:
        warnings.append(
            f"Phase {phase.name!r}: tpe with n_trials={phase.n_trials} <= "
            f"n_startup_trials={sampler.n_startup_trials}, so every trial is sampled at random "
            "and TPE never fits its model. Use sampler.type: random with a seed (same "
            "coverage), or set sampler.n_startup_trials below n_trials."
        )
    repeats = phase.repeats
    if repeats is not None and phase.max_consecutive_failures <= len(repeats.values):
        count = len(repeats.values)
        warnings.append(
            f"Phase {phase.name!r}: max_consecutive_failures={phase.max_consecutive_failures} "
            f"<= {count} repeats.values, so one configuration that fails on every repeat value "
            "can abort the phase by itself; under seeded random its repeats run back to back. "
            f"Set max_consecutive_failures above {count}."
        )
    return warnings


def fixed_configuration_target_refusal(
    phase: Phase, *, finished: int, completed: int
) -> str | None:
    """Refuse extra measurements of an already completed fixed configuration.

    A fixed configuration keeps at most one COMPLETE result, feasible or not.
    Without one, the target may allow one attempt beyond its finished failures.
    Interrupted reruns do not count. Sampled phases are unrestricted here.

    :param Phase phase: Phase whose ``n_trials`` is checked.
    :param int finished: Finished attempts its study already holds, interrupted
        ones left out, zero when the study does not exist yet.
    :param int completed: How many of those attempts are COMPLETE, whether or
        not they are feasible.
    :return str | None: The refusal, which names the fix, or ``None`` when the
        target is allowed.
    """
    if phase_draws(phase):
        return None
    name, target = phase.name, phase.n_trials
    if completed >= 1:
        if target <= finished:
            return None
        return (
            f"Phase {name!r} is a fixed configuration that already has a completed trial; "
            f"n_trials={target} would run the same configuration again. Keep n_trials at "
            f"{finished}, or use repeats to measure seed noise."
        )
    if target <= finished + 1:
        return None
    if finished == 0:
        return (
            f"Phase {name!r}: n_trials={target} with an empty search_space would run {target} "
            "identical trials. Use n_trials: 1 for a fixed configuration, or repeats to "
            "measure seed noise."
        )
    plural = "" if finished == 1 else "s"
    return (
        f"Phase {name!r} is a fixed configuration with {finished} failed attempt{plural} and "
        "no completed one; raising n_trials retries it one attempt at a time, so use at most "
        f"{finished + 1}, not {target}."
    )


def _validate_sampler_search_space(phase: Phase) -> None:
    """Validate sampler compatibility and the concrete trial target at config load.

    Grid counts include the repeat axis, float rounding, step fidelity, bounds,
    and the cardinality cap without materializing values. Fixed configurations
    instead use :func:`fixed_configuration_target_refusal`.

    :param Phase phase: Phase whose sampler and search space are checked together.
    :raises ValueError: If ``allow_partial_grid`` is set with a non-grid sampler
        or with ``repeats``, if ``repeats`` is set without grid or seeded random
        or with a trial count that does not cover every repeat, if the CMA-ES
        sampler is paired with categorical params or the ``cmaes`` package is
        not importable, if the grid sampler cannot enumerate a parameter
        (raised by :func:`_grid_value_counts`), or if ``n_trials`` exceeds the
        grid cardinality or does not equal it without ``allow_partial_grid``.
    """
    sampler_type = phase.sampler.type
    space = phase.search_space

    if phase.allow_partial_grid and sampler_type != "grid":
        raise ValueError(
            f"Phase {phase.name!r}: allow_partial_grid: true applies only to "
            f"sampler.type='grid', but this phase uses sampler.type={sampler_type!r}, "
            "which would silently ignore it. Remove allow_partial_grid or set "
            "sampler.type: grid."
        )
    if phase.repeats is not None:
        _validate_repeat_sampling(phase, phase.repeats)

    if sampler_type == "cmaes":
        cats = [name for name, p in space.items() if isinstance(p, CategoricalParam)]
        if cats:
            raise ValueError(
                f"Phase {phase.name!r}: sampler.type='cmaes' does not support "
                f"categorical parameters: {cats}. Use sampler.type='tpe' or "
                f"remove the categorical params from this phase."
            )
        try:
            import cmaes  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"Phase {phase.name!r}: sampler.type='cmaes' requires the "
                "'cmaes' package, which is not installed. Reinstall phasesweep "
                "or install it directly with `pip install cmaes`."
            ) from exc

    if sampler_type == "grid" and phase_draws(phase):
        cardinality = math.prod(
            _grid_value_counts(sampler_search_space(phase), phase_name=phase.name).values()
        )
        repeat_note = (
            ""
            if phase.repeats is None
            else f" ({cardinality // len(phase.repeats.values)} configurations x "
            f"{len(phase.repeats.values)} repeats)"
        )
        if phase.n_trials > cardinality:
            raise ValueError(
                f"Phase {phase.name!r}: n_trials={phase.n_trials} exceeds the grid "
                f"cardinality {cardinality}{repeat_note}."
            )
        if not phase.allow_partial_grid and phase.n_trials != cardinality:
            remedy = (
                "increase n_trials or set allow_partial_grid: true."
                if phase.repeats is None
                else f"a grid with repeats always runs the full matrix, so set n_trials: "
                f"{cardinality}."
            )
            raise ValueError(
                f"Phase {phase.name!r}: grid sampler has {cardinality} combinations"
                f"{repeat_note} but n_trials={phase.n_trials}. Grid phases run the full "
                f"matrix by default; {remedy}"
            )


def _validate_repeat_sampling(phase: Phase, repeats: Repeats) -> None:
    """Require complete repeat groups under grid or seeded random sampling.

    Partial grids are refused because they can leave configuration groups incomplete.

    :param Phase phase: Phase whose sampler and trial target are checked.
    :param Repeats repeats: The phase's repeats block.
    :raises ValueError: The sampler is neither grid nor seeded random,
        ``allow_partial_grid`` is set, an empty search space has ``n_trials``
        other than the repeat count, or seeded random has ``n_trials`` that is
        not a multiple of it.
    """
    sampler = phase.sampler
    count = len(repeats.values)
    if sampler.type not in ("grid", "random") or (
        sampler.type == "random" and sampler.seed is None
    ):
        spelled = f"sampler.type={sampler.type!r}" + (
            " without a seed" if sampler.type == "random" else ""
        )
        raise ValueError(
            f"Phase {phase.name!r}: repeats needs sampler.type grid, or sampler.type random "
            f"with a seed, and this phase uses {spelled}. Only those two keep every "
            "configuration's repeats together: grid runs each configuration once per repeat "
            "value, and seeded random draws each configuration once and runs it once per "
            "repeat value."
        )
    if phase.allow_partial_grid:
        raise ValueError(
            f"Phase {phase.name!r}: allow_partial_grid cannot be combined with repeats. A grid "
            "visits the configuration x repeat matrix in a shuffled order, so a partial grid "
            "leaves most configurations without all their repeats, and selection ranks only "
            "complete groups. To run a subset of configurations with all their repeats, use "
            "sampler.type random with a seed."
        )
    if not phase.search_space and phase.n_trials != count:
        raise ValueError(
            f"Phase {phase.name!r}: repeats over an empty search_space runs one configuration "
            f"once per repeat value, so n_trials must be {count} (the number of "
            f"repeats.values), not {phase.n_trials}."
        )
    if sampler.type == "random" and phase.n_trials % count:
        raise ValueError(
            f"Phase {phase.name!r}: seeded random with repeats runs each configuration "
            f"{count} times, once per repeats.values entry, so n_trials must be a multiple "
            f"of {count}; got {phase.n_trials}."
        )


def _grid_value_counts(
    search_space: Mapping[str, SearchParam],
    *,
    phase_name: str,
) -> dict[str, int]:
    """Return GridSampler value counts before materializing parameter lists.

    Float counts include validation of 12-decimal rounding against uniqueness,
    exact configured steps, and bounds. No parameter value lists are retained.

    :param Mapping[str, SearchParam] search_space: Search-space specification to inspect.
    :param str phase_name: Phase name included in validation errors.
    :raises ValueError: If a parameter cannot be represented faithfully as a
        grid or the concrete matrix exceeds the supported cardinality.
    :return dict[str, int]: Number of concrete values for each parameter.
    """
    counts: dict[str, int] = {}
    for name, param in search_space.items():
        if isinstance(param, CategoricalParam):
            counts[name] = len(param.choices)
        elif isinstance(param, IntParam):
            if param.log:
                raise ValueError(
                    f"Phase {phase_name!r}: grid sampler does not support "
                    f"log-scale int param {name!r}."
                )
            counts[name] = (param.high - param.low) // param.step + 1
        else:
            if param.log:
                raise ValueError(
                    f"Phase {phase_name!r}: grid sampler does not support "
                    f"log-scale float param {name!r}."
                )
            if param.step is None:
                raise ValueError(
                    f"Phase {phase_name!r}: grid sampler requires 'step' for float param {name!r}."
                )
            counts[name] = _float_step_count(param.low, param.high, param.step) + 1
    cardinality = math.prod(counts.values())
    if cardinality > _MAX_GRID_CARDINALITY:
        raise ValueError(
            f"Phase {phase_name!r}: grid sampler has {cardinality:,} combinations, "
            f"which exceeds the supported maximum of {_MAX_GRID_CARDINALITY:,}. "
            "Reduce the search space or use a stochastic sampler."
        )

    for name, param in search_space.items():
        if isinstance(param, FloatParam):
            assert param.step is not None
            count = counts[name]
            unique = len({round(param.low + i * param.step, 12) for i in range(count)})
            if unique != count:
                raise ValueError(
                    f"Phase {phase_name!r}: grid float param {name!r} collapses to "
                    f"{unique} unique value(s) instead of {count} after rounding "
                    f"to 12 decimal places (low={param.low}, high={param.high}, "
                    f"step={param.step}). Rescale the parameter — sweep an exponent or "
                    "a multiplier — so adjacent grid points differ by more than 1e-12."
                )
            origin = Fraction(str(param.low))
            spacing = Fraction(str(param.step))
            if any(
                not (param.low <= value <= param.high)
                or Fraction(str(value)) != origin + index * spacing
                for index, value in enumerate(
                    round(param.low + i * param.step, 12) for i in range(count)
                )
            ):
                raise ValueError(
                    f"Phase {phase_name!r}: grid float param {name!r} cannot preserve its "
                    "configured bounds and step after rounding to 12 decimal places. "
                    "Rescale the parameter or use explicit categorical choices."
                )
    return counts


def grid_search_space(
    search_space: dict[str, SearchParam],
    *,
    phase_name: str = "<direct>",
) -> dict[str, list[StrictJsonScalar]]:
    """Build Optuna ``GridSampler`` values and validate grid-only constraints.

    :param dict[str, SearchParam] search_space: Search-space specification to enumerate.
    :param str phase_name: Phase name included in validation errors.
    :raises ValueError: If a parameter cannot be represented as a grid or the
        concrete matrix exceeds the supported cardinality.
    :return dict[str, list[StrictJsonScalar]]: Concrete grid values keyed by parameter name.
    """
    counts = _grid_value_counts(search_space, phase_name=phase_name)
    grid: dict[str, list[StrictJsonScalar]] = {}
    for name, param in search_space.items():
        if isinstance(param, CategoricalParam):
            grid[name] = list(param.choices)
        elif isinstance(param, IntParam):
            grid[name] = list(range(param.low, param.high + 1, param.step))
        else:
            assert param.step is not None
            grid[name] = [round(param.low + i * param.step, 12) for i in range(counts[name])]
    return grid


def _placeholder_value_for(param: SearchParam) -> Any:
    """Choose one valid value for command-template preflight.

    :param SearchParam param: Configured search parameter.
    :return: The numeric low bound or first categorical choice.
    """
    if isinstance(param, (FloatParam, IntParam)):
        return param.low
    return param.choices[0]


def _placeholder_values_for(search_space: Mapping[str, SearchParam]) -> dict[str, Any]:
    """Choose deterministic preflight values for every sampler axis.

    :param Mapping[str, SearchParam] search_space: Parameters keyed by override name.
    :return: One valid value per parameter.
    """
    return {name: _placeholder_value_for(param) for name, param in search_space.items()}
