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
        """Reject non-finite bounds, ``low > high``, log+nonpositive, log+step combos.

        Returns:
            Self, unchanged. Pydantic ``mode='after'`` validator protocol.

        Raises:
            ValueError: ``low`` or ``high`` is non-finite, ``low > high``,
                ``log`` is set with ``low <= 0``, ``step`` is non-finite or
                ``<= 0``, ``log`` and ``step`` are combined, or the stepped
                interval does not include the configured high bound.

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
        """Reject unsafe bounds, log+nonpositive, non-positive step, and log+step!=1.

        Returns:
            Self, unchanged. Pydantic ``mode='after'`` validator protocol.

        Raises:
            ValueError: Bounds cannot round-trip exactly through Optuna,
                ``log`` is set with ``low <= 0``, ``step <= 0``, or ``log`` is
                combined with ``step != 1`` (which Optuna's
                ``IntDistribution`` rejects at construction time), or the
                stepped interval does not include the configured high bound.

        """
        _validate_optuna_integer_domain(self.low, self.high)
        if self.log and self.low <= 0:
            raise ValueError("log-scale int param requires low > 0")
        if self.step <= 0:
            raise ValueError("int param step must be > 0")
        if self.log and self.step != 1:
            # Optuna's IntDistribution rejects this at construction time.
            # Catch it here so config-load fails instead of trial-launch.
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
        """Reject choices Optuna can't store (lists, dicts, NaN, ...) and equal choices.

        Optuna keeps a duplicated choice verbatim — ``CategoricalDistribution([1, 1, 2])``
        has three choices and ``GridSampler`` enumerates three points — so
        ``[1, 1, 2]`` would run three trials over two distinct assignments and
        still report the grid complete. Duplicates also silently double a
        choice's sampling weight under TPE/random.

        Uniqueness is by plain Python equality, across types, because that is
        the comparison Optuna itself uses. ``CategoricalDistribution.
        to_internal_repr`` locates a sampled value with ``choices.index(value)``,
        i.e. by ``==``, when it records ``FrozenTrial.params``. So choices that
        are distinct objects but compare equal — ``1``/``1.0``/``True``,
        ``0``/``False``, ``0.0``/``-0.0`` — all collapse onto the *first* equal
        choice the moment the trial is recorded. The trainer still receives the
        live suggested value, but the published winner, the inherited overrides
        of every child phase, and TPE's own history are rebuilt from the
        collapsed ``params``: a phase that ran ``--x true`` publishes a winner
        claiming ``x: 1``. Rejecting equal choices at config load is what makes
        every accepted choice round-trip identically.

        Args:
            choices: The candidate choices list pre-validation.

        Returns:
            The same list, unchanged.

        Raises:
            ValueError: An element is not an Optuna-compatible scalar
                (``None``/``bool``/``int``/``float``/``str``), is a non-finite
                float, or compares equal to an earlier choice.

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
    # Optuna only accepts None|bool|int|float|str as categorical choices.
    # Anything else (lists, dicts, custom objects) fails at suggest time.
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
        # Pairwise ``==`` rather than a hash/identity key: equal objects of
        # different types (1 vs 1.0 vs True) are exactly the collision
        # Optuna's index lookup cannot distinguish, and this also holds if
        # a choice type ever stops being hashable.
        for earlier_index in range(index):
            earlier = choices[earlier_index]
            if c == earlier:
                raise ValueError(
                    f"{label} must remain distinguishable after "
                    f"Optuna persistence; {c!r} at index {index} compares equal "
                    f"to {earlier!r} at index {earlier_index}. Optuna records a "
                    "sampled value as its `==` index into choices, so equal "
                    "choices collapse onto the first of them in "
                    "FrozenTrial.params, the published winner, and every "
                    "inherited override. A repeat also inflates grid "
                    "cardinality (the phase runs an extra trial on an "
                    "assignment it already evaluated, yet still reports a "
                    "complete grid) and doubles that value's sampling weight."
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


def repeat_param(repeats: Repeats) -> CategoricalParam:
    """Return the categorical axis the sampler draws repeat values from.

    :param Repeats repeats: The phase's repeats block.
    :return CategoricalParam: One choice per repeat value, in configured order.
    """
    return CategoricalParam(type="categorical", choices=list(repeats.values))


def sampler_search_space(phase: Phase) -> dict[str, SearchParam]:
    """Return every axis the phase's sampler draws: the search space plus the repeat axis.

    The only way to get a sampler's search space, so suggestion, grid
    enumeration, cardinality, wire-form checks and command preflight all see
    the repeat axis.

    :param Phase phase: Phase whose axes are listed.
    :return dict[str, SearchParam]: ``search_space`` with ``repeats.key`` appended when set.
    """
    space: dict[str, SearchParam] = dict(phase.search_space)
    if phase.repeats is not None:
        space[phase.repeats.key] = repeat_param(phase.repeats)
    return space


# Samplers whose suggestions depend on process-local RNG/optimizer state that
# Optuna storage does not persist. `phasesweep.engine.study_policy.
# _validate_sampler_continuation` refuses to resume a phase that draws from one
# of these mid-target, so each trial target must be run in a single invocation.
NON_RESUMABLE_SAMPLERS = frozenset({"tpe", "cmaes"})
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
    acknowledge_nonresumable: bool = Field(
        default=False,
        description=(
            "Acknowledge that this sampler's trial target must run in one "
            "invocation. TPE and CMA-ES suggestions depend on process-local "
            "state Optuna storage does not persist, so PhaseSweep refuses to "
            "resume such a phase mid-target. Required for sampler.type 'tpe' "
            "or 'cmaes' on persistent storage when the phase has a "
            "search_space; rejected for 'grid' and 'random', which resume safely."
        ),
    )

    @model_validator(mode="after")
    def _validate_acknowledgement_applies(self) -> Sampler:
        """Reject an acknowledgement of a restriction this sampler does not impose.

        :raises ValueError: ``acknowledge_nonresumable`` is set for a sampler
            type outside :data:`NON_RESUMABLE_SAMPLERS`.
        :return Sampler: Self, unchanged.
        """
        if self.acknowledge_nonresumable and self.type not in NON_RESUMABLE_SAMPLERS:
            raise ValueError(
                f"sampler.acknowledge_nonresumable is set for sampler.type={self.type!r}, "
                "which resumes safely: PhaseSweep reattaches it to an existing study and "
                "tops the study up. Only "
                f"{sorted(NON_RESUMABLE_SAMPLERS)} carry the run-the-target-in-one-invocation "
                "contract this flag acknowledges. Remove acknowledge_nonresumable."
            )
        return self

    @model_validator(mode="after")
    def _validate_n_startup_trials_applies(self) -> Sampler:
        """Reject a non-default ``n_startup_trials`` for a sampler that ignores it.

        Checked by value, not by whether the field was explicitly set, so a
        config that spells out the default (e.g. a config snapshot dump)
        keeps loading; only a value that would actually change behavior on a
        sampler that never reads this field is refused.

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
    """Say whether a phase asks its sampler for any values.

    A phase with an empty search space and no repeats is a fixed
    configuration: its sampler is never consulted, so the seed and
    resumability requirements that guard sampler draws do not apply to it. A
    repeats phase always draws its repeat values, from grid or seeded random.

    :param Phase phase: Phase to inspect.
    :return bool: ``True`` when the sampler has any axis to draw.
    """
    return bool(sampler_search_space(phase))


def sampler_capability_line(phase: Phase) -> str:
    """Render the one-line resume/reproduce contract disclosed for ``phase``.

    Shared by ``phasesweep validate`` and ``phasesweep run --dry-run`` so both
    surfaces state the same contract in the same words before any trial runs.

    :param Phase phase: Phase whose sampler capability is described.
    :return str: One line naming the phase and either its sampler type, seed,
        capability and repeat axis, or that it is a fixed configuration with no
        sampler draws.
    """
    if not phase_draws(phase):
        return f"phase {phase.name!r}: fixed configuration, no sampler draws (resumable)"
    sampler = phase.sampler
    seed = "" if sampler.seed is None else f" seed={sampler.seed}"
    if sampler.type in NON_RESUMABLE_SAMPLERS:
        capability = "non-resumable: run each target in one invocation"
    elif sampler.seed is None:
        capability = "resumable"
    else:
        capability = "resumable, reproducible"
    repeats = phase.repeats
    axis = (
        ""
        if repeats is None
        else f", repeats={repeats.key}×{len(repeats.values)} ({repeats.reduce})"
    )
    return f"phase {phase.name!r}: sampler={sampler.type}{seed} ({capability}){axis}"


def sampler_warnings(phase: Phase) -> list[str]:
    """List the advisory warnings about a phase's sampler and trial budget.

    Shared by ``phasesweep validate`` and ``phasesweep run --dry-run``. A
    warning never blocks a run. A phase that draws nothing has no sampler to
    warn about, but a fixed configuration asking for several trials is told
    that only a launch can say whether they are retries.

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
            "and TPE never fits its model. Use sampler.type: random with a seed (resumable, "
            "same coverage), or set sampler.n_startup_trials below n_trials."
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
    """Say why a fixed configuration may not run to ``phase.n_trials``, if it may not.

    A fixed configuration runs the same trial every time, so it keeps at most
    one completed result, and ``n_trials`` above 1 only replaces failed
    attempts. With ``finished`` terminal attempts and none completed, the
    target may reach ``finished + 1``: one more attempt than the failures it
    replaces. Once one attempt completed, feasible or not, the configuration
    has been measured and the target may not pass the attempts already made, or
    a rerun would shop for a better draw of the same configuration. A phase
    that draws is never refused here.

    :param Phase phase: Phase whose ``n_trials`` is checked.
    :param int finished: Terminal attempts its study already holds, zero when
        the study does not exist yet.
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
    """Reject sampler/search-space combinations Optuna will not accept at runtime.

    Run at config-load. Catches:

    * ``allow_partial_grid: true`` with a sampler other than ``grid``; the
      switch is read only for grid, so any other sampler would silently ignore it.
    * CMA-ES with categorical parameters — Optuna's ``CmaEsSampler`` is float-only;
      categorical params silently fail every trial trying to cast 'b' to float.
    * CMA-ES without the ``cmaes`` package importable. ``cmaes`` is a declared
      hard dependency, but a declared dependency is not an enforced one: an
      environment can lose it (partial install, manual uninstall, a checkout
      run without installing). Without this preflight the failure surfaces
      inside ``optuna.samplers.CmaEsSampler`` during ``_build_sampler``, i.e.
      after the generation is claimed, the experiment lock is held, and earlier
      phases have already burned GPU time — and with an Optuna-internal message
      instead of an install hint. This is an environment check, not a
      closed-contract type check; do not "simplify" it away.
    * Grid sampler with log-scale floats or ints — Optuna's ``GridSampler`` does
      not enumerate log-spaced values.
    * Grid sampler with float param missing ``step``.
    * Grid sampler with a float param whose enumerated points collapse under the
      12-decimal canonical rounding, which would make the cardinality below
      overcount the distinct configurations.

    The cardinality is a product of the enumerated value-list lengths, so it is
    only truthful while those lists are duplicate-free. Categorical duplicates
    are rejected by :class:`CategoricalParam` and float collapse by
    :func:`grid_search_space`, both before this product is taken. The repeat
    axis is part of the grid, so it counts toward the cardinality and its cap.
    The grid checks cover only a phase that draws: a fixed configuration
    consults no sampler, so a one-point grid does not bound its target. The
    retry rule in :func:`fixed_configuration_target_refusal` does.

    :param Phase phase: Phase whose sampler and search space are checked together.
    :raises ValueError: If ``allow_partial_grid`` is set with a non-grid sampler
        or with ``repeats``, if ``repeats`` is set without grid or seeded random
        or with a trial count that does not cover every repeat, if the CMA-ES
        sampler is paired with categorical params or the ``cmaes`` package is
        not importable, if the grid sampler cannot enumerate a parameter
        (raised by :func:`grid_search_space`), or if ``n_trials`` exceeds the
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
            len(values)
            for values in grid_search_space(
                sampler_search_space(phase), phase_name=phase.name
            ).values()
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
    """Require a sampler and trial count that run every configuration once per repeat value.

    Only grid, whose cells each hold one configuration and one repeat value,
    and seeded random, whose draws are keyed by configuration slot, keep a
    configuration's repeats together. A partial grid would visit the shuffled
    configuration x repeat matrix in part and leave most groups incomplete.

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

    :param Mapping[str, SearchParam] search_space: Search-space specification to inspect.
    :param str phase_name: Phase name included in validation errors.
    :raises ValueError: If a parameter cannot be represented as a grid.
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
    return counts


def grid_search_space(
    search_space: dict[str, SearchParam],
    *,
    phase_name: str = "<direct>",
) -> dict[str, list[Any]]:
    """Build Optuna ``GridSampler`` values and validate grid-only constraints.

    :param dict[str, SearchParam] search_space: Search-space specification to enumerate.
    :param str phase_name: Phase name included in validation errors.
    :raises ValueError: If a parameter cannot be represented as a grid or the
        concrete matrix exceeds the supported cardinality.
    :return dict[str, list[Any]]: Concrete grid values keyed by parameter name.
    """
    counts = _grid_value_counts(search_space, phase_name=phase_name)
    cardinality = math.prod(counts.values())
    if cardinality > _MAX_GRID_CARDINALITY:
        raise ValueError(
            f"Phase {phase_name!r}: grid sampler has {cardinality:,} combinations, "
            f"which exceeds the supported maximum of {_MAX_GRID_CARDINALITY:,}. "
            "Reduce the search space or use a stochastic sampler."
        )

    grid: dict[str, list[Any]] = {}
    for name, param in search_space.items():
        if isinstance(param, CategoricalParam):
            grid[name] = list(param.choices)
        elif isinstance(param, IntParam):
            grid[name] = list(range(param.low, param.high + 1, param.step))
        else:
            assert param.step is not None
            values = [round(param.low + i * param.step, 12) for i in range(counts[name])]
            # Post-canonicalization collapse: the round(..., 12) above maps
            # adjacent points onto the same float once the step drops below
            # ~1e-12, so the grid would publish fewer unique configurations
            # than the cardinality check counts — the same "complete grid" lie
            # duplicate categorical choices would produce.
            unique = len(set(values))
            if unique != len(values):
                raise ValueError(
                    f"Phase {phase_name!r}: grid float param {name!r} collapses to "
                    f"{unique} unique value(s) instead of {len(values)} after rounding "
                    f"to 12 decimal places (low={param.low}, high={param.high}, "
                    f"step={param.step}). Rescale the parameter — sweep an exponent or "
                    "a multiplier — so adjacent grid points differ by more than 1e-12."
                )
            origin = Fraction(str(param.low))
            spacing = Fraction(str(param.step))
            if any(
                not (param.low <= value <= param.high)
                or Fraction(str(value)) != origin + index * spacing
                for index, value in enumerate(values)
            ):
                raise ValueError(
                    f"Phase {phase_name!r}: grid float param {name!r} cannot preserve its "
                    "configured bounds and step after rounding to 12 decimal places. "
                    "Rescale the parameter or use explicit categorical choices."
                )
            grid[name] = values
    return grid


def _placeholder_value_for(param: SearchParam) -> Any:
    """Synthesize one valid value for a search-space param (used in template render preflight).

    Args:
        param: A concrete search parameter from a phase's ``search_space``.

    Returns:
        For ``FloatParam`` and ``IntParam`` the configured ``low`` bound,
        which the param validators keep finite, on the step lattice, and
        positive under ``log=True``; for ``CategoricalParam`` the first
        listed choice.

    """
    if isinstance(param, (FloatParam, IntParam)):
        return param.low
    return param.choices[0]


def _placeholder_values_for(search_space: Mapping[str, SearchParam]) -> dict[str, Any]:
    """Synthesize one deterministic valid value for each search-space param.

    Args:
        search_space: Search parameters keyed by override name.

    Returns:
        Placeholder values keyed by override name.

    """
    return {name: _placeholder_value_for(param) for name, param in search_space.items()}
