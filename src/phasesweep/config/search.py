"""Search-space and sampler config models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator

from phasesweep.config.common import _Frozen, _require_finite

if TYPE_CHECKING:
    from phasesweep.config.models import Phase


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

        Raises:
            ValueError: ``low`` or ``high`` is non-finite, ``low > high``,
                ``log`` is set with ``low <= 0``, ``step`` is non-finite or
                ``<= 0``, or ``log`` and ``step`` are combined.

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

        Raises:
            ValueError: ``low > high``, ``log`` is set with ``low <= 0``,
                ``step <= 0``, or ``log`` is combined with ``step != 1`` (which
                Optuna's ``IntDistribution`` rejects at construction time).

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
    def _choices_are_unique_optuna_scalars(cls, choices: list[Any]) -> list[Any]:
        """Reject choices Optuna can't store (lists, dicts, NaN, ...) and duplicates.

        Optuna keeps a duplicated choice verbatim — ``CategoricalDistribution([1, 1, 2])``
        has three choices and ``GridSampler`` enumerates three points — so
        ``[1, 1, 2]`` used to run three trials over two distinct assignments and
        still report the grid complete (review v0.5.17 / finding C). Duplicates
        also silently double a choice's sampling weight under TPE/random.

        Uniqueness is type-aware: ``1``, ``1.0``, and ``true`` are three
        different overrides on the wire even though Python calls them equal, so
        the identity key is ``(type name, repr)`` rather than the value itself.

        Args:
            choices: The candidate choices list pre-validation.

        Returns:
            The same list, unchanged.

        Raises:
            ValueError: An element is not an Optuna-compatible scalar
                (``None``/``bool``/``int``/``float``/``str``), is a non-finite
                float, or repeats an earlier choice under the type-aware
                identity key.

        """
        # Optuna only accepts None|bool|int|float|str as categorical choices.
        # Anything else (lists, dicts, custom objects) fails at suggest time.
        allowed = (str, int, float, bool, type(None))
        first_index: dict[tuple[str, str], int] = {}
        for index, c in enumerate(choices):
            if not isinstance(c, allowed):
                raise ValueError(
                    "categorical choices must be Optuna-compatible scalars "
                    "(None, bool, int, float, or str); "
                    f"got {type(c).__name__}: {c!r}"
                )
            if isinstance(c, float) and not math.isfinite(c):
                raise ValueError(f"categorical float choices must be finite; got {c!r}")
            identity = (type(c).__name__, repr(c))
            if identity in first_index:
                raise ValueError(
                    f"categorical choices must be unique; {c!r} appears at index "
                    f"{first_index[identity]} and index {index}. A repeated choice "
                    "inflates grid cardinality (the phase runs an extra trial on an "
                    "assignment it already evaluated, yet still reports a complete "
                    "grid) and doubles that value's sampling weight. Uniqueness is "
                    "type-aware: 1, 1.0, and true remain three distinct choices."
                )
            first_index[identity] = index
        return choices


SearchParam = FloatParam | IntParam | CategoricalParam


class Sampler(_Frozen):
    """Optuna sampler configuration."""

    type: Literal["tpe", "random", "grid", "cmaes"] = "tpe"
    seed: int | None = None
    n_startup_trials: int = Field(default=10, ge=0)  # tpe only


def _validate_sampler_search_space(phase: Phase) -> None:
    """Reject sampler/search-space combinations Optuna will not accept at runtime.

    Run at config-load (review v0.5.2 / blocker 2). Catches:

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
    * Grid sampler with float ``(high - low)`` not an integer multiple of ``step`` —
      naive enumeration emits values above ``high`` (review v0.5.2 / blocker 4).
    * Grid sampler with a float param whose enumerated points collapse under the
      12-decimal canonical rounding, which would make the cardinality below
      overcount the distinct configurations (review v0.5.17 / finding C).

    The cardinality is a product of the enumerated value-list lengths, so it is
    only truthful while those lists are duplicate-free. Categorical duplicates
    are rejected by :class:`CategoricalParam` and float collapse by
    :func:`grid_search_space`, both before this product is taken.

    :param Phase phase: Phase whose sampler and search space are checked together.
    :raises ValueError: If the CMA-ES sampler is paired with categorical params or
        the ``cmaes`` package is not importable, if the grid sampler cannot
        enumerate a parameter (raised by :func:`grid_search_space`), or if
        ``n_trials`` exceeds the grid cardinality or does not equal it without
        ``allow_partial_grid``.
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
        try:
            import cmaes  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"Phase {phase.name!r}: sampler.type='cmaes' requires the "
                "'cmaes' package, which is not installed. Reinstall phasesweep "
                "or install it directly with `pip install cmaes`."
            ) from exc

    if sampler_type == "grid":
        cardinality = math.prod(
            len(values)
            for values in grid_search_space(phase.search_space, phase_name=phase.name).values()
        )
        if phase.n_trials > cardinality:
            raise ValueError(
                f"Phase {phase.name!r}: n_trials={phase.n_trials} exceeds the grid "
                f"cardinality {cardinality}."
            )
        if not phase.allow_partial_grid and phase.n_trials != cardinality:
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


def grid_search_space(
    search_space: dict[str, SearchParam],
    *,
    phase_name: str = "<direct>",
) -> dict[str, list[Any]]:
    """Build Optuna ``GridSampler`` values and validate grid-only constraints.

    :param dict[str, SearchParam] search_space: Search-space specification to enumerate.
    :param str phase_name: Phase name included in validation errors.
    :raises ValueError: If a parameter cannot be represented as a grid.
    :return dict[str, list[Any]]: Concrete grid values keyed by parameter name.
    """
    grid: dict[str, list[Any]] = {}
    for name, param in search_space.items():
        if isinstance(param, CategoricalParam):
            grid[name] = list(param.choices)
        elif isinstance(param, IntParam):
            if param.log:
                raise ValueError(
                    f"Phase {phase_name!r}: grid sampler does not support "
                    f"log-scale int param {name!r}."
                )
            grid[name] = list(range(param.low, param.high + 1, param.step))
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
            _validate_float_grid_divides(phase_name, name, param)
            n_steps = int(round((param.high - param.low) / param.step))
            values = [round(param.low + i * param.step, 12) for i in range(n_steps + 1)]
            # Post-canonicalization collapse (review v0.5.17 / finding C): the
            # round(..., 12) above maps adjacent points onto the same float once
            # the step drops below ~1e-12, so the grid would publish fewer unique
            # configurations than the cardinality check counts — the same
            # "complete grid" lie duplicate categorical choices produced.
            unique = len(set(values))
            if unique != len(values):
                raise ValueError(
                    f"Phase {phase_name!r}: grid float param {name!r} collapses to "
                    f"{unique} unique value(s) instead of {len(values)} after rounding "
                    f"to 12 decimal places (low={param.low}, high={param.high}, "
                    f"step={param.step}). Rescale the parameter — sweep an exponent or "
                    "a multiplier — so adjacent grid points differ by more than 1e-12."
                )
            grid[name] = values
    return grid


def _placeholder_value_for(param: SearchParam) -> Any:
    """Synthesize one valid value for a search-space param (used in template render preflight).

    Args:
        param: A concrete search parameter from a phase's ``search_space``.

    Returns:
        For ``FloatParam`` the interval midpoint; for ``IntParam`` the
        integer midpoint; for ``CategoricalParam`` the first listed choice.

    """
    if isinstance(param, FloatParam):
        return (param.low + param.high) / 2
    if isinstance(param, IntParam):
        return (param.low + param.high) // 2
    return param.choices[0]


def _placeholder_values_for(search_space: Mapping[str, SearchParam]) -> dict[str, Any]:
    """Synthesize one deterministic valid value for each search-space param.

    Args:
        search_space: Search parameters keyed by override name.

    Returns:
        Placeholder values keyed by override name.

    """
    return {name: _placeholder_value_for(param) for name, param in search_space.items()}
