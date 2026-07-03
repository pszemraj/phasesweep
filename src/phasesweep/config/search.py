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
        elif isinstance(param, FloatParam):
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
            grid[name] = [round(param.low + i * param.step, 12) for i in range(n_steps + 1)]
        else:  # pragma: no cover
            raise ValueError(f"Unhandled param type for grid: {param!r}")
    return grid


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


def _placeholder_values_for(search_space: Mapping[str, SearchParam]) -> dict[str, Any]:
    """Synthesize one deterministic valid value for each search-space param.

    Args:
        search_space: Search parameters keyed by override name.

    Returns:
        Placeholder values keyed by override name.

    """
    return {name: _placeholder_value_for(param) for name, param in search_space.items()}
