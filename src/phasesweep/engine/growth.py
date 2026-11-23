"""Whether a changed search-space key only gains values, so its study can continue in place.

A phase's study keeps its finished trials when its new ``search_space`` holds every value its
trials could have drawn and Optuna accepts the changed parameter. Values compare by their JSON
text, so ``1`` and ``1.0``, or ``true`` and ``1``, are a removal plus an addition.

- A grid gains values when its enumerated values are a superset of the recorded ones and the
  axis stays an int or float axis (:func:`~phasesweep.engine.optuna._grid_axis_kind`): Optuna
  fixes a categorical's choices for the life of a study, and its journal drops a param whose
  choices changed when another process replays it.
- Seeded random and TPE gain values when an int or float range keeps its type and ``log``,
  reaches at least as low and as high, and keeps its drawable values. A singleton contributes
  one point to the new lattice regardless of its old step. TPE re-encodes earlier trials with
  the current distribution, so it continues soundly. Choices
  under these samplers are categorical, so they cannot grow.
- CMA-ES uses only the params whose distribution is the same in every trial, so a changed range
  would be sampled at random for the rest of the study; it never grows in place.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from pydantic import TypeAdapter

from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Sampler,
    SearchParam,
    grid_search_space,
)
from phasesweep.engine.optuna import _grid_axis_kind
from phasesweep.engine.state import PhaseIdentityRecord
from phasesweep.runtime.json import JsonObject

_SEARCH_PARAM: TypeAdapter[SearchParam] = TypeAdapter(SearchParam)

#: What a phase that retrains nothing and loads carries: an inheriting phase may not search a key
#: the phase it inherits sampled, so it takes no ``inherits`` and sets the other values itself.
#: ``{phase}`` is the phase being grown.
STANDALONE_PHASE = "with no `inherits` and `{phase}`'s other values set in `fixed_overrides`"
#: The step after choices that cannot grow in place.
CHOICES_ROUTE = (
    f"Restore the choices and add a phase over only the added choices, {STANDALONE_PHASE}"
)
#: The step after a range that changed under CMA-ES.
CMAES_ROUTE = (
    f"Restore the range and add a seeded random or TPE phase over the wider range, "
    f"{STANDALONE_PHASE}"
)


@dataclass(frozen=True)
class _GrowthVerdict:
    """What a change to one search-space key does to the values its trials could have drawn."""

    #: The added values or the widened range, as the operator reads them.
    gained: tuple[str, ...] = ()
    #: The recorded values or range ends the new space no longer holds.
    dropped: tuple[str, ...] = ()
    #: Why the key cannot change in place although nothing was dropped.
    refusal: str | None = None
    #: The step that retrains nothing for :attr:`refusal`, or None when renaming is the way.
    route: str | None = None

    @property
    def grows(self) -> bool:
        """Say whether the study can continue under the new key.

        :return bool: ``True`` when nothing was dropped and nothing else refuses the change.
        """
        return not self.dropped and self.refusal is None


def _text(value: object) -> str:
    """Render a value as the JSON text it compares by.

    :param object value: A configured value.
    :return str: Its compact JSON text.
    """
    return json.dumps(value, separators=(",", ":"))


def _grid_kind(param: SearchParam) -> str:
    """Return how a grid axis registers with Optuna.

    :param SearchParam param: A grid axis.
    :return str: ``int``, ``float`` or ``categorical``.
    """
    if isinstance(param, CategoricalParam):
        return _grid_axis_kind(param.choices)
    return param.type


def _grid_growth(old: SearchParam, new: SearchParam) -> _GrowthVerdict:
    """Judge a grid axis by its enumerated values.

    :param SearchParam old: The recorded axis.
    :param SearchParam new: The configured axis.
    :return _GrowthVerdict: The verdict.
    """
    old_values = [_text(value) for value in grid_search_space({"axis": old})["axis"]]
    new_values = [_text(value) for value in grid_search_space({"axis": new})["axis"]]
    dropped = tuple(value for value in old_values if value not in new_values)
    if dropped:
        return _GrowthVerdict(dropped=dropped)
    if "categorical" in (_grid_kind(old), _grid_kind(new)):
        return _GrowthVerdict(
            refusal="holds choices that are not all ints or all floats, which Optuna fixes for "
            "the life of a study",
            route=CHOICES_ROUTE,
        )
    return _GrowthVerdict(gained=tuple(value for value in new_values if value not in old_values))


def _on_lattice(value: float, origin: float, step: float) -> bool:
    """Say whether ``value`` lies on the lattice ``origin + k * step``.

    :param float value: The value to place.
    :param float origin: The lattice's origin.
    :param float step: The lattice's spacing.
    :return bool: Whether the distance from ``origin`` is a whole number of steps.
    """
    return ((Fraction(str(value)) - Fraction(str(origin))) / Fraction(str(step))).denominator == 1


def _range_growth(old: SearchParam, new: SearchParam) -> _GrowthVerdict:
    """Judge a seeded random or TPE parameter.

    :param SearchParam old: The recorded parameter.
    :param SearchParam new: The configured parameter.
    :return _GrowthVerdict: The verdict.
    """
    if isinstance(old, CategoricalParam) and isinstance(new, CategoricalParam):
        new_choices = [_text(choice) for choice in new.choices]
        removed = tuple(text for c in old.choices if (text := _text(c)) not in new_choices)
        if removed:
            return _GrowthVerdict(dropped=removed)
        return _GrowthVerdict(
            refusal="holds choices, which Optuna fixes for the life of a study", route=CHOICES_ROUTE
        )
    if not isinstance(old, IntParam | FloatParam) or not isinstance(new, IntParam | FloatParam):
        return _GrowthVerdict(refusal=f"changes from {old.type} to {new.type}")
    if old.type != new.type or old.log != new.log:
        return _GrowthVerdict(refusal="changes its type or log scale")
    dropped: list[str] = []
    if new.low > old.low:
        dropped.append(f"low {_text(old.low)}")
    if new.high < old.high:
        dropped.append(f"high {_text(old.high)}")
    if new.step is not None:
        aligned = _on_lattice(old.low, new.low, new.step)
        if old.low == old.high:
            if not dropped and not aligned:
                dropped.append(_text(old.low))
        elif old.step is None:
            dropped.append("its continuous values")
        elif not aligned or not _on_lattice(old.step, 0, new.step):
            dropped.append(f"step {_text(old.step)} lattice")
    if dropped:
        return _GrowthVerdict(dropped=tuple(dropped))
    step = "" if old.step == new.step else f" step {_text(old.step)} -> {_text(new.step)}"
    return _GrowthVerdict(
        gained=(
            f"{_text(old.low)}..{_text(old.high)} -> {_text(new.low)}..{_text(new.high)}{step}",
        )
    )


def _search_space_growth(
    recorded: JsonObject, current: JsonObject, *, sampler: Sampler
) -> _GrowthVerdict:
    """Judge whether one search-space key only gains values under the phase's sampler.

    :param JsonObject recorded: The key's parameter as the study recorded it.
    :param JsonObject current: The key's parameter as configured now.
    :param Sampler sampler: The phase's sampler, the same in both configs.
    :raises pydantic.ValidationError: Either side is not a valid search parameter.
    :return _GrowthVerdict: What the change gains or drops, or why it cannot happen in place.
    """
    old = _SEARCH_PARAM.validate_python(recorded)
    new = _SEARCH_PARAM.validate_python(current)
    if sampler.type == "cmaes":
        return _GrowthVerdict(
            refusal="changed under CMA-ES, which would sample it at random for the rest of the "
            "study",
            route=CMAES_ROUTE,
        )
    if sampler.type == "grid":
        return _grid_growth(old, new)
    return _range_growth(old, new)


def describe_gains(gained: Mapping[str, Sequence[str]], *, sampler: Sampler) -> str:
    """Name the values a grown search space adds, key by key, the same way on every surface.

    A grid's added values read ``+value``; the other samplers gain a widened range, which
    already reads ``old -> new``.

    :param Mapping[str, Sequence[str]] gained: Added values or widened ranges, by key.
    :param Sampler sampler: The sampler the growth was judged under.
    :return str: Such as ``lr +0.003, +0.01``, or ``wd 0..0.1 -> 0..0.3`` for a range.
    """
    sign = "+" if sampler.type == "grid" else ""
    return "; ".join(
        f"{key} {', '.join(f'{sign}{value}' for value in values)}" for key, values in gained.items()
    )


def recorded_growth(record: PhaseIdentityRecord) -> tuple[tuple[int, str], ...]:
    """Name each growth a study's identity records, as the draw position it took effect at.

    Each recorded search space after the first gained values over the one before it, judged under
    the sampler the study recorded, so the text reads the same whatever the config says now.

    :param PhaseIdentityRecord record: The study's identity record.
    :raises pydantic.ValidationError: The record holds an invalid sampler or search parameter.
    :return tuple[tuple[int, str], ...]: ``(first_position, gains)`` per growth, such as
        ``(12, "lr +0.003, +0.01")``; a range reads ``wd 0..0.1 -> 0..0.3``.
    """
    phase = record["payload"].get("phase")
    raw_sampler = phase.get("sampler") if isinstance(phase, dict) else None
    sampler = Sampler.model_validate(raw_sampler)
    steps: list[tuple[int, str]] = []
    for before, entry in itertools.pairwise(record["search_spaces"]):
        gained: dict[str, tuple[str, ...]] = {}
        for key, new in sorted(entry["search_space"].items()):
            old = before["search_space"].get(key)
            if old == new or not isinstance(old, dict) or not isinstance(new, dict):
                continue
            if values := _search_space_growth(old, new, sampler=sampler).gained:
                gained[key] = values
        if gained:
            steps.append((entry["first_position"], describe_gains(gained, sampler=sampler)))
    return tuple(steps)
