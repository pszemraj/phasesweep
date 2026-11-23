"""The order in which a grid phase draws its cells, across every search space it ran under.

A grid study assigns draw position ``p`` the cell at index ``p`` of one order, and records
``p`` as the trial's ``grid_id``. The order is built from the study's search-space history
(:class:`~phasesweep.engine.state.SearchSpaceEntry`): the first space gets a seeded shuffle of
its whole grid, and each later space keeps every position drawn before it took effect and
appends a seeded shuffle of the cells of its grid not among them. A position's cell therefore
never changes once it is drawn, failed-cell retries keyed by ``grid_id`` stay on their cell, and
a grid that grew draws only cells it has not run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Sequence

from optuna.samplers._lazy_random_state import LazyRandomState
from pydantic import TypeAdapter

from phasesweep.config.common import StrictJsonScalar
from phasesweep.config.search import (
    CategoricalParam,
    Repeats,
    SearchParam,
    grid_search_space,
)
from phasesweep.engine.state import SearchSpaceEntry
from phasesweep.runtime.json import JsonObject

GridCell = tuple[StrictJsonScalar, ...]
"""One grid point: a value per axis, in :func:`grid_axis_names` order."""

_SEARCH_PARAM: TypeAdapter[SearchParam] = TypeAdapter(SearchParam)


def recorded_search_space(space: JsonObject) -> dict[str, SearchParam]:
    """Rebuild the search parameters a recorded search space holds.

    :param JsonObject space: A phase ``search_space`` as its fingerprint payload holds it.
    :raises pydantic.ValidationError: An entry is not a valid search parameter.
    :return dict[str, SearchParam]: The parameters, by key.
    """
    return {name: _SEARCH_PARAM.validate_python(param) for name, param in space.items()}


def grid_axis_names(space: Sequence[str], repeats: Repeats | None) -> list[str]:
    """Return a grid's axis names in the order its cells hold their values.

    Sorted, as Optuna's grid sampler sorts them, so the order of keys in a config file never
    reorders a grid.

    :param Sequence[str] space: The phase's ``search_space`` keys.
    :param Repeats | None repeats: The phase's repeat axis, whose key is an axis too.
    :return list[str]: The axis names, repeat key included.
    """
    return sorted([*space, *([] if repeats is None else [repeats.key])])


def _cell_key(cell: GridCell) -> str:
    """Key a cell by the JSON text of its values, so ``1`` and ``1.0`` are different cells.

    :param GridCell cell: A grid point.
    :return str: Its canonical JSON text.
    """
    return json.dumps(list(cell), separators=(",", ":"))


def _grid_cells(space: JsonObject, repeats: Repeats | None, phase_name: str) -> list[GridCell]:
    """Enumerate every cell of one recorded search space, in Optuna's product order.

    :param JsonObject space: A recorded phase ``search_space``.
    :param Repeats | None repeats: The phase's repeat axis.
    :param str phase_name: Phase named in an enumeration error.
    :raises ValueError: The space cannot be enumerated as a grid.
    :return list[GridCell]: The cells, each in :func:`grid_axis_names` order.
    """
    params = recorded_search_space(space)
    if repeats is not None:
        params[repeats.key] = CategoricalParam(type="categorical", choices=list(repeats.values))
    axes = grid_search_space(params, phase_name=phase_name)
    return list(itertools.product(*(axes[name] for name in grid_axis_names(list(space), repeats))))


def _shuffled(cells: list[GridCell], seed: int) -> list[GridCell]:
    """Shuffle cells as Optuna's grid sampler shuffles its grid for the same seed.

    :param list[GridCell] cells: Cells to shuffle; left unchanged.
    :param int seed: Seed of the shuffle.
    :return list[GridCell]: The shuffled copy.
    """
    return [cells[int(index)] for index in LazyRandomState(seed).rng.permutation(len(cells))]


def _layer_seed(seed: int | None, layer: int) -> int:
    """Derive the shuffle seed of a later search space from the phase seed and its index.

    :param int | None seed: The phase's ``sampler.seed``.
    :param int layer: Index of the search space in the study's history, at least 1.
    :return int: A 32-bit seed, the same in every process.
    """
    material = json.dumps([seed, layer], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _configurations(cells: Sequence[GridCell], repeat_index: int) -> list[GridCell]:
    """List the distinct configurations of some cells, in order of first appearance.

    :param Sequence[GridCell] cells: Grid cells.
    :param int repeat_index: Position of the repeat value in a cell.
    :return list[GridCell]: Each cell with its repeat value removed, once.
    """
    seen: dict[str, GridCell] = {}
    for cell in cells:
        configuration = cell[:repeat_index] + cell[repeat_index + 1 :]
        seen.setdefault(_cell_key(configuration), configuration)
    return list(seen.values())


def _expanded(
    configurations: Sequence[GridCell], repeat_index: int, repeats: Repeats
) -> list[GridCell]:
    """Expand configurations into cells, each configuration's repeat values consecutive.

    :param Sequence[GridCell] configurations: Configurations without their repeat value.
    :param int repeat_index: Position of the repeat value in a cell.
    :param Repeats repeats: The phase's repeat axis, in configured value order.
    :return list[GridCell]: The cells.
    """
    return [
        configuration[:repeat_index] + (value,) + configuration[repeat_index:]
        for configuration in configurations
        for value in repeats.values
    ]


def grid_cell_order(
    search_spaces: Sequence[SearchSpaceEntry],
    *,
    seed: int | None,
    repeats: Repeats | None,
    phase_name: str,
) -> list[GridCell]:
    """Return the cell each draw position of a grid study is assigned, across its history.

    The first space's order is Optuna's seeded shuffle of its grid. Each later space keeps the
    first ``first_position`` cells of the order before it and appends a shuffle, seeded from
    ``[seed, index]``, of every other cell of its own grid, including earlier cells never
    drawn, so a partial grid still spreads over the grown space. Under repeats the shuffle is
    over configurations and each configuration's repeat values follow in configured order, so a
    ``first_position`` that is a whole number of configurations keeps every group whole.

    :param Sequence[SearchSpaceEntry] search_spaces: The study's search-space history.
    :param int | None seed: The phase's ``sampler.seed``; None shuffles as seed 0 does.
    :param Repeats | None repeats: The phase's repeat axis.
    :param str phase_name: Phase named in an enumeration error.
    :raises ValueError: The history is empty, a space cannot be enumerated, or a
        ``first_position`` lies past the order before it.
    :return list[GridCell]: Cells by draw position, each in :func:`grid_axis_names` order.
    """
    if not search_spaces:
        raise ValueError(f"Phase {phase_name!r} has no recorded search space.")
    names = grid_axis_names(list(search_spaces[0]["search_space"]), repeats)
    repeat_index = None if repeats is None else names.index(repeats.key)
    order: list[GridCell] = []
    for layer, entry in enumerate(search_spaces):
        first = entry["first_position"]
        if first > len(order):
            raise ValueError(
                f"Phase {phase_name!r} search space {layer} takes effect at draw position "
                f"{first}, past the {len(order)} cells before it."
            )
        kept = order[:first]
        cells = _grid_cells(entry["search_space"], repeats, phase_name)
        shuffle_seed = (seed or 0) if layer == 0 else _layer_seed(seed, layer)
        if repeats is None or repeat_index is None:
            drawn = {_cell_key(cell) for cell in kept}
            fresh = [cell for cell in cells if _cell_key(cell) not in drawn]
            order = kept + _shuffled(fresh, shuffle_seed)
            continue
        if layer == 0:
            # Optuna shuffles whole cells; configurations follow their first appearance.
            configurations = _configurations(_shuffled(cells, shuffle_seed), repeat_index)
        else:
            drawn = {_cell_key(item) for item in _configurations(kept, repeat_index)}
            configurations = _shuffled(
                [
                    item
                    for item in _configurations(cells, repeat_index)
                    if _cell_key(item) not in drawn
                ],
                shuffle_seed,
            )
        order = kept + _expanded(configurations, repeat_index, repeats)
    return order
