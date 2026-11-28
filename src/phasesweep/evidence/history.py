"""Retain tracker history across attempts of one trial."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")


def merge_attempt_histories(
    histories: Sequence[Sequence[tuple[int | float, _T]]],
) -> list[_T]:
    """Replace a metric's earlier suffix from each newer attempt's first position.

    :param Sequence[Sequence[tuple[int | float, _T]]] histories: Positions and records
        for one metric, oldest attempt first. An empty attempt preserves the prefix.
    :return list[_T]: Retained records, in newest-attempt order.
    """
    records = []
    cutoff: int | float | None = None
    for history in reversed(histories):
        next_cutoff = cutoff
        for step, record in history:
            if cutoff is None or step < cutoff:
                records.append(record)
                if next_cutoff is None or step < next_cutoff:
                    next_cutoff = step
        cutoff = next_cutoff
    return records
