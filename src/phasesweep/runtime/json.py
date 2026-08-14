"""Strict JSON parsing shared by result evidence and installer edits."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import partial
from typing import Any, NoReturn


def _reject_constant(value: str) -> NoReturn:
    """Reject non-standard constants accepted by Python's JSON parser.

    :param str value: Non-standard constant token.
    :raises ValueError: Always.
    """
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting duplicate member names.

    :param list[tuple[str, Any]] pairs: Parsed members in source order.
    :return dict[str, Any]: Mapping containing each unique member.
    :raises ValueError: A member name repeats, which the standard ``json``
        parser would silently resolve last-wins.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    """Parse a JSON float and reject overflow to infinity.

    :param str value: Raw JSON numeric token.
    :return float: Finite parsed value.
    :raises ValueError: The token is not a valid float, or it overflows to
        infinity.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _finite_then(hook: Callable[[str], Any], value: str) -> Any:
    """Enforce finiteness on a float token before handing it to a caller's hook.

    :param Callable[[str], Any] hook: Caller hook receiving the validated token.
    :param str value: Raw JSON numeric token.
    :return Any: Whatever ``hook`` builds from the token.
    :raises ValueError: The token is not a valid float, or it overflows to
        infinity.
    """
    _finite_float(value)
    return hook(value)


def strict_json_loads(
    text: str,
    *,
    finite_floats: bool = False,
    parse_float: Callable[[str], Any] | None = None,
) -> Any:
    """Parse strict JSON with unique keys and optional finite-float enforcement.

    ``parse_float`` lets a caller keep each float's raw source token instead of
    a ``float``; it never weakens ``finite_floats``, which still rejects an
    overflowing token before the hook sees it. Non-standard ``Infinity``,
    ``-Infinity``, and ``NaN`` reach ``parse_constant`` rather than either float
    path and are always rejected.

    :param str text: Complete JSON document.
    :param bool finite_floats: Reject finite-syntax floats that overflow to infinity.
    :param Callable[[str], Any] | None parse_float: Hook building each float
        value from its raw token, or ``None`` for standard ``float`` parsing.
    :return Any: Parsed JSON value.
    """
    hook: Callable[[str], Any] | None = parse_float
    if finite_floats:
        hook = _finite_float if parse_float is None else partial(_finite_then, parse_float)
    if hook is None:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_float=hook,
        parse_constant=_reject_constant,
    )
