"""Strict JSON parsing shared by result evidence and installer edits."""

from __future__ import annotations

import json
import math
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
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def strict_json_loads(text: str, *, finite_floats: bool = False) -> Any:
    """Parse strict JSON with unique keys and optional finite-float enforcement.

    :param str text: Complete JSON document.
    :param bool finite_floats: Reject finite-syntax floats that overflow to infinity.
    :return Any: Parsed JSON value.
    """
    if finite_floats:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
