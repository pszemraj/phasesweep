"""Strict JSON parsing shared by result evidence and runtime state."""

from __future__ import annotations

import json
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


def strict_json_loads(text: str) -> Any:
    """Parse strict JSON with unique keys and no non-standard constants.

    Non-standard ``Infinity``, ``-Infinity``, and ``NaN`` reach
    ``parse_constant`` and are rejected.

    :param str text: Complete JSON document.
    :return Any: Parsed JSON value.
    """
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
