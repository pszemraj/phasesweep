"""Strict JSON parsing shared by result evidence and runtime state."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, NoReturn, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import BaseModel

JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"
"""Any value the JSON data model can carry, for payloads whose keys are open."""

JsonObject: TypeAlias = dict[str, JsonValue]
"""A JSON object: string keys over :data:`JsonValue`."""


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


def progress_position(value: Any) -> int | float | None:
    """Return a valid training-progress position, or ``None``.

    A position is where on an evaluation axis (step, tokens, a W&B step
    metric) an objective was measured: a finite, non-negative JSON number.
    Integers stay integers so step groups compare and render exactly.

    :param Any value: Candidate position read from evidence.
    :return int | float | None: ``value`` when valid, else ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def json_value(value: object) -> JsonValue:
    """Narrow a decoded value to the JSON data model, rebuilding its containers.

    :param object value: Value decoded from JSON or built from JSON-compatible parts.
    :raises ValueError: An object key is not a string, or a value lies outside the JSON model.
    :return JsonValue: An equal value whose lists and objects carry the narrowed type.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        narrowed: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object keys must be strings, not {type(key).__name__}.")
            narrowed[key] = json_value(item)
        return narrowed
    raise ValueError(f"{type(value).__name__} is not a JSON value.")


def json_object(value: object) -> JsonObject:
    """Narrow a decoded value to a JSON object.

    :param object value: Value decoded from JSON or built from JSON-compatible parts.
    :raises ValueError: ``value`` is not a JSON object.
    :return JsonObject: An equal object whose members carry the narrowed type.
    """
    narrowed = json_value(value)
    if not isinstance(narrowed, dict):
        raise ValueError(f"Expected a JSON object, not {type(value).__name__}.")
    return narrowed


def dump_json(model: BaseModel) -> JsonObject:
    """Dump a Pydantic model to its JSON-mode object.

    :param BaseModel model: Model to serialize.
    :return JsonObject: The model's fields in JSON-compatible form.
    """
    dumped: object = model.model_dump(mode="json")
    return json_object(dumped)


def json_text(record: Mapping[str, object], key: str) -> str:
    """Read a string member of a decoded JSON object.

    :param Mapping[str, object] record: Decoded JSON object.
    :param str key: Member name.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is not a string.
    :return str: The member's value.
    """
    value = record[key]
    if not isinstance(value, str):
        raise ValueError(f"JSON member {key!r} must be a string, not {type(value).__name__}.")
    return value


def json_optional_text(record: Mapping[str, object], key: str) -> str | None:
    """Read a string-or-null member of a decoded JSON object.

    :param Mapping[str, object] record: Decoded JSON object.
    :param str key: Member name.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is neither a string nor ``null``.
    :return str | None: The member's value.
    """
    value = record[key]
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"JSON member {key!r} must be a string or null, not {type(value).__name__}."
        )
    return value


def json_integer(record: Mapping[str, object], key: str) -> int:
    """Read an integer member of a decoded JSON object.

    :param Mapping[str, object] record: Decoded JSON object.
    :param str key: Member name.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is not an integer, or is a boolean.
    :return int: The member's value.
    """
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"JSON member {key!r} must be an integer, not {type(value).__name__}.")
    return value


def json_number(record: Mapping[str, object], key: str) -> float:
    """Read a numeric member of a decoded JSON object, keeping integers exact.

    :param Mapping[str, object] record: Decoded JSON object.
    :param str key: Member name.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is not a number, or is a boolean.
    :return float: The member's value, unchanged.
    """
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"JSON member {key!r} must be a number, not {type(value).__name__}.")
    return value


def json_text_map(value: object) -> dict[str, str] | None:
    """Narrow a decoded value to an object whose members are all strings.

    :param object value: Decoded JSON value.
    :return dict[str, str] | None: A copy of ``value``, or ``None`` when it is not such an object.
    """
    if not isinstance(value, dict):
        return None
    narrowed: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            return None
        narrowed[key] = item
    return narrowed


def json_number_map(value: object) -> dict[str, float] | None:
    """Narrow a decoded value to an object whose members are all numbers.

    :param object value: Decoded JSON value.
    :return dict[str, float] | None: A copy of ``value`` with members unchanged, or ``None``
        when it is not such an object; booleans are not numbers here.
    """
    if not isinstance(value, dict):
        return None
    narrowed: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int | float):
            return None
        narrowed[key] = item
    return narrowed
