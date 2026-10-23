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


def nonempty_text(value: object, *, label: str) -> str:
    """Require a nonempty string.

    :param object value: Candidate string.
    :param str label: Field name used in the refusal.
    :raises ValueError: The value is not a nonempty string.
    :return str: The supplied string, unchanged.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _reject_constant(value: str) -> NoReturn:
    """Reject Python's non-standard JSON constants with ValueError."""
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object with unique member names.

    :param pairs: Decoded JSON members in source order.
    :raises ValueError: A member name occurs more than once.
    :return: A mapping of the supplied member names to their values.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Parse JSON with unique keys, rejecting Infinity, -Infinity and NaN.

    :param str text: JSON document to decode.
    :raises json.JSONDecodeError: The document has invalid JSON syntax.
    :raises ValueError: An object has duplicate keys or a nonstandard numeric constant occurs.
    :return: The decoded JSON value.
    """
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def progress_position(value: Any) -> int | float | None:
    """Return a finite, nonnegative JSON number or None, excluding booleans.

    Keep integers exact for evaluation-axis grouping.

    :param value: Candidate evaluation position.
    :return int | float | None: The valid position unchanged, or None for an unsupported value.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def json_value(value: object) -> JsonValue:
    """Rebuild decoded JSON containers with narrowed types.

    :param object value: Decoded value whose containers need narrowing.
    :raises ValueError: An object key is not a string or a value is outside the JSON model.
    :return JsonValue: The scalar or recursively rebuilt JSON container.
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
    """Narrow a decoded JSON object.

    :param object value: Decoded value expected to be an object.
    :raises ValueError: The value is not an object or contains unsupported keys or values.
    :return JsonObject: The recursively narrowed object.
    """
    narrowed = json_value(value)
    if not isinstance(narrowed, dict):
        raise ValueError(f"Expected a JSON object, not {type(value).__name__}.")
    return narrowed


def dump_json(model: BaseModel) -> JsonObject:
    """Dump a Pydantic model's fields as a JSON-mode object.

    :param BaseModel model: Model to serialize.
    :raises ValueError: The serialized value is not a JSON object.
    :return JsonObject: The model's JSON-mode fields.
    """
    dumped: object = model.model_dump(mode="json")
    return json_object(dumped)


def json_text(record: Mapping[str, object], key: str) -> str:
    """Read a string member.

    :param Mapping[str, object] record: Object containing the member.
    :param str key: Member name to read.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is not a string.
    :return str: The member value, unchanged.
    """
    value = record[key]
    if not isinstance(value, str):
        raise ValueError(f"JSON member {key!r} must be a string, not {type(value).__name__}.")
    return value


def json_optional_text(record: Mapping[str, object], key: str) -> str | None:
    """Read a string-or-null member.

    :param Mapping[str, object] record: Object containing the member.
    :param str key: Member name to read.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is neither a string nor None.
    :return str | None: The member value, unchanged.
    """
    value = record[key]
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"JSON member {key!r} must be a string or null, not {type(value).__name__}."
        )
    return value


def json_integer(record: Mapping[str, object], key: str) -> int:
    """Read an integer member, excluding booleans.

    :param Mapping[str, object] record: Object containing the member.
    :param str key: Member name to read.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is not an integer or is a boolean.
    :return int: The member value, unchanged.
    """
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"JSON member {key!r} must be an integer, not {type(value).__name__}.")
    return value


def json_number(record: Mapping[str, object], key: str) -> float:
    """Read a numeric member unchanged, rejecting booleans; absence raises KeyError.

    Wrong types raise ValueError. Integers remain exact; finiteness is not checked.

    :param Mapping[str, object] record: Object containing the member.
    :param str key: Member name to read.
    :raises KeyError: The member is absent.
    :raises ValueError: The member is not numeric or is a boolean.
    :return float: The numeric member unchanged, retaining an integer's exact value.
    """
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"JSON member {key!r} must be a number, not {type(value).__name__}.")
    return value


def json_text_map(value: object) -> dict[str, str] | None:
    """Copy a string-to-string object, or return None for other shapes.

    :param object value: Candidate object to narrow.
    :return dict[str, str] | None: A copy when every key and value is a string, otherwise None.
    """
    if not isinstance(value, dict):
        return None
    narrowed: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            return None
        narrowed[key] = item
    return narrowed


def json_text_list(value: object) -> list[str] | None:
    """Copy a list of strings, or return None for other shapes.

    :param object value: Candidate list to narrow.
    :return list[str] | None: A copy when every item is a string, otherwise None.
    """
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, str)]
    return items if len(items) == len(value) else None


def json_number_map(value: object) -> dict[str, float] | None:
    """Copy a string-to-number object unchanged, or return None for other shapes.

    Booleans are excluded; integers remain exact and finiteness is not checked.

    :param object value: Candidate object to narrow.
    :return dict[str, float] | None: A copy with string keys and numeric values, or None for other shapes.
    """
    if not isinstance(value, dict):
        return None
    narrowed: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int | float):
            return None
        narrowed[key] = item
    return narrowed
