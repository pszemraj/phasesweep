"""Shared config model primitives, scalar checks, and override-key validation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated, TypeAlias, TypeGuard
from urllib.parse import SplitResult, urlunsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
)

SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

#: Validation-context key holding the absolute directory of the config file
#: being loaded. Relative path fields resolve against it.
CONFIG_DIR_CONTEXT = "config_dir"


def _normalize_http_endpoint(parsed: SplitResult) -> str | None:
    """Normalize an HTTP(S) endpoint without nonempty credentials, query, or fragment.

    :param SplitResult parsed: Endpoint already parsed by ``urlsplit``.
    :return str | None: Lowercase scheme and authority with trailing path slashes
        removed, or ``None`` when the endpoint is invalid.
    """
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any((parsed.username, parsed.password, parsed.query, parsed.fragment))
    ):
        return None
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )


def anchor_relative_path(value: str, info: ValidationInfo) -> str:
    """Resolve a relative config path against the loaded config file's directory.

    Absolute and home-relative paths stay unchanged. Without a config directory
    in the validation context, relative paths keep working-directory semantics.

    :param str value: Nonempty configured path.
    :param ValidationInfo info: Validation info whose context may carry
        :data:`CONFIG_DIR_CONTEXT`.
    :return str: ``value`` joined onto the config directory when it is relative,
        otherwise ``value`` unchanged.
    """
    context = info.context
    config_dir = context.get(CONFIG_DIR_CONTEXT) if isinstance(context, dict) else None
    if not isinstance(config_dir, Path) or value.startswith("~") or Path(value).is_absolute():
        return value
    return str(config_dir / value)


_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")


def is_sha256_hex(value: object) -> TypeGuard[str]:
    """Recognize a SHA-256 digest as written by PhaseSweep.

    :param object value: Candidate digest to check.
    :return bool: Whether ``value`` is a lowercase 64-character hex digest.
    """
    return isinstance(value, str) and bool(_SHA256_HEX_PATTERN.fullmatch(value))


def _reject_boolean_number(value: object) -> object:
    """Reject YAML booleans before Pydantic coerces them to zero or one.

    :param object value: Raw numeric config value.
    :raises ValueError: If the value is a boolean.
    :return object: The unchanged non-boolean value.
    """
    if isinstance(value, bool):
        raise ValueError("numeric config values must be numbers, not booleans")
    return value


ConfigInt: TypeAlias = Annotated[int, BeforeValidator(_reject_boolean_number)]
ConfigFloat: TypeAlias = Annotated[float, BeforeValidator(_reject_boolean_number)]

StrictJsonScalar: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr | None
"""The exact set of values a JSON document can hold at a leaf position.

Strict (non-coercing) members on purpose: ``StrictInt`` rejects ``bool`` and
``StrictStr`` rejects a YAML ``date``, so nothing is silently widened into a
neighbouring type. Anything outside this union — a mapping, a sequence, a
``datetime.date`` — is not representable in parsed JSON and is rejected at
config load rather than normalized later.
"""


class _Frozen(BaseModel):
    """Base for all config models: frozen + reject unknown keys."""

    # Config values can contain storage credentials and environment secrets.
    # Pydantic normally repeats the rejected input in ValidationError text,
    # which would leak those values through CLI diagnostics.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


def _require_finite(label: str, value: float) -> None:
    """Reject NaN and infinite numeric values.

    :param str label: Field label included in the error.
    :param float value: Numeric value to validate.
    :raises ValueError: The value is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite; got {value!r}")


def check_bounds(value: float, *, min_value: float | None, max_value: float | None) -> bool:
    """Return whether ``value`` is finite and inside optional numeric bounds.

    :param float value: The numeric value to test.
    :param float | None min_value: Optional inclusive lower bound.
    :param float | None max_value: Optional inclusive upper bound.
    :return bool: ``True`` when ``value`` is finite and within all supplied bounds.
    """
    if not math.isfinite(value):
        return False
    if min_value is not None and value < min_value:
        return False
    return not (max_value is not None and value > max_value)


def _validate_optional_bounds(
    *,
    label: str,
    min_value: float | None,
    max_value: float | None,
) -> None:
    """Reject empty, non-finite, or inverted optional bounds.

    :param str label: Human-readable label included in validation errors.
    :param float | None min_value: Optional inclusive lower bound.
    :param float | None max_value: Optional inclusive upper bound.
    :raises ValueError: If no bound is set, a bound is non-finite, or ``min > max``.
    """
    if min_value is None and max_value is None:
        raise ValueError(f"{label} must define at least one of min/max.")
    if min_value is not None:
        _require_finite(f"{label} min", min_value)
    if max_value is not None:
        _require_finite(f"{label} max", max_value)
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"{label}: min ({min_value}) > max ({max_value}).")


def _validate_safe_name(kind: str, value: str) -> str:
    """Reject empty names and characters unsafe for path/study-name components.

    :param str kind: Human-readable object kind included in validation errors.
    :param str value: Candidate name to validate.
    :raises ValueError: If ``value`` is empty or contains an unsafe character.
    :return str: The validated name, unchanged.
    """
    if not SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{kind} name {value!r} must be non-empty and [A-Za-z0-9_-] only.")
    return value


_OVERRIDE_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_override_key(key: str, *, label: str) -> None:
    """Require nonempty dotted override keys with ``[A-Za-z0-9_-]`` segments.

    :param str key: Candidate fixed or search-space key.
    :param str label: Context included in validation errors.
    :raises ValueError: A segment is empty, padded, or contains disallowed characters.
    """
    if not key:
        raise ValueError(f"{label}: override key cannot be empty.")
    if key != key.strip():
        raise ValueError(f"{label}: override key {key!r} has leading or trailing whitespace.")
    if any(c.isspace() for c in key):
        raise ValueError(f"{label}: override key {key!r} contains whitespace.")
    parts = key.split(".")
    for i, part in enumerate(parts):
        if not part:
            raise ValueError(
                f"{label}: override key {key!r} has an empty dotted segment at "
                f"position {i}. Use simple keys like 'lr' or dotted keys like "
                f"'model.depth'; do not use leading/trailing dots or '..'."
            )
        if not _OVERRIDE_KEY_SEGMENT.match(part):
            raise ValueError(
                f"{label}: override key {key!r} segment {part!r} contains "
                "invalid characters (allowed: alphanumerics, underscore, dash, "
                "and '.' as the segment separator)."
            )


def _find_prefix_collisions(keys: set[str]) -> list[tuple[str, str]]:
    """Find override keys that are strict dotted-path prefixes of other keys.

    ``model`` conflicts with ``model.depth``; ``model.depths`` does not.

    :param set[str] keys: Fixed, sampled, and inherited override keys.
    :return list[tuple[str, str]]: Lexically sorted ``(short, long)`` collisions.
    """
    parts_by_key = {key: tuple(key.split(".")) for key in keys}
    seen: set[tuple[str, str]] = set()

    items = list(parts_by_key.items())
    for i, (a, a_parts) in enumerate(items):
        for b, b_parts in items[i + 1 :]:
            if a_parts == b_parts:
                continue
            if len(a_parts) < len(b_parts):
                shorter_parts, shorter_key, longer_key = a_parts, a, b
                longer_parts = b_parts
            else:
                shorter_parts, shorter_key, longer_key = b_parts, b, a
                longer_parts = a_parts
            if longer_parts[: len(shorter_parts)] == shorter_parts:
                seen.add((shorter_key, longer_key))

    return sorted(seen)
