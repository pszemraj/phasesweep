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
    """Normalize an HTTP(S) endpoint, or return None when invalid.

    Reject nonempty credentials, query or fragment; lowercase scheme and authority
    and strip trailing path slashes.
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
    """Anchor relative paths to the config file's directory in validation context.

    Absolute and home-relative paths stay unchanged. Without a config directory,
    relative paths keep working-directory semantics.
    """
    context = info.context
    config_dir = context.get(CONFIG_DIR_CONTEXT) if isinstance(context, dict) else None
    if not isinstance(config_dir, Path) or value.startswith("~") or Path(value).is_absolute():
        return value
    return str(config_dir / value)


_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")


def is_sha256_hex(value: object) -> TypeGuard[str]:
    """Recognize lowercase, 64-character SHA-256 hex digests."""
    return isinstance(value, str) and bool(_SHA256_HEX_PATTERN.fullmatch(value))


def _reject_boolean_number(value: object) -> object:
    """Reject YAML booleans with ValueError before numeric coercion."""
    if isinstance(value, bool):
        raise ValueError("numeric config values must be numbers, not booleans")
    return value


ConfigInt: TypeAlias = Annotated[int, BeforeValidator(_reject_boolean_number)]
ConfigFloat: TypeAlias = Annotated[float, BeforeValidator(_reject_boolean_number)]

StrictJsonScalar: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr | None
"""Non-coercing JSON leaf scalars; reject booleans as integers and YAML dates.

Mappings and sequences are not scalars.
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
    """Reject NaN and infinity with a field-labeled ValueError."""
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite; got {value!r}")


def check_bounds(value: float, *, min_value: float | None, max_value: float | None) -> bool:
    """Test finiteness and optional inclusive bounds."""
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
    """Reject empty, non-finite or inverted inclusive bounds with ValueError."""
    if min_value is None and max_value is None:
        raise ValueError(f"{label} must define at least one of min/max.")
    if min_value is not None:
        _require_finite(f"{label} min", min_value)
    if max_value is not None:
        _require_finite(f"{label} max", max_value)
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"{label}: min ({min_value}) > max ({max_value}).")


def _validate_safe_name(kind: str, value: str) -> str:
    """Require nonempty [A-Za-z0-9_-] path/study names, raising ValueError otherwise."""
    if not SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{kind} name {value!r} must be non-empty and [A-Za-z0-9_-] only.")
    return value


def _validate_override_key(key: str, *, label: str) -> None:
    """Require nonempty dotted [A-Za-z0-9_-] segments, raising ValueError otherwise."""
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
        if not SAFE_NAME_PATTERN.fullmatch(part):
            raise ValueError(
                f"{label}: override key {key!r} segment {part!r} contains "
                "invalid characters (allowed: alphanumerics, underscore, dash, "
                "and '.' as the segment separator)."
            )


def _find_prefix_collisions(keys: set[str]) -> list[tuple[str, str]]:
    """Return sorted pairs of override keys with a strict dotted-prefix collision.

    model conflicts with model.depth; model.depths does not.
    """
    collisions: list[tuple[str, str]] = []
    for key in keys:
        prefix = key
        while "." in prefix:
            prefix = prefix.rsplit(".", 1)[0]
            if prefix in keys:
                collisions.append((prefix, key))
    return sorted(collisions)
