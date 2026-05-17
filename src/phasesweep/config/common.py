"""Shared config model primitives, scalar checks, and override-key validation."""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Base for all config models: frozen + reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_finite(label: str, value: float) -> None:
    """Raise ``ValueError`` if ``value`` is NaN or +/-inf.

    Used at config-load to keep silently-broken bounds out of the runtime
    (review v0.5.2 / blocker 3): a constraint with ``max: .nan`` would otherwise
    be vacuous because ``x > nan`` is always ``False``.

    Args:
        label: Human-readable name of the field (e.g. ``"constraint.max"``);
            included verbatim in the error message.
        value: The numeric value to validate.

    Raises:
        ValueError: If ``value`` is not finite (``math.isfinite`` returns False).

    """
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite; got {value!r}")


def check_bounds(value: float, *, min_value: float | None, max_value: float | None) -> bool:
    """Return whether ``value`` is finite and inside optional numeric bounds."""
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
    """Reject empty, non-finite, or inverted optional bounds."""
    if min_value is None and max_value is None:
        raise ValueError(f"{label} must define at least one of min/max.")
    if min_value is not None:
        _require_finite(f"{label} min", min_value)
    if max_value is not None:
        _require_finite(f"{label} max", max_value)
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"{label}: min ({min_value}) > max ({max_value}).")


def _validate_safe_name(kind: str, value: str) -> str:
    """Reject empty names and characters unsafe for path/study-name components."""
    if not value or not all(c.isalnum() or c in "_-" for c in value):
        raise ValueError(f"{kind} name {value!r} must be non-empty and [A-Za-z0-9_-] only.")
    return value


_OVERRIDE_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_override_key(key: object, *, label: str) -> None:
    """Reject override keys that would render to ambiguous shell arguments.

    Args:
        key: The candidate key from ``fixed_overrides`` or ``search_space``.
            Must be a non-empty string of dotted ``[A-Za-z0-9_-]`` segments.
        label: Human-readable context prepended to any error message
            (e.g. ``"phase 'lr' search_space"``).

    Raises:
        ValueError: ``key`` is not a string; empty; has leading/trailing or
            embedded whitespace; or contains an empty dotted segment or a
            segment with disallowed characters.

    """
    if not isinstance(key, str):
        raise ValueError(
            f"{label}: override key must be a string, got {type(key).__name__}: {key!r}."
        )
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


def _key_parts(key: str) -> tuple[str, ...]:
    """Split a validated dotted override key into path components."""
    return tuple(key.split("."))


def _find_prefix_collisions(keys: set[str]) -> list[tuple[str, str]]:
    """Return pairs ``(short, long)`` where ``short`` is a strict path-prefix of ``long``.

    Two keys collide when one's dot-path is a strict prefix of the other's.

    Examples:
        * ``model`` and ``model.depth`` collide — argparse/Hydra renders
          contradictory flags or ``model=llama model.depth=16``, and ``json_file``
          cannot represent both a scalar and a nested object at the same key.
        * ``model.depth`` and ``model.depths`` do **not** collide (different
          siblings, same depth).
        * ``a.b.c`` and ``a.b.c.d`` do collide.

    Used in ``Experiment._validate_phase_graph`` (review v0.5.3 / blocker 5).

    Args:
        keys: The combined fixed-overrides / search-space / inherited key set
            for one phase.

    Returns:
        Lexically-sorted ``(short, long)`` pairs where ``short`` is a strict
        path-prefix of ``long``. Empty list when no collisions are present.

    """
    parts_by_key = {key: _key_parts(key) for key in keys}
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
