"""Time helpers for MCP persistence and audit records."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_utc_iso(value: object) -> datetime | None:
    """Parse a timezone-aware ISO-8601 timestamp, tolerating malformed values.

    :param object value: Raw timestamp from persisted MCP state.
    :return datetime | None: Parsed timestamp, or ``None`` for non-string,
        malformed, or timezone-naive values.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timezone-aware UTC timestamp.

    """
    return datetime.now(timezone.utc).isoformat()
