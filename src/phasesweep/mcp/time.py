"""Time helpers for MCP persistence and audit records."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timezone-aware UTC timestamp.

    """
    return datetime.now(timezone.utc).isoformat()
