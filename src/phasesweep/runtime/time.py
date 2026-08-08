"""UTC timestamp helpers shared across runtime contexts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal


def utc_now_iso(
    *,
    timespec: Literal[
        "auto", "hours", "minutes", "seconds", "milliseconds", "microseconds"
    ] = "auto",
) -> str:
    """Return the current UTC time as an ISO-8601 string.

    :param timespec: Timestamp precision accepted by :meth:`datetime.isoformat`.
    :return str: Timezone-aware UTC timestamp.
    """
    return datetime.now(UTC).isoformat(timespec=timespec)
