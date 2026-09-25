"""Date / time helpers with timezone awareness.

Uses :mod:`zoneinfo` (stdlib ≥ 3.9) so no third-party dependency is needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_DEFAULT_TZ = ZoneInfo("UTC")


def now(tz: str | ZoneInfo = _DEFAULT_TZ) -> datetime:
    """Return the current timezone-aware datetime."""
    if isinstance(tz, str):
        tz = ZoneInfo(tz)
    return datetime.now(tz=tz)


def date_range_for(label: str, tz: str | ZoneInfo = _DEFAULT_TZ) -> tuple[str, str]:
    """Convert a human date-range label into ``(start, end)`` ISO-8601 strings.

    Supported labels:
        ``today``, ``this_week``, ``this_month``, ``past_hour``,
        ``past_24h``, ``past_week``, ``past_month``, ``past_year``,
        ``recent`` (alias for ``past_week``).

    Returns ISO-8601 date strings (``YYYY-MM-DD``).
    """
    current = now(tz)
    today_str = current.strftime("%Y-%m-%d")

    mapping: dict[str, tuple[timedelta, timedelta]] = {
        "today": (timedelta(days=0), timedelta(days=0)),
        "past_hour": (timedelta(hours=1), timedelta(0)),
        "past_24h": (timedelta(days=1), timedelta(0)),
        "this_week": (timedelta(days=current.weekday()), timedelta(0)),
        "past_week": (timedelta(days=7), timedelta(0)),
        "recent": (timedelta(days=7), timedelta(0)),
        "this_month": (timedelta(days=current.day - 1), timedelta(0)),
        "past_month": (timedelta(days=30), timedelta(0)),
        "past_year": (timedelta(days=365), timedelta(0)),
    }

    label_lower = label.lower().strip()
    if label_lower not in mapping:
        # Unknown label — return a generous 7-day window
        start = current - timedelta(days=7)
        return start.strftime("%Y-%m-%d"), today_str

    delta_back, _ = mapping[label_lower]
    start = current - delta_back
    return start.strftime("%Y-%m-%d"), today_str
