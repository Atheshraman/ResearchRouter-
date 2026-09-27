"""Input validation helpers."""

from __future__ import annotations

import re


def validate_query(query: str) -> str:
    """Validate and sanitize a research query string.

    Raises ``ValueError`` when the query is empty or too long.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    cleaned = query.strip()

    if len(cleaned) > 2000:
        raise ValueError("Query exceeds the maximum length of 2 000 characters.")

    return cleaned


MIN_CONTEXT_BUDGET = 200
MAX_CONTEXT_BUDGET = 100_000
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")


def validate_context_budget(budget: int | None) -> int | None:
    """``None`` means "use the configured default"."""
    if budget is None:
        return None
    if not MIN_CONTEXT_BUDGET <= budget <= MAX_CONTEXT_BUDGET:
        raise ValueError(
            f"context_budget must be between {MIN_CONTEXT_BUDGET} and {MAX_CONTEXT_BUDGET} tokens."
        )
    return budget


def validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ValueError(
            "session_id must be 1-64 characters of letters, digits, '_', '.', ':', '-'."
        )
    return session_id
