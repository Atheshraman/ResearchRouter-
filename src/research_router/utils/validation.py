"""Input validation helpers."""

from __future__ import annotations


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
