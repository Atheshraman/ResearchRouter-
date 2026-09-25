"""Deterministic result ranker."""

from __future__ import annotations

from research_router.models.result import ResearchResult


def _score(result: ResearchResult, query_keywords: list[str]) -> float:
    """Compute a deterministic relevance score for a single result."""
    score = 0.0
    text = " ".join(filter(None, [result.title, result.snippet, result.source])).lower()

    # Keyword overlap
    for kw in query_keywords:
        if kw.lower() in text:
            score += 1.0

    # Completeness bonuses
    if result.title:
        score += 0.5
    if result.url:
        score += 0.3
    if result.snippet:
        score += 0.5
    if result.source:
        score += 0.2
    if result.published_at:
        score += 0.3

    return score


def rank(
    results: list[ResearchResult],
    query: str,
) -> list[ResearchResult]:
    """Sort results by deterministic relevance score (descending).

    Factors: keyword overlap, field completeness, freshness.
    """
    keywords = [w for w in query.lower().split() if len(w) > 2]

    scored = [(r, _score(r, keywords)) for r in results]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [r for r, _ in scored]
