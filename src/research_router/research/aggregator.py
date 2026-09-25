"""Result aggregator — merges results from multiple searches."""

from __future__ import annotations

from research_router.models.result import ResearchResult, SearchError


def aggregate(
    all_results: list[list[ResearchResult]],
    all_errors: list[SearchError] | None = None,
) -> tuple[list[ResearchResult], list[SearchError]]:
    """Flatten multiple result lists into a single list.

    Returns ``(flat_results, errors)`` preserving original order within
    each sub-list.
    """
    flat: list[ResearchResult] = []
    for batch in all_results:
        flat.extend(batch)
    return flat, all_errors or []
