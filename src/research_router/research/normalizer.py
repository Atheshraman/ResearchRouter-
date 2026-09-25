"""Result normalizer — ensures all results conform to ResearchResult shape."""

from __future__ import annotations

from research_router.models.result import ResearchResult


def normalise_results(results: list[ResearchResult]) -> list[ResearchResult]:
    """Clean and validate a list of results.

    * Strips whitespace from title / snippet.
    * Removes results with no title AND no URL.
    """
    cleaned: list[ResearchResult] = []
    for r in results:
        title = (r.title or "").strip() or None
        url = (r.url or "").strip() or None
        snippet = (r.snippet or "").strip() or None

        if not title and not url:
            continue  # discard empty results

        cleaned.append(
            r.model_copy(
                update={
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
        )

    return cleaned
