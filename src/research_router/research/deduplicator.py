"""URL-based result deduplication."""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from research_router.models.result import ResearchResult

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "msclkid",
        "ref",
        "source",
        "sxsrf",
    }
)


def _normalise_url(url: str) -> str:
    """Normalise a URL for deduplication purposes.

    * Lower-case scheme and host
    * Remove trailing slash
    * Remove fragment
    * Remove common tracking query parameters
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return url.lower().rstrip("/")

    # Remove tracking params
    qs = parse_qs(parsed.query, keep_blank_values=False)
    cleaned_qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
    new_query = urlencode(cleaned_qs, doseq=True) if cleaned_qs else ""

    normalised = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            parsed.params,
            new_query,
            "",  # no fragment
        )
    )
    return normalised


def deduplicate(results: list[ResearchResult]) -> list[ResearchResult]:
    """Remove duplicate results based on normalised URLs.

    Keeps the first occurrence of each URL.  Results without URLs are
    always kept (they can't be deduped).
    """
    seen_urls: set[str] = set()
    unique: list[ResearchResult] = []

    for r in results:
        if not r.url:
            unique.append(r)
            continue

        norm = _normalise_url(r.url)
        if norm not in seen_urls:
            seen_urls.add(norm)
            unique.append(r)

    return unique
