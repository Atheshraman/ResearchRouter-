"""Google News engine adapter."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient
from research_router.utils.dates import now


class NewsEngine(SearchEngine):
    """Adapter for the ``google_news`` SerpApi engine."""

    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "google_news"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self.engine_name,
            "q": plan.query,
        }
        if plan.parameters.get("location"):
            params["gl"] = plan.parameters["location"]
        return params

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for item in raw.get("news_results", []):
            results.append(
                ResearchResult(
                    title=item.get("title"),
                    url=item.get("link"),
                    snippet=item.get("snippet"),
                    source=item.get("source", {}).get("name")
                    if isinstance(item.get("source"), dict)
                    else item.get("source"),
                    metadata={
                        "date": item.get("date"),
                        "thumbnail": item.get("thumbnail"),
                    },
                    published_at=_parse_news_date(item.get("date")),
                )
            )
        return results


def _parse_news_date(value: object) -> Any:
    """Convert common Google News relative dates to timezone-aware datetimes."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in {"just now", "now"}:
        return now()
    match = re.fullmatch(r"(\d+)\s+(minute|hour|day|week)s?\s+ago", text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = {
            "minute": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
        }[unit]
        return now().astimezone(UTC) - delta
    for pattern in (
        "%m/%d/%Y, %I:%M %p, %z UTC",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value.strip(), pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None
