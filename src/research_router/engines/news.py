"""Google News engine adapter."""

from __future__ import annotations

from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


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
                )
            )
        return results
