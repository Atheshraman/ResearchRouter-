"""Google general web search engine adapter."""

from __future__ import annotations

from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


class GoogleEngine(SearchEngine):
    """Adapter for the ``google`` SerpApi engine (organic web results)."""

    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "google"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self.engine_name,
            "q": plan.query,
            "num": plan.max_results,
        }
        if plan.parameters.get("location"):
            params["location"] = plan.parameters["location"]
        if plan.parameters.get("gl"):
            params["gl"] = plan.parameters["gl"]
        if plan.parameters.get("hl"):
            params["hl"] = plan.parameters["hl"]
        return params

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for item in raw.get("organic_results", []):
            results.append(
                ResearchResult(
                    title=item.get("title"),
                    url=item.get("link"),
                    snippet=item.get("snippet"),
                    source=item.get("source"),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in ("title", "link", "snippet", "source", "position")
                    },
                )
            )
        return results
