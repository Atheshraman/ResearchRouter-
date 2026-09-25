"""Google Scholar engine adapter."""

from __future__ import annotations

from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


class ScholarEngine(SearchEngine):
    """Adapter for the ``google_scholar`` SerpApi engine."""

    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "google_scholar"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self.engine_name,
            "q": plan.query,
            "num": min(plan.max_results, 20),  # Scholar caps at 20
        }
        if plan.parameters.get("date_range") in ("recent", "past_year"):
            params["as_ylo"] = "2024"  # last ~2 years
        return params

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for item in raw.get("organic_results", []):
            pub_info = item.get("publication_info", {})
            results.append(
                ResearchResult(
                    title=item.get("title"),
                    url=item.get("link"),
                    snippet=item.get("snippet"),
                    source=pub_info.get("summary") if isinstance(pub_info, dict) else None,
                    metadata={
                        "cited_by": item.get("inline_links", {}).get("cited_by", {}).get("total"),
                        "year": pub_info.get("summary", "").split(",")[-1].strip()
                        if isinstance(pub_info, dict)
                        else None,
                        "resources": item.get("resources"),
                    },
                )
            )
        return results
