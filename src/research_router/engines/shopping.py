"""Google Shopping engine adapter."""

from __future__ import annotations

from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


class ShoppingEngine(SearchEngine):
    """Adapter for the ``google_shopping`` SerpApi engine."""

    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "google_shopping"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self.engine_name,
            "q": plan.query,
            "num": plan.max_results,
        }
        if plan.parameters.get("location"):
            params["location"] = plan.parameters["location"]

        # Price filters
        price_min = plan.parameters.get("price_min")
        price_max = plan.parameters.get("price_max")
        if price_min is not None or price_max is not None:
            tbs_parts: list[str] = []
            if price_min is not None:
                tbs_parts.append(f"price:1,ppr_min:{int(price_min)}")
            if price_max is not None:
                tbs_parts.append(f"price:1,ppr_max:{int(price_max)}")
            params["tbs"] = ",".join(tbs_parts)

        return params

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for item in raw.get("shopping_results", []):
            results.append(
                ResearchResult(
                    title=item.get("title"),
                    url=item.get("link"),
                    snippet=item.get("snippet"),
                    source=item.get("source"),
                    metadata={
                        "price": item.get("price"),
                        "extracted_price": item.get("extracted_price"),
                        "rating": item.get("rating"),
                        "reviews": item.get("reviews"),
                        "thumbnail": item.get("thumbnail"),
                        "delivery": item.get("delivery"),
                    },
                )
            )
        return results
