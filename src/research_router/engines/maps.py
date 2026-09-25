"""Google Maps / Places engine adapter."""

from __future__ import annotations

from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


class MapsEngine(SearchEngine):
    """Adapter for the ``google_maps`` SerpApi engine."""

    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "google_maps"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self.engine_name,
            "q": plan.query,
            "type": "search",
        }
        if plan.parameters.get("location"):
            params["ll"] = plan.parameters.get("ll", "")
            # If no lat/long, set the textual location in the query
            if not params["ll"]:
                params["q"] = f"{plan.query} in {plan.parameters['location']}"
                del params["ll"]
        return params

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for item in raw.get("local_results", []):
            results.append(
                ResearchResult(
                    title=item.get("title"),
                    url=item.get("website") or item.get("link"),
                    snippet=item.get("description"),
                    source="Google Maps",
                    metadata={
                        "address": item.get("address"),
                        "phone": item.get("phone"),
                        "rating": item.get("rating"),
                        "reviews": item.get("reviews"),
                        "type": item.get("type"),
                        "hours": item.get("hours"),
                        "gps_coordinates": item.get("gps_coordinates"),
                        "place_id": item.get("place_id"),
                    },
                )
            )
        return results
