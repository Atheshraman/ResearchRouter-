"""Google Jobs engine adapter."""

from __future__ import annotations

from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


class JobsEngine(SearchEngine):
    """Adapter for the ``google_jobs`` SerpApi engine."""

    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "google_jobs"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self.engine_name,
            "q": plan.query,
        }
        if plan.parameters.get("location"):
            params["location"] = plan.parameters["location"]
        # Date posted filter
        date_range = plan.parameters.get("date_range")
        chip_map = {
            "today": "date_posted:today",
            "this_week": "date_posted:week",
            "past_week": "date_posted:week",
            "past_month": "date_posted:month",
        }
        if date_range and date_range in chip_map:
            params["chips"] = chip_map[date_range]
        return params

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for item in raw.get("jobs_results", []):
            results.append(
                ResearchResult(
                    title=item.get("title"),
                    url=item.get("apply_options", [{}])[0].get("link")
                    if item.get("apply_options")
                    else item.get("share_link"),
                    snippet=item.get("description", "")[:500],
                    source=item.get("company_name"),
                    metadata={
                        "company": item.get("company_name"),
                        "location": item.get("location"),
                        "via": item.get("via"),
                        "posted_at": item.get("detected_extensions", {}).get("posted_at"),
                        "schedule": item.get("detected_extensions", {}).get("schedule_type"),
                        "salary": item.get("detected_extensions", {}).get("salary"),
                    },
                )
            )
        return results
