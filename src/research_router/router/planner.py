"""Research planner — converts a ResearchIntent into a SearchPlan."""

from __future__ import annotations

from research_router.models.intent import ResearchDepth, ResearchDomain, ResearchIntent
from research_router.models.plan import SearchPlan
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

# ── domain → engine mapping ───────────────────────────────────────────

_DOMAIN_ENGINE_MAP: dict[ResearchDomain, str] = {
    ResearchDomain.GENERAL: "google",
    ResearchDomain.NEWS: "google_news",
    ResearchDomain.ACADEMIC: "google_scholar",
    ResearchDomain.JOBS: "google_jobs",
    ResearchDomain.SHOPPING: "google_shopping",
    ResearchDomain.PLACES: "google_maps",
}


class ResearchPlanner:
    """Convert a ``ResearchIntent`` into an executable ``SearchPlan``.

    Responsibilities:
    * Engine selection
    * Query construction
    * Parameter mapping
    * Sub-query generation for deep research
    """

    def __init__(self, max_results: int = 10, max_sub_queries: int = 5) -> None:
        self._max_results = max_results
        self._max_sub_queries = max_sub_queries

    def plan(self, intent: ResearchIntent) -> SearchPlan:
        """Build a ``SearchPlan`` from *intent*."""
        engine = _DOMAIN_ENGINE_MAP.get(intent.domain, "google")
        params = self._build_parameters(intent)
        query = intent.query

        sub_queries: list[SearchPlan] = []
        if intent.research_depth == ResearchDepth.DEEP or intent.requires_multiple_searches:
            sub_queries = self._generate_sub_queries(intent, engine)

        plan = SearchPlan(
            domain=intent.domain,
            engine=engine,
            query=query,
            parameters=params,
            max_results=self._max_results,
            research_depth=intent.research_depth,
            sub_queries=sub_queries,
            confidence=intent.confidence,
        )

        logger.info(
            "Search plan created",
            extra={
                "extra_data": {
                    "domain": plan.domain.value,
                    "engine": plan.engine,
                    "sub_queries": len(plan.sub_queries),
                }
            },
        )
        return plan

    # ── parameter building ────────────────────────────────────────

    def _build_parameters(self, intent: ResearchIntent) -> dict[str, object]:
        params: dict[str, object] = {}

        if intent.location:
            params["location"] = intent.location
        if intent.date_range:
            params["date_range"] = intent.date_range

        # Shopping-specific
        if intent.price_min is not None:
            params["price_min"] = intent.price_min
        if intent.price_max is not None:
            params["price_max"] = intent.price_max
        if intent.currency:
            params["currency"] = intent.currency

        # Jobs-specific
        if intent.job_type:
            params["job_type"] = intent.job_type
        if intent.remote is not None:
            params["remote"] = intent.remote
        if intent.experience_level:
            params["experience_level"] = intent.experience_level

        # Freshness
        if intent.freshness:
            params["freshness"] = intent.freshness

        # Pass through any extra constraints
        params.update(intent.constraints)

        return params

    # ── sub-query generation ──────────────────────────────────────

    def _generate_sub_queries(self, intent: ResearchIntent, engine: str) -> list[SearchPlan]:
        """Generate sub-queries for deep / multi-search research."""
        keywords = intent.keywords or intent.query.split()
        base_query = intent.query
        sub_queries: list[SearchPlan] = []

        # Strategy: create angle-shifted queries
        angles = self._query_angles(base_query, keywords)
        for angle_query in angles[: self._max_sub_queries]:
            sub_queries.append(
                SearchPlan(
                    domain=intent.domain,
                    engine=engine,
                    query=angle_query,
                    parameters=self._build_parameters(intent),
                    max_results=self._max_results,
                    research_depth=ResearchDepth.QUICK,
                    confidence=intent.confidence,
                )
            )

        return sub_queries

    @staticmethod
    def _query_angles(base: str, keywords: list[str]) -> list[str]:
        """Create varied search angles from a base query."""
        angles: list[str] = []

        # Add the base query as-is (it will be the primary search)
        # Generate supplementary angles
        if len(keywords) >= 2:
            angles.append(f"{base} techniques")
            angles.append(f"{base} best practices")
            angles.append(f"{base} recent advances")
            angles.append(f"{base} challenges")

        return angles
