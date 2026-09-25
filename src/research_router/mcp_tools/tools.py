"""MCP tool implementations.

Exposes:
* ``research`` — the primary intelligent research tool
* ``explain_research_plan`` — debug / explainability tool
"""

from __future__ import annotations

from typing import Any

from research_router.config import Settings
from research_router.engines.google import GoogleEngine
from research_router.engines.jobs import JobsEngine
from research_router.engines.maps import MapsEngine
from research_router.engines.news import NewsEngine
from research_router.engines.registry import EngineRegistry
from research_router.engines.scholar import ScholarEngine
from research_router.engines.shopping import ShoppingEngine
from research_router.llm.base import LLMProvider
from research_router.llm.factory import create_llm_provider
from research_router.models.intent import ResearchDepth
from research_router.research.cache import TTLCache
from research_router.research.executor import ResearchExecutor
from research_router.router.classifier import QueryClassifier
from research_router.router.intent import IntentAnalyser
from research_router.router.planner import ResearchPlanner
from research_router.serpapi.client import SerpApiClient
from research_router.utils.logging import get_logger
from research_router.utils.validation import validate_query

logger = get_logger(__name__)


class ResearchRouter:
    """High-level orchestrator wiring all subsystems together.

    Used by MCP tools and the CLI.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # SerpApi client
        self._serpapi = SerpApiClient(
            api_key=settings.serpapi_api_key,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

        # Engine registry
        self._registry = EngineRegistry()
        self._registry.register("google", GoogleEngine(self._serpapi))
        self._registry.register("google_news", NewsEngine(self._serpapi))
        self._registry.register("google_scholar", ScholarEngine(self._serpapi))
        self._registry.register("google_jobs", JobsEngine(self._serpapi))
        self._registry.register("google_shopping", ShoppingEngine(self._serpapi))
        self._registry.register("google_maps", MapsEngine(self._serpapi))

        # Classifier + LLM
        self._classifier = QueryClassifier()
        self._llm: LLMProvider | None = None
        if settings.google_api_key:
            try:
                self._llm = create_llm_provider(settings)
            except Exception:
                logger.warning("LLM provider init failed; using deterministic only")

        # Intent analyser
        self._analyser = IntentAnalyser(
            classifier=self._classifier,
            llm_provider=self._llm,
            confidence_threshold=settings.classifier_confidence_threshold,
        )

        # Planner + executor
        self._planner = ResearchPlanner(
            max_results=settings.max_results,
            max_sub_queries=settings.max_queries_per_request,
        )
        self._executor = ResearchExecutor(
            registry=self._registry,
            max_concurrent=settings.max_concurrent_searches,
        )

        # Cache
        self._cache = TTLCache(
            ttl=settings.cache_ttl,
            enabled=settings.cache_enabled,
        )

    async def research(
        self,
        query: str,
        max_results: int | None = None,
        depth: str = "standard",
    ) -> dict[str, Any]:
        """Execute the full research pipeline."""
        query = validate_query(query)

        # Override depth
        try:
            research_depth = ResearchDepth(depth)
        except ValueError:
            research_depth = ResearchDepth.STANDARD

        # Analyse intent
        intent = await self._analyser.analyse(query)
        intent = intent.model_copy(update={"research_depth": research_depth})
        if max_results:
            self._planner._max_results = max_results

        # Plan
        plan = self._planner.plan(intent)

        # Check cache
        cache_key_params = plan.parameters.copy()
        cached = self._cache.get(plan.engine, plan.query, cache_key_params)
        if cached is not None:
            return dict(cached)

        # Execute
        response = await self._executor.execute(plan, query)
        result = response.model_dump()

        # Cache result
        self._cache.set(plan.engine, plan.query, cache_key_params, result)

        return result

    async def explain(self, query: str) -> dict[str, Any]:
        """Return the research plan without executing it."""
        query = validate_query(query)
        classification = self._classifier.classify(query)
        intent = await self._analyser.analyse(query)
        plan = self._planner.plan(intent)

        return {
            "domain": plan.domain.value,
            "confidence": classification.confidence,
            "entities": {
                "location": intent.location,
                "date_range": intent.date_range,
                "price_min": intent.price_min,
                "price_max": intent.price_max,
                "currency": intent.currency,
            },
            "engine": plan.engine,
            "generated_query": plan.query,
            "parameters": plan.parameters,
            "research_depth": plan.research_depth.value,
            "sub_queries": len(plan.sub_queries),
            "reason": f"{plan.domain.value.title()} intent detected"
            + (
                f" with {classification.confidence:.0%} confidence"
                if classification.confidence > 0
                else ""
            ),
        }

    async def close(self) -> None:
        """Release resources."""
        await self._serpapi.close()
        if self._llm:
            await self._llm.close()
