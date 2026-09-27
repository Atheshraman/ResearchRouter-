"""MCP tool implementations.

Exposes:
* ``research`` — the primary intelligent research tool
* ``explain_research_plan`` — debug / explainability tool
"""

from __future__ import annotations

from typing import Any

from research_router.agent.metrics import MetricsAggregator, RequestMetrics
from research_router.agent.orchestrator import ResearchAgent
from research_router.agent.runner import TaskRunner
from research_router.agent.task_planner import TaskPlanner
from research_router.config import Settings
from research_router.context.manager import ContextManager
from research_router.context.reducer import ContextReducer
from research_router.context.tokens import estimate_tokens
from research_router.engines.google import GoogleEngine
from research_router.engines.jobs import JobsEngine
from research_router.engines.maps import MapsEngine
from research_router.engines.news import NewsEngine
from research_router.engines.registry import EngineRegistry
from research_router.engines.scholar import ScholarEngine
from research_router.engines.shopping import ShoppingEngine
from research_router.llm.base import LLMProvider
from research_router.llm.factory import create_llm_provider
from research_router.memory.base import MemoryStore, NullMemory
from research_router.memory.obsidian import ObsidianMemory
from research_router.models.intent import ResearchDepth
from research_router.research.cache import TTLCache
from research_router.research.executor import ResearchExecutor
from research_router.router.classifier import QueryClassifier
from research_router.router.intent import IntentAnalyser
from research_router.router.planner import ResearchPlanner
from research_router.serpapi.client import SerpApiClient
from research_router.utils.logging import get_logger
from research_router.utils.validation import (
    validate_context_budget,
    validate_query,
    validate_session_id,
)

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

        # Context-aware agent (wraps the components above; nothing is replaced)
        self._metrics = MetricsAggregator()
        self._memory: MemoryStore = (
            ObsidianMemory(settings.obsidian_vault_path, settings.obsidian_research_folder)
            if settings.obsidian_vault_path
            else NullMemory()
        )
        if settings.obsidian_vault_path and not self._memory.available:
            logger.warning("Obsidian vault not found; continuing without persistent memory")
        self._agent = ResearchAgent(
            task_planner=TaskPlanner(
                classifier=self._classifier,
                analyser=self._analyser,
                planner=self._planner,
                max_tasks=settings.max_tasks_per_request,
            ),
            runner=TaskRunner(
                self._executor,
                self._cache,
                timeout=settings.task_timeout,
                adaptive_expansion=settings.adaptive_expansion,
            ),
            reducer=ContextReducer(),
            context_manager=ContextManager(max_turns=settings.session_max_turns),
            memory=self._memory,
            aggregator=self._metrics,
            default_budget=settings.context_budget,
            auto_save=settings.obsidian_auto_save,
            memory_max_notes=settings.memory_max_notes,
        )

    async def research(
        self,
        query: str,
        max_results: int | None = None,
        depth: str = "standard",
        *,
        context_budget: int | None = None,
        debug: bool = False,
        session_id: str = "default",
        use_memory: bool = True,
        save_to_memory: bool | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Execute research.

        ``mode`` is ``"agent"`` (context-aware pipeline) or ``"legacy"``
        (original single-plan pipeline); ``None`` follows ``AGENT_MODE``.
        """
        query = validate_query(query)
        context_budget = validate_context_budget(context_budget)
        session_id = validate_session_id(session_id)
        use_agent = self._settings.agent_mode if mode is None else mode == "agent"
        if not use_agent:
            return await self.legacy_research(query, max_results=max_results, depth=depth)

        try:
            research_depth = ResearchDepth(depth)
        except ValueError:
            research_depth = ResearchDepth.STANDARD
        return await self._agent.research(
            query,
            max_results=max_results or self._settings.max_results,
            depth=research_depth,
            context_budget=context_budget,
            debug=debug,
            session_id=session_id,
            use_memory=use_memory,
            save_to_memory=save_to_memory,
        )

    async def legacy_research(
        self,
        query: str,
        max_results: int | None = None,
        depth: str = "standard",
    ) -> dict[str, Any]:
        """The original pipeline: one plan → execute → normalise/dedup/rank."""
        query = validate_query(query)
        metrics = RequestMetrics(mode="legacy")

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
            metrics.cache_hits = 1
            self._record_legacy(metrics, dict(cached))
            return dict(cached)

        # Execute
        response = await self._executor.execute(plan, query)
        result = response.model_dump()
        metrics.tool_calls = 1 + len(plan.sub_queries)
        metrics.parallel_tool_calls = max(1, len(plan.sub_queries))

        # Cache result
        self._cache.set(plan.engine, plan.query, cache_key_params, result)

        self._record_legacy(metrics, result)
        return result

    def _record_legacy(self, metrics: RequestMetrics, result: dict[str, Any]) -> None:
        """Measure the legacy pipeline so it can be compared with the agent."""
        meta = result.get("metadata", {})
        metrics.raw_results = int(meta.get("total_raw_results", 0))
        metrics.deduplicated_results = int(meta.get("after_dedup", 0))
        metrics.relevant_results = metrics.final_evidence = int(result.get("total_results", 0))
        # The legacy pipeline returns results unreduced: input == output.
        metrics.output_context_tokens = estimate_tokens(result.get("results", []))
        metrics.input_tokens_estimated = metrics.output_context_tokens
        metrics.response_tokens = estimate_tokens(result)
        metrics.errors = len(result.get("errors", []))
        metrics.finish()
        self._metrics.record(metrics)

    def metrics_summary(self) -> dict[str, Any]:
        """Average measured metrics per mode since the server started."""
        return self._metrics.summary()

    async def explain(self, query: str, session_id: str = "default") -> dict[str, Any]:
        """Return the research plan without executing it."""
        query = validate_query(query)
        session_id = validate_session_id(session_id)
        classification = self._classifier.classify(query)
        intent = await self._analyser.analyse(query)
        plan = self._planner.plan(intent)
        resolved, execution_plan = await self._agent.plan(
            query,
            session_id=session_id,
            depth=intent.research_depth,
            max_results=self._settings.max_results,
        )

        return {
            "agent_mode": self._settings.agent_mode,
            "execution_plan": execution_plan.as_dict(),
            "context": resolved.as_dict(),
            "memory": self._memory.status(),
            "session_metrics": self.metrics_summary(),
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
