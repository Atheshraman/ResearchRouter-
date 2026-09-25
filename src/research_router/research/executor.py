"""Research executor — runs search plans through engine adapters.

Handles:
* Single-query execution
* Multi-query (deep) execution with bounded concurrency
* Partial failure handling
* Result normalization → deduplication → ranking pipeline
"""

from __future__ import annotations

import asyncio
import time

from research_router.engines.registry import EngineRegistry
from research_router.models.plan import SearchPlan
from research_router.models.result import (
    ResearchResponse,
    ResearchResult,
    SearchError,
)
from research_router.research.aggregator import aggregate
from research_router.research.deduplicator import deduplicate
from research_router.research.normalizer import normalise_results
from research_router.research.ranker import rank
from research_router.utils.logging import generate_request_id, get_logger

logger = get_logger(__name__)


class ResearchExecutor:
    """Execute a ``SearchPlan`` and return a ``ResearchResponse``."""

    def __init__(
        self,
        registry: EngineRegistry,
        max_concurrent: int = 3,
    ) -> None:
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(self, plan: SearchPlan, original_query: str) -> ResearchResponse:
        """Run *plan* and return a normalised, deduplicated, ranked response."""
        request_id = generate_request_id()
        t0 = time.monotonic()

        all_results: list[list[ResearchResult]] = []
        all_errors: list[SearchError] = []

        # Primary search
        primary_results, primary_errors = await self._run_single(plan)
        all_results.append(primary_results)
        all_errors.extend(primary_errors)

        # Sub-queries (deep research)
        if plan.sub_queries:
            sub_results, sub_errors = await self._run_sub_queries(plan.sub_queries)
            all_results.extend(sub_results)
            all_errors.extend(sub_errors)

        # Pipeline: aggregate → normalise → dedup → rank
        flat, errors = aggregate(all_results, all_errors)
        normalised = normalise_results(flat)
        deduped = deduplicate(normalised)
        ranked = rank(deduped, original_query)

        # Limit to requested count
        final = ranked[: plan.max_results]

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        sources = list({r.source for r in final if r.source})

        response = ResearchResponse(
            query=original_query,
            domain=plan.domain,
            engine=plan.engine,
            results=final,
            total_results=len(final),
            sources=sources,
            errors=errors,
            metadata={
                "request_id": request_id,
                "execution_time_ms": elapsed_ms,
                "total_raw_results": len(flat),
                "after_dedup": len(deduped),
                "sub_queries_executed": len(plan.sub_queries),
            },
        )

        logger.info(
            "Research execution complete",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "domain": plan.domain.value,
                    "engine": plan.engine,
                    "result_count": len(final),
                    "execution_time_ms": elapsed_ms,
                    "errors": len(errors),
                }
            },
        )

        return response

    # ── single search ─────────────────────────────────────────────

    async def _run_single(self, plan: SearchPlan) -> tuple[list[ResearchResult], list[SearchError]]:
        try:
            engine = self._registry.get(plan.engine)
        except KeyError as exc:
            return [], [
                SearchError(
                    engine=plan.engine,
                    query=plan.query,
                    error_type="UnsupportedEngine",
                    message=str(exc),
                )
            ]

        try:
            async with self._semaphore:
                results = await engine.search(plan)
            return results, []
        except Exception as exc:
            logger.warning(
                "Search failed: %s",
                exc,
                extra={"extra_data": {"engine": plan.engine, "query": plan.query}},
            )
            return [], [
                SearchError(
                    engine=plan.engine,
                    query=plan.query,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            ]

    # ── multi-query ───────────────────────────────────────────────

    async def _run_sub_queries(
        self, sub_plans: list[SearchPlan]
    ) -> tuple[list[list[ResearchResult]], list[SearchError]]:
        tasks = [self._run_single(sp) for sp in sub_plans]
        outcomes = await asyncio.gather(*tasks)

        all_results: list[list[ResearchResult]] = []
        all_errors: list[SearchError] = []
        for results, errors in outcomes:
            all_results.append(results)
            all_errors.extend(errors)

        return all_results, all_errors
