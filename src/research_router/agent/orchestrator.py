"""Context-aware, memory-aware research agent.

``research()`` pipeline::

    resolve context → plan tasks → [web tasks ∥ memory recall] → evidence
      → context reducer (per source type, under one token budget)
      → compact response (+ debug report) → record turn → optional memory save

Fresh web evidence, earlier-session context and recalled memory are kept in
separate, labelled sections so old information is never presented as newly
retrieved.
"""

from __future__ import annotations

import asyncio
from typing import Any

from research_router.agent.metrics import (
    MetricsAggregator,
    RequestMetrics,
    log_research_summary,
    render_debug_report,
)
from research_router.agent.runner import TaskOutcome, TaskRunner
from research_router.agent.task_planner import ExecutionPlan, TaskPlanner
from research_router.context.evidence import Evidence, EvidenceSource, SourceType
from research_router.context.manager import ContextManager, ResolvedContext
from research_router.context.reducer import ContextReducer, ReductionResult
from research_router.context.text import salient_terms
from research_router.context.tokens import estimate_tokens
from research_router.memory.base import MemoryHit, MemoryStore, ResearchRecord
from research_router.memory.obsidian import hits_to_evidence
from research_router.models.intent import ResearchDepth
from research_router.models.result import ResearchResult, SearchError
from research_router.research.normalizer import normalise_results
from research_router.utils.logging import generate_request_id, get_logger

logger = get_logger(__name__)

MEMORY_LABEL = (
    "OLD MEMORY — recalled from saved research notes, not retrieved in this request. "
    "Cite with its recorded date."
)
CONTEXT_LABEL = "EARLIER IN THIS SESSION — results returned by a previous research turn."
_ENVELOPE_RESERVE = 100  # tokens reserved for keys, labels and compact metrics
_MEMORY_SHARE = 0.25
_CONTEXT_SHARE = 0.30
_COMPACT_METRIC_KEYS = (
    "tool_calls",
    "parallel_tool_calls",
    "cache_hits",
    "raw_results",
    "final_evidence",
    "input_tokens_estimated",
    "output_context_tokens",
    "response_tokens",
    "compression_ratio",
    "memory_hits",
    "execution_time_ms",
)


def _evidence_sections(response: dict[str, Any]) -> list[Any]:
    """The research context proper: fresh, earlier-session and memory evidence."""
    return [
        response.get("results", []),
        response.get("previous_context", {}).get("evidence", []),
        response.get("memory", {}).get("evidence", []),
    ]


class ResearchAgent:
    def __init__(
        self,
        *,
        task_planner: TaskPlanner,
        runner: TaskRunner,
        reducer: ContextReducer,
        context_manager: ContextManager,
        memory: MemoryStore,
        aggregator: MetricsAggregator,
        default_budget: int,
        auto_save: bool = False,
        memory_max_notes: int = 3,
    ) -> None:
        self._task_planner = task_planner
        self._runner = runner
        self._reducer = reducer
        self._ctx = context_manager
        self._memory = memory
        self._aggregator = aggregator
        self._default_budget = default_budget
        self._auto_save = auto_save
        self._memory_max_notes = memory_max_notes

    @property
    def memory(self) -> MemoryStore:
        return self._memory

    # ── public API ────────────────────────────────────────────────

    async def plan(
        self, query: str, *, session_id: str, depth: ResearchDepth, max_results: int
    ) -> tuple[ResolvedContext, ExecutionPlan]:
        """Resolve context and build the execution plan without running it."""
        resolved = self._ctx.resolve(session_id, query)
        plan = await self._task_planner.plan(resolved, depth, max_results)
        return resolved, plan

    async def research(
        self,
        query: str,
        *,
        max_results: int = 10,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        context_budget: int | None = None,
        debug: bool = False,
        session_id: str = "default",
        use_memory: bool = True,
        save_to_memory: bool | None = None,
    ) -> dict[str, Any]:
        m = RequestMetrics(mode="agent")
        budget = context_budget or self._default_budget
        warnings: list[str] = []

        with m.stage("planning"):
            resolved, plan = await self.plan(
                query, session_id=session_id, depth=depth, max_results=max_results
            )

        # Web tasks and memory recall are independent → run concurrently.
        with m.stage("retrieval"):
            outcomes, hits = await asyncio.gather(
                self._runner.run(plan.tasks, m),
                self._recall(resolved, use_memory, m, warnings),
            )

        with m.stage("reduction"):
            web_candidates, raw_tokens = self._web_candidates(outcomes, resolved)
            m.raw_results = len(web_candidates)
            web, memory_red, context_red = self._reduce_all(
                plan, resolved, web_candidates, hits, budget, max_results, m, raw_tokens
            )

        errors = [e for o in outcomes for e in o.errors]
        m.errors = len(errors)
        response = self._assemble(query, resolved, plan, web, memory_red, context_red, errors)
        if warnings:
            response["warnings"] = warnings

        # Persist compact state for follow-up questions in this session.
        self._ctx.record(session_id, resolved, plan.intent, web.evidence)

        should_save = self._auto_save if save_to_memory is None else save_to_memory
        if should_save:
            with m.stage("memory_save"):
                await self._save(query, plan, web.evidence, resolved, hits, m, response)

        m.finish()
        response["metadata"]["execution_time_ms"] = m.execution_time_ms
        self._set_compact_metrics(response, m)
        self._enforce_budget(response, budget)
        m.final_evidence = len(response["results"])
        m.output_context_tokens = estimate_tokens(_evidence_sections(response))
        m.response_tokens = estimate_tokens(response)
        self._set_compact_metrics(response, m)
        self._aggregator.record(m)
        log_research_summary(query, sorted({t.tool for t in plan.tasks}), m)

        if debug:  # debug output is diagnostic and not counted against the budget
            response["debug"] = {
                "report": render_debug_report(
                    query=query,
                    intent=plan.intent,
                    plan=plan.as_dict(),
                    metrics=m,
                    memory_status=self._memory.status(),
                ),
                "plan": plan.as_dict(),
                "context": resolved.as_dict(),
                "metrics": m.as_dict(),
                "reduction": {
                    "web": web.stats.as_dict(),
                    "memory": memory_red.stats.as_dict(),
                    "previous_context": context_red.stats.as_dict(),
                },
                "memory_status": self._memory.status(),
            }
        return response

    # ── retrieval helpers ─────────────────────────────────────────

    async def _recall(
        self,
        resolved: ResolvedContext,
        use_memory: bool,
        m: RequestMetrics,
        warnings: list[str],
    ) -> list[MemoryHit]:
        if not use_memory:
            return []
        if not self._memory.available:
            if "memory" in resolved.required_context:
                reason = self._memory.status().get("reason") or "not configured"
                warnings.append(f"Persistent memory requested but unavailable ({reason}).")
            return []
        query = resolved.memory_query or resolved.resolved_query
        try:
            hits = await self._memory.search(query, limit=self._memory_max_notes)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            warnings.append(f"Memory search failed ({type(exc).__name__}); continued without it.")
            return []
        if hits:
            m.memory_hits += 1
            m.memory_notes_used = [h.note for h in hits]
        else:
            m.memory_misses += 1
        return hits

    @staticmethod
    def _web_candidates(
        outcomes: list[TaskOutcome], resolved: ResolvedContext
    ) -> tuple[list[Evidence], int]:
        raw: list[ResearchResult] = [r for o in outcomes for r in o.results]
        raw_tokens = estimate_tokens([r.model_dump(mode="json") for r in raw]) if raw else 0
        skip = resolved.seen_urls if resolved.user_intent == "continue" else set()

        candidates: list[Evidence] = []
        for o in outcomes:
            try:
                results = normalise_results(o.results)
            except Exception:
                logger.warning("Malformed results from %s dropped", o.task.engine)
                continue
            for rank, r in enumerate(results):
                if r.url and r.url in skip:
                    continue  # already shown earlier in this session
                candidates.append(
                    Evidence(
                        claim=r.snippet or r.title or "",
                        source=EvidenceSource(
                            title=r.title,
                            url=r.url,
                            date=r.published_at.date().isoformat() if r.published_at else None,
                            publisher=r.source,
                        ),
                        task_id=o.task.id,
                        engine=o.task.engine,
                        metadata={"rank": rank},
                    )
                )
        return candidates, raw_tokens

    def _reduce_all(
        self,
        plan: ExecutionPlan,
        resolved: ResolvedContext,
        web_candidates: list[Evidence],
        hits: list[MemoryHit],
        budget: int,
        max_results: int,
        m: RequestMetrics,
        raw_tokens: int,
    ) -> tuple[ReductionResult, ReductionResult, ReductionResult]:
        available = max(budget - _ENVELOPE_RESERVE, 50)
        focus = resolved.resolved_query

        # Memory and earlier context get bounded shares; unused budget flows to web.
        memory = self._reducer.reduce(
            hits_to_evidence(hits),
            resolved.memory_query or focus,
            int(available * _MEMORY_SHARE) if web_candidates else available // 2,
            max_items=8,
        )
        available -= memory.stats.output_tokens

        context = self._reducer.reduce(
            resolved.previous_results,
            focus,
            int(available * _CONTEXT_SHARE) if web_candidates else available,
            max_items=6,
        )
        available -= context.stats.output_tokens

        web = self._reducer.reduce(
            web_candidates,
            focus,
            max(available, 0),
            max_items=max_results * max(1, len(plan.tasks)),
            freshness_required=plan.freshness_required,
            input_tokens=raw_tokens,
        )
        m.deduplicated_results = web.stats.deduplicated
        m.relevant_results = web.stats.relevant
        m.input_tokens_estimated = (
            raw_tokens + memory.stats.input_tokens + context.stats.input_tokens
        )
        return web, memory, context

    # ── response ──────────────────────────────────────────────────

    @staticmethod
    def _assemble(
        query: str,
        resolved: ResolvedContext,
        plan: ExecutionPlan,
        web: ReductionResult,
        memory: ReductionResult,
        context: ReductionResult,
        errors: list[SearchError],
    ) -> dict[str, Any]:
        primary = plan.primary
        response: dict[str, Any] = {
            "query": query,
            "domain": plan.intent,
            "engine": primary.engine if primary else "none",
            "results": web.context,
            "total_results": len(web.context),
            "sources": sorted({e.source.publisher for e in web.evidence if e.source.publisher}),
            "errors": [e.model_dump() for e in errors],
            "metadata": {
                "request_id": generate_request_id(),
                "tools_used": sorted({t.tool for t in plan.tasks}),
                "total_raw_results": web.stats.raw,
                "after_dedup": web.stats.deduplicated,
            },
        }
        if resolved.resolved_query != query:
            response["resolved_query"] = resolved.resolved_query
        if plan.context_only:
            response["note"] = (
                "Answered from earlier session context and/or saved memory; no new search was run."
                if context.context or memory.context
                else "The query refers to earlier research, but no matching context was found."
            )
        if context.context:
            response["previous_context"] = {"label": CONTEXT_LABEL, "evidence": context.context}
        if memory.context:
            notes = sorted({str(e.metadata.get("note")) for e in memory.evidence})
            response["memory"] = {"label": MEMORY_LABEL, "notes": notes, "evidence": memory.context}
        return response

    @staticmethod
    def _set_compact_metrics(response: dict[str, Any], m: RequestMetrics) -> None:
        full = m.as_dict()
        response["metadata"]["metrics"] = {k: full[k] for k in _COMPACT_METRIC_KEYS}

    @staticmethod
    def _enforce_budget(response: dict[str, Any], budget: int) -> None:
        """Final guard: trim lowest-ranked evidence if the whole envelope overshoots."""
        sections = [
            response["results"],
            response.get("previous_context", {}).get("evidence", []),
            response.get("memory", {}).get("evidence", []),
        ]
        preserve_minimum = budget <= _ENVELOPE_RESERVE * 2
        while estimate_tokens(response) > budget:
            target = next(
                (s for s in sections if len(s) > 1 or (s and not preserve_minimum)),
                None,
            )
            if target is None:
                break  # preserve one item per section when the envelope cannot fit
            target.pop()
        response["total_results"] = len(response["results"])

    async def _save(
        self,
        query: str,
        plan: ExecutionPlan,
        evidence: list[Evidence],
        resolved: ResolvedContext,
        hits: list[MemoryHit],
        m: RequestMetrics,
        response: dict[str, Any],
    ) -> None:
        if not self._memory.available:
            response.setdefault("warnings", []).append(
                "save_to_memory requested but persistent memory is unavailable."
            )
            return
        findings = [e for e in evidence if e.source_type is SourceType.WEB][:10]
        if not findings:
            return  # nothing fresh worth remembering
        record = ResearchRecord(
            query=query,
            domain=plan.intent,
            tools=sorted({t.tool for t in plan.tasks}),
            findings=findings,
            entities=resolved.entities,
            tags=salient_terms(resolved.resolved_query, drop_generic=True)[:5],
            related_notes=[h.note for h in hits],
        )
        try:
            m.memory_saved = await self._memory.save(record)
            if m.memory_saved:
                response["metadata"]["memory_note"] = m.memory_saved
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)
            response.setdefault("warnings", []).append(
                f"Could not save research note ({type(exc).__name__})."
            )
