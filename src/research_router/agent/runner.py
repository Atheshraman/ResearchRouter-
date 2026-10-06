"""Parallel task runner.

* Independent tasks run concurrently (bounded by the executor's semaphore).
* Tasks with ``depends_on`` run in later waves, after their dependencies.
* Every tool call has a timeout; failures become ``SearchError`` entries and
  never abort the other tasks.
* Adaptive expansion: a task's extra "angle" searches (deep research) run
  only when its primary search did not return enough relevant results.
* Fallback: a specialised engine that fails or returns nothing is retried
  once on general web search, so one tool failure doesn't empty a task.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from research_router.agent.metrics import RequestMetrics
from research_router.agent.task_planner import PlannedTask
from research_router.context.text import salient_terms, tokenize
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult, SearchError
from research_router.research.cache import TTLCache
from research_router.research.executor import ResearchExecutor
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

_NO_FALLBACK_ERRORS = frozenset({"SerpApiAuthError", "SerpApiRateLimitError"})


@dataclass
class TaskOutcome:
    task: PlannedTask
    results: list[ResearchResult] = field(default_factory=list)
    errors: list[SearchError] = field(default_factory=list)
    execution_time_ms: int = 0

    @property
    def status(self) -> str:
        if self.results:
            return "completed" if not self.errors else "completed_with_errors"
        return "failed"


class TaskRunner:
    def __init__(
        self,
        executor: ResearchExecutor,
        cache: TTLCache,
        *,
        timeout: float = 30.0,
        adaptive_expansion: bool = True,
    ) -> None:
        self._executor = executor
        self._cache = cache
        self._timeout = timeout
        self._adaptive = adaptive_expansion

    async def run(self, tasks: list[PlannedTask], metrics: RequestMetrics) -> list[TaskOutcome]:
        started = time.perf_counter()
        outcomes: dict[str, TaskOutcome] = {}
        pending = list(tasks)
        while pending:
            ready = [t for t in pending if all(d in outcomes for d in t.depends_on)]
            if not ready:  # unsatisfiable dependency — run the rest rather than hang
                ready = pending
            pending = [t for t in pending if t not in ready]
            wave = await self._run_wave(ready, metrics)
            outcomes.update({o.task.id: o for o in wave})
        ordered = [outcomes[t.id] for t in tasks]
        metrics.parallel_wall_time_ms += int((time.perf_counter() - started) * 1000)
        metrics.sum_task_time_ms += sum(o.execution_time_ms for o in ordered)
        return ordered

    # ── waves ─────────────────────────────────────────────────────

    async def _run_wave(
        self, tasks: list[PlannedTask], metrics: RequestMetrics
    ) -> list[TaskOutcome]:
        primaries = await self._call_many([t.plan for t in tasks], metrics)
        outcomes = [
            TaskOutcome(task=t, results=res, errors=err, execution_time_ms=elapsed)
            for t, (res, err, elapsed) in zip(tasks, primaries, strict=True)
        ]

        # Expansions/fallbacks depend on the primary results, so they run after.
        follow_ups: list[tuple[TaskOutcome, SearchPlan, str]] = []
        for o in outcomes:
            if self._needs_expansion(o):
                follow_ups += [(o, p, "expansion") for p in o.task.expansions]
            elif self._needs_fallback(o):
                fb = o.task.plan.model_copy(
                    update={"engine": "google", "parameters": _fallback_params(o.task.plan)}
                )
                follow_ups.append((o, fb, "fallback"))
        if follow_ups:
            extra = await self._call_many([p for _, p, _ in follow_ups], metrics)
            for (o, _, kind), (res, err, _) in zip(follow_ups, extra, strict=True):
                o.results.extend(res)
                o.errors.extend(err)
                if kind == "expansion":
                    metrics.expansion_calls += 1
                else:
                    metrics.fallback_calls += 1
        return outcomes

    def _needs_expansion(self, o: TaskOutcome) -> bool:
        if not o.task.expansions:
            return False
        if any(e.error_type in _NO_FALLBACK_ERRORS for e in o.errors):
            return False  # bad key / rate limit: more calls would fail the same way
        if not self._adaptive:
            return True
        return _relevant_count(o.results, o.task.query) < o.task.plan.max_results

    @staticmethod
    def _needs_fallback(o: TaskOutcome) -> bool:
        if o.task.engine == "google" or o.task.plan.domain.value == "jobs" or o.results:
            return False
        return not any(e.error_type in _NO_FALLBACK_ERRORS for e in o.errors)

    # ── tool calls ────────────────────────────────────────────────

    async def _call_many(
        self, plans: list[SearchPlan], metrics: RequestMetrics
    ) -> list[tuple[list[ResearchResult], list[SearchError], int]]:
        uncached = sum(1 for p in plans if self._cache_get(p) is None)
        metrics.parallel_tool_calls = max(metrics.parallel_tool_calls, uncached)
        return list(await asyncio.gather(*(self._call(p, metrics) for p in plans)))

    async def _call(
        self, plan: SearchPlan, metrics: RequestMetrics
    ) -> tuple[list[ResearchResult], list[SearchError], int]:
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        cached = self._cache_get(plan)
        if cached is not None:
            metrics.cache_hits += 1
            return [ResearchResult.model_validate(r) for r in cached], [], elapsed()

        metrics.tool_calls += 1
        try:
            results, errors = await asyncio.wait_for(
                self._executor.run_plan(plan), timeout=self._timeout
            )
        except TimeoutError:
            logger.warning("Tool call timed out", extra={"extra_data": {"engine": plan.engine}})
            return [], [
                SearchError(
                    engine=plan.engine,
                    query=plan.query,
                    error_type="Timeout",
                    message=f"No response within {self._timeout:.0f}s",
                )
            ], elapsed()
        except Exception as exc:  # defensive: run_plan already converts errors
            return [], [
                SearchError(
                    engine=plan.engine,
                    query=plan.query,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            ], elapsed()
        if not errors:
            self._cache.set(
                plan.engine, plan.query, _cache_params(plan), [r.model_dump() for r in results]
            )
        return results, errors, elapsed()

    def _cache_get(self, plan: SearchPlan) -> list[dict[str, object]] | None:
        value = self._cache.get(plan.engine, plan.query, _cache_params(plan))
        return value if isinstance(value, list) else None


def _cache_params(plan: SearchPlan) -> dict[str, object]:
    return {"__agent__": True, "max_results": plan.max_results, **plan.parameters}


def _fallback_params(plan: SearchPlan) -> dict[str, object]:
    keep = ("location", "date_range", "freshness")
    return {k: v for k, v in plan.parameters.items() if k in keep}


def _relevant_count(results: list[ResearchResult], query: str) -> int:
    terms = set(salient_terms(query, drop_generic=True)) or set(salient_terms(query))
    if not terms:
        return len(results)
    need = max(1, (len(terms) + 1) // 2)
    count = 0
    for r in results:
        words = set(tokenize(f"{r.title or ''} {r.snippet or ''}"))
        if len(terms & words) >= need:
            count += 1
    return count
