"""End-to-end tests of the context-aware agent with a mocked SerpApi client."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from research_router.config import Settings
from research_router.context.tokens import estimate_tokens
from research_router.mcp_tools.tools import ResearchRouter
from research_router.models.intent import ResearchDepth
from research_router.serpapi.exceptions import SerpApiError

FIXTURES = Path(__file__).parent.parent / "fixtures"
_FIXTURE_FOR = {
    "google": "google",
    "google_news": "news",
    "google_scholar": "scholar",
    "google_jobs": "jobs",
    "google_shopping": "shopping",
    "google_maps": "maps",
}


class FakeSerpApi:
    """Records calls; per-engine delay/failure injection."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.delay = delay
        self.fail: set[str] = set()
        self.data = {
            e: json.loads((FIXTURES / f"{n}.json").read_text()) for e, n in _FIXTURE_FOR.items()
        }

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        engine = params["engine"]
        self.calls.append((engine, str(params.get("q"))))
        await asyncio.sleep(self.delay)
        if engine in self.fail:
            raise SerpApiError("boom")
        return self.data[engine]


def _router(fake: FakeSerpApi, **overrides: Any) -> ResearchRouter:
    settings = Settings(
        serpapi_api_key="test",
        google_api_key="",
        cache_enabled=overrides.pop("cache_enabled", False),
        _env_file=None,  # type: ignore[call-arg]
        **overrides,
    )
    router = ResearchRouter(settings)
    router._serpapi.search = fake.search  # type: ignore[method-assign]
    return router


# ═══════════════════════════════════════════════════════════════════════
# Task planning / tool selection
# ═══════════════════════════════════════════════════════════════════════


class TestTaskPlanning:
    async def _plan(self, query: str) -> Any:
        router = _router(FakeSerpApi())
        _, plan = await router._agent.plan(
            query, session_id="p", depth=ResearchDepth.STANDARD, max_results=10
        )
        return plan

    async def test_single_intent_uses_one_tool(self) -> None:
        plan = await self._plan("Find recent research papers about RAG hallucination")
        assert [t.tool for t in plan.tasks] == ["academic_search"]
        assert plan.freshness_required and not plan.parallelizable

    async def test_papers_and_implementations_decompose_in_parallel(self) -> None:
        plan = await self._plan("Find recent RAG papers and their GitHub implementations")
        assert [t.tool for t in plan.tasks] == ["academic_search", "github_search"]
        assert "site:github.com" in plan.tasks[1].plan.query
        assert "rag" in plan.tasks[1].plan.query.lower()
        assert plan.parallelizable

    async def test_same_tool_conjunction_stays_one_search(self) -> None:
        plan = await self._plan("compare Python and the JVM garbage collectors")
        assert len(plan.tasks) == 1

    async def test_different_domains_are_split(self) -> None:
        plan = await self._plan("latest AI news and the best laptops under ₹90000")
        assert {t.tool for t in plan.tasks} == {"news_search", "shopping_search"}

    async def test_plan_is_internal_unless_debug(self) -> None:
        router = _router(FakeSerpApi())
        out = await router.research("RAG hallucination papers")
        assert "debug" not in out and "plan" not in out
        out = await router.research("RAG hallucination papers", debug=True)
        assert out["debug"]["plan"]["tasks"][0]["tool"] == "academic_search"
        assert "Tool calls:" in out["debug"]["report"]


# ═══════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════


class TestExecution:
    async def test_independent_tasks_run_concurrently(self) -> None:
        fake = FakeSerpApi(delay=0.2)
        router = _router(fake)
        t0 = time.perf_counter()
        out = await router.research("Find recent RAG papers and their GitHub implementations")
        elapsed = time.perf_counter() - t0
        assert len(fake.calls) == 2
        assert elapsed < 0.35, "two 0.2s calls should overlap"
        assert out["metadata"]["metrics"]["parallel_tool_calls"] == 2

    async def test_failed_specialist_falls_back_to_web(self) -> None:
        fake = FakeSerpApi()
        fake.fail.add("google_scholar")
        router = _router(fake, max_retries=0)
        out = await router.research("RAG hallucination research papers", debug=True)
        assert [c[0] for c in fake.calls] == ["google_scholar", "google"]
        assert out["errors"][0]["engine"] == "google_scholar"
        assert out["total_results"] > 0
        assert out["debug"]["metrics"]["fallback_calls"] == 1

    async def test_timeout_is_reported_not_raised(self) -> None:
        fake = FakeSerpApi(delay=0.5)
        router = _router(fake, task_timeout=0.05)
        out = await router.research("what is retrieval augmented generation")
        assert out["errors"] and out["errors"][0]["error_type"] == "Timeout"
        assert out["results"] == []

    async def test_cache_avoids_repeat_tool_calls(self) -> None:
        fake = FakeSerpApi()
        router = _router(fake, cache_enabled=True)
        await router.research("RAG hallucination papers", session_id="a")
        out = await router.research("RAG hallucination papers", session_id="b")
        assert len(fake.calls) == 1
        assert out["metadata"]["metrics"]["tool_calls"] == 0
        assert out["metadata"]["metrics"]["cache_hits"] == 1

    async def test_budget_bounds_whole_response(self) -> None:
        router = _router(FakeSerpApi())
        out = await router.research("RAG hallucination papers", context_budget=250)
        assert estimate_tokens(out) <= 250
        with pytest.raises(ValueError, match="context_budget"):
            await router.research("RAG", context_budget=10)

    async def test_source_attribution_preserved(self) -> None:
        out = await _router(FakeSerpApi()).research("RAG hallucination research papers")
        for r in out["results"]:
            assert r["url"].startswith("http") and r["source_type"] == "web"


# ═══════════════════════════════════════════════════════════════════════
# Context + memory
# ═══════════════════════════════════════════════════════════════════════


class TestContextAndMemory:
    async def test_follow_up_answered_from_context_without_tools(self) -> None:
        fake = FakeSerpApi()
        router = _router(fake)
        await router.research("RAG hallucination research papers", session_id="s")
        fake.calls.clear()
        out = await router.research("compare them", session_id="s")
        assert fake.calls == []
        assert out["previous_context"]["evidence"]
        assert out["previous_context"]["label"].startswith("EARLIER IN THIS SESSION")
        assert out["results"] == []  # nothing presented as fresh

    async def test_memory_saved_and_recalled_as_old_memory(self, tmp_path: Path) -> None:
        fake = FakeSerpApi()
        router = _router(fake, obsidian_vault_path=str(tmp_path))
        first = await router.research(
            "Find recent research papers about RAG hallucination", save_to_memory=True
        )
        assert first["metadata"]["memory_note"] == "Research/RAG/Hallucination"

        fake.calls.clear()
        out = await router.research(
            "What did my previous RAG hallucination research find?", session_id="new"
        )
        assert fake.calls == []  # recall only — no web search needed
        assert out["memory"]["label"].startswith("OLD MEMORY")
        assert out["memory"]["notes"] == ["Research/RAG/Hallucination"]
        assert all(e["source_type"] == "memory" for e in out["memory"]["evidence"])
        assert out["metadata"]["metrics"]["memory_hits"] == 1

    async def test_missing_vault_never_blocks_research(self, tmp_path: Path) -> None:
        router = _router(FakeSerpApi(), obsidian_vault_path=str(tmp_path / "nope"))
        out = await router.research(
            "Find RAG papers and compare with my previous RAG research", save_to_memory=True
        )
        assert out["total_results"] > 0
        assert any("unavailable" in w for w in out["warnings"])

    async def test_memory_search_exception_is_contained(self, tmp_path: Path) -> None:
        router = _router(FakeSerpApi(), obsidian_vault_path=str(tmp_path))

        async def broken(*_: Any, **__: Any) -> Any:
            raise OSError("disk gone")

        router._memory.search = broken  # type: ignore[method-assign]
        out = await router.research("RAG hallucination research papers")
        assert out["total_results"] > 0
        assert any("Memory search failed" in w for w in out["warnings"])


# ═══════════════════════════════════════════════════════════════════════
# Backward compatibility + metrics
# ═══════════════════════════════════════════════════════════════════════


class TestCompatibility:
    async def test_legacy_mode_keeps_original_response(self) -> None:
        router = _router(FakeSerpApi(), agent_mode=False)
        out = await router.research("RAG hallucination research papers")
        assert set(out) == {
            "query", "domain", "engine", "results", "total_results",
            "sources", "errors", "metadata",
        }  # fmt: skip
        assert "snippet" in out["results"][0]

    async def test_response_keeps_legacy_top_level_keys(self) -> None:
        out = await _router(FakeSerpApi()).research("RAG hallucination research papers")
        for key in ("query", "domain", "engine", "results", "total_results", "sources", "errors"):
            assert key in out
        assert out["metadata"]["request_id"]

    async def test_metrics_summary_compares_modes(self) -> None:
        router = _router(FakeSerpApi())
        await router.research("RAG hallucination research papers", mode="legacy")
        await router.research("RAG hallucination research papers", mode="agent")
        summary = router.metrics_summary()
        assert summary["legacy"]["requests"] == 1 and summary["agent"]["requests"] == 1
        assert summary["legacy"]["avg_tool_calls"] == 1.0

    async def test_explain_includes_execution_plan(self) -> None:
        out = await _router(FakeSerpApi()).explain("RAG papers and their GitHub implementations")
        assert len(out["execution_plan"]["tasks"]) == 2
        assert out["engine"]  # legacy explain keys still present
