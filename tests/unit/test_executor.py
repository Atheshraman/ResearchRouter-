"""Executor, cache, and deduplicator+ranker integration tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

from research_router.engines.google import GoogleEngine
from research_router.engines.registry import EngineRegistry
from research_router.models.intent import ResearchDepth, ResearchDomain
from research_router.models.plan import SearchPlan
from research_router.research.cache import TTLCache
from research_router.research.executor import ResearchExecutor
from research_router.serpapi.client import SerpApiClient

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _mock_client(fixture: str) -> SerpApiClient:
    with open(FIXTURES / fixture) as f:
        data = json.load(f)
    client = SerpApiClient.__new__(SerpApiClient)
    client.search = AsyncMock(return_value=data)  # type: ignore[method-assign]
    return client


def _registry() -> EngineRegistry:
    reg = EngineRegistry()
    reg.register("google", GoogleEngine(_mock_client("google.json")))
    return reg


# ═══════════════════════════════════════════════════════════════════════
# Executor
# ═══════════════════════════════════════════════════════════════════════


class TestExecutor:
    async def test_basic_execution(self) -> None:
        executor = ResearchExecutor(_registry())
        plan = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="google",
            query="test query",
        )
        response = await executor.execute(plan, "test query")
        assert response.total_results > 0
        assert response.domain == ResearchDomain.GENERAL
        assert response.metadata["request_id"]
        assert response.metadata["execution_time_ms"] >= 0

    async def test_unsupported_engine_returns_error(self) -> None:
        executor = ResearchExecutor(_registry())
        plan = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="nonexistent",
            query="test",
        )
        response = await executor.execute(plan, "test")
        assert len(response.errors) == 1
        assert response.errors[0].error_type == "UnsupportedEngine"

    async def test_deep_research_with_sub_queries(self) -> None:
        executor = ResearchExecutor(_registry())
        sub1 = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="google",
            query="sub query 1",
        )
        sub2 = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="google",
            query="sub query 2",
        )
        plan = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="google",
            query="main query",
            research_depth=ResearchDepth.DEEP,
            sub_queries=[sub1, sub2],
        )
        response = await executor.execute(plan, "main query")
        assert response.total_results > 0
        assert response.metadata["sub_queries_executed"] == 2

    async def test_partial_failure(self) -> None:
        reg = _registry()
        executor = ResearchExecutor(reg)
        # main query succeeds, sub-query uses missing engine
        sub_bad = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="missing_engine",
            query="will fail",
        )
        plan = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="google",
            query="main",
            sub_queries=[sub_bad],
        )
        response = await executor.execute(plan, "main")
        assert response.total_results > 0  # main succeeded
        assert len(response.errors) == 1  # sub failed


# ═══════════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════════


class TestCache:
    def test_set_and_get(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("google", "test", {}, {"result": 1})
        assert cache.get("google", "test", {}) == {"result": 1}

    def test_miss(self) -> None:
        cache = TTLCache(ttl=60)
        assert cache.get("google", "test", {}) is None

    def test_expiry(self) -> None:
        cache = TTLCache(ttl=0)  # immediate expiry
        cache.set("google", "test", {}, {"result": 1})
        # After TTL=0, next get should miss
        time.sleep(0.01)
        assert cache.get("google", "test", {}) is None

    def test_disabled(self) -> None:
        cache = TTLCache(ttl=60, enabled=False)
        cache.set("google", "test", {}, {"result": 1})
        assert cache.get("google", "test", {}) is None

    def test_different_params_different_keys(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("google", "test", {"a": 1}, "val1")
        cache.set("google", "test", {"a": 2}, "val2")
        assert cache.get("google", "test", {"a": 1}) == "val1"
        assert cache.get("google", "test", {"a": 2}) == "val2"

    def test_clear(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("google", "test", {}, "val")
        assert cache.size() == 1
        cache.clear()
        assert cache.size() == 0
