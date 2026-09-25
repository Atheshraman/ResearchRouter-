"""Engine adapter normalisation tests — verify each adapter correctly
normalises its SerpApi response format into ResearchResult lists."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from research_router.engines.google import GoogleEngine
from research_router.engines.jobs import JobsEngine
from research_router.engines.maps import MapsEngine
from research_router.engines.news import NewsEngine
from research_router.engines.registry import EngineRegistry
from research_router.engines.scholar import ScholarEngine
from research_router.engines.shopping import ShoppingEngine
from research_router.models.intent import ResearchDomain
from research_router.models.plan import SearchPlan
from research_router.serpapi.client import SerpApiClient

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _mock_client(fixture_name: str) -> SerpApiClient:
    """Return a SerpApiClient whose .search() returns fixture data."""
    with open(FIXTURES / fixture_name) as f:
        data = json.load(f)
    client = SerpApiClient.__new__(SerpApiClient)
    client.search = AsyncMock(return_value=data)  # type: ignore[method-assign]
    return client


def _plan(engine: str, query: str = "test", **params: object) -> SearchPlan:
    return SearchPlan(
        domain=ResearchDomain.GENERAL,
        engine=engine,
        query=query,
        parameters=dict(params),
    )


# ═══════════════════════════════════════════════════════════════════════


class TestGoogleEngine:
    async def test_normalises_organic_results(self) -> None:
        engine = GoogleEngine(_mock_client("google.json"))
        results = await engine.search(_plan("google"))
        assert len(results) == 2
        assert results[0].title == "What is Retrieval Augmented Generation?"
        assert results[0].url == "https://example.com/rag-explained"
        assert results[0].source == "Example Blog"


class TestNewsEngine:
    async def test_normalises_news_results(self) -> None:
        engine = NewsEngine(_mock_client("news.json"))
        results = await engine.search(_plan("google_news"))
        assert len(results) == 2
        assert results[0].title == "AI Breakthrough: New Model Surpasses GPT-4"
        assert results[0].source == "Tech News Daily"
        assert results[0].metadata["date"] == "2 hours ago"


class TestScholarEngine:
    async def test_normalises_scholar_results(self) -> None:
        engine = ScholarEngine(_mock_client("scholar.json"))
        results = await engine.search(_plan("google_scholar"))
        assert len(results) == 2
        assert "Retrieval-Augmented Generation" in results[0].title  # type: ignore[operator]
        assert results[0].metadata["cited_by"] == 3500


class TestJobsEngine:
    async def test_normalises_jobs_results(self) -> None:
        engine = JobsEngine(_mock_client("jobs.json"))
        results = await engine.search(_plan("google_jobs"))
        assert len(results) == 2
        assert results[0].title == "AI/ML Intern"
        assert results[0].source == "TechCorp India"
        assert results[0].metadata["location"] == "Chennai, Tamil Nadu"
        assert "25,000/month" in results[0].metadata["salary"]


class TestShoppingEngine:
    async def test_normalises_shopping_results(self) -> None:
        engine = ShoppingEngine(_mock_client("shopping.json"))
        results = await engine.search(_plan("google_shopping"))
        assert len(results) == 2
        assert "RTX 4060" in results[0].title  # type: ignore[operator]
        assert results[0].metadata["extracted_price"] == 89990
        assert results[0].metadata["rating"] == 4.5


class TestMapsEngine:
    async def test_normalises_maps_results(self) -> None:
        engine = MapsEngine(_mock_client("maps.json"))
        results = await engine.search(_plan("google_maps"))
        assert len(results) == 2
        assert "Coffee Day - Airport" in results[0].title  # type: ignore[operator]
        assert results[0].source == "Google Maps"
        assert results[0].metadata["rating"] == 4.1
        assert results[0].metadata["address"] is not None


# ═══════════════════════════════════════════════════════════════════════


class TestEngineRegistry:
    def test_register_and_get(self) -> None:
        registry = EngineRegistry()
        engine = GoogleEngine(_mock_client("google.json"))
        registry.register("google", engine)
        assert registry.get("google") is engine

    def test_missing_engine_raises(self) -> None:
        registry = EngineRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_list_engines(self) -> None:
        registry = EngineRegistry()
        registry.register("a", GoogleEngine(_mock_client("google.json")))
        registry.register("b", GoogleEngine(_mock_client("google.json")))
        assert registry.list_engines() == ["a", "b"]

    def test_has(self) -> None:
        registry = EngineRegistry()
        registry.register("google", GoogleEngine(_mock_client("google.json")))
        assert registry.has("google") is True
        assert registry.has("youtube") is False
