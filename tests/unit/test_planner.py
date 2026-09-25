"""Phase 7 — Planner tests."""

from __future__ import annotations

import pytest

from research_router.models.intent import ResearchDepth, ResearchDomain, ResearchIntent
from research_router.router.planner import ResearchPlanner


@pytest.fixture
def planner() -> ResearchPlanner:
    return ResearchPlanner(max_results=10, max_sub_queries=4)


class TestPlannerEngineSelection:
    def test_general_maps_to_google(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(query="What is Python?", domain=ResearchDomain.GENERAL)
        plan = planner.plan(intent)
        assert plan.engine == "google"

    def test_news_maps_to_google_news(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(query="AI news", domain=ResearchDomain.NEWS)
        plan = planner.plan(intent)
        assert plan.engine == "google_news"

    def test_academic_maps_to_google_scholar(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(query="RAG papers", domain=ResearchDomain.ACADEMIC)
        plan = planner.plan(intent)
        assert plan.engine == "google_scholar"

    def test_jobs_maps_to_google_jobs(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(query="AI internships", domain=ResearchDomain.JOBS)
        plan = planner.plan(intent)
        assert plan.engine == "google_jobs"

    def test_shopping_maps_to_google_shopping(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(query="laptops", domain=ResearchDomain.SHOPPING)
        plan = planner.plan(intent)
        assert plan.engine == "google_shopping"

    def test_places_maps_to_google_maps(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(query="cafes nearby", domain=ResearchDomain.PLACES)
        plan = planner.plan(intent)
        assert plan.engine == "google_maps"


class TestPlannerParameters:
    def test_location_parameter(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(
            query="AI internships",
            domain=ResearchDomain.JOBS,
            location="Chennai",
        )
        plan = planner.plan(intent)
        assert plan.parameters["location"] == "Chennai"

    def test_price_parameters(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(
            query="laptops",
            domain=ResearchDomain.SHOPPING,
            price_max=80000,
            currency="INR",
        )
        plan = planner.plan(intent)
        assert plan.parameters["price_max"] == 80000
        assert plan.parameters["currency"] == "INR"

    def test_date_range_parameter(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(
            query="AI news",
            domain=ResearchDomain.NEWS,
            date_range="this_week",
        )
        plan = planner.plan(intent)
        assert plan.parameters["date_range"] == "this_week"


class TestPlannerDeepResearch:
    def test_deep_generates_sub_queries(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(
            query="RAG hallucination mitigation",
            domain=ResearchDomain.ACADEMIC,
            keywords=["RAG", "hallucination", "mitigation"],
            research_depth=ResearchDepth.DEEP,
        )
        plan = planner.plan(intent)
        assert len(plan.sub_queries) > 0
        assert plan.research_depth == ResearchDepth.DEEP

    def test_quick_no_sub_queries(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(
            query="AI news",
            domain=ResearchDomain.NEWS,
            research_depth=ResearchDepth.QUICK,
        )
        plan = planner.plan(intent)
        assert len(plan.sub_queries) == 0

    def test_requires_multiple_searches(self, planner: ResearchPlanner) -> None:
        intent = ResearchIntent(
            query="RAG techniques",
            domain=ResearchDomain.ACADEMIC,
            keywords=["RAG", "techniques"],
            requires_multiple_searches=True,
        )
        plan = planner.plan(intent)
        assert len(plan.sub_queries) > 0
