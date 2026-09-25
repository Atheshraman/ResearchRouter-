"""Phase 1 — data model validation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from research_router.models.intent import (
    ResearchDepth,
    ResearchDomain,
    ResearchIntent,
)
from research_router.models.plan import SearchPlan
from research_router.models.result import (
    ResearchResponse,
    ResearchResult,
    SearchError,
)

# ═══════════════════════════════════════════════════════════════════════
# ResearchIntent
# ═══════════════════════════════════════════════════════════════════════


class TestResearchIntent:
    def test_minimal_intent(self) -> None:
        intent = ResearchIntent(query="AI internships")
        assert intent.domain == ResearchDomain.GENERAL
        assert intent.query == "AI internships"
        assert intent.confidence == 0.0

    def test_full_intent(self) -> None:
        intent = ResearchIntent(
            domain=ResearchDomain.JOBS,
            query="AI internships",
            keywords=["AI", "internships"],
            location="Chennai",
            date_range="this_week",
            job_type="internship",
            remote=False,
            experience_level="entry",
            research_depth=ResearchDepth.QUICK,
            confidence=0.94,
            requires_multiple_searches=False,
        )
        assert intent.domain == ResearchDomain.JOBS
        assert intent.location == "Chennai"
        assert intent.confidence == 0.94
        assert intent.research_depth == ResearchDepth.QUICK

    def test_shopping_intent(self) -> None:
        intent = ResearchIntent(
            domain=ResearchDomain.SHOPPING,
            query="RTX laptops",
            price_max=90000,
            currency="INR",
            confidence=0.88,
        )
        assert intent.price_max == 90000
        assert intent.currency == "INR"

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchIntent(query="")

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ResearchIntent(query="test", confidence=1.5)
        with pytest.raises(ValidationError):
            ResearchIntent(query="test", confidence=-0.1)

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchIntent(query="test", price_min=-10)

    def test_domains_enum(self) -> None:
        for domain in ResearchDomain:
            intent = ResearchIntent(query="test", domain=domain)
            assert intent.domain == domain

    def test_depths_enum(self) -> None:
        for depth in ResearchDepth:
            intent = ResearchIntent(query="test", research_depth=depth)
            assert intent.research_depth == depth

    def test_constraints_dict(self) -> None:
        intent = ResearchIntent(
            query="test",
            constraints={"gpu": "RTX", "purpose": "machine learning"},
        )
        assert intent.constraints["gpu"] == "RTX"

    def test_serialization_roundtrip(self) -> None:
        intent = ResearchIntent(
            domain=ResearchDomain.NEWS,
            query="AI news",
            location="India",
            freshness="recent",
            confidence=0.9,
        )
        data = intent.model_dump()
        restored = ResearchIntent(**data)
        assert restored == intent

    def test_json_roundtrip(self) -> None:
        intent = ResearchIntent(query="test query", domain=ResearchDomain.ACADEMIC)
        json_str = intent.model_dump_json()
        restored = ResearchIntent.model_validate_json(json_str)
        assert restored == intent


# ═══════════════════════════════════════════════════════════════════════
# SearchPlan
# ═══════════════════════════════════════════════════════════════════════


class TestSearchPlan:
    def test_minimal_plan(self) -> None:
        plan = SearchPlan(
            domain=ResearchDomain.GENERAL,
            engine="google",
            query="hello world",
        )
        assert plan.engine == "google"
        assert plan.max_results == 10
        assert plan.sub_queries == []

    def test_jobs_plan(self) -> None:
        plan = SearchPlan(
            domain=ResearchDomain.JOBS,
            engine="google_jobs",
            query="AI internships",
            parameters={"location": "Chennai"},
            max_results=15,
            confidence=0.94,
        )
        assert plan.parameters["location"] == "Chennai"
        assert plan.max_results == 15

    def test_deep_plan_with_sub_queries(self) -> None:
        sub1 = SearchPlan(
            domain=ResearchDomain.ACADEMIC,
            engine="google_scholar",
            query="RAG hallucination research",
        )
        sub2 = SearchPlan(
            domain=ResearchDomain.ACADEMIC,
            engine="google_scholar",
            query="RAG hallucination mitigation",
        )
        plan = SearchPlan(
            domain=ResearchDomain.ACADEMIC,
            engine="google_scholar",
            query="RAG hallucination",
            research_depth=ResearchDepth.DEEP,
            sub_queries=[sub1, sub2],
        )
        assert len(plan.sub_queries) == 2
        assert plan.research_depth == ResearchDepth.DEEP

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchPlan(domain=ResearchDomain.GENERAL, engine="google", query="")

    def test_max_results_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SearchPlan(
                domain=ResearchDomain.GENERAL,
                engine="google",
                query="test",
                max_results=0,
            )

    def test_serialization_roundtrip(self) -> None:
        plan = SearchPlan(
            domain=ResearchDomain.SHOPPING,
            engine="google_shopping",
            query="RTX laptop",
            parameters={"price_max": 90000},
        )
        data = plan.model_dump()
        restored = SearchPlan(**data)
        assert restored == plan


# ═══════════════════════════════════════════════════════════════════════
# ResearchResult
# ═══════════════════════════════════════════════════════════════════════


class TestResearchResult:
    def test_minimal_result(self) -> None:
        r = ResearchResult()
        assert r.title is None
        assert r.url is None
        assert r.metadata == {}

    def test_full_result(self) -> None:
        r = ResearchResult(
            title="AI Engineer Intern",
            url="https://example.com/job/123",
            snippet="Exciting AI internship in Chennai...",
            source="LinkedIn",
            published_at=datetime(2026, 9, 20, tzinfo=UTC),
            metadata={"salary": "₹25,000/month"},
        )
        assert r.title == "AI Engineer Intern"
        assert r.source == "LinkedIn"
        assert r.published_at is not None
        assert r.published_at.tzinfo is not None

    def test_null_fields_allowed(self) -> None:
        r = ResearchResult(title="Test", url=None, snippet=None)
        assert r.url is None


# ═══════════════════════════════════════════════════════════════════════
# SearchError
# ═══════════════════════════════════════════════════════════════════════


class TestSearchError:
    def test_error_model(self) -> None:
        err = SearchError(
            engine="google_scholar",
            query="test query",
            error_type="TimeoutError",
            message="Request timed out after 20s",
        )
        assert err.engine == "google_scholar"
        assert err.error_type == "TimeoutError"


# ═══════════════════════════════════════════════════════════════════════
# ResearchResponse
# ═══════════════════════════════════════════════════════════════════════


class TestResearchResponse:
    def test_empty_response(self) -> None:
        resp = ResearchResponse(query="test")
        assert resp.results == []
        assert resp.errors == []
        assert resp.total_results == 0
        assert resp.domain == ResearchDomain.GENERAL

    def test_response_with_results(self) -> None:
        results = [
            ResearchResult(title="Result 1", url="https://a.com"),
            ResearchResult(title="Result 2", url="https://b.com"),
        ]
        resp = ResearchResponse(
            query="AI news",
            domain=ResearchDomain.NEWS,
            engine="google_news",
            results=results,
            total_results=2,
            sources=["a.com", "b.com"],
        )
        assert len(resp.results) == 2
        assert resp.total_results == 2

    def test_response_with_errors(self) -> None:
        err = SearchError(
            engine="google_scholar",
            error_type="RateLimitError",
            message="429 Too Many Requests",
        )
        resp = ResearchResponse(
            query="test",
            results=[ResearchResult(title="Partial result")],
            total_results=1,
            errors=[err],
        )
        assert len(resp.errors) == 1
        assert len(resp.results) == 1

    def test_metadata_field(self) -> None:
        resp = ResearchResponse(
            query="test",
            metadata={
                "request_id": "abc123",
                "execution_time_ms": 450,
            },
        )
        assert resp.metadata["request_id"] == "abc123"

    def test_json_roundtrip(self) -> None:
        resp = ResearchResponse(
            query="test",
            domain=ResearchDomain.JOBS,
            engine="google_jobs",
            results=[
                ResearchResult(title="Job 1", url="https://example.com/1"),
            ],
            total_results=1,
        )
        json_str = resp.model_dump_json()
        restored = ResearchResponse.model_validate_json(json_str)
        assert restored == resp
