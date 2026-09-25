"""Phase 6 — Intent analysis tests."""

from __future__ import annotations

from typing import Any

import pytest

from research_router.llm.base import LLMProvider
from research_router.models.intent import ResearchDepth, ResearchDomain
from research_router.router.classifier import QueryClassifier
from research_router.router.intent import IntentAnalyser


class MockLLMProvider(LLMProvider):
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, response: dict[str, Any] | None = None, fail: bool = False) -> None:
        self._response = response or {}
        self._fail = fail

    async def analyse_intent(self, query: str) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError("LLM is down!")
        return self._response

    async def close(self) -> None:
        pass


@pytest.fixture
def classifier() -> QueryClassifier:
    return QueryClassifier()


class TestDeterministicIntent:
    async def test_news_intent(self, classifier: QueryClassifier) -> None:
        analyser = IntentAnalyser(classifier, confidence_threshold=0.30)
        intent = await analyser.analyse("Latest AI news")
        assert intent.domain == ResearchDomain.NEWS

    async def test_jobs_intent_with_location(self, classifier: QueryClassifier) -> None:
        analyser = IntentAnalyser(classifier, confidence_threshold=0.30)
        intent = await analyser.analyse("AI internships in Chennai")
        assert intent.domain == ResearchDomain.JOBS
        assert intent.location == "Chennai"

    async def test_shopping_intent_with_price(self, classifier: QueryClassifier) -> None:
        analyser = IntentAnalyser(classifier, confidence_threshold=0.30)
        intent = await analyser.analyse("Buy laptop under ₹80000")
        assert intent.domain == ResearchDomain.SHOPPING
        assert intent.price_max == 80000
        assert intent.currency == "INR"

    async def test_date_range_extraction(self, classifier: QueryClassifier) -> None:
        analyser = IntentAnalyser(classifier, confidence_threshold=0.30)
        intent = await analyser.analyse("Latest news this week")
        assert intent.date_range == "this_week"

    async def test_general_fallback(self, classifier: QueryClassifier) -> None:
        analyser = IntentAnalyser(classifier, confidence_threshold=0.99)  # force low confidence
        intent = await analyser.analyse("What is Python?")
        assert intent.domain == ResearchDomain.GENERAL


class TestLLMIntent:
    async def test_llm_called_when_low_confidence(self, classifier: QueryClassifier) -> None:
        mock_llm = MockLLMProvider(
            response={
                "domain": "academic",
                "query": "RAG hallucination mitigation",
                "keywords": ["RAG", "hallucination"],
                "confidence": 0.92,
                "research_depth": "deep",
                "requires_multiple_searches": True,
            }
        )
        # Use a truly ambiguous query that won't trigger strong deterministic signals
        analyser = IntentAnalyser(classifier, llm_provider=mock_llm, confidence_threshold=0.75)
        intent = await analyser.analyse(
            "practical techniques suitable for production Java backends"
        )
        assert intent.domain == ResearchDomain.ACADEMIC
        assert intent.requires_multiple_searches is True
        assert intent.research_depth == ResearchDepth.DEEP

    async def test_llm_failure_falls_back(self, classifier: QueryClassifier) -> None:
        mock_llm = MockLLMProvider(fail=True)
        analyser = IntentAnalyser(classifier, llm_provider=mock_llm, confidence_threshold=0.01)
        # Should not crash — falls back to deterministic
        intent = await analyser.analyse("Latest AI news")
        assert intent.domain == ResearchDomain.NEWS

    async def test_no_llm_uses_deterministic(self, classifier: QueryClassifier) -> None:
        analyser = IntentAnalyser(classifier, llm_provider=None, confidence_threshold=0.01)
        intent = await analyser.analyse("Research papers about transformers")
        assert intent.domain == ResearchDomain.ACADEMIC
