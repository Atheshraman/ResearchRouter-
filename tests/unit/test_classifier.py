"""Phase 4 — classifier unit tests (20+ test cases)."""

from __future__ import annotations

import pytest

from research_router.models.intent import ResearchDomain
from research_router.router.classifier import ClassificationResult, QueryClassifier


@pytest.fixture
def classifier() -> QueryClassifier:
    return QueryClassifier()


# ═══════════════════════════════════════════════════════════════════════
# News domain
# ═══════════════════════════════════════════════════════════════════════


class TestNewsClassification:
    def test_latest_ai_news(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Latest AI news")
        assert r.domain == ResearchDomain.NEWS
        assert r.confidence >= 0.40

    def test_breaking_news(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Breaking news about climate change")
        assert r.domain == ResearchDomain.NEWS

    def test_today_headlines(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Headlines today")
        assert r.domain == ResearchDomain.NEWS

    def test_news_coverage(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Media coverage of the election")
        assert r.domain == ResearchDomain.NEWS


# ═══════════════════════════════════════════════════════════════════════
# Academic domain
# ═══════════════════════════════════════════════════════════════════════


class TestAcademicClassification:
    def test_research_papers_rag(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Research papers about RAG hallucination")
        assert r.domain == ResearchDomain.ACADEMIC
        assert r.confidence >= 0.40

    def test_scholarly_articles(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Scholarly articles on machine learning")
        assert r.domain == ResearchDomain.ACADEMIC

    def test_peer_review(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Peer-reviewed study on climate change")
        assert r.domain == ResearchDomain.ACADEMIC

    def test_arxiv_papers(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("arxiv papers about transformers")
        assert r.domain == ResearchDomain.ACADEMIC


# ═══════════════════════════════════════════════════════════════════════
# Jobs domain
# ═══════════════════════════════════════════════════════════════════════


class TestJobsClassification:
    def test_java_internships(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Java internships in Chennai")
        assert r.domain == ResearchDomain.JOBS
        assert r.confidence >= 0.40

    def test_remote_jobs(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Remote work software developer jobs")
        assert r.domain == ResearchDomain.JOBS

    def test_hiring_vacancy(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Companies hiring for data science")
        assert r.domain == ResearchDomain.JOBS

    def test_career_openings(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Career openings at Google")
        assert r.domain == ResearchDomain.JOBS


# ═══════════════════════════════════════════════════════════════════════
# Shopping domain
# ═══════════════════════════════════════════════════════════════════════


class TestShoppingClassification:
    def test_buy_laptop(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Buy laptop under ₹80000")
        assert r.domain == ResearchDomain.SHOPPING
        assert r.confidence >= 0.40

    def test_best_laptops_price(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Best laptops under ₹80000")
        assert r.domain == ResearchDomain.SHOPPING

    def test_rtx_shopping(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("RTX laptops under ₹90000 for machine learning")
        assert r.domain == ResearchDomain.SHOPPING

    def test_cheap_headphones(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Cheap wireless headphones deal")
        assert r.domain == ResearchDomain.SHOPPING

    def test_price_in_dollars(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Best phone under $500")
        assert r.domain == ResearchDomain.SHOPPING


# ═══════════════════════════════════════════════════════════════════════
# Places domain
# ═══════════════════════════════════════════════════════════════════════


class TestPlacesClassification:
    def test_cafes_near_airport(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Best cafes near Chennai airport")
        assert r.domain == ResearchDomain.PLACES
        assert r.confidence >= 0.30

    def test_restaurants_near_me(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Restaurants near me")
        assert r.domain == ResearchDomain.PLACES

    def test_hospital_location(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Hospital near me open now")
        assert r.domain == ResearchDomain.PLACES

    def test_gym_nearby(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Gym nearby with parking")
        assert r.domain == ResearchDomain.PLACES


# ═══════════════════════════════════════════════════════════════════════
# General / fallback domain
# ═══════════════════════════════════════════════════════════════════════


class TestGeneralClassification:
    def test_generic_question(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("What is retrieval augmented generation?")
        assert r.domain == ResearchDomain.GENERAL

    def test_ambiguous_query(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("How to cook pasta")
        assert r.domain == ResearchDomain.GENERAL

    def test_empty_signals(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("asdfghjkl random gibberish")
        assert r.domain == ResearchDomain.GENERAL


# ═══════════════════════════════════════════════════════════════════════
# Structural tests
# ═══════════════════════════════════════════════════════════════════════


class TestClassifierStructure:
    def test_confidence_bounds(self, classifier: QueryClassifier) -> None:
        for q in [
            "Latest AI news",
            "Research papers about RAG",
            "Java internships in Chennai",
            "Buy laptop under ₹80000",
            "Cafes near me",
            "What is Python?",
        ]:
            r = classifier.classify(q)
            assert 0.0 <= r.confidence <= 1.0

    def test_classification_result_type(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("test")
        assert isinstance(r, ClassificationResult)

    def test_signals_list(self, classifier: QueryClassifier) -> None:
        r = classifier.classify("Buy cheap laptop deal")
        assert isinstance(r.signals, list)
        assert len(r.signals) >= 1
