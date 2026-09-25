"""Phase 10 — Normalizer, deduplicator, and ranker tests."""

from __future__ import annotations

from research_router.models.result import ResearchResult
from research_router.research.deduplicator import deduplicate
from research_router.research.normalizer import normalise_results
from research_router.research.ranker import rank

# ═══════════════════════════════════════════════════════════════════════
# Normalizer
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizer:
    def test_strips_whitespace(self) -> None:
        results = [ResearchResult(title="  Hello  ", url="https://a.com", snippet="  world  ")]
        cleaned = normalise_results(results)
        assert cleaned[0].title == "Hello"
        assert cleaned[0].snippet == "world"

    def test_removes_empty_results(self) -> None:
        results = [
            ResearchResult(title="Valid", url="https://a.com"),
            ResearchResult(title=None, url=None),
            ResearchResult(title="", url=""),
        ]
        cleaned = normalise_results(results)
        assert len(cleaned) == 1

    def test_keeps_url_only_result(self) -> None:
        results = [ResearchResult(url="https://a.com")]
        cleaned = normalise_results(results)
        assert len(cleaned) == 1

    def test_keeps_title_only_result(self) -> None:
        results = [ResearchResult(title="Just a title")]
        cleaned = normalise_results(results)
        assert len(cleaned) == 1


# ═══════════════════════════════════════════════════════════════════════
# Deduplicator
# ═══════════════════════════════════════════════════════════════════════


class TestDeduplicator:
    def test_removes_exact_duplicates(self) -> None:
        results = [
            ResearchResult(title="A", url="https://example.com/page"),
            ResearchResult(title="A copy", url="https://example.com/page"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].title == "A"  # first wins

    def test_trailing_slash_normalised(self) -> None:
        results = [
            ResearchResult(title="A", url="https://example.com/page"),
            ResearchResult(title="B", url="https://example.com/page/"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1

    def test_utm_params_stripped(self) -> None:
        results = [
            ResearchResult(title="A", url="https://example.com/page"),
            ResearchResult(
                title="B", url="https://example.com/page?utm_source=google&utm_medium=cpc"
            ),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1

    def test_fragment_stripped(self) -> None:
        results = [
            ResearchResult(title="A", url="https://example.com/page"),
            ResearchResult(title="B", url="https://example.com/page#section2"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1

    def test_different_urls_kept(self) -> None:
        results = [
            ResearchResult(title="A", url="https://example.com/a"),
            ResearchResult(title="B", url="https://example.com/b"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2

    def test_no_url_results_always_kept(self) -> None:
        results = [
            ResearchResult(title="A"),
            ResearchResult(title="B"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2

    def test_case_insensitive_host(self) -> None:
        results = [
            ResearchResult(title="A", url="https://Example.COM/page"),
            ResearchResult(title="B", url="https://example.com/page"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1


# ═══════════════════════════════════════════════════════════════════════
# Ranker
# ═══════════════════════════════════════════════════════════════════════


class TestRanker:
    def test_keyword_match_ranks_higher(self) -> None:
        results = [
            ResearchResult(title="Unrelated stuff", url="https://a.com"),
            ResearchResult(title="AI internships in Chennai", url="https://b.com"),
        ]
        ranked = rank(results, "AI internships Chennai")
        assert ranked[0].title == "AI internships in Chennai"

    def test_completeness_bonus(self) -> None:
        results = [
            ResearchResult(title="Sparse"),
            ResearchResult(
                title="Complete",
                url="https://x.com",
                snippet="Full result",
                source="Source",
            ),
        ]
        ranked = rank(results, "anything")
        assert ranked[0].title == "Complete"

    def test_stable_order_for_equal_scores(self) -> None:
        results = [
            ResearchResult(title="First", url="https://a.com"),
            ResearchResult(title="Second", url="https://b.com"),
        ]
        ranked = rank(results, "unrelated query")
        # Both have same score — order preserved
        assert ranked[0].title == "First"

    def test_empty_list(self) -> None:
        ranked = rank([], "test")
        assert ranked == []
