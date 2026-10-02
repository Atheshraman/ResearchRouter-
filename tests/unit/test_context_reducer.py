"""Context reducer, token estimation and evidence tests."""

from __future__ import annotations

from research_router.context.evidence import Evidence, EvidenceSource, SourceType
from research_router.context.reducer import ContextReducer
from research_router.context.text import clean_text
from research_router.context.tokens import estimate_tokens


def _ev(
    claim: str,
    url: str | None,
    title: str = "",
    task: str = "t1",
    rank: int = 0,
    source_type: SourceType = SourceType.WEB,
    date: str | None = None,
) -> Evidence:
    return Evidence(
        claim=claim,
        source=EvidenceSource(title=title or None, url=url, date=date),
        task_id=task,
        source_type=source_type,
        metadata={"rank": rank},
    )


LONG = (
    "Retrieval augmented generation reduces hallucination in large language models. "
    "The study evaluates RAG on 12 benchmarks and finds a 23% drop in factual errors. "
    "Cookie policy. Subscribe now for updates. "
    "Unrelated sentence about the weather in Paris during spring time."
)


class TestTokens:
    def test_estimate_is_monotonic(self) -> None:
        short, long = estimate_tokens("hello world"), estimate_tokens("hello world " * 50)
        assert 0 < short < long

    def test_estimate_accepts_structures_and_unserialisable(self) -> None:
        assert estimate_tokens({"a": [1, 2, 3]}) > 0
        assert estimate_tokens(object()) > 0  # falls back to str(), never raises


class TestText:
    def test_clean_text_strips_html_and_boilerplate(self) -> None:
        out = clean_text("<b>RAG</b> works well. Read more... Accept cookies")
        assert "<b>" not in out and "Read more" not in out and "cookies" not in out
        assert out.startswith("RAG works well")


class TestReducer:
    def test_dedups_urls_and_near_duplicate_text(self) -> None:
        dup = "RAG reduces hallucination in LLMs by grounding answers"
        items = [
            _ev(dup, "https://a.com/x?utm_source=t", "A"),
            _ev(dup, "https://b.com/y", "B"),
            _ev("Other finding about RAG hallucination metrics", "https://a.com/x", "C"),
            _ev("Reranking improves RAG hallucination rates", "https://c.com/z", "D"),
        ]
        out = ContextReducer().reduce(items, "RAG hallucination", 5000)
        assert len(out.evidence) == 2
        assert out.stats.raw == 4 and out.stats.deduplicated == 2

    def test_under_budget_applies_no_compression(self) -> None:
        items = [_ev(LONG, "https://a.com", "RAG study")]
        out = ContextReducer().reduce(items, "RAG hallucination", 5000)
        assert out.stats.compression_level == 0
        assert out.stats.dropped_for_budget == 0
        # extraction keeps relevant/numeric sentences and drops the off-topic one
        claim = out.evidence[0].claim
        assert "23%" in claim and "weather" not in claim

    def test_budget_is_enforced_adaptively(self) -> None:
        items = [
            _ev(
                # distinct vocabulary per item so near-dup detection keeps them all
                "RAG hallucination " + " ".join(f"term{i}x{j}" for j in range(40)) + ".",
                f"https://s{i}.com/p",
                f"RAG paper {i}",
                rank=i,
            )
            for i in range(30)
        ]
        budget = 400
        out = ContextReducer(min_relevance=0.0).reduce(items, "RAG hallucination", budget)
        assert out.stats.output_tokens <= budget
        assert out.stats.compression_level > 0  # escalated only because over budget
        assert out.evidence, "keeps the highest-value evidence instead of dropping everything"
        assert all(e.source.url for e in out.evidence), "provenance preserved"

    def test_high_relevance_survives_budget_pressure(self) -> None:
        items = [
            _ev("Totally unrelated gardening advice for roses and tulips", f"https://g{i}.com")
            for i in range(10)
        ]
        key = _ev("RAG hallucination mitigation with reranking", "https://key.com", "RAG")
        items.append(key)
        out = ContextReducer().reduce(items, "RAG hallucination mitigation", 150)
        assert out.evidence[0].source.url == "https://key.com"

    def test_keeps_one_evidence_item_when_budget_is_too_small(self) -> None:
        items = [
            _ev("RAG hallucination mitigation with reranking", "https://key.com", "RAG")
        ]
        out = ContextReducer().reduce(items, "RAG hallucination mitigation", 1)
        assert len(out.evidence) == 1

    def test_low_relevance_removed_but_each_task_keeps_minimum(self) -> None:
        items = [
            _ev("RAG hallucination benchmark results", "https://a.com", task="t1"),
            _ev("Gardening tips for spring", "https://b.com", task="t2"),
            _ev("More gardening tips for autumn", "https://c.com", task="t2"),
        ]
        out = ContextReducer(min_per_task=1).reduce(items, "RAG hallucination", 5000)
        assert {e.task_id for e in out.evidence} == {"t1", "t2"}
        assert len(out.evidence) == 2

    def test_on_topic_results_kept_when_a_rare_query_term_is_missing(self) -> None:
        # Every result matches the core topic ("RAG hallucination"); only one also
        # mentions "evaluation". Rarity must not make the other nine look irrelevant.
        topics = [
            "retrieval", "hallucination", "grounding", "reranking", "citations",
            "faithfulness", "evaluation", "chunking", "embeddings", "knowledge graphs",
        ]  # fmt: skip
        filler = "Large language models frequently generate fluent but unsupported statements."
        items = [
            _ev(
                f"RAG {t} reduces hallucination by {10 + i}% on benchmark {i}. {filler}",
                f"https://s{i}.org/{i}",
                f"RAG hallucination and {t}: a study",
                rank=i,
            )
            for i, t in enumerate(topics)
        ]
        query = "RAG hallucination mitigation techniques and evaluation"
        out = ContextReducer().reduce(items, query, 5000)
        assert out.stats.relevant == 10  # IDF-dominated scoring kept only 4

    def test_web_preferred_over_memory_duplicate(self) -> None:
        items = [
            _ev("Old note about RAG", "https://a.com", source_type=SourceType.MEMORY),
            _ev("Fresh RAG result", "https://a.com"),
        ]
        out = ContextReducer().reduce(items, "RAG", 5000)
        assert [e.source_type for e in out.evidence] == [SourceType.WEB]

    def test_malformed_candidates_are_skipped(self) -> None:
        items = [_ev("RAG result", "https://a.com"), "not evidence"]
        out = ContextReducer().reduce(items, "RAG", 5000)  # type: ignore[arg-type]
        assert len(out.evidence) == 1

    def test_does_not_mutate_inputs(self) -> None:
        item = _ev(LONG, "https://a.com", "RAG")
        ContextReducer().reduce([item], "RAG hallucination", 60)
        assert item.claim == LONG and "_full_claim" not in item.metadata

    def test_empty_input(self) -> None:
        out = ContextReducer().reduce([], "anything", 1000)
        assert out.evidence == [] and out.stats.output_tokens == 0

    def test_to_context_keeps_provenance_and_labels_memory(self) -> None:
        e = _ev("claim", "https://a.com", "T", source_type=SourceType.MEMORY, date="2025-01-02")
        e.recorded_at = "2025-02-01T00:00:00+00:00"
        ctx = e.to_context()
        assert ctx["url"] == "https://a.com" and ctx["date"] == "2025-01-02"
        assert ctx["source_type"] == "memory" and ctx["recorded_at"].startswith("2025-02-01")
