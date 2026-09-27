"""Reference resolution tests for the session context manager."""

from __future__ import annotations

from research_router.context.evidence import Evidence, EvidenceSource, SourceType
from research_router.context.manager import ContextManager


def _paper(title: str, url: str, engine: str = "google_scholar") -> Evidence:
    return Evidence(
        claim=f"{title} abstract", source=EvidenceSource(title=title, url=url), engine=engine
    )


def _seeded() -> ContextManager:
    mgr = ContextManager()
    first = mgr.resolve("s", "Find recent research papers about RAG hallucination")
    mgr.record(
        "s",
        first,
        "academic",
        [
            _paper("Self-RAG: Learning to Retrieve", "https://arxiv.org/1"),
            _paper("RAG tutorial blog", "https://blog.example/rag", engine="google"),
            _paper("Corrective RAG", "https://arxiv.org/2"),
        ],
    )
    return mgr


class TestResolution:
    def test_new_query_without_history(self) -> None:
        ctx = ContextManager().resolve("s", "RAG hallucination papers")
        assert ctx.user_intent == "new_research" and ctx.needs_search
        assert "rag" in ctx.entities and ctx.previous_task is None

    def test_ordinal_paper_counts_only_papers(self) -> None:
        ctx = _seeded().resolve("s", "What about the second paper?")
        assert ctx.user_intent == "follow_up_item"
        assert ctx.resolved_query.startswith("Corrective RAG")
        assert [e.source_type for e in ctx.previous_results] == [SourceType.CONTEXT]

    def test_ordinal_approach_counts_all_results(self) -> None:
        ctx = _seeded().resolve("s", "what about the second approach?")
        assert ctx.resolved_query.startswith("RAG tutorial blog")

    def test_compare_them_needs_no_search(self) -> None:
        ctx = _seeded().resolve("s", "compare them")
        assert ctx.user_intent == "compare"
        assert ctx.needs_search is False
        assert len(ctx.previous_results) == 3

    def test_those_papers_with_new_directive_builds_subjects(self) -> None:
        ctx = _seeded().resolve("s", "find GitHub implementations of those papers")
        assert ctx.needs_search
        assert ctx.search_subjects and all("github" in s for s in ctx.search_subjects)

    def test_continue_uses_previous_topic(self) -> None:
        ctx = _seeded().resolve("s", "continue my research")
        assert ctx.user_intent == "continue"
        assert "rag" in ctx.resolved_query.lower()
        assert "hallucination" in ctx.resolved_query.lower()
        assert "https://arxiv.org/1" in ctx.seen_urls

    def test_memory_reference_extracts_topic(self) -> None:
        ctx = _seeded().resolve("s", "Compare this with my previous RAG research.")
        assert "memory" in ctx.required_context
        assert ctx.memory_query == "rag"
        assert ctx.needs_search is False

    def test_unresolvable_reference_without_history(self) -> None:
        ctx = ContextManager().resolve("fresh", "compare them")
        assert ctx.needs_search is False and ctx.user_intent == "unresolved_reference"

    def test_sessions_are_isolated_and_bounded(self) -> None:
        mgr = ContextManager(max_turns=2, max_sessions=2)
        for sid in ("a", "b", "c"):
            for i in range(3):
                mgr.record(sid, mgr.resolve(sid, f"topic {i}"), "general", [])
        assert mgr.history("a") == []  # evicted (LRU)
        assert len(mgr.history("c")) == 2
