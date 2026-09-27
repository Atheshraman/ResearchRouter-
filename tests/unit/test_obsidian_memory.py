"""Obsidian persistent-memory tests (use a temporary vault)."""

from __future__ import annotations

from pathlib import Path

from research_router.context.evidence import Evidence, EvidenceSource
from research_router.memory.base import NullMemory, ResearchRecord
from research_router.memory.obsidian import ObsidianMemory, hits_to_evidence, note_location


def _record(query: str = "Find recent research papers about RAG hallucination") -> ResearchRecord:
    return ResearchRecord(
        query=query,
        domain="academic",
        tools=["academic_search"],
        findings=[
            Evidence(
                claim="Self-RAG reduces hallucination via self-reflection tokens.",
                source=EvidenceSource(
                    title="Self-RAG", url="https://arxiv.org/abs/2310.11511", date="2024-01-10"
                ),
            )
        ],
        entities=["rag", "hallucination"],
        tags=["rag", "hallucination"],
    )


class TestNoteLocation:
    def test_acronym_becomes_topic(self) -> None:
        assert note_location("Find recent research papers about RAG hallucination") == (
            "RAG",
            "Hallucination",
        )

    def test_path_segments_are_sanitised(self) -> None:
        topic, sub = note_location("../../etc/passwd secrets")
        assert "/" not in topic and ".." not in topic and "/" not in sub


class TestObsidianMemory:
    async def test_unconfigured_and_missing_vault_degrade(self, tmp_path: Path) -> None:
        for mem in (ObsidianMemory(""), ObsidianMemory(str(tmp_path / "missing"))):
            assert mem.available is False
            assert await mem.search("rag") == []
            assert await mem.save(_record()) is None
            assert mem.status()["reason"]
        assert await NullMemory().search("x") == []

    async def test_save_then_search_roundtrip(self, tmp_path: Path) -> None:
        mem = ObsidianMemory(str(tmp_path))
        note = await mem.save(_record())
        assert note == "Research/RAG/Hallucination"
        text = (tmp_path / "Research/RAG/Hallucination.md").read_text()
        assert text.startswith("---\n") and "created:" in text and "#research" in text
        assert "[Self-RAG](https://arxiv.org/abs/2310.11511) (2024-01-10)" in text

        hits = await mem.search("RAG hallucination")
        assert hits and hits[0].note == "Research/RAG/Hallucination"
        evidence = hits_to_evidence(hits)
        assert evidence[0].source_type.value == "memory"
        assert evidence[0].recorded_at  # date preserved so old memory is identifiable
        assert any(e.source.url == "https://arxiv.org/abs/2310.11511" for e in evidence)

    async def test_update_appends_session_and_merges_frontmatter(self, tmp_path: Path) -> None:
        mem = ObsidianMemory(str(tmp_path))
        await mem.save(_record())
        await mem.save(_record())
        text = (tmp_path / "Research/RAG/Hallucination.md").read_text()
        assert text.count("## Session ") == 2
        assert text.count("\n---\n") == 1  # still a single frontmatter block
        assert text.count('  - "Find recent research papers about RAG hallucination"') == 1

    async def test_search_only_reads_research_folder(self, tmp_path: Path) -> None:
        (tmp_path / "Personal").mkdir()
        (tmp_path / "Personal/diary.md").write_text("RAG hallucination secret diary")
        mem = ObsidianMemory(str(tmp_path))
        assert await mem.search("RAG hallucination") == []

    async def test_irrelevant_notes_not_returned(self, tmp_path: Path) -> None:
        mem = ObsidianMemory(str(tmp_path))
        await mem.save(_record())
        assert await mem.search("sourdough bread baking") == []
