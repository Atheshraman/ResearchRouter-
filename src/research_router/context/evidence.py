"""Source-aware evidence model.

Every piece of information that reaches the LLM is an ``Evidence`` block that
keeps its provenance (title, URL, date, origin) through compression.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    WEB = "web"  # freshly retrieved in this request
    MEMORY = "memory"  # recalled from persistent (Obsidian) memory
    CONTEXT = "context"  # carried over from earlier turns in this session


class EvidenceSource(BaseModel):
    title: str | None = None
    url: str | None = None
    date: str | None = None
    publisher: str | None = None


class Evidence(BaseModel):
    """A single claim plus the provenance needed to cite it."""

    claim: str
    why_relevant: str | None = None
    source: EvidenceSource = Field(default_factory=EvidenceSource)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type: SourceType = SourceType.WEB
    task_id: str | None = None
    domain: str | None = None
    engine: str | None = None
    # When the evidence was recorded (memory) or retrieved (web), ISO-8601.
    recorded_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_context(self, *, compact: bool = False) -> dict[str, Any]:
        """Serialise to the minimal dict handed to the LLM."""
        out: dict[str, Any] = {"claim": self.claim}
        if self.source.title:
            out["title"] = self.source.title
        if self.source.url:
            out["url"] = self.source.url
        if self.source.date:
            out["date"] = self.source.date
        if not compact and self.source.publisher:
            out["source"] = self.source.publisher  # same key as legacy ResearchResult
        out["relevance"] = round(self.relevance_score, 2)
        out["source_type"] = self.source_type.value
        if self.source_type is not SourceType.WEB and self.recorded_at:
            out["recorded_at"] = self.recorded_at
        if self.task_id:
            out["task_id"] = self.task_id
        if self.domain:
            out["domain"] = self.domain
        if self.engine:
            out["engine"] = self.engine
        if self.why_relevant:
            out["why_relevant"] = self.why_relevant
        return out
