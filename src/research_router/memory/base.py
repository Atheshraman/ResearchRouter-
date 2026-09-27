"""Memory-store contract.

Persistent memory is an *enhancement*: every implementation must degrade to
"no hits" / "not saved" rather than raise into the research pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from research_router.context.evidence import Evidence


@dataclass
class MemoryPassage:
    text: str
    url: str | None = None
    date: str | None = None


@dataclass
class MemoryHit:
    note: str  # vault-relative note path without extension, e.g. "Research/RAG/Hallucination"
    title: str
    score: float
    recorded_at: str | None
    passages: list[MemoryPassage] = field(default_factory=list)


@dataclass
class ResearchRecord:
    """A completed research task worth remembering."""

    query: str
    domain: str
    tools: list[str]
    findings: list[Evidence]
    entities: list[str]
    tags: list[str]
    related_notes: list[str] = field(default_factory=list)


class MemoryStore(ABC):
    name: str = "memory"

    @property
    @abstractmethod
    def available(self) -> bool:
        """True when the store is configured and reachable."""

    @abstractmethod
    async def search(self, query: str, limit: int = 3) -> list[MemoryHit]:
        """Return the notes most relevant to *query* (never the whole store)."""

    @abstractmethod
    async def save(self, record: ResearchRecord) -> str | None:
        """Persist *record*; return the note identifier, or ``None`` if skipped."""

    def status(self) -> dict[str, Any]:
        return {"backend": self.name, "available": self.available}


class NullMemory(MemoryStore):
    """Used when no persistent memory is configured."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    async def search(self, query: str, limit: int = 3) -> list[MemoryHit]:
        return []

    async def save(self, record: ResearchRecord) -> str | None:
        return None
