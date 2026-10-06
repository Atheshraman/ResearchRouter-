"""Conversation/task context manager.

Keeps a small, per-session history of research turns (query, topic,
entities and the *compact* evidence returned) and resolves references such
as "those papers", "the second approach", "compare them" or "continue my
research" against it.

Only the slice of history relevant to the current query is returned — never
the whole conversation.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from research_router.context.evidence import Evidence, SourceType
from research_router.context.text import salient_terms, truncate_words
from research_router.utils.dates import now

_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
}  # fmt: skip
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(" + "|".join(_ORDINALS) + r")\s+(one|approach|paper|result|article|method|"
    r"link|option|study|source|item|technique|model|framework|repo|tool)\b",
    re.I,
)
_PLURAL_REF_RE = re.compile(
    r"\b(those|these|them|they|the above|above results|(?:the\s+)?previous results)\b"
    r"|\bthis\b(?=\s+(?:with|to|against|and)\b|\s*[?.!]?\s*$)",
    re.I,
)
_CONTINUE_RE = re.compile(
    r"\b(continue|keep going|go deeper|dig deeper|more on (?:this|that)|follow up on|expand on)\b",
    re.I,
)
_COMPARE_RE = re.compile(r"\b(compare|contrast|differences?|versus|vs\.?)\b", re.I)
_MEMORY_RE = re.compile(
    r"\b(my\s+(?:previous|past|earlier|old|prior|saved)\s+(?:[\w-]+\s+){0,3}(?:research|notes|findings|work)"
    r"|my\s+notes|what\s+i\s+(?:found|researched|learned)\s+(?:before|earlier|previously)"
    r"|previously\s+researched|from\s+(?:my\s+)?memory)\b",
    re.I,
)
# Words that express *what to do* with context rather than a new topic.
_COMMAND_WORDS = frozenset(
    """
    compare contrast summarize summarise summary explain list recap difference differences
    between versus vs them those these they results result previous above continue research
    more detail details deeper dig keep going go expand follow up on about what how why
    approach approaches one ones item items option options paper papers this that it
    my notes past earlier old prior saved findings work memory found learned before
    researched previously first second third fourth fifth last 1st 2nd 3rd 4th 5th
    tell give show describe discuss key main points
    """.split()  # noqa: SIM905
)

_MAX_STORED_RESULTS = 10


@dataclass
class TurnRecord:
    query: str
    resolved_query: str
    domain: str
    topic: str
    entities: list[str]
    results: list[Evidence]
    timestamp: str


@dataclass
class ResolvedContext:
    """Internal representation of the current task after reference resolution."""

    current_task: str
    resolved_query: str
    previous_task: str | None = None
    entities: list[str] = field(default_factory=list)
    previous_results: list[Evidence] = field(default_factory=list)
    user_intent: str = "new_research"
    required_context: list[str] = field(default_factory=list)
    needs_search: bool = True
    references: list[str] = field(default_factory=list)
    # Extra, independent search subjects derived from referenced items.
    search_subjects: list[str] = field(default_factory=list)
    memory_query: str | None = None
    seen_urls: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_task": self.current_task,
            "resolved_query": self.resolved_query,
            "previous_task": self.previous_task,
            "entities": self.entities,
            "previous_results": [
                {"title": e.source.title, "url": e.source.url} for e in self.previous_results
            ],
            "user_intent": self.user_intent,
            "required_context": self.required_context,
            "needs_search": self.needs_search,
            "references": self.references,
            "search_subjects": self.search_subjects,
        }


class ContextManager:
    """In-process, per-session research context (LRU-bounded)."""

    def __init__(self, max_turns: int = 10, max_sessions: int = 100) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[TurnRecord]] = OrderedDict()

    # ── history ───────────────────────────────────────────────────

    def history(self, session_id: str) -> list[TurnRecord]:
        return list(self._sessions.get(session_id, []))

    def record(
        self,
        session_id: str,
        resolved: ResolvedContext,
        domain: str,
        evidence: list[Evidence],
    ) -> None:
        """Store a compact record of a completed turn."""
        stored = [
            e.model_copy(
                update={
                    "claim": truncate_words(e.claim, 40) if e.claim else None,
                    "metadata": {},
                }
            )
            for e in evidence
            if e.source_type is SourceType.WEB
        ][:_MAX_STORED_RESULTS]
        # A context-only turn (e.g. "compare them") keeps pointing at the
        # results it discussed so a following "the second one" still resolves.
        if not stored and resolved.previous_results:
            stored = list(resolved.previous_results[:_MAX_STORED_RESULTS])
        topic = " ".join(salient_terms(resolved.resolved_query, drop_generic=True)[:8])
        turn = TurnRecord(
            query=resolved.current_task,
            resolved_query=resolved.resolved_query,
            domain=domain,
            topic=topic or resolved.resolved_query,
            entities=resolved.entities[:10],
            results=stored,
            timestamp=now().isoformat(timespec="seconds"),
        )
        turns = self._sessions.setdefault(session_id, [])
        turns.append(turn)
        del turns[: -self._max_turns]
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    # ── resolution ────────────────────────────────────────────────

    def resolve(self, session_id: str, query: str) -> ResolvedContext:
        """Resolve references in *query* against this session's history."""
        turns = self._sessions.get(session_id, [])
        prev = turns[-1] if turns else None
        ctx = ResolvedContext(current_task=query, resolved_query=query)

        memory_ref = _MEMORY_RE.search(query)
        ordinal = _ORDINAL_RE.search(query)
        plural = _PLURAL_REF_RE.search(query)
        cont = _CONTINUE_RE.search(query)
        compare = _COMPARE_RE.search(query)
        new_terms = self._new_terms(query)

        if memory_ref:
            ctx.references.append(memory_ref.group(0))
            ctx.required_context.append("memory")
            ctx.user_intent = "recall_memory"
            # "my previous RAG research" names its own topic; prefer that.
            named = [w for w in salient_terms(memory_ref.group(0)) if w not in _COMMAND_WORDS]
            ctx.memory_query = " ".join(named or new_terms) or (prev.topic if prev else query)

        if prev is None:
            ctx.entities = salient_terms(query, drop_generic=True)[:10]
            if memory_ref:
                # Recalling notes needs no web search unless a new topic is named.
                ctx.needs_search = bool(new_terms)
            elif (cont or ordinal or plural) and not new_terms:
                ctx.needs_search = False
                ctx.user_intent = "unresolved_reference"
            return ctx

        ctx.previous_task = prev.query
        ctx.seen_urls = {e.source.url for t in turns for e in t.results if e.source.url}

        if ordinal and prev.results:
            idx = _ORDINALS[ordinal.group(1).lower()]
            pool = prev.results
            if ordinal.group(2).lower() in ("paper", "study"):
                # "the second paper" counts papers, not every mixed result.
                papers = [e for e in pool if e.engine == "google_scholar"]
                pool = papers or pool
            if -len(pool) <= idx < len(pool):
                item = pool[idx]
                ctx.references.append(ordinal.group(0))
                ctx.previous_results = [_as_context(item)]
                ctx.required_context.append("previous_results")
                subject = truncate_words(item.source.title or item.claim, 12).rstrip(" …")
                ctx.resolved_query = " ".join([subject, *new_terms]).strip()
                ctx.user_intent = "follow_up_item"
                ctx.entities = [subject, *new_terms]
                return ctx

        if cont:
            ctx.references.append(cont.group(0))
            ctx.user_intent = "continue"
            ctx.previous_results = [_as_context(e) for e in prev.results[:5]]
            ctx.required_context.append("previous_results")
            ctx.resolved_query = " ".join([prev.topic, *new_terms]).strip()
            if not new_terms:
                ctx.resolved_query = f"{prev.topic} advances challenges"
            ctx.entities = salient_terms(ctx.resolved_query, drop_generic=True)[:10]
            return ctx

        if plural or (compare and not new_terms):
            if plural:
                ctx.references.append(plural.group(0))
            ctx.previous_results = [_as_context(e) for e in prev.results[:6]]
            ctx.required_context.append("previous_results")
            ctx.entities = [e.source.title for e in prev.results[:6] if e.source.title]
            ctx.user_intent = "compare" if compare else "follow_up"
            if not new_terms:
                # "summarise those papers", "compare them" — answerable from
                # context alone; no tool call needed.
                ctx.needs_search = False
                ctx.resolved_query = prev.resolved_query
            elif compare:
                # "compare them with LoRA" — search only the new subject.
                ctx.resolved_query = " ".join(new_terms)
            else:
                # "find GitHub implementations of those papers"
                directive = " ".join(new_terms)
                ctx.resolved_query = f"{prev.topic} {directive}".strip()
                ctx.search_subjects = [
                    f"{truncate_words(e.source.title, 8).rstrip(' …')} {directive}"
                    for e in prev.results[:3]
                    if e.source.title
                ]
            return ctx

        # No references to the session — independent new research.
        ctx.entities = salient_terms(query, drop_generic=True)[:10]
        if memory_ref:
            ctx.needs_search = bool(new_terms)
        return ctx

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _new_terms(query: str) -> list[str]:
        """Terms in *query* that introduce new subject matter."""
        stripped = _MEMORY_RE.sub(" ", query)
        stripped = _ORDINAL_RE.sub(" ", stripped)
        return [w for w in salient_terms(stripped) if w not in _COMMAND_WORDS and not w.isdigit()]


def _as_context(e: Evidence) -> Evidence:
    return e.model_copy(update={"source_type": SourceType.CONTEXT})
