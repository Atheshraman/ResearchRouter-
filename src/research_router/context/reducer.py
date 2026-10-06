"""Context reducer — turns raw tool results into a compact, budgeted context.

Pipeline::

    raw evidence → normalise → deduplicate → rank → filter → extract
                 → adaptive compression loop (until under the token budget)

The loop is *adaptive*: when the context already fits, nothing beyond
dedup/filtering is applied.  Otherwise it escalates one step at a time —
tighter claim extraction, then compact serialisation, then dropping the
lowest-value evidence — re-checking the budget after every step so
high-value evidence is preserved as long as possible.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from research_router.context.evidence import Evidence, SourceType
from research_router.context.text import (
    GENERIC_TERMS,
    clean_text,
    jaccard,
    salient_terms,
    shingles,
    split_sentences,
    strip_leading_date,
    tokenize,
    truncate_words,
)
from research_router.context.tokens import estimate_tokens
from research_router.research.deduplicator import _normalise_url
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

# Claim word limits per compression level (None = untruncated).
_LEVEL_WORDS: tuple[int | None, ...] = (None, 45, 25, 12)
_NEAR_DUP_THRESHOLD = 0.7
_FULL_CLAIM = "_full_claim"


@dataclass
class ReductionStats:
    raw: int = 0
    normalised: int = 0
    deduplicated: int = 0
    relevant: int = 0
    final: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compression_level: int = 0
    dropped_for_budget: int = 0
    steps: list[str] = field(default_factory=list)
    fallback: bool = False

    @property
    def compression_ratio(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return round(1 - self.output_tokens / self.input_tokens, 4)

    @property
    def remaining_ratio(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return round(self.output_tokens / self.input_tokens, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_results": self.raw,
            "normalised_results": self.normalised,
            "deduplicated_results": self.deduplicated,
            "relevant_results": self.relevant,
            "final_evidence": self.final,
            "input_tokens_estimated": self.input_tokens,
            "output_context_tokens": self.output_tokens,
            "compression_ratio": self.compression_ratio,
            "reduction_ratio": self.compression_ratio,
            "remaining_ratio": self.remaining_ratio,
            "reduction_percent": round(self.compression_ratio * 100, 2),
            "remaining_percent": round(self.remaining_ratio * 100, 2),
            "compression_level": self.compression_level,
            "dropped_for_budget": self.dropped_for_budget,
            "steps": self.steps,
            "fallback": self.fallback,
        }


@dataclass
class ReductionResult:
    evidence: list[Evidence]
    context: list[dict[str, Any]]
    stats: ReductionStats


class ContextReducer:
    """Deterministic, budget-aware evidence reducer."""

    def __init__(
        self,
        *,
        min_relevance: float = 0.2,
        min_per_task: int = 2,
        max_sentences: int = 3,
    ) -> None:
        self._min_relevance = min_relevance
        self._min_per_task = min_per_task
        self._max_sentences = max_sentences

    # ── public API ────────────────────────────────────────────────

    def reduce(
        self,
        candidates: list[Evidence],
        query: str,
        budget_tokens: int,
        *,
        max_items: int | None = None,
        freshness_required: bool = False,
        input_tokens: int | None = None,
    ) -> ReductionResult:
        """Reduce *candidates* to at most *budget_tokens* of context.

        ``input_tokens`` lets the caller pass the size of the raw payload the
        candidates were built from; otherwise it is estimated here.
        """
        stats = ReductionStats(raw=len(candidates))
        stats.input_tokens = (
            input_tokens
            if input_tokens is not None
            else estimate_tokens([c.to_context() for c in candidates if isinstance(c, Evidence)])
        )
        try:
            return self._reduce(
                candidates, query, budget_tokens, max_items, freshness_required, stats
            )
        except Exception:
            logger.warning("Context reduction failed; using truncation fallback", exc_info=True)
            return self._fallback(candidates, budget_tokens, stats)

    # ── pipeline ──────────────────────────────────────────────────

    def _reduce(
        self,
        candidates: list[Evidence],
        query: str,
        budget: int,
        max_items: int | None,
        freshness_required: bool,
        stats: ReductionStats,
    ) -> ReductionResult:
        items = self._normalise(candidates)
        stats.normalised = len(items)

        items = self._deduplicate(items)
        stats.deduplicated = len(items)

        self._score(items, query, freshness_required)
        items.sort(key=lambda e: e.relevance_score, reverse=True)

        items = self._filter(items)
        stats.relevant = len(items)
        if max_items is not None and len(items) > max_items:
            items = items[:max_items]
            stats.steps.append(f"capped_to_{max_items}_items")

        # Information extraction: keep only the sentences that carry the answer.
        for e in items:
            e.metadata[_FULL_CLAIM] = self._extract(e.claim, query)
        level = 0
        items = [_at_level(e, level) for e in items]
        tokens = self._measure(items, level)
        stats.steps.append("dedup+filter+extract")

        # Adaptive escalation: only as much compression as the budget needs.
        while tokens > budget and level < len(_LEVEL_WORDS) - 1:
            level += 1
            items = [_at_level(e, level) for e in items]
            tokens = self._measure(items, level)
            stats.steps.append(f"compress_level_{level}")

        while tokens > budget and len(items) > 1:
            items = self._drop_one(items)
            stats.dropped_for_budget += 1
            tokens = self._measure(items, level)
        if stats.dropped_for_budget:
            stats.steps.append(f"dropped_{stats.dropped_for_budget}_low_value")

        compact = level >= 2
        stats.compression_level = level
        stats.final = len(items)
        for e in items:
            e.metadata.pop(_FULL_CLAIM, None)
        context = [e.to_context(compact=compact) for e in items]
        stats.output_tokens = estimate_tokens(context) if context else 0
        return ReductionResult(evidence=items, context=context, stats=stats)

    def _fallback(
        self, candidates: list[Evidence], budget: int, stats: ReductionStats
    ) -> ReductionResult:
        """Last-resort reduction: keep candidates in order until the budget is hit."""
        stats.fallback = True
        kept: list[Evidence] = []
        context: list[dict[str, Any]] = []
        for c in candidates:
            try:
                entry = c.to_context(compact=True)
                entry["claim"] = truncate_words(str(entry.get("claim", "")), 25)
            except Exception:
                continue
            if estimate_tokens([*context, entry]) > budget:
                break
            kept.append(c)
            context.append(entry)
        stats.final = len(kept)
        stats.output_tokens = estimate_tokens(context) if context else 0
        stats.steps.append("fallback_truncation")
        return ReductionResult(evidence=kept, context=context, stats=stats)

    # ── stages ────────────────────────────────────────────────────

    @staticmethod
    def _normalise(candidates: list[Evidence]) -> list[Evidence]:
        out: list[Evidence] = []
        for c in candidates:
            if not isinstance(c, Evidence):
                continue  # malformed input — skip rather than fail
            claim = strip_leading_date(clean_text(c.claim))
            title = clean_text(c.source.title) or None
            if not claim and not title:
                continue
            source = c.source.model_copy(update={"title": title})
            out.append(
                c.model_copy(
                    update={
                        "claim": claim or title or "",
                        "source": source,
                        "metadata": dict(c.metadata),  # never mutate the caller's objects
                    }
                )
            )
        return out

    @staticmethod
    def _deduplicate(items: list[Evidence]) -> list[Evidence]:
        # Web evidence wins over memory/context duplicates: it is fresher.
        ordered = sorted(items, key=lambda e: e.source_type is not SourceType.WEB)
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        kept: list[Evidence] = []
        kept_shingles: list[set[tuple[str, ...]]] = []
        for e in ordered:
            norm = _normalise_url(e.source.url) if e.source.url else None
            if norm and norm in seen_urls:
                continue
            title_key = (e.task_id, e.domain, e.engine, " ".join(tokenize(e.source.title or "")))
            if len(title_key[3]) > 20 and title_key in seen_titles:
                continue
            sh = shingles(e.claim)
            if any(jaccard(sh, other) >= _NEAR_DUP_THRESHOLD for other in kept_shingles):
                continue
            if norm:
                seen_urls.add(norm)
            if title_key[3]:
                seen_titles.add(title_key)
            kept_shingles.append(sh)
            kept.append(e)
        return kept

    @staticmethod
    def _score(items: list[Evidence], query: str, freshness_required: bool) -> None:
        terms = salient_terms(query)
        if not items:
            return
        docs = [set(tokenize(f"{e.source.title or ''} {e.claim}")) for e in items]
        titles = [set(tokenize(e.source.title or "")) for e in items]
        n = len(docs)
        df = Counter(t for d in docs for t in set(terms) & d)
        # Plain weights measure how much of the query a result covers. Rarity (IDF)
        # only breaks ties: core topic terms appear in *every* result, so an
        # IDF-dominated score would rank on-topic results as irrelevant.
        plain = {t: 0.5 if t in GENERIC_TERMS else 1.0 for t in terms}
        rare = {t: plain[t] * (1.0 + math.log((n + 1) / (df[t] + 0.5))) for t in terms}
        plain_total = sum(plain.values()) or 1.0
        rare_total = sum(rare.values()) or 1.0
        cutoff = datetime.now(UTC) - timedelta(days=365)

        for e, doc, title in zip(items, docs, titles, strict=True):
            local_terms = salient_terms(str(e.metadata.get("task_query", query))) or terms
            local_plain = {t: plain.get(t, 1.0) for t in local_terms}
            local_total = sum(local_plain.values()) or 1.0
            coverage = sum(w for t, w in local_plain.items() if t in doc) / local_total
            distinct = sum(rare.get(t, 1.0) for t in local_terms if t in doc) / rare_total
            title_hit = sum(w for t, w in local_plain.items() if t in title) / local_total
            score = 0.55 * coverage + 0.15 * distinct + 0.2 * title_hit
            rank = e.metadata.get("rank")
            if isinstance(rank, int) and rank >= 0:
                score += 0.1 / (1 + rank)  # engine's own ordering as a weak prior
            if freshness_required and _is_recent(e.source.date, cutoff):
                score += 0.1
            if e.domain == "news" and e.source.date:
                if _is_recent(e.source.date, cutoff):
                    score += 0.1
                if e.source.publisher:
                    score += 0.03
            if not terms:
                score = max(score, 0.5)
            e.relevance_score = max(0.0, min(1.0, score))

    def _filter(self, items: list[Evidence]) -> list[Evidence]:
        per_task: Counter[str | None] = Counter()
        kept: list[Evidence] = []
        for e in items:  # already sorted by score
            if e.relevance_score >= self._min_relevance or per_task[e.task_id] < self._min_per_task:
                kept.append(e)
                per_task[e.task_id] += 1
        return kept

    def _extract(self, claim: str, query: str) -> str:
        terms = set(salient_terms(query, drop_generic=True)) or set(salient_terms(query))
        sentences = split_sentences(claim)
        if len(sentences) <= 1:
            return claim
        useful = [s for s in sentences if terms & set(tokenize(s))]
        chosen = useful or sentences[:1]
        return " ".join(chosen[: self._max_sentences])

    # ── compression helpers ───────────────────────────────────────

    @staticmethod
    def _measure(items: list[Evidence], level: int) -> int:
        if not items:
            return 0
        return estimate_tokens([e.to_context(compact=level >= 2) for e in items])

    @staticmethod
    def _drop_one(items: list[Evidence]) -> list[Evidence]:
        """Drop the lowest-value item from the most-represented task."""
        counts = Counter(e.task_id for e in items)
        heaviest = max(counts.values())
        for idx in range(len(items) - 1, -1, -1):
            if counts[items[idx].task_id] == heaviest:
                return items[:idx] + items[idx + 1 :]
        return items[:-1]


def _at_level(e: Evidence, level: int) -> Evidence:
    full = str(e.metadata.get(_FULL_CLAIM, e.claim))
    limit = _LEVEL_WORDS[level]
    claim = full if limit is None else truncate_words(full, limit)
    return e.model_copy(update={"claim": claim})


def _is_recent(date_str: str | None, cutoff: datetime) -> bool:
    if not date_str:
        return False
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt >= cutoff
