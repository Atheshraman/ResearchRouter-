"""Deterministic query classifier.

Uses keyword / pattern signals to classify queries into research domains
without calling an LLM.  Falls back to LLM only when confidence is below
the configured threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from research_router.models.intent import ResearchDomain


@dataclass(frozen=True)
class ClassificationResult:
    """Deterministic classification output."""

    domain: ResearchDomain
    confidence: float
    signals: list[str] = field(default_factory=list)


# ── Signal tables ─────────────────────────────────────────────────────

# Each entry: (compiled regex, weight).  Weights are summed to compute
# a confidence score which is then clamped to [0, 1].

_SIGNALS: dict[ResearchDomain, list[tuple[re.Pattern[str], float]]] = {
    ResearchDomain.NEWS: [
        (re.compile(r"\b(latest|breaking|headlines?|news|current events)\b", re.I), 0.40),
        (re.compile(r"\b(today|yesterday|this morning|this week)\b", re.I), 0.15),
        (re.compile(r"\b(reported|coverage|press)\b", re.I), 0.20),
        (re.compile(r"\b(media|journalist|newspaper)\b", re.I), 0.20),
        (re.compile(r"\b(update|updates|announcement)\b", re.I), 0.15),
    ],
    ResearchDomain.ACADEMIC: [
        (re.compile(r"\b(research paper|research papers|journal|publication)\b", re.I), 0.45),
        (re.compile(r"\b(study|studies|findings|analysis)\b", re.I), 0.25),
        (re.compile(r"\b(scholar|scholarly|academic|peer.?review\w*)\b", re.I), 0.40),
        (re.compile(r"\b(thesis|dissertation|conference paper)\b", re.I), 0.40),
        (re.compile(r"\b(arxiv|ieee|acm|springer)\b", re.I), 0.35),
        (re.compile(r"\b(cite|citation|bibliography)\b", re.I), 0.25),
    ],
    ResearchDomain.JOBS: [
        (re.compile(r"\b(job|jobs|career|careers|hiring|vacancy|vacancies)\b", re.I), 0.40),
        (re.compile(r"\b(internship|internships|intern)\b", re.I), 0.45),
        (re.compile(r"\b(salary|compensation|remote work|work from home)\b", re.I), 0.25),
        (re.compile(r"\b(apply|application|resume|CV)\b", re.I), 0.20),
        (re.compile(r"\b(full.?time|part.?time|contract|freelance)\b", re.I), 0.25),
        (re.compile(r"\b(recruit|recruiter|openings)\b", re.I), 0.30),
    ],
    ResearchDomain.SHOPPING: [
        (re.compile(r"\b(buy|purchase|shop|shopping|order)\b", re.I), 0.40),
        (re.compile(r"\b(price|prices|cheap|cheapest|affordable|expensive)\b", re.I), 0.30),
        (re.compile(r"\b(under|below|budget)\b", re.I), 0.15),
        (re.compile(r"[₹$€£]\s?\d+", re.I), 0.35),
        (re.compile(r"\b\d+\s?(INR|USD|EUR|GBP|rupees|dollars)\b", re.I), 0.30),
        (re.compile(r"\b(best|top|review|reviews|rating|compare)\b", re.I), 0.15),
        (re.compile(r"\b(deal|deals|discount|offer|sale)\b", re.I), 0.25),
        (re.compile(r"\b(laptop|phone|headphone|camera|tablet|monitor|GPU|RTX)\b", re.I), 0.20),
    ],
    ResearchDomain.PLACES: [
        (re.compile(r"\b(near me|near by|nearby|close to|around)\b", re.I), 0.45),
        (re.compile(r"\b(restaurant|cafe|hotel|hospital|gym|store|mall)\b", re.I), 0.30),
        (re.compile(r"\b(directions|located|location|address|map)\b", re.I), 0.30),
        (re.compile(r"\b(open now|hours|timings|open)\b", re.I), 0.15),
        (re.compile(r"\b(best .+ in |top .+ in |places in )\b", re.I), 0.20),
        (re.compile(r"\b(near .+ airport|near .+ station)\b", re.I), 0.35),
    ],
}

# Minimum raw score to consider a domain a candidate.
_MIN_RAW_SCORE = 0.30


class QueryClassifier:
    """Deterministic keyword-based query classifier.

    Returns the highest-confidence domain.  If no domain exceeds
    ``_MIN_RAW_SCORE`` the result is ``ResearchDomain.GENERAL`` with
    low confidence.
    """

    def classify(self, query: str) -> ClassificationResult:
        """Classify *query* into a ``ResearchDomain``."""
        best_domain = ResearchDomain.GENERAL
        best_score = 0.0
        best_signals: list[str] = []

        for domain, patterns in _SIGNALS.items():
            score = 0.0
            signals: list[str] = []
            for pattern, weight in patterns:
                if pattern.search(query):
                    score += weight
                    signals.append(pattern.pattern)
            # Clamp to 1.0
            score = min(score, 1.0)
            if score > best_score:
                best_score = score
                best_domain = domain
                best_signals = signals

        if best_score < _MIN_RAW_SCORE:
            # No strong domain signal — fall back to GENERAL with moderate
            # confidence so ambiguous queries can still trigger LLM analysis.
            return ClassificationResult(
                domain=ResearchDomain.GENERAL,
                confidence=min(0.5, max(0.3, 1.0 - best_score)),
                signals=[],
            )

        return ClassificationResult(
            domain=best_domain,
            confidence=min(best_score, 1.0),
            signals=best_signals,
        )
