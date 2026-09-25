"""Intent analysis — hybrid classifier + optional LLM."""

from __future__ import annotations

import re
from typing import Any

from research_router.llm.base import LLMProvider
from research_router.models.intent import ResearchDepth, ResearchDomain, ResearchIntent
from research_router.router.classifier import ClassificationResult, QueryClassifier
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

# ── entity extraction patterns ────────────────────────────────────────

_LOCATION_RE = re.compile(r"\b(?:in|at|near|around)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")
_PRICE_MAX_RE = re.compile(r"(?:under|below|less than|max|upto|up to)\s*[₹$€£]?\s?(\d[\d,]*)", re.I)
_PRICE_MIN_RE = re.compile(r"(?:above|over|more than|min|starting)\s*[₹$€£]?\s?(\d[\d,]*)", re.I)
_CURRENCY_RE = re.compile(r"[₹]|INR", re.I)
_CURRENCY_USD_RE = re.compile(r"[$]|USD", re.I)
_CURRENCY_EUR_RE = re.compile(r"[€]|EUR", re.I)
_CURRENCY_GBP_RE = re.compile(r"[£]|GBP", re.I)

_DATE_RANGE_RE = re.compile(
    r"\b(today|this week|this month|past week|past month|past year"
    r"|recently|recent|last week|last month)\b",
    re.I,
)
_DATE_RANGE_MAP: dict[str, str] = {
    "today": "today",
    "this week": "this_week",
    "this month": "this_month",
    "past week": "past_week",
    "last week": "past_week",
    "past month": "past_month",
    "last month": "past_month",
    "past year": "past_year",
    "recently": "recent",
    "recent": "recent",
}


def _parse_number(s: str) -> float:
    return float(s.replace(",", ""))


class IntentAnalyser:
    """Hybrid intent analyser.

    1. Run the fast deterministic classifier.
    2. If confidence ≥ threshold → use deterministic result + entity extraction.
    3. If confidence < threshold → call the LLM for structured intent.
    4. If the LLM fails → fall back to deterministic.
    """

    def __init__(
        self,
        classifier: QueryClassifier,
        llm_provider: LLMProvider | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self._classifier = classifier
        self._llm = llm_provider
        self._threshold = confidence_threshold

    async def analyse(self, query: str) -> ResearchIntent:
        """Analyse *query* and return a ``ResearchIntent``."""
        classification = self._classifier.classify(query)

        if classification.confidence >= self._threshold:
            logger.info(
                "Deterministic classification accepted",
                extra={
                    "extra_data": {
                        "domain": classification.domain.value,
                        "confidence": classification.confidence,
                    }
                },
            )
            return self._build_intent_from_classification(query, classification)

        # Complex / ambiguous → try LLM
        if self._llm is not None:
            try:
                return await self._analyse_with_llm(query, classification)
            except Exception:
                logger.warning(
                    "LLM analysis failed, falling back to deterministic",
                    exc_info=True,
                )

        # Fallback
        return self._build_intent_from_classification(query, classification)

    # ── deterministic intent building ─────────────────────────────

    def _build_intent_from_classification(
        self,
        query: str,
        classification: ClassificationResult,
    ) -> ResearchIntent:
        location = self._extract_location(query)
        date_range = self._extract_date_range(query)
        price_max = self._extract_price_max(query)
        price_min = self._extract_price_min(query)
        currency = self._extract_currency(query)
        freshness = "recent" if date_range in ("recent", "past_week") else None

        return ResearchIntent(
            domain=classification.domain,
            query=query,
            keywords=query.lower().split(),
            location=location,
            date_range=date_range,
            freshness=freshness,
            price_max=price_max,
            price_min=price_min,
            currency=currency,
            research_depth=ResearchDepth.STANDARD,
            confidence=classification.confidence,
            requires_multiple_searches=False,
        )

    # ── LLM-assisted analysis ─────────────────────────────────────

    async def _analyse_with_llm(
        self,
        query: str,
        fallback: ClassificationResult,
    ) -> ResearchIntent:
        assert self._llm is not None
        data: dict[str, Any] = await self._llm.analyse_intent(query)
        logger.info(
            "LLM intent analysis complete",
            extra={"extra_data": {"llm_domain": data.get("domain")}},
        )

        # Validate domain
        domain_str = data.get("domain", "general")
        try:
            domain = ResearchDomain(domain_str)
        except ValueError:
            domain = fallback.domain

        # Validate depth
        depth_str = data.get("research_depth", "standard")
        try:
            depth = ResearchDepth(depth_str)
        except ValueError:
            depth = ResearchDepth.STANDARD

        return ResearchIntent(
            domain=domain,
            query=data.get("query", query),
            keywords=data.get("keywords", []),
            location=data.get("location"),
            date_range=data.get("date_range"),
            freshness=data.get("freshness"),
            price_min=data.get("price_min"),
            price_max=data.get("price_max"),
            currency=data.get("currency"),
            job_type=data.get("job_type"),
            remote=data.get("remote"),
            experience_level=data.get("experience_level"),
            research_depth=depth,
            confidence=float(data.get("confidence", fallback.confidence)),
            requires_multiple_searches=bool(data.get("requires_multiple_searches", False)),
        )

    # ── entity extractors ─────────────────────────────────────────

    @staticmethod
    def _extract_location(query: str) -> str | None:
        m = _LOCATION_RE.search(query)
        return m.group(1) if m else None

    @staticmethod
    def _extract_date_range(query: str) -> str | None:
        m = _DATE_RANGE_RE.search(query)
        if m:
            return _DATE_RANGE_MAP.get(m.group(1).lower())
        return None

    @staticmethod
    def _extract_price_max(query: str) -> float | None:
        m = _PRICE_MAX_RE.search(query)
        return _parse_number(m.group(1)) if m else None

    @staticmethod
    def _extract_price_min(query: str) -> float | None:
        m = _PRICE_MIN_RE.search(query)
        return _parse_number(m.group(1)) if m else None

    @staticmethod
    def _extract_currency(query: str) -> str | None:
        if _CURRENCY_RE.search(query):
            return "INR"
        if _CURRENCY_USD_RE.search(query):
            return "USD"
        if _CURRENCY_EUR_RE.search(query):
            return "EUR"
        if _CURRENCY_GBP_RE.search(query):
            return "GBP"
        return None
