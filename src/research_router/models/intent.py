"""Research intent model — structured representation of a user's research need."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResearchDomain(StrEnum):
    """Supported research domains."""

    GENERAL = "general"
    NEWS = "news"
    ACADEMIC = "academic"
    JOBS = "jobs"
    SHOPPING = "shopping"
    PLACES = "places"


class ResearchDepth(StrEnum):
    """How thoroughly to research a query."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ResearchIntent(BaseModel):
    """Structured representation of a user's research intent.

    Built by the classifier (deterministic) or the LLM analyzer for
    complex queries.  Consumed by the ``ResearchPlanner`` to produce a
    ``SearchPlan``.
    """

    domain: ResearchDomain = Field(
        default=ResearchDomain.GENERAL,
        description="Detected research domain.",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="Core search query (cleaned / refined).",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords.",
    )
    location: str | None = Field(
        default=None,
        description="Geographic location (e.g. 'Chennai').",
    )
    date_range: str | None = Field(
        default=None,
        description="Human label such as 'this_week' or 'past_month'.",
    )
    freshness: str | None = Field(
        default=None,
        description="Freshness requirement (e.g. 'recent').",
    )
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code (e.g. 'INR', 'USD').",
    )
    job_type: str | None = Field(
        default=None,
        description="full_time | part_time | contract | internship",
    )
    remote: bool | None = Field(
        default=None,
        description="True = remote, False = on-site, None = unspecified.",
    )
    experience_level: str | None = Field(
        default=None,
        description="entry | mid | senior",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional domain-specific constraints.",
    )
    research_depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
        description="Desired research depth.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Classifier confidence in domain assignment.",
    )
    requires_multiple_searches: bool = Field(
        default=False,
        description="Whether the planner should decompose into sub-queries.",
    )
