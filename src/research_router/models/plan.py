"""Search plan model — the executable search specification produced by the planner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from research_router.models.intent import ResearchDepth, ResearchDomain


class SearchPlan(BaseModel):
    """Concrete search plan ready for execution by an engine adapter.

    Produced by the ``ResearchPlanner`` from a ``ResearchIntent``.
    May contain ``sub_queries`` for deep research that requires multiple
    searches.
    """

    domain: ResearchDomain = Field(
        ...,
        description="Target research domain.",
    )
    engine: str = Field(
        ...,
        description="Engine key in the registry (e.g. 'google_jobs').",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="Primary search query.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific parameters (location, date filters, etc.).",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results to return.",
    )
    research_depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
    )
    sub_queries: list[SearchPlan] = Field(
        default_factory=list,
        description="Additional queries for deep research.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Inherited classification confidence.",
    )
