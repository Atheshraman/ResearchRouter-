"""Result models — normalised research results and the top-level response."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from research_router.models.intent import ResearchDomain


class ResearchResult(BaseModel):
    """A single normalised search result.

    Every engine adapter must map its raw response into this shape.
    Engine-specific extras are preserved in ``metadata``.
    """

    title: str | None = Field(default=None, description="Result title.")
    url: str | None = Field(default=None, description="Canonical URL.")
    snippet: str | None = Field(default=None, description="Short description / snippet.")
    source: str | None = Field(default=None, description="Source name (e.g. 'LinkedIn').")
    published_at: datetime | None = Field(
        default=None,
        description="Publication timestamp (timezone-aware when available).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific extra fields.",
    )


class SearchError(BaseModel):
    """Structured error from a single search execution."""

    engine: str = Field(..., description="Engine that failed.")
    query: str = Field(default="", description="Query that caused the error.")
    error_type: str = Field(..., description="Error class name.")
    message: str = Field(..., description="Human-readable error message.")


class ResearchResponse(BaseModel):
    """Top-level response returned by the ``research`` MCP tool.

    Contains normalised results, source attribution, and any errors
    from partial multi-query failures.
    """

    query: str = Field(..., description="Original user query.")
    domain: ResearchDomain = Field(
        default=ResearchDomain.GENERAL,
        description="Detected domain.",
    )
    engine: str = Field(default="google", description="Primary engine used.")
    results: list[ResearchResult] = Field(
        default_factory=list,
        description="Normalised results.",
    )
    total_results: int = Field(
        default=0,
        ge=0,
        description="Total number of results returned.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Unique source domains.",
    )
    errors: list[SearchError] = Field(
        default_factory=list,
        description="Errors from failed sub-queries.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (timing, request_id, etc.).",
    )
