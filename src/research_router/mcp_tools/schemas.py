"""Pydantic schemas for MCP tool inputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchInput(BaseModel):
    """Input schema for the ``research`` MCP tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language research query.",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return.",
    )
    depth: str = Field(
        default="standard",
        description="Research depth: 'quick', 'standard', or 'deep'.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source attribution.",
    )


class ExplainInput(BaseModel):
    """Input schema for the ``explain_research_plan`` MCP tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Query to explain the research plan for.",
    )
