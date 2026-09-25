"""Abstract base class for all search engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult


class SearchEngine(ABC):
    """Contract that every engine adapter must implement.

    Each adapter is responsible for:
    1. Converting a ``SearchPlan`` into SerpApi parameters.
    2. Calling SerpApi through the shared client.
    3. Normalising raw results into ``ResearchResult`` instances.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """SerpApi ``engine`` parameter value (e.g. ``'google'``)."""

    @abstractmethod
    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        """Execute a search and return normalised results."""

    @abstractmethod
    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        """Build the SerpApi query-string parameters from a plan."""
