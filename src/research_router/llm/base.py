"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Contract for all LLM providers.

    Each implementation must convert a natural-language query into
    structured JSON representing a ``ResearchIntent``.
    """

    @abstractmethod
    async def analyse_intent(self, query: str) -> dict[str, Any]:
        """Return a structured intent dictionary from a natural-language query.

        The returned dict must be compatible with
        ``ResearchIntent.model_validate(result)``.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the provider."""
