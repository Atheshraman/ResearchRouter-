"""LLM provider factory — creates the right provider from configuration."""

from __future__ import annotations

from research_router.config import Settings
from research_router.llm.base import LLMProvider
from research_router.llm.gemini import GeminiProvider
from research_router.llm.ollama import OllamaProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured LLM provider.

    Raises ``ValueError`` if the provider name is unknown.
    """
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.google_api_key,
            model=settings.llm_model,
        )

    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: gemini, ollama")
