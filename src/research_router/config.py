"""Typed, validated application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings.

    Values are loaded from environment variables and/or a ``.env`` file.
    Every field has a sensible default so the server can start with only
    the required API keys provided.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── SerpApi ────────────────────────────────────────────────────────
    serpapi_api_key: str = ""

    # ── LLM Provider ──────────────────────────────────────────────────
    llm_provider: str = Field(default="gemini", description="gemini | ollama")
    llm_model: str = Field(default="gemini-2.5-flash")
    google_api_key: str = ""

    # Ollama-specific
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Research Defaults ─────────────────────────────────────────────
    max_results: int = Field(default=10, ge=1, le=100)
    max_concurrent_searches: int = Field(default=3, ge=1, le=10)
    max_queries_per_request: int = Field(default=5, ge=1, le=20)

    # ── HTTP / SerpApi Client ─────────────────────────────────────────
    request_timeout: int = Field(default=20, ge=1, le=120)
    max_retries: int = Field(default=2, ge=0, le=10)

    # ── Cache ─────────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_ttl: int = Field(default=300, ge=0, description="TTL in seconds")

    # ── Classifier ────────────────────────────────────────────────────
    classifier_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    """Create and return a validated ``Settings`` instance."""
    return Settings()
