"""Typed, validated application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field, field_validator
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

    # ── Context-aware agent ───────────────────────────────────────────
    agent_mode: bool = Field(
        default=True, description="False restores the original single-plan pipeline."
    )
    context_budget: int = Field(
        default=4000,
        ge=200,
        le=100_000,
        description="Default output context budget (estimated tokens) per research call.",
    )
    task_timeout: float = Field(
        default=50.0,
        gt=0,
        le=300,
        description=(
            "Per tool-call timeout including SerpApi retries. The default leaves room for "
            "one retry (2 x REQUEST_TIMEOUT + backoff) while staying under the ~60 s tool "
            "timeout MCP clients commonly use."
        ),
    )
    adaptive_expansion: bool = Field(
        default=True, description="Run deep-research angle searches only when needed."
    )
    session_max_turns: int = Field(default=10, ge=1, le=100)
    max_tasks_per_request: int = Field(default=4, ge=1, le=10)

    # ── Obsidian memory (optional) ────────────────────────────────────
    obsidian_vault_path: str = Field(default="", description="Empty disables persistent memory.")
    obsidian_research_folder: str = Field(default="Research")
    obsidian_auto_save: bool = Field(default=False)
    memory_max_notes: int = Field(default=3, ge=1, le=20)

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @field_validator("serpapi_api_key", "google_api_key", "obsidian_vault_path", mode="before")
    @classmethod
    def _clean_secret_or_path(cls, value: object) -> object:
        """Trim pasted whitespace; treat unfilled ``${user_config.*}`` placeholders as unset.

        MCP bundle hosts substitute user settings into env vars; an optional
        setting the user left blank can arrive as the literal placeholder.
        """
        if not isinstance(value, str):
            return value
        value = value.strip().strip('"').strip("'").strip()
        return "" if value.startswith("${") and value.endswith("}") else value


def get_settings() -> Settings:
    """Create and return a validated ``Settings`` instance."""
    return Settings()
