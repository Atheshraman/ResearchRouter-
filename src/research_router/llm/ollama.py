"""Ollama LLM provider (local models)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from research_router.llm.base import LLMProvider
from research_router.llm.gemini import _SYSTEM_PROMPT
from research_router.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance.

    Uses Ollama's ``/api/chat`` endpoint with JSON mode.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60)
        return self._client

    async def analyse_intent(self, query: str) -> dict[str, Any]:
        """Send the query to Ollama and parse the JSON response."""
        client = await self._get_client()

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"User query: {query}"},
            ],
            "format": "json",
            "stream": False,
        }

        response = await client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()

        body = response.json()
        text = body.get("message", {}).get("content", "")

        logger.info(
            "Ollama response received",
            extra={"extra_data": {"model": self._model, "response_length": len(text)}},
        )

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Ollama returned invalid JSON: %s", text[:200])
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc

        return data

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
