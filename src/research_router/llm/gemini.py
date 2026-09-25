"""Gemini LLM provider using the google-genai SDK."""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from research_router.llm.base import LLMProvider
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a research intent classifier. Given a user's natural-language
research query, extract a structured JSON object with the following fields.
Return ONLY valid JSON — no markdown fences, no explanation.

Fields:
- domain: one of "general", "news", "academic", "jobs", "shopping", "places"
- query: the cleaned core search query
- keywords: list of extracted keywords
- location: geographic location if mentioned, else null
- date_range: one of "today", "this_week", "past_week", "past_month",
              "past_year", "recent", or null
- freshness: "recent" if the user wants fresh results, else null
- price_min: number or null
- price_max: number or null
- currency: ISO 4217 code (e.g. "INR", "USD") or null
- job_type: "full_time", "part_time", "contract", "internship" or null
- remote: true / false / null
- experience_level: "entry", "mid", "senior" or null
- research_depth: "quick", "standard", or "deep"
- confidence: your confidence in the domain classification (0.0–1.0)
- requires_multiple_searches: true if the query should be decomposed
"""


class GeminiProvider(LLMProvider):
    """LLM provider backed by Google Gemini (via google-genai SDK).

    Uses ``response_mime_type="application/json"`` so Gemini returns
    structured JSON directly.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GeminiProvider.")
        self._model = model
        self._client = genai.Client(api_key=api_key)

    async def analyse_intent(self, query: str) -> dict[str, Any]:
        """Send the query to Gemini and parse the structured response."""
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=f"User query: {query}",
            config=types.GenerateContentConfig(
                system_instruction=[_SYSTEM_PROMPT],
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )

        text = response.text or ""
        logger.info(
            "Gemini response received",
            extra={"extra_data": {"model": self._model, "response_length": len(text)}},
        )

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Gemini returned invalid JSON: %s", text[:200])
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

        return data

    async def close(self) -> None:
        """No persistent connections to close for google-genai SDK."""
