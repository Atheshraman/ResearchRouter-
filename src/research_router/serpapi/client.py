"""Centralised, async SerpApi HTTP client.

Responsibilities:
* API authentication (via query-string ``api_key``)
* Async HTTP with httpx
* Configurable timeout
* Retry with exponential back-off
* Rate-limit detection (HTTP 429)
* Error classification
* Response validation
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from research_router.serpapi.exceptions import (
    SerpApiAuthError,
    SerpApiError,
    SerpApiRateLimitError,
    SerpApiResponseError,
    SerpApiTimeoutError,
)
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://serpapi.com/search"


class SerpApiClient:
    """Async SerpApi client with retry and error handling.

    Parameters
    ----------
    api_key:
        SerpApi API key.  **Never** logged or exposed.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Maximum number of retries on transient errors.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 20,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise SerpApiAuthError("SERPAPI_API_KEY is required but was empty.")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    # ── lifecycle ──────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── public API ─────────────────────────────────────────────────

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a SerpApi search request.

        Parameters
        ----------
        params:
            Query parameters **excluding** ``api_key`` (injected automatically).
            Must include at least ``engine`` and ``q`` (or engine-specific key).

        Returns
        -------
        dict
            Parsed JSON response from SerpApi.

        Raises
        ------
        SerpApiAuthError
            Invalid or missing API key (HTTP 401 / 403).
        SerpApiRateLimitError
            Rate limit exceeded (HTTP 429).
        SerpApiTimeoutError
            Request timed out.
        SerpApiResponseError
            Non-JSON or unexpected response body.
        SerpApiError
            Any other HTTP error.
        """
        full_params: dict[str, Any] = {
            **params,
            "api_key": self._api_key,
            "output": "json",
        }

        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            try:
                return await self._do_request(full_params)
            except SerpApiRateLimitError:
                raise  # never retry 429
            except SerpApiAuthError:
                raise  # never retry auth errors
            except (SerpApiTimeoutError, SerpApiError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = 2**attempt  # 1 s, 2 s, …
                    logger.warning(
                        "SerpApi request failed (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        1 + self._max_retries,
                        wait,
                        extra={"extra_data": {"error": str(exc)}},
                    )
                    await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ── internals ──────────────────────────────────────────────────

    async def _do_request(self, params: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.get(_BASE_URL, params=params)
        except httpx.TimeoutException as exc:
            raise SerpApiTimeoutError(f"Request timed out after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise SerpApiError(f"HTTP error: {exc}") from exc

        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: httpx.Response) -> dict[str, Any]:
        status = response.status_code

        if status in (401, 403):
            raise SerpApiAuthError(
                "Authentication failed — check SERPAPI_API_KEY.",
                status_code=status,
            )
        if status == 429:
            raise SerpApiRateLimitError(
                "Rate limit exceeded. Slow down.",
                status_code=429,
            )
        if status >= 500:
            raise SerpApiError(
                f"SerpApi server error (HTTP {status}).",
                status_code=status,
            )
        if status >= 400:
            raise SerpApiError(
                f"SerpApi client error (HTTP {status}): {response.text[:300]}",
                status_code=status,
            )

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:
            raise SerpApiResponseError(f"Failed to parse JSON response: {exc}") from exc

        # SerpApi sometimes returns errors inside the JSON body
        if "error" in data:
            raise SerpApiResponseError(f"SerpApi error: {data['error']}")

        return data
