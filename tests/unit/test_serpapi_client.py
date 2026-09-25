"""Phase 2 — SerpApi client tests (fully mocked, no real API key needed)."""

from __future__ import annotations

import httpx
import pytest
import respx

from research_router.serpapi.client import _BASE_URL, SerpApiClient
from research_router.serpapi.exceptions import (
    SerpApiAuthError,
    SerpApiError,
    SerpApiRateLimitError,
    SerpApiResponseError,
    SerpApiTimeoutError,
)

# ── helpers ────────────────────────────────────────────────────────────

FAKE_KEY = "test-api-key-000"


def _client(**kwargs: object) -> SerpApiClient:
    return SerpApiClient(api_key=FAKE_KEY, max_retries=0, **kwargs)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_empty_key_rejected(self) -> None:
        with pytest.raises(SerpApiAuthError, match="required"):
            SerpApiClient(api_key="")

    def test_valid_construction(self) -> None:
        c = SerpApiClient(api_key="sk-test")
        assert c._timeout == 20  # default
        assert c._max_retries == 2


# ═══════════════════════════════════════════════════════════════════════
# Successful search
# ═══════════════════════════════════════════════════════════════════════


class TestSuccessfulSearch:
    @respx.mock
    async def test_basic_search(self) -> None:
        route = respx.get(_BASE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "search_metadata": {"id": "abc"},
                    "organic_results": [{"title": "Hello"}],
                },
            )
        )
        client = _client()
        result = await client.search({"engine": "google", "q": "test"})
        assert "organic_results" in result
        assert route.called
        await client.close()

    @respx.mock
    async def test_api_key_injected(self) -> None:
        """The api_key param must appear in the request."""

        def _check(request: httpx.Request) -> httpx.Response:
            assert f"api_key={FAKE_KEY}" in str(request.url)
            return httpx.Response(200, json={"ok": True})

        respx.get(_BASE_URL).mock(side_effect=_check)
        client = _client()
        await client.search({"engine": "google", "q": "test"})
        await client.close()


# ═══════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    @respx.mock
    async def test_auth_error_401(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(401, text="Unauthorized"))
        client = _client()
        with pytest.raises(SerpApiAuthError):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_auth_error_403(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(403, text="Forbidden"))
        client = _client()
        with pytest.raises(SerpApiAuthError):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_rate_limit_429(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(429, text="Rate limited"))
        client = _client()
        with pytest.raises(SerpApiRateLimitError):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_server_error_500(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        client = _client()
        with pytest.raises(SerpApiError, match="server error"):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_malformed_json(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, text="not json at all"))
        client = _client()
        with pytest.raises(SerpApiResponseError, match="parse JSON"):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_error_in_json_body(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json={"error": "Invalid query"}))
        client = _client()
        with pytest.raises(SerpApiResponseError, match="Invalid query"):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_timeout(self) -> None:
        respx.get(_BASE_URL).mock(side_effect=httpx.ReadTimeout("timeout"))
        client = _client()
        with pytest.raises(SerpApiTimeoutError):
            await client.search({"engine": "google", "q": "test"})
        await client.close()

    @respx.mock
    async def test_generic_http_error(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(418, text="I'm a teapot"))
        client = _client()
        with pytest.raises(SerpApiError, match="client error"):
            await client.search({"engine": "google", "q": "test"})
        await client.close()


# ═══════════════════════════════════════════════════════════════════════
# Retry logic
# ═══════════════════════════════════════════════════════════════════════


class TestRetry:
    @respx.mock
    async def test_retry_on_server_error(self) -> None:
        """Should retry once then succeed."""
        call_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(500, text="error")
            return httpx.Response(200, json={"ok": True})

        respx.get(_BASE_URL).mock(side_effect=_handler)
        client = SerpApiClient(api_key=FAKE_KEY, max_retries=1)
        result = await client.search({"engine": "google", "q": "test"})
        assert result == {"ok": True}
        assert call_count == 2
        await client.close()

    @respx.mock
    async def test_no_retry_on_rate_limit(self) -> None:
        """429 should NOT be retried."""
        call_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, text="rate limited")

        respx.get(_BASE_URL).mock(side_effect=_handler)
        client = SerpApiClient(api_key=FAKE_KEY, max_retries=3)
        with pytest.raises(SerpApiRateLimitError):
            await client.search({"engine": "google", "q": "test"})
        assert call_count == 1  # no retries
        await client.close()

    @respx.mock
    async def test_no_retry_on_auth_error(self) -> None:
        call_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(401, text="unauthorized")

        respx.get(_BASE_URL).mock(side_effect=_handler)
        client = SerpApiClient(api_key=FAKE_KEY, max_retries=3)
        with pytest.raises(SerpApiAuthError):
            await client.search({"engine": "google", "q": "test"})
        assert call_count == 1
        await client.close()


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════


class TestLifecycle:
    async def test_close_idempotent(self) -> None:
        client = _client()
        await client.close()
        await client.close()  # should not raise
