"""SerpApi-specific exceptions."""

from __future__ import annotations


class SerpApiError(Exception):
    """Base exception for all SerpApi errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class SerpApiAuthError(SerpApiError):
    """Invalid or missing API key."""


class SerpApiRateLimitError(SerpApiError):
    """Rate limit exceeded (HTTP 429)."""


class SerpApiTimeoutError(SerpApiError):
    """Request timed out."""


class SerpApiResponseError(SerpApiError):
    """Malformed or unexpected response body."""
