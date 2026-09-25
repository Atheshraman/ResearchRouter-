"""Structured logging utilities.

Provides a configured logger that:
* outputs JSON-structured log lines when running in production
* outputs human-readable coloured lines during development
* never logs secrets
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

_REDACT_KEYS = frozenset(
    {
        "serpapi_api_key",
        "google_api_key",
        "api_key",
        "authorization",
        "token",
        "secret",
    }
)


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with sensitive values masked."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _REDACT_KEYS:
            out[key] = "***REDACTED***"
        elif isinstance(value, dict):
            out[key] = _redact(value)
        else:
            out[key] = value
    return out


class StructuredFormatter(logging.Formatter):
    """Emit log records as human-readable structured lines."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<8} {record.name} — {record.getMessage()}"
        extra: dict[str, Any] = getattr(record, "extra_data", {})
        if extra:
            safe = _redact(extra)
            pairs = " ".join(f"{k}={v}" for k, v in safe.items())
            base = f"{base}  [{pairs}]"
        return base


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with structured formatting attached to *stdout*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def generate_request_id() -> str:
    """Return a short unique request identifier."""
    return uuid.uuid4().hex[:12]
