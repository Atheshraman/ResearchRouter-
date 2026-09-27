"""Structured logging utilities.

Provides a configured logger that:
* outputs JSON-structured log lines when running in production
* outputs human-readable coloured lines during development
* never logs secrets
"""

from __future__ import annotations

import logging
import re
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


_KEY_IN_TEXT_RE = re.compile(r"\b(api_key|key|token)=[^&\s'\"]+", re.I)


class StructuredFormatter(logging.Formatter):
    """Emit log records as human-readable structured lines."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<8} {record.name} — {record.getMessage()}"
        extra: dict[str, Any] = getattr(record, "extra_data", {})
        if extra:
            safe = _redact(extra)
            pairs = " ".join(f"{k}={v}" for k, v in safe.items())
            base = f"{base}  [{pairs}]"
        if record.exc_info and record.exc_info[1] is not None:
            # One-line reason instead of a silent failure (full tracebacks are noise here).
            exc = record.exc_info[1]
            reason = str(exc).splitlines()[0][:300] if str(exc) else ""
            base = f"{base}  ({type(exc).__name__}: {reason})"
        return _KEY_IN_TEXT_RE.sub(r"\1=***", base)


# httpx logs every request URL at INFO, and SerpApi URLs carry ``api_key`` in
# the query string — keep those out of logs (and out of MCP client log files).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with structured formatting attached to *stderr*.

    Never stdout: in stdio MCP mode stdout carries the JSON-RPC stream.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        # The MCP SDK installs its own root handler; don't print every line twice.
        logger.propagate = False
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def generate_request_id() -> str:
    """Return a short unique request identifier."""
    return uuid.uuid4().hex[:12]
