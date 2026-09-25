"""Phase 0 — bootstrap smoke tests.

Verify that the package is importable, configuration loads correctly,
and core utilities work.
"""

from __future__ import annotations

import pytest

from research_router import __version__
from research_router.config import Settings, get_settings
from research_router.utils.dates import date_range_for, now
from research_router.utils.logging import _redact, generate_request_id, get_logger
from research_router.utils.validation import validate_query

# ── Package ────────────────────────────────────────────────────────────


class TestPackage:
    def test_version_exists(self) -> None:
        assert __version__

    def test_version_format(self) -> None:
        parts = __version__.split(".")
        assert len(parts) == 3


# ── Configuration ──────────────────────────────────────────────────────


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings(
            serpapi_api_key="test",
            google_api_key="test",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert s.llm_provider == "gemini"
        assert s.llm_model == "gemini-2.5-flash"
        assert s.max_results == 10
        assert s.max_concurrent_searches == 3
        assert s.request_timeout == 20
        assert s.cache_enabled is True
        assert s.cache_ttl == 300
        assert s.classifier_confidence_threshold == 0.75

    def test_custom_values(self) -> None:
        s = Settings(
            serpapi_api_key="sk-test",
            google_api_key="gk-test",
            llm_provider="ollama",
            max_results=25,
            cache_ttl=600,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert s.llm_provider == "ollama"
        assert s.max_results == 25
        assert s.cache_ttl == 600

    def test_max_results_bounds(self) -> None:
        with pytest.raises(Exception):
            Settings(max_results=0, _env_file=None)  # type: ignore[call-arg]

    def test_get_settings_returns_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)


# ── Logging ────────────────────────────────────────────────────────────


class TestLogging:
    def test_get_logger(self) -> None:
        log = get_logger("test_logger")
        assert log.name == "test_logger"

    def test_generate_request_id_uniqueness(self) -> None:
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_redact_secrets(self) -> None:
        data = {
            "api_key": "super-secret",
            "query": "hello",
            "nested": {"token": "abc123", "safe": "yes"},
        }
        clean = _redact(data)
        assert clean["api_key"] == "***REDACTED***"
        assert clean["query"] == "hello"
        assert clean["nested"]["token"] == "***REDACTED***"
        assert clean["nested"]["safe"] == "yes"


# ── Validation ─────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_query("")

    def test_whitespace_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_query("   ")

    def test_long_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="maximum length"):
            validate_query("x" * 2001)

    def test_valid_query_trimmed(self) -> None:
        assert validate_query("  hello world  ") == "hello world"

    def test_normal_query(self) -> None:
        q = "Find AI internships in Chennai"
        assert validate_query(q) == q


# ── Dates ──────────────────────────────────────────────────────────────


class TestDates:
    def test_now_returns_aware_datetime(self) -> None:
        dt = now()
        assert dt.tzinfo is not None

    def test_date_range_today(self) -> None:
        start, end = date_range_for("today")
        assert start == end  # same day

    def test_date_range_this_week(self) -> None:
        start, end = date_range_for("this_week")
        assert start <= end

    def test_date_range_recent(self) -> None:
        start, end = date_range_for("recent")
        assert start < end

    def test_date_range_unknown_label(self) -> None:
        # Unknown label should return a 7-day window, not crash
        start, end = date_range_for("next_century")
        assert start < end

    def test_date_range_past_month(self) -> None:
        start, end = date_range_for("past_month")
        assert start < end


# ── Server stub ────────────────────────────────────────────────────────


class TestServerStub:
    def test_create_mcp_server_does_not_crash(self) -> None:
        from research_router.server import create_mcp_server

        mcp = create_mcp_server()
        assert mcp is not None
        assert mcp.name == "research-router"
