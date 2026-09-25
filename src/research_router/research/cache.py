"""In-memory TTL cache for search results."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from research_router.utils.logging import get_logger

logger = get_logger(__name__)


class CacheEntry:
    """A single cache entry with expiration."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: int) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl


class TTLCache:
    """Simple in-memory TTL cache.

    Designed so the interface can later be implemented by a Redis backend.
    """

    def __init__(self, ttl: int = 300, enabled: bool = True) -> None:
        self._ttl = ttl
        self._enabled = enabled
        self._store: dict[str, CacheEntry] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _make_key(engine: str, query: str, params: dict[str, Any]) -> str:
        """Build a deterministic cache key."""
        raw = json.dumps({"engine": engine, "query": query, "params": params}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, engine: str, query: str, params: dict[str, Any]) -> Any | None:
        """Return cached value or ``None`` on miss / disabled / expired."""
        if not self._enabled:
            return None

        key = self._make_key(engine, query, params)
        entry = self._store.get(key)
        if entry is None:
            return None

        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None

        logger.info("Cache hit", extra={"extra_data": {"engine": engine}})
        return entry.value

    def set(self, engine: str, query: str, params: dict[str, Any], value: Any) -> None:
        """Store a value in the cache."""
        if not self._enabled:
            return

        key = self._make_key(engine, query, params)
        self._store[key] = CacheEntry(value, self._ttl)

    def clear(self) -> None:
        """Evict all entries."""
        self._store.clear()

    def size(self) -> int:
        """Number of (possibly expired) entries."""
        return len(self._store)
