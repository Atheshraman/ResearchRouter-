"""Engine registry — central look-up for engine adapters."""

from __future__ import annotations

from research_router.engines.base import SearchEngine
from research_router.utils.logging import get_logger

logger = get_logger(__name__)


class EngineRegistry:
    """Thread-safe registry mapping engine keys to adapter instances.

    Usage::

        registry = EngineRegistry()
        registry.register("google", GoogleEngine(client))
        engine = registry.get("google")
    """

    def __init__(self) -> None:
        self._engines: dict[str, SearchEngine] = {}

    def register(self, key: str, engine: SearchEngine) -> None:
        """Register an engine adapter under *key*."""
        self._engines[key] = engine
        logger.info("Registered engine: %s", key)

    def get(self, key: str) -> SearchEngine:
        """Look up an engine by key.

        Raises ``KeyError`` when the engine is not registered.
        """
        if key not in self._engines:
            available = ", ".join(sorted(self._engines)) or "(none)"
            raise KeyError(f"Engine '{key}' is not registered. Available: {available}")
        return self._engines[key]

    def list_engines(self) -> list[str]:
        """Return all registered engine keys."""
        return sorted(self._engines)

    def has(self, key: str) -> bool:
        """Check whether an engine is registered."""
        return key in self._engines
