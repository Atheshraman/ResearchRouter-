"""Token estimation.

The MCP server cannot know the exact context window (or remaining space) of
the calling LLM, so budgets here are an *output* budget: an upper bound on
how many tokens of research context this server returns.

Estimation uses ``tiktoken`` when it happens to be installed and otherwise a
character/word heuristic that slightly over-estimates for English text,
which is the safe direction for a budget.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from research_router.utils.logging import get_logger

logger = get_logger(__name__)


def _heuristic(text: str) -> int:
    # ~4 chars/token for English prose; JSON punctuation and URLs tokenise
    # worse, so take the larger of the char- and word-based estimates.
    return max(math.ceil(len(text) / 3.8), math.ceil(len(text.split()) * 1.35))


def _load_encoder() -> Callable[[str], int] | None:
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:
        return None


_ENCODER = _load_encoder()


def estimate_tokens(value: Any) -> int:
    """Estimate the token count of a string or JSON-serialisable value.

    Never raises: falls back to the heuristic, then to a length-based bound.
    """
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if _ENCODER is not None:
        try:
            return _ENCODER(text)
        except Exception:
            logger.debug("tiktoken estimation failed; using heuristic")
    try:
        return _heuristic(text)
    except Exception:
        return len(text) // 3 + 1
