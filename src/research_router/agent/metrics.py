"""Per-request metrics and a process-wide aggregator.

All numbers are measured during execution — nothing here is estimated
except token counts, which come from ``context.tokens.estimate_tokens``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from research_router.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetrics:
    mode: str = "agent"
    tool_calls: int = 0
    parallel_tool_calls: int = 0  # widest concurrent wave of tool calls
    cache_hits: int = 0
    expansion_calls: int = 0
    fallback_calls: int = 0
    raw_results: int = 0
    deduplicated_results: int = 0
    relevant_results: int = 0
    final_evidence: int = 0
    input_tokens_estimated: int = 0  # raw tool payload (+ recalled memory/context candidates)
    output_context_tokens: int = 0  # evidence actually returned to the LLM
    response_tokens: int = 0  # whole response envelope incl. metadata
    memory_hits: int = 0
    memory_misses: int = 0
    memory_notes_used: list[str] = field(default_factory=list)
    memory_saved: str | None = None
    errors: int = 0
    execution_time_ms: int = 0
    sum_task_time_ms: int = 0
    parallel_wall_time_ms: int = 0
    stage_ms: dict[str, int] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    @property
    def compression_ratio(self) -> float:
        if self.input_tokens_estimated <= 0:
            return 0.0
        return round(1 - self.output_context_tokens / self.input_tokens_estimated, 4)

    @property
    def remaining_ratio(self) -> float:
        if self.input_tokens_estimated <= 0:
            return 0.0
        return round(self.output_context_tokens / self.input_tokens_estimated, 4)

    @property
    def parallelism_gain(self) -> float:
        if self.parallel_wall_time_ms <= 0:
            return 0.0
        return round(self.sum_task_time_ms / self.parallel_wall_time_ms, 2)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stage_ms[name] = self.stage_ms.get(name, 0) + int((time.perf_counter() - t) * 1000)

    def finish(self) -> None:
        self.execution_time_ms = int((time.perf_counter() - self._t0) * 1000)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_t0", None)
        data["compression_ratio"] = self.compression_ratio
        data["reduction_ratio"] = self.compression_ratio
        data["remaining_ratio"] = self.remaining_ratio
        data["reduction_percent"] = round(self.compression_ratio * 100, 2)
        data["remaining_percent"] = round(self.remaining_ratio * 100, 2)
        data["parallelism_gain"] = self.parallelism_gain
        return data


_SUMMED = (
    "tool_calls",
    "parallel_tool_calls",
    "cache_hits",
    "raw_results",
    "deduplicated_results",
    "relevant_results",
    "final_evidence",
    "input_tokens_estimated",
    "output_context_tokens",
    "response_tokens",
    "memory_hits",
    "memory_misses",
    "errors",
    "execution_time_ms",
)


class MetricsAggregator:
    """Running totals per mode ("agent" / "legacy") for before/after comparison."""

    def __init__(self) -> None:
        self._totals: dict[str, dict[str, float]] = {}
        self._counts: dict[str, int] = {}

    def record(self, m: RequestMetrics) -> None:
        totals = self._totals.setdefault(m.mode, dict.fromkeys(_SUMMED, 0.0))
        for key in _SUMMED:
            totals[key] += float(getattr(m, key))
        self._counts[m.mode] = self._counts.get(m.mode, 0) + 1

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for mode, totals in self._totals.items():
            n = self._counts[mode]
            out[mode] = {"requests": n, **{f"avg_{k}": round(v / n, 2) for k, v in totals.items()}}
        return out


def log_research_summary(query: str, tools: list[str], m: RequestMetrics) -> None:
    """One INFO line per research call, so token savings are visible in client logs."""
    logger.info(
        "Research complete",
        extra={
            "extra_data": {
                "mode": m.mode,
                "query": query[:80],
                "tools": ",".join(tools) or "none",
                "tool_calls": m.tool_calls,
                "cache_hits": m.cache_hits,
                "results": f"{m.raw_results}->{m.final_evidence}",
                "tokens": f"{m.input_tokens_estimated}->{m.output_context_tokens}",
                "response_tokens": m.response_tokens,
                "reduction": f"{m.compression_ratio:.1%}",
                "time_ms": m.execution_time_ms,
            }
        },
    )


def render_debug_report(
    *,
    query: str,
    intent: str,
    plan: dict[str, Any],
    metrics: RequestMetrics,
    memory_status: dict[str, Any],
) -> str:
    """Human-readable debug block (mirrors the metrics dict)."""
    tasks = plan.get("tasks", [])
    tools = sorted({t["tool"] for t in tasks})
    parallel = metrics.parallel_tool_calls
    lines = [
        "ResearchRouter Debug",
        "",
        f"Query: {query}",
        f"Intent: {intent}",
        "",
        "Tools selected:" if tools else "Tools selected: none (answered from context/memory)",
        *[f"- {t}" for t in tools],
        "",
        f"Tool calls: {metrics.tool_calls}"
        + (f" (+{metrics.cache_hits} cache hits)" if metrics.cache_hits else ""),
        f"Parallel execution: {'yes' if parallel > 1 else 'no'}"
        + (f" ({parallel} concurrent)" if parallel > 1 else ""),
        "",
        f"Raw results: {metrics.raw_results}",
        f"After deduplication: {metrics.deduplicated_results}",
        f"Relevant: {metrics.relevant_results}",
        f"Final evidence blocks: {metrics.final_evidence}",
        "",
        f"Estimated raw context: {metrics.input_tokens_estimated:,} tokens",
        f"Final context: {metrics.output_context_tokens:,} tokens"
        f" (full response {metrics.response_tokens:,})",
        f"Reduction: {metrics.compression_ratio:.1%}",
        "",
        f"Execution time: {metrics.execution_time_ms / 1000:.2f}s",
    ]
    if memory_status.get("available"):
        lines.append(
            f"Obsidian memory used: {len(metrics.memory_notes_used)} notes"
            + (f" (saved: {metrics.memory_saved})" if metrics.memory_saved else "")
        )
    else:
        reason = memory_status.get("reason") or "not configured"
        lines.append(f"Obsidian memory: unavailable ({reason})")
    return "\n".join(lines)
