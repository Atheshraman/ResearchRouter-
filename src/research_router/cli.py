"""CLI entry-point for ResearchRouter.

Usage::

    research-router "Find AI internships in Chennai"
    research-router --debug "Latest AI news"
    research-router --depth deep "RAG hallucination mitigation"
    research-router --trace --budget 3000 "Find recent RAG papers and their GitHub implementations"
    research-router --compare "RAG hallucination mitigation"
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from research_router.config import get_settings
from research_router.mcp_tools.tools import ResearchRouter


def main() -> None:
    """CLI entry-point (synchronous wrapper)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="research-router",
        description="ResearchRouter — Intelligent research from the command line.",
    )
    parser.add_argument("query", help="Research query")
    parser.add_argument("--debug", action="store_true", help="Show routing plan instead of results")
    parser.add_argument(
        "--depth",
        default="standard",
        choices=["quick", "standard", "deep"],
        help="Research depth (default: standard)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output raw JSON",
    )
    parser.add_argument(
        "--budget", type=int, default=None, help="Output context budget in estimated tokens"
    )
    parser.add_argument("--trace", action="store_true", help="Execute and print the debug report")
    parser.add_argument("--save", action="store_true", help="Save findings to the Obsidian vault")
    parser.add_argument("--no-memory", action="store_true", help="Do not consult saved memory")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the legacy and context-aware pipelines on the query and compare metrics",
    )

    args = parser.parse_args()
    asyncio.run(_run(args))


async def _run(args: object) -> None:
    settings = get_settings()
    router = ResearchRouter(settings)

    try:
        if getattr(args, "compare", False):
            await _compare(router, args)
        elif getattr(args, "debug", False):
            result = await router.explain(args.query)  # type: ignore[attr-defined]
            if getattr(args, "json_output", False):
                print(json.dumps(result, indent=2, default=str))
            else:
                _print_explain(result)
        else:
            result = await router.research(
                args.query,  # type: ignore[attr-defined]
                max_results=getattr(args, "max_results", 10),
                depth=getattr(args, "depth", "standard"),
                context_budget=getattr(args, "budget", None),
                debug=getattr(args, "trace", False),
                use_memory=not getattr(args, "no_memory", False),
                save_to_memory=True if getattr(args, "save", False) else None,
            )
            if getattr(args, "json_output", False):
                print(json.dumps(result, indent=2, default=str))
            else:
                _print_results(result)
                if "debug" in result:
                    print("\n" + result["debug"]["report"])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        await router.close()


def _print_explain(data: dict[str, Any]) -> None:
    """Pretty-print a research plan explanation."""
    print(f"Domain:      {data.get('domain', 'unknown')}")
    print(f"Confidence:  {data.get('confidence', 0):.0%}")
    print(f"Engine:      {data.get('engine', 'unknown')}")
    print(f"Query:       {data.get('generated_query', '')}")
    print(f"Depth:       {data.get('research_depth', 'standard')}")
    print(f"Sub-queries: {data.get('sub_queries', 0)}")
    print(f"Reason:      {data.get('reason', '')}")

    entities = data.get("entities", {})
    non_null = {k: v for k, v in entities.items() if v is not None}
    if non_null:
        print(f"Entities:    {json.dumps(non_null)}")

    params = data.get("parameters", {})
    if params:
        print(f"Parameters:  {json.dumps(params, default=str)}")


def _print_results(data: dict[str, Any]) -> None:
    """Pretty-print research results."""
    print(f"Domain:  {data.get('domain', 'unknown')}")
    print(f"Engine:  {data.get('engine', 'unknown')}")
    print(f"Results: {data.get('total_results', 0)}")
    print()

    if data.get("note"):
        print(f"Note: {data['note']}\n")

    for i, r in enumerate(data.get("results", []), 1):
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        snippet = r.get("snippet") or r.get("claim") or ""
        source = r.get("source") or ""

        print(f"  {i}. {title}")
        if url:
            print(f"     {url}")
        if snippet:
            print(f"     {snippet[:200]}")
        if source:
            print(f"     Source: {source}")
        print()

    for key in ("previous_context", "memory"):
        section = data.get(key)
        if section:
            print(f"[{section['label']}]")
            for e in section["evidence"]:
                when = e.get("recorded_at") or e.get("date") or ""
                print(f"  - {e.get('claim', '')[:160]}  {e.get('url') or ''} {when}".rstrip())
            print()

    meta = data.get("metadata", {})
    if meta.get("execution_time_ms"):
        print(f"Execution time: {meta['execution_time_ms']}ms")

    errors = data.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - [{err.get('engine')}] {err.get('message')}")

    for warning in data.get("warnings", []):
        print(f"Warning: {warning}")


_COMPARE_ROWS = (
    ("requests", "requests"),
    ("tool calls", "avg_tool_calls"),
    ("parallel tool calls", "avg_parallel_tool_calls"),
    ("raw results", "avg_raw_results"),
    ("deduplicated", "avg_deduplicated_results"),
    ("relevant results", "avg_relevant_results"),
    ("returned results", "avg_final_evidence"),
    ("input tokens (est.)", "avg_input_tokens_estimated"),
    ("context tokens (est.)", "avg_output_context_tokens"),
    ("response tokens (est.)", "avg_response_tokens"),
    ("execution time (ms)", "avg_execution_time_ms"),
)


async def _compare(router: ResearchRouter, args: object) -> None:
    """Run both pipelines on the same query and print the measured metrics.

    The legacy run goes first and the agent's cache is separate from the
    legacy result cache, so the agent does not benefit from the legacy run.
    """
    query = args.query  # type: ignore[attr-defined]
    for mode in ("legacy", "agent"):
        await router.research(
            query,
            max_results=getattr(args, "max_results", 10),
            depth=getattr(args, "depth", "standard"),
            context_budget=getattr(args, "budget", None),
            use_memory=not getattr(args, "no_memory", False),
            mode=mode,
        )
    summary = router.metrics_summary()
    if getattr(args, "json_output", False):
        print(json.dumps(summary, indent=2))
        return
    legacy, agent = summary.get("legacy", {}), summary.get("agent", {})
    print(f"Query: {query}\n")
    print(f"{'metric':<24}{'legacy':>12}{'agent':>12}")
    for label, key in _COMPARE_ROWS:
        print(f"{label:<24}{legacy.get(key, 0):>12}{agent.get(key, 0):>12}")


if __name__ == "__main__":
    main()
