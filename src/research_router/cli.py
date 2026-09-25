"""CLI entry-point for ResearchRouter.

Usage::

    research-router "Find AI internships in Chennai"
    research-router --debug "Latest AI news"
    research-router --depth deep "RAG hallucination mitigation"
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

    args = parser.parse_args()
    asyncio.run(_run(args))


async def _run(args: object) -> None:
    settings = get_settings()
    router = ResearchRouter(settings)

    try:
        if getattr(args, "debug", False):
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
            )
            if getattr(args, "json_output", False):
                print(json.dumps(result, indent=2, default=str))
            else:
                _print_results(result)
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

    for i, r in enumerate(data.get("results", []), 1):
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        source = r.get("source") or ""

        print(f"  {i}. {title}")
        if url:
            print(f"     {url}")
        if snippet:
            print(f"     {snippet[:200]}")
        if source:
            print(f"     Source: {source}")
        print()

    meta = data.get("metadata", {})
    if meta.get("execution_time_ms"):
        print(f"Execution time: {meta['execution_time_ms']}ms")

    errors = data.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - [{err.get('engine')}] {err.get('message')}")


if __name__ == "__main__":
    main()
