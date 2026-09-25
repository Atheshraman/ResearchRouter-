"""MCP server entry-point.

Wires the MCPServer (MCP SDK v2) with the ResearchRouter and exposes
the ``research`` and ``explain_research_plan`` tools.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from research_router.config import Settings, get_settings
from research_router.mcp_tools.tools import ResearchRouter
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

# ── global state (initialised lazily on first tool call) ──────────────

_router: ResearchRouter | None = None
_settings: Settings | None = None


def _get_router() -> ResearchRouter:
    global _router, _settings
    if _router is None:
        _settings = get_settings()
        _router = ResearchRouter(_settings)
        logger.info("ResearchRouter initialised")
    return _router


def create_mcp_server() -> MCPServer:
    """Build and return the configured MCP server."""
    mcp = MCPServer(
        "research-router",
        version="0.1.0",
        instructions=(
            "ResearchRouter MCP — One intelligent research tool.\n"
            "Use the 'research' tool to search across Google, News, Scholar, "
            "Jobs, Shopping, and Maps. The server automatically detects intent "
            "and routes to the right engine."
        ),
    )

    @mcp.tool(
        name="research",
        description=(
            "Intelligent research tool. Provide a natural-language query and "
            "the system automatically determines the search domain (news, academic, "
            "jobs, shopping, places, or general web), selects the right SerpApi "
            "engine, and returns normalised results.\n\n"
            "Examples:\n"
            "- 'Latest AI news'\n"
            "- 'Research papers about RAG hallucination'\n"
            "- 'AI internships in Chennai posted this week'\n"
            "- 'RTX laptops under ₹90000'\n"
            "- 'Best cafes near Chennai airport'\n"
        ),
    )
    async def research(
        query: str,
        max_results: int = 10,
        depth: str = "standard",
        include_sources: bool = True,
    ) -> str:
        """Execute an intelligent research query."""
        router = _get_router()
        try:
            result = await router.research(query, max_results=max_results, depth=depth)
            if not include_sources:
                result.pop("sources", None)
            return json.dumps(result, indent=2, default=str)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.error("Research failed: %s", exc, exc_info=True)
            return json.dumps({"error": f"Research failed: {type(exc).__name__}: {exc}"})

    @mcp.tool(
        name="explain_research_plan",
        description=(
            "Debug tool: shows how ResearchRouter would handle a query without "
            "executing it. Returns the detected domain, confidence, extracted "
            "entities, selected engine, and generated search plan."
        ),
    )
    async def explain_research_plan(query: str) -> str:
        """Explain how a query would be routed."""
        router = _get_router()
        try:
            explanation = await router.explain(query)
            return json.dumps(explanation, indent=2, default=str)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.error("Explain failed: %s", exc, exc_info=True)
            return json.dumps({"error": f"Explain failed: {type(exc).__name__}: {exc}"})

    return mcp


# ── __main__ support ──────────────────────────────────────────────────


async def run_server() -> None:
    """Start the MCP server on stdio transport."""
    mcp = create_mcp_server()
    logger.info("Starting ResearchRouter MCP server (stdio)")
    await mcp.run_stdio_async()
