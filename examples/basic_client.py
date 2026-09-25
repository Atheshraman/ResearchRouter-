"""Basic ResearchRouter client example.

Usage:
    python examples/basic_client.py
"""

import asyncio

from research_router.config import get_settings
from research_router.mcp_tools.tools import ResearchRouter


async def main() -> None:
    settings = get_settings()
    router = ResearchRouter(settings)

    try:
        # Simple general query
        result = await router.research("What is retrieval augmented generation?")
        print(f"Domain: {result['domain']}")
        print(f"Engine: {result['engine']}")
        print(f"Results: {result['total_results']}")
        for r in result.get("results", [])[:3]:
            print(f"  - {r['title']}")
            print(f"    {r.get('url', '')}")
    finally:
        await router.close()


if __name__ == "__main__":
    asyncio.run(main())
