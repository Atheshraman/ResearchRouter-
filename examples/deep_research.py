"""Deep research example.

Usage:
    python examples/deep_research.py
"""

import asyncio

from research_router.config import get_settings
from research_router.mcp_tools.tools import ResearchRouter


async def main() -> None:
    settings = get_settings()
    router = ResearchRouter(settings)

    try:
        # Deep research query
        query = "Research recent RAG hallucination mitigation techniques"
        print(f"Executing deep research for: {query}\n")

        # First, explain the plan
        plan = await router.explain(query)
        print("Plan:")
        print(f"  Engine: {plan['engine']}")
        print(f"  Depth: {plan['research_depth']}")
        print(f"  Sub-queries: {plan['sub_queries']}\n")

        # Execute
        result = await router.research(query, depth="deep")
        
        print(f"Total Unique Results: {result['total_results']}")
        print(f"Execution Time: {result['metadata'].get('execution_time_ms')}ms")
        print(f"Raw Results (before dedup): {result['metadata'].get('total_raw_results')}")
        print("\nTop 5 Results:")
        
        for i, r in enumerate(result.get("results", [])[:5], 1):
            print(f"  {i}. {r['title']}")
            if r.get('source'):
                print(f"     Source: {r['source']}")
            if r.get('url'):
                print(f"     {r['url']}")
            print()
    finally:
        await router.close()


if __name__ == "__main__":
    asyncio.run(main())
