# Adding a New Engine

This guide shows how to add a new SerpApi engine to ResearchRouter without modifying the core system.

## Steps

### 1. Create the Engine Adapter

Create a new file in `src/research_router/engines/`, e.g. `youtube.py`:

```python
"""YouTube engine adapter."""
from __future__ import annotations
from typing import Any

from research_router.engines.base import SearchEngine
from research_router.models.plan import SearchPlan
from research_router.models.result import ResearchResult
from research_router.serpapi.client import SerpApiClient


class YouTubeEngine(SearchEngine):
    def __init__(self, client: SerpApiClient) -> None:
        self._client = client

    @property
    def engine_name(self) -> str:
        return "youtube"

    def build_params(self, plan: SearchPlan) -> dict[str, Any]:
        return {
            "engine": "youtube",
            "search_query": plan.query,
        }

    async def search(self, plan: SearchPlan) -> list[ResearchResult]:
        params = self.build_params(plan)
        raw = await self._client.search(params)
        return [
            ResearchResult(
                title=item.get("title"),
                url=item.get("link"),
                snippet=item.get("description"),
                source="YouTube",
                metadata={
                    "channel": item.get("channel", {}).get("name"),
                    "views": item.get("views"),
                    "length": item.get("length"),
                },
            )
            for item in raw.get("video_results", [])
        ]
```

### 2. Register the Engine

In `src/research_router/mcp_tools/tools.py`, add to the `ResearchRouter.__init__`:

```python
from research_router.engines.youtube import YouTubeEngine

self._registry.register("youtube", YouTubeEngine(self._serpapi))
```

### 3. Add Domain + Classifier Signals

In `src/research_router/models/intent.py`, add to `ResearchDomain`:

```python
YOUTUBE = "youtube"
```

In `src/research_router/router/classifier.py`, add signals:

```python
ResearchDomain.YOUTUBE: [
    (re.compile(r"\b(youtube|video|watch|tutorial)\b", re.I), 0.40),
    (re.compile(r"\b(channel|subscribe|stream)\b", re.I), 0.25),
],
```

### 4. Map Domain → Engine

In `src/research_router/router/planner.py`, add to `_DOMAIN_ENGINE_MAP`:

```python
ResearchDomain.YOUTUBE: "youtube",
```

### 5. Name the Tool for the Agent Layer

The context-aware planner reports tools by logical name (in debug output, metrics and `tools_used`). In `src/research_router/agent/task_planner.py`, add:

```python
TOOL_NAMES["youtube"] = "video_search"                   # in the TOOL_NAMES dict
_TASK_NAMES[ResearchDomain.YOUTUBE] = "find_videos"      # in the _TASK_NAMES dict
```

Without these entries the engine still works; it just appears under its raw engine key.

### 6. Write Tests

Create `tests/unit/test_youtube_engine.py` and test with mock SerpApi responses.

### 7. Add Test Fixture

Create `tests/fixtures/youtube.json` with a sample SerpApi response.

## That's It

No changes needed to:
- `server.py`
- `executor.py`
- MCP tool definitions
- The agent pipeline (context manager, runner, reducer, memory)

The new engine is automatically available through the same `research()` tool. The planner can also select it for one part of a multi-part request, and it gets the same timeout, cache, fallback, deduplication and budget handling as the built-in engines.
