# Architecture

## System Overview

ResearchRouter MCP is an agent-native research router that sits between MCP clients and SerpApi.

```
MCP Client  →  research(query)  →  ResearchRouter  →  SerpApi  →  Results
```

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Server                           │
│  ┌─────────┐    ┌───────────┐    ┌──────────────────┐      │
│  │  Tools   │───▶│  Router   │───▶│    Executor      │      │
│  │ research │    │           │    │                  │      │
│  │ explain  │    │ Classifier│    │ ┌──────────────┐ │      │
│  └─────────┘    │ Intent    │    │ │Engine Registry│ │      │
│                 │ Planner   │    │ └──────┬───────┘ │      │
│                 └───────────┘    │        │         │      │
│                       │          │  ┌─────┴─────┐   │      │
│                 ┌─────┴─────┐    │  │  Engines   │   │      │
│                 │    LLM    │    │  │ Google     │   │      │
│                 │ (optional)│    │  │ News       │   │      │
│                 │ Gemini    │    │  │ Scholar    │   │      │
│                 │ Ollama    │    │  │ Jobs       │   │      │
│                 └───────────┘    │  │ Shopping   │   │      │
│                                  │  │ Maps       │   │      │
│                                  │  └─────┬──────┘  │      │
│                                  │        │         │      │
│                                  │  ┌─────┴──────┐  │      │
│                                  │  │ SerpApi    │  │      │
│                                  │  │ Client     │  │      │
│                                  │  └────────────┘  │      │
│                                  │        │         │      │
│                                  │  ┌─────┴──────┐  │      │
│                                  │  │ Normalizer │  │      │
│                                  │  │ Dedup      │  │      │
│                                  │  │ Ranker     │  │      │
│                                  │  └────────────┘  │      │
│                                  └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **MCP Client** sends `research(query)`
2. **QueryClassifier** runs fast deterministic classification
3. If confidence < threshold → **LLM** extracts structured intent
4. **ResearchPlanner** converts intent into `SearchPlan`
5. **ResearchExecutor** runs plan through **EngineRegistry**
6. **Engine adapter** calls **SerpApi** and normalises results
7. Pipeline: **Normalise** → **Deduplicate** → **Rank**
8. **ResearchResponse** returned to client

## Key Design Decisions

- **Hybrid classification**: Fast regex-based classification handles ~80% of queries without LLM calls
- **Engine abstraction**: Adding a new search domain requires only a new adapter + registry entry
- **Bounded concurrency**: Deep research uses asyncio.Semaphore to limit parallel requests
- **Graceful degradation**: LLM failure falls back to deterministic; partial search failure doesn't crash the whole request
