# Architecture

## System Overview

ResearchRouter is an MCP server that sits between an LLM and SerpApi. It exposes one tool, `research(query)`. Everything else (intent analysis, planning, tool selection, parallel execution, memory, context reduction and budget enforcement) happens inside the server.

```
LLM  →  research(query)  →  ResearchRouter  →  SerpApi (+ optional Obsidian vault)
     ←  compact, cited, budgeted context  ←
```

## Pipeline

```
research(query, context_budget?, session_id?, use_memory?, save_to_memory?, debug?)
   │
   ├─ ContextManager        resolve references against this session's history
   │                        ("those papers", "the second approach", "compare them")
   ├─ TaskPlanner           intent per task → decomposition → minimum tool set
   │                        (context-only follow-ups produce zero tasks)
   ├─ TaskRunner ∥ Memory   independent tasks run concurrently; memory recall
   │                        runs alongside them
   │     └─ wave 2          adaptive expansion / fallback, only when needed
   ├─ ContextReducer        normalise → dedup → rank → filter → extract
   │                        → adaptive compression (per section)
   ├─ Budget enforcement    web, earlier-session and memory sections share one
   │                        estimated-token budget
   ├─ Response              labelled sections, source-attributed evidence, metrics
   └─ Record / Save         store a compact turn for follow-ups; optionally
                            write an Obsidian note
```

## Components

```
┌──────────────────────────────────────────────────────────────────────┐
│ MCP server (server.py)            tools: research, explain_research_plan
├──────────────────────────────────────────────────────────────────────┤
│ ResearchRouter (mcp_tools/tools.py)   AGENT_MODE switch: agent | legacy
├───────────────────────────────┬──────────────────────────────────────┤
│ Agent layer (new)             │ Core (original, reused)              │
│  agent/orchestrator.py        │  router/classifier.py   regex intent │
│  agent/task_planner.py        │  router/intent.py       + LLM fallback│
│  agent/runner.py              │  router/planner.py      SearchPlan   │
│  agent/metrics.py             │  research/executor.py   semaphore    │
│  context/manager.py           │  research/cache.py      TTL cache    │
│  context/reducer.py           │  engines/*              6 adapters   │
│  context/evidence.py          │  serpapi/client.py      retries      │
│  context/tokens.py            │  llm/*                  Gemini/Ollama│
│  memory/base.py, obsidian.py  │                                      │
└───────────────────────────────┴──────────────────────────────────────┘
```

| Module | Responsibility |
|--------|----------------|
| `context/manager.py` | Per-session turn history (LRU-bounded, compact evidence only) and reference resolution. Produces a `ResolvedContext` (`current_task`, `previous_task`, `entities`, `previous_results`, `user_intent`, `required_context`, `needs_search`). Only the relevant slice of history is used. |
| `agent/task_planner.py` | Builds the internal `ExecutionPlan` (`intent`, `tasks[{id, task, tool, engine, query, depends_on}]`, `parallelizable`). Splits a query only when the parts need different tools, the later part has no subject of its own ("…and their implementations"), or the user explicitly chains requests. Identical searches are merged. |
| `agent/runner.py` | Wave-based execution. Independent tasks run concurrently (bounded by `MAX_CONCURRENT_SEARCHES`), and dependent tasks run in later waves. Each call has a timeout (`TASK_TIMEOUT`) and uses the cache. Deep-research angle searches run only if the primary search returned too few relevant results. A failed or empty specialist engine is retried once on web search. |
| `context/reducer.py` | Deterministic reduction: clean HTML and boilerplate → dedup by normalised URL, title and near-duplicate text (3-gram Jaccard) → BM25-style relevance with a freshness bonus → filter (keeping a minimum per task) → extract the relevant or numeric sentences → compress step by step only while over budget (shorter claims, compact fields, then drop the lowest-value items from the most-represented task). |
| `context/evidence.py` | `Evidence{claim, source{title, url, date, publisher}, relevance_score, source_type, recorded_at}`. Provenance survives every compression step. |
| `context/tokens.py` | Token estimation: `tiktoken` if installed, otherwise a conservative character/word heuristic. It never raises. |
| `memory/obsidian.py` | Optional vault read/write. Scans only `<vault>/Research/`, bounded by file count and bytes per file, and returns the best passages. Writes are atomic, and paths are sanitised and confined to the research folder. |
| `agent/metrics.py` | Per-request metrics, the debug report, and per-mode averages used by `--compare`. |

## Data Flow Example

`"Find recent RAG papers and their GitHub implementations"`, then `"compare them"`:

1. **Turn 1.** No earlier references to resolve → `needs_search = true`.
2. The planner splits at "and their": `t1` "Find recent RAG papers" → classifier *academic* → `google_scholar` (freshness filter). `t2` "RAG GitHub implementations" → code intent → `google` with `site:github.com`.
3. The runner calls both engines concurrently; `parallel_tool_calls = 2`.
4. The reducer merges the results, removes duplicates, ranks them against the query, and fits them into the budget.
5. The turn is recorded (compact evidence only) and optionally saved to Obsidian.
6. **Turn 2.** "compare them" contains a reference and no new topic → `needs_search = false`. The earlier results come back in `previous_context`, with 0 tool calls.

## Source Separation

| `source_type` | Where it appears | Meaning |
|---|---|---|
| `web` | `results` | Retrieved during this request |
| `context` | `previous_context.evidence` | Returned by an earlier turn in this session |
| `memory` | `memory.evidence` (with `recorded_at` and the note path) | Recalled from saved notes; old information |

Old information is never presented as newly retrieved.

## Context Budget

- The budget is an **output** budget in estimated tokens. The server cannot see the LLM's remaining context window and does not claim to.
- Allocation: a reserve for the response envelope, then memory (≤ 25%), then earlier-session context (≤ 30% of the remainder). Fresh web evidence gets everything left over, including any unused share.
- A final guard trims the lowest-ranked evidence until the whole response (excluding `debug`) fits.

## Failure Handling

| Failure | Behaviour |
|---------|-----------|
| One engine errors | Recorded in `errors`; other tasks continue; a specialist engine falls back to web search once (not for auth or rate-limit errors) |
| Timeout | Per-call `Timeout` error; the request still returns |
| SerpApi auth error | `SerpApiAuthError`, including SerpApi's own reason (never the key) |
| Malformed results | Skipped during normalisation/reduction |
| Token estimation failure | Falls back to the heuristic, then to a length bound |
| Reducer failure | Truncation fallback in rank order (`fallback: true` in stats) |
| LLM unavailable/fails | Deterministic classifier result is used |
| Obsidian not configured / missing / unreadable / write fails | Research continues; a warning is added to the response |

## Key Design Decisions

- **One tool, internal intelligence.** Internal modules are not exposed as separate MCP tools.
- **Wrap, don't replace.** The original router, planner, executor, engines and cache are reused unchanged, and `AGENT_MODE=false` restores the original pipeline.
- **Hybrid classification.** The fast regex classifier handles most queries; the LLM is consulted only below the confidence threshold.
- **Deterministic reduction.** No LLM calls in the reduction path, so it's fast, cheap and testable.
- **Memory is an enhancement.** It is never a single point of failure.
- **Measured, not claimed.** Every metric is recorded during execution; `--compare` measures both pipelines on the same query.
- **stdio safety.** Logs go to stderr, and HTTP request logging (which would include the API key in URLs) is suppressed.
