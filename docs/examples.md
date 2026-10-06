# Examples

All examples call the single MCP tool `research`. Arguments other than `query` are optional.

## Single-domain queries (one tool call each)

```python
research("What is retrieval augmented generation?")     # web_search       → google
research("Latest AI news")                               # news_search      → google_news
research("Recent papers about RAG hallucination")        # academic_search  → google_scholar (freshness filter)
research("AI internships in Chennai posted this week")   # jobs_search      → google_jobs (location + date)
research("RTX laptops under ₹90000 for machine learning")# shopping_search  → google_shopping (price ≤ 90000 INR)
research("Best cafes near Chennai airport")              # places_search    → google_maps
```

## Multi-part request (decomposed and run in parallel)

```python
research("Find recent RAG papers and their GitHub implementations", debug=True)
```

Internal plan (shown only because `debug=True`):

```json
{
  "intent": "multi",
  "tasks": [
    {"id": "t1", "task_id": "t1", "task": "find_papers",          "tool": "academic_search", "engine": "google_scholar",
     "query": "Find recent RAG papers", "parameters": {"date_range": "recent", "freshness": "recent"}},
    {"id": "t2", "task_id": "t2", "task": "find_implementations", "tool": "github_search",   "engine": "google",
     "query": "rag implementations site:github.com"}
  ],
  "parallelizable": true
}
```

A request is split only when the parts need different tools. A query like `"compare Python and the JVM garbage collectors"` stays a single search.

```python
research("latest AI news and the best laptops under ₹90000")
# → news_search + shopping_search, in parallel
```

## Follow-ups in a session

Pass the same `session_id` (the default is `"default"`):

```python
research("Find recent research papers about RAG hallucination", session_id="rag")
research("What about the second paper?", session_id="rag")
# → resolves to the 2nd paper's title; one targeted search; the referenced item is
#   returned in `previous_context`

research("compare them", session_id="rag")
# → 0 tool calls; answered from earlier results:
#   "note": "Answered from earlier session context and/or saved memory; no new search was run."

research("find GitHub implementations of those papers", session_id="rag")
# → one github_search per referenced paper (up to 3), in parallel

research("continue my research", session_id="rag")
# → searches the previous topic again for new angles and skips URLs already shown
```

## Persistent memory (Obsidian)

With `OBSIDIAN_VAULT_PATH` set:

```python
research("Find recent research papers about RAG hallucination", save_to_memory=True)
# metadata.memory_note → "Research/RAG/Hallucination"

# Later (even after a restart):
research("What did my previous RAG hallucination research find?")
# → no web search; returns a `memory` section:
# {
#   "label": "OLD MEMORY — recalled from saved research notes, not retrieved in this request. …",
#   "notes": ["Research/RAG/Hallucination"],
#   "evidence": [{"claim": null, "title": "A paper title", "url": "https://arxiv.org/abs/…",
#                 "source_type": "memory", "recorded_at": "2026-09-27T15:31:34+00:00"}]
# }

research("Find RAG reranking papers and compare with my previous RAG research")
# → fresh academic search (`results`) + recalled notes (`memory`), kept separate
```

If the vault is missing, the same calls still return web results plus a `warnings` entry.

## Context budget

```python
research("RAG hallucination mitigation", context_budget=1500)
# The whole response (excluding `debug`) stays within ~1500 estimated tokens.
# Compression escalates only as far as needed; the highest-relevance evidence is kept.
```

For a multi-domain response, the flattened top-level results and task sections
describe the same selected evidence:

```jsonc
{
  "domain": "multi",
  "engine": null,
  "engines": ["google_scholar", "google_jobs", "google_news"],
  "results": [
    {"task_id": "t1", "domain": "academic", "engine": "google_scholar", "claim": null},
    {"task_id": "t2", "domain": "jobs", "engine": "google_jobs", "title": "AI/ML Intern"},
    {"task_id": "t3", "domain": "news", "engine": "google_news", "freshness_score": 0.71, "ranking_score": 0.63}
  ],
  "total_results": 3,
  "tasks": [
    {"task_id": "t1", "selected_results": 1},
    {"task_id": "t2", "selected_results": 1},
    {"task_id": "t3", "selected_results": 1}
  ]
}
```

The invariants are `total_results == len(results)` and
`task.selected_results == len(task.results)`. `duplicates_removed` is reported
as `raw_results - after_dedup`; it is zero when the search response contains no
duplicates.

Valid range: 200–100000. The default is `CONTEXT_BUDGET` (4000).

## Deep research

```python
research("RAG hallucination mitigation techniques", depth="deep")
# Primary search first; the extra "techniques / best practices / recent advances /
# challenges" searches run only if it returned too few relevant results
# (ADAPTIVE_EXPANSION=true).
```

## Debug / explain

```python
explain_research_plan("Find AI internships in Chennai posted this week")
```

Returns the plan without searching:

```json
{
  "agent_mode": true,
  "execution_plan": {"intent": "jobs", "tasks": [{"tool": "jobs_search", "engine": "google_jobs"}]},
  "context": {"user_intent": "new_research", "needs_search": true},
  "memory": {"backend": "none", "available": false},
  "domain": "jobs",
  "engine": "google_jobs",
  "entities": {"location": "Chennai", "date_range": "this_week"},
  "reason": "Jobs intent detected with …% confidence"
}
```

## Measuring before/after

```bash
uv run research-router --compare "RAG hallucination mitigation techniques"
```

This prints measured averages for the legacy pipeline and the context-aware pipeline on the same query: tool calls, parallel calls, results, estimated tokens and execution time.

Live output for that query:

```text
metric                          legacy     agent
tool calls                         1.0       1.0
raw results                        9.0       9.0
results returned                   9.0       8.0
result tokens (est.)           12668.0    1423.0
whole response (est.)          12792.0    1662.0
```

In deep mode the legacy pipeline runs the query plus up to 4 angle searches every time (5 searches for multi-word queries). The agent runs the extra angles only if the first search returns too few relevant results. More live numbers are in the [README](../README.md#-measured-results-live-serpapi-run).
