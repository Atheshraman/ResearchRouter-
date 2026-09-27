# ResearchRouter MCP

> A context-aware, memory-aware research agent for LLMs, exposed as **one** MCP tool: `research(query)`.

**Built for the SerpApi India Hackathon 2026 (Open-Source Integrations Track)**

---

## 🎯 The Problem

When an LLM agent researches something it has to decide which search engine to use (Google, News, Scholar, Jobs, Shopping or Maps), how many searches to run, and how much of the raw output to put in its context window. Exposing every SerpApi engine as a separate MCP tool pushes those decisions onto the LLM. It wastes tool calls, repeats searches, and floods the context with duplicated snippets and boilerplate.

## 🚀 The Solution

ResearchRouter keeps the interface to **one tool** and does the work internally:

> **Retrieve less, retrieve smarter, remember what matters, run only the necessary tools, and return the smallest useful context.**

| Capability | What it does |
|---|---|
| **Intelligent tool selection** | Works out the intent, splits multi-part requests into tasks, and calls only the engines each task needs |
| **Parallel execution** | Runs independent tasks at the same time; dependent steps (fallbacks, extra searches) run after them |
| **Context-aware follow-ups** | Resolves "those papers", "the second approach", "compare them" and "continue my research" against the session |
| **Context reducer** | Normalises, deduplicates, ranks and filters results, extracts the key sentences, and compresses only as much as needed |
| **Token budget** | Keeps each response within a configurable estimated-token budget (`context_budget`) |
| **Source-aware evidence** | Every claim keeps its title, URL, date and publisher, even after compression |
| **Persistent memory (optional)** | Saves findings to an Obsidian vault and recalls relevant notes later, labelled as old information |
| **Observability** | Debug report, measured per-request metrics, and a side-by-side comparison with the original pipeline |

Example of what happens inside a single call:

```text
"Find recent RAG papers and their GitHub implementations"
  → task t1  find_papers           academic_search (google_scholar)  ┐ run in
  → task t2  find_implementations  github_search   (google)          ┘ parallel
  → merge → dedup → rank → compress to budget → cited evidence
```

## 🧠 Architecture

```mermaid
graph TD
  A[MCP client / LLM] -->|research query| B[Context Manager]
  B -->|resolved query| C[Task Planner]
  C -->|classifier + optional LLM| D[Intent per task]
  D --> E[Task Runner]
  B -.->|memory query| M[(Obsidian memory<br/>optional)]
  E -->|parallel, cached, timed out| F[Engine Registry]
  F --> G1[Google] & G2[News] & G3[Scholar] & G4[Jobs] & G5[Shopping] & G6[Maps]
  G1 & G2 & G3 & G4 & G5 & G6 --> S[SerpApi]
  S --> R[Context Reducer]
  M --> R
  B -.->|earlier-session results| R
  R -->|dedup → rank → extract → compress| T[Token Budget]
  T --> O[Compact, cited response]
  O --> A
  O -.->|optional save| M
```

The original classifier, planner, executor, engines and cache are all still there; the new layer wraps them. Setting `AGENT_MODE=false` switches back to the original single-plan pipeline. See [`docs/architecture.md`](docs/architecture.md).

## 🛠️ Supported Domains

| Domain | Tool / SerpApi engine | Example query |
|--------|----------------------|---------------|
| General | `web_search` → `google` | "What is retrieval augmented generation?" |
| News | `news_search` → `google_news` | "Latest AI news" |
| Academic | `academic_search` → `google_scholar` | "Recent papers about RAG hallucination" |
| Code | `github_search` → `google` + `site:github.com` | "…and their GitHub implementations" |
| Jobs | `jobs_search` → `google_jobs` | "AI internships in Chennai posted this week" |
| Shopping | `shopping_search` → `google_shopping` | "RTX laptops under ₹90,000" |
| Places | `places_search` → `google_maps` | "Best cafes near Chennai airport" |

## 📋 Requirements

| Item | Required | Notes |
|------|----------|-------|
| Python ≥ 3.11 | Yes | |
| [uv](https://docs.astral.sh/uv/) (or pip) | Yes | uv is recommended; `uv.lock` is included |
| SerpApi API key | Yes | [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) |
| Google Gemini API key | No | Only used to classify ambiguous queries; without it, built-in rules are used |
| Obsidian vault | No | Any folder works; the Obsidian app doesn't need to be running |
| `tiktoken` | No | If installed, token counts are exact instead of estimated |

## 💻 Installation

```bash
git clone https://github.com/your-org/research-router-mcp.git
cd research-router-mcp

uv sync --extra dev          # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env         # then add SERPAPI_API_KEY
```

Minimal `.env`:

```bash
SERPAPI_API_KEY=your_serpapi_key
# GOOGLE_API_KEY=            # optional
# OBSIDIAN_VAULT_PATH=/Users/you/Documents/MyVault   # optional memory
# CONTEXT_BUDGET=4000        # default response budget (estimated tokens)
```

All options are listed in [`docs/configuration.md`](docs/configuration.md).

## 🔌 Using it from Claude

### Option A: install as a Claude Desktop extension (`.mcpb`)

```bash
npx @anthropic-ai/mcpb pack . research-router.mcpb
```

Double-click `research-router.mcpb` (or use Claude Desktop → Settings → Extensions → Install). The settings screen asks for your **SerpApi API key**. The **Gemini key** and **Obsidian vault** are optional.

### Option B: manual MCP config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or your client's MCP config:

```json
{
  "mcpServers": {
    "research-router": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/research-router-mcp",
               "python", "-m", "research_router"],
      "env": {
        "SERPAPI_API_KEY": "your_serpapi_key",
        "OBSIDIAN_VAULT_PATH": "/absolute/path/to/vault"
      }
    }
  }
}
```

Restart the client. You'll see two tools: `research` and `explain_research_plan`.

## 🧭 The `research` tool

```python
research(
    query,                  # natural-language request (required)
    max_results=10,         # per task
    depth="standard",       # "quick" | "standard" | "deep"
    include_sources=True,
    context_budget=None,    # estimated-token budget for the response (default 4000)
    session_id="default",   # keep the same id so follow-ups resolve
    use_memory=True,        # consult the Obsidian vault if configured
    save_to_memory=None,    # True saves findings (default: OBSIDIAN_AUTO_SAVE)
    debug=False,            # include plan, reduction stats and a readable report
)
```

Every argument except `query` is optional, so existing callers keep working.

### A research conversation

```text
research("Find recent RAG papers and their GitHub implementations")
  → 2 tool calls, in parallel (academic_search + github_search)

research("What about the second paper?")
  → resolves to that paper's title, 1 targeted search

research("compare them")
  → 0 tool calls, answered from earlier results in this session

research("Compare this with my previous RAG research")
  → recalls matching Obsidian notes, labelled as OLD MEMORY
```

### Response format

The top-level keys are the same as the original response (`query`, `domain`, `engine`, `results`, `total_results`, `sources`, `errors`, `metadata`). Fresh, older and remembered information are kept in separate sections:

```jsonc
{
  "query": "Find recent RAG papers and their GitHub implementations",
  "domain": "multi",
  "engine": "google_scholar",
  "results": [                        // FRESH evidence from this request only
    {
      "claim": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "url": "https://arxiv.org/abs/2005.11401",
      "source": "P Lewis, E Perez - NeurIPS, 2020",
      "relevance": 0.1,
      "source_type": "web",
      "task": "t1"
    }
  ],
  "previous_context": {               // only for follow-ups
    "label": "EARLIER IN THIS SESSION — results returned by a previous research turn.",
    "evidence": [ /* source_type: "context" */ ]
  },
  "memory": {                         // only when notes match
    "label": "OLD MEMORY — recalled from saved research notes, not retrieved in this request. …",
    "notes": ["Research/RAG/Hallucination"],
    "evidence": [ /* source_type: "memory", with recorded_at */ ]
  },
  "warnings": [],                     // e.g. memory unavailable; research still completes
  "errors": [],                       // per-tool failures, e.g. a timeout on one engine
  "metadata": {
    "tools_used": ["academic_search", "github_search"],
    "metrics": {
      "tool_calls": 2, "parallel_tool_calls": 2, "cache_hits": 0,
      "raw_results": 4, "final_evidence": 4,
      "input_tokens_estimated": 323, "output_context_tokens": 292,
      "response_tokens": 481, "compression_ratio": 0.096,
      "memory_hits": 0, "execution_time_ms": 5
    }
  }
}
```

*(Values in these examples come from the test fixtures; real queries return more results and larger reductions.)*

### Debug report (`debug=True` or `--trace`)

```text
ResearchRouter Debug

Query: Find recent RAG papers and their GitHub implementations
Intent: multi

Tools selected:
- academic_search
- github_search

Tool calls: 2
Parallel execution: yes (2 concurrent)

Raw results: 4
After deduplication: 4
Relevant: 4
Final evidence blocks: 4

Estimated raw context: 323 tokens
Final context: 292 tokens (full response 481)
Reduction: 9.6%

Execution time: 0.01s
Obsidian memory used: 0 notes (saved: Research/RAG/Overview)
```

With `debug=True` the response also includes the internal execution plan, the resolved context, and reduction statistics for each section. Otherwise the plan stays internal.

## 🗂️ Persistent research memory (Obsidian)

Set `OBSIDIAN_VAULT_PATH` to turn it on. Each saved research task creates or updates one note:

```text
<vault>/Research/
└── RAG/
    ├── Hallucination.md
    ├── Reranking.md
    └── Evaluation.md
```

Each note has YAML frontmatter (`created`, `updated`, `tags`, `queries`) and one `## Session <timestamp>` section per save. Each section contains a summary, key findings with `[title](url) (date)` citations, entities, sources, `[[links]]` to related notes, and tags.

- **Retrieval** reads only the `Research/` folder and returns the few best-matching *passages*, never whole notes or the whole vault.
- Recalled notes always appear in the separate `memory` section with their recorded date, never mixed into fresh `results`.
- **Failure-safe:** if the vault is missing or unreadable, research continues and the response includes a warning.

## 🖥️ CLI

```bash
# Research (context-aware pipeline)
uv run research-router "Find AI internships in Chennai"

# Show the routing/execution plan without searching (no SerpApi credits used)
uv run research-router --debug "Find recent RAG papers and their GitHub implementations"

# Execute and print the debug report, with a smaller budget
uv run research-router --trace --budget 3000 "RAG hallucination mitigation"

# Deep research (extra searches only if the first one is insufficient)
uv run research-router --depth deep "RAG hallucination mitigation techniques"

# Save findings to the vault / ignore memory
uv run research-router --save "RAG hallucination papers"
uv run research-router --no-memory "RAG hallucination papers"

# Measure the original pipeline against the new one on the same query
uv run research-router --compare "RAG hallucination mitigation techniques"
```

`--compare` runs both pipelines and prints **measured** averages (tool calls, parallel calls, raw/deduplicated/returned results, estimated input/context/response tokens, and execution time). No numbers are made up. Run it on your own queries to demonstrate the difference.

## 🛟 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Authentication failed — check SERPAPI_API_KEY (SerpApi says: …)` | The key reached SerpApi and was rejected; the message in brackets is SerpApi's reason. Check the key with `curl "https://serpapi.com/account.json?api_key=YOUR_KEY"` (free, uses no searches) and re-enter it in the extension settings. |
| `SERPAPI_API_KEY is required but was empty` | No key was passed to the server; set it in `.env`, the MCP `env` block, or the extension settings. |
| Follow-ups like "compare them" don't resolve | Use the same `session_id` for every call. History lives in the server process and resets when it restarts. |
| `Persistent memory requested but unavailable` | `OBSIDIAN_VAULT_PATH` isn't set or doesn't exist. Research still works without it. |
| Logs | Server logs go to stderr (in Claude Desktop: `~/Library/Logs/Claude/mcp-server-*.log`). API keys are never logged. |

## 🧪 Testing

```bash
uv run pytest -q          # 196 tests, fully mocked (no API key needed)
uv run mypy src/research_router
```

The tests cover reference resolution, task decomposition, parallel execution timing, tool failure and timeout handling, cache hits, budget enforcement, deduplication, the Obsidian round-trip and failure modes, and compatibility with the original response format.

## 📖 Documentation

- [Architecture](docs/architecture.md): pipeline, modules, failure handling
- [Configuration](docs/configuration.md): all environment variables
- [Usage examples](docs/examples.md): single, multi-task, follow-up and memory flows
- [Adding a new engine](docs/adding-engine.md)

## 🐳 Docker

```bash
docker build -t research-router-mcp .
docker run -i --env-file .env research-router-mcp
```

To use memory in Docker, mount your vault and point `OBSIDIAN_VAULT_PATH` at the mount path.
