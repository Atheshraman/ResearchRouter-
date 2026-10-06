# Configuration

All configuration is done through environment variables or a `.env` file. When the server is installed as a Claude Desktop extension (`.mcpb`), the extension settings screen fills in `SERPAPI_API_KEY`, `GOOGLE_API_KEY` and `OBSIDIAN_VAULT_PATH`.

Surrounding whitespace and quotes are removed from keys and paths, and an optional extension setting left blank (an unfilled `${user_config.…}` placeholder) is treated as unset.

| Where you run it | Where settings come from |
|---|---|
| CLI (`uv run research-router …`) | `.env` in the project folder (save the file!), or shell environment variables |
| Manual MCP config (`uv run --directory <project> …`) | The config's `env` block, then `.env` in that folder |
| Claude Desktop extension (`.mcpb`) | The extension's settings screen only; `.env` is not read and is never packed into the bundle |

Environment variables take precedence over `.env`. At startup the server logs a warning if `SERPAPI_API_KEY` doesn't look like a SerpApi key. The warning shows only the key's length and last 4 characters.

## Required

| Variable | Description |
|----------|-------------|
| `SERPAPI_API_KEY` | Your SerpApi API key ([manage-api-key](https://serpapi.com/manage-api-key)) |

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(empty)* | Optional Gemini key. Only used when the built-in classifier is unsure; without it, the classifier result is used |
| `LLM_PROVIDER` | `gemini` | LLM provider (`gemini` or `ollama`) |
| `LLM_MODEL` | `gemini-2.5-flash` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `MAX_RESULTS` | `10` | Default max results per search |
| `MAX_CONCURRENT_SEARCHES` | `3` | Max parallel SerpApi requests |
| `MAX_QUERIES_PER_REQUEST` | `5` | Max sub-queries for deep research |
| `REQUEST_TIMEOUT` | `20` | HTTP timeout in seconds |
| `MAX_RETRIES` | `2` | Max retry attempts |
| `CACHE_ENABLED` | `true` | Enable in-memory result cache |
| `CACHE_TTL` | `300` | Cache TTL in seconds |
| `CLASSIFIER_CONFIDENCE_THRESHOLD` | `0.75` | Min confidence for deterministic classification |
| `LOG_LEVEL` | `INFO` | Logging level. Logs go to **stderr** (stdout is the MCP channel); HTTP request URLs, which contain the API key, are never logged |

## Context-aware agent

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODE` | `true` | `false` restores the original single-plan pipeline and response shape |
| `CONTEXT_BUDGET` | `4000` | Default output budget (estimated tokens) for one `research` response; callers can override per call with `context_budget` (200–100000) |
| `TASK_TIMEOUT` | `50` | Timeout (seconds) for each tool call, including SerpApi retries. Keep it above `2 × REQUEST_TIMEOUT + 1` so a retry can finish, and below your MCP client's tool timeout (often ~60 s) |
| `ADAPTIVE_EXPANSION` | `true` | Deep research runs its extra angle searches only when the primary search returns too few relevant results |
| `SESSION_MAX_TURNS` | `10` | Turns remembered per `session_id` for follow-up questions (in-process only) |
| `MAX_TASKS_PER_REQUEST` | `4` | Max independent sub-tasks a query is decomposed into |

The budget is an **output** budget. The server cannot see the calling LLM's
context window, so it guarantees only that its own response stays within the
estimate. Tokens are counted with `tiktoken` (`cl100k_base`, a close
approximation of Claude's tokenizer; installed as a dependency). If its
encoding file can't be downloaded on first use (e.g. offline), a
conservative character/word heuristic is used instead.

## Response diagnostics

Pass `debug=true` to `research` when investigating routing or evidence quality.
The response then includes the execution plan, per-task status and timing,
reduction statistics, and measured parallelism. Multi-domain responses retain
both the flattened `results` view and task-specific `tasks` sections. The
following invariants hold after budget enforcement:

```text
total_results == len(results)
task.selected_results == len(task.results)
duplicates_removed == raw_results - after_dedup
```

News queries with recency intent (`latest`, `recent`, `today`, `current`,
`breaking`, and similar terms) expose `freshness_score` and `ranking_score` in
evidence. Academic evidence keeps the original Scholar `snippet` and
publication metadata when available; incomplete snippets produce `claim: null`.

For local MCP Inspector validation:

```powershell
npx --yes @modelcontextprotocol/inspector --cli `
	--cwd E:\SerpAPI\ResearchRouter- `
	--method tools/list --format json `
	E:\SerpAPI\ResearchRouter-\.venv\Scripts\python.exe -m research_router
```

## Obsidian memory (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | *(empty)* | Absolute path to a vault. Empty disables persistent memory |
| `OBSIDIAN_RESEARCH_FOLDER` | `Research` | Folder inside the vault that is read and written; nothing else in the vault is scanned |
| `OBSIDIAN_AUTO_SAVE` | `false` | Save every research result; otherwise pass `save_to_memory=true` per call |
| `MEMORY_MAX_NOTES` | `3` | Max notes recalled per query (only their best-matching passages are returned) |

Notes are written to `Research/<Topic>/<Subtopic>.md` (for example
`Research/RAG/Hallucination.md`), each with YAML frontmatter, one dated
`## Session …` section per save, cited findings, sources, tags and
`[[links]]` to related notes. If the vault is missing or unreadable, research
continues without memory and the response includes a warning.
