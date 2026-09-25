# Configuration

All configuration is done through environment variables or a `.env` file.

## Required

| Variable | Description |
|----------|-------------|
| `SERPAPI_API_KEY` | Your SerpApi API key |
| `GOOGLE_API_KEY` | Google AI API key (for Gemini LLM) |

## Optional

| Variable | Default | Description |
|----------|---------|-------------|
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
| `LOG_LEVEL` | `INFO` | Logging level |
