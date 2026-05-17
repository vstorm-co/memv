# MCP Server

memv ships an [MCP](https://modelcontextprotocol.io) server that exposes its memory operations as tools any MCP-compatible client (Claude Desktop, Claude Code, Cursor, custom agents) can call.

## Install

```bash
uv add "memvee[mcp]"
# or
pip install "memvee[mcp]"
```

This pulls in the `mcp` package alongside memv. Combine with other extras as needed, e.g. `memvee[mcp,postgres]`.

## Run

```bash
memv-mcp --db-url memory.db --llm-model openai:gpt-4o-mini
```

By default the server speaks `stdio` — the transport every desktop MCP client expects.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--db-url` | *required* | SQLite path or `postgresql://...` URL. |
| `--user-id` | `default` | Default `user_id` applied to every tool call when the caller doesn't pass one. |
| `--embedding-provider` | `openai` | `openai`, `voyage`, `cohere`, or `local` (FastEmbed). |
| `--embedding-model` | provider default | Override the embedding model. |
| `--embedding-dimensions` | provider default | Override vector dimensions. Must match the model. |
| `--llm-model` | *none* | PydanticAI model string (e.g. `openai:gpt-4o-mini`). Without this, knowledge extraction is disabled. |
| `--transport` | `stdio` | `stdio` or `streamable-http`. |

!!! note "LLM is optional"
    Without `--llm-model`, `add_conversation` stores messages but does not extract knowledge. `search_memory` and `add_memory` still work — they don't need an LLM.

!!! warning "add_conversation latency"
    With an LLM configured, `add_conversation` runs segmentation and predict-calibrate extraction inline before returning. This can take 10–30+ seconds on long histories. Raise your MCP client's tool-call timeout accordingly (Claude Desktop defaults to ~60 s).

## Tools

| Tool | Purpose |
|------|---------|
| `search_memory(query, user_id?, top_k=10)` | Hybrid retrieval (vector + BM25 + RRF). Returns an LLM-ready prompt block. |
| `add_memory(statement, user_id?)` | Store a fact directly. Deduplicates against existing knowledge. |
| `add_conversation(user_message, assistant_message, user_id?)` | Append an exchange. Triggers extraction when an LLM is configured. |
| `list_memories(user_id?, limit=20, offset=0)` | Page through stored knowledge. |
| `delete_memory(knowledge_id)` | Permanently remove an entry by UUID. |

All `user_id` arguments are optional — the server falls back to the `--user-id` default when omitted.

## Client setup

=== "Claude Desktop"

    Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent path on your platform:

    ```json
    {
      "mcpServers": {
        "memv": {
          "command": "memv-mcp",
          "args": [
            "--db-url", "/absolute/path/to/memory.db",
            "--user-id", "your-name",
            "--llm-model", "openai:gpt-4o-mini"
          ],
          "env": {
            "OPENAI_API_KEY": "sk-..."
          }
        }
      }
    }
    ```

=== "Claude Code"

    ```bash
    claude mcp add memv -- memv-mcp \
      --db-url /absolute/path/to/memory.db \
      --user-id your-name \
      --llm-model openai:gpt-4o-mini
    ```

=== "Cursor"

    In `~/.cursor/mcp.json`:

    ```json
    {
      "mcpServers": {
        "memv": {
          "command": "memv-mcp",
          "args": ["--db-url", "/absolute/path/to/memory.db", "--llm-model", "openai:gpt-4o-mini"]
        }
      }
    }
    ```

## HTTP transport

For remote agents, run with `--transport streamable-http`:

```bash
memv-mcp --db-url memory.db --llm-model openai:gpt-4o-mini --transport streamable-http
```

The server listens on the default MCP HTTP port. Put it behind your own auth/proxy before exposing it.

## Programmatic use

The server factory is importable, so you can mount it inside an existing process or inject custom clients (e.g. for tests):

```python
from memv.mcp.server import create_server

server = create_server(
    db_url="memory.db",
    default_user_id="alice",
    embedding_client=my_embedder,
    llm_client=my_llm,
)
server.run(transport="stdio")
```

The tool implementations are exported as plain `do_*` coroutines (`do_search_memory`, `do_add_memory`, …) so you can unit-test them without an MCP runtime.
