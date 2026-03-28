# SQLite

The default backend. Single-file storage using [sqlite-vec](https://github.com/asg017/sqlite-vec) for vectors and [FTS5](https://www.sqlite.org/fts5.html) for text search. No configuration required.

## When to use

- Local development
- Single-process applications
- Prototyping and testing
- Embedded or edge deployments

## Setup

No extra dependencies — SQLite is included by default.

```bash
uv add memvee
# or: pip install memvee
```

```python
from memv import Memory
from memv.embeddings import OpenAIEmbedAdapter
from memv.llm import PydanticAIAdapter

memory = Memory(
    db_url="memory.db",
    embedding_client=OpenAIEmbedAdapter(),
    llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
)
```

Parent directories are created if they don't exist. `db_url=".db/my_app/memory.db"` works.

## How it works

| Component | Implementation |
|-----------|---------------|
| MessageStore | Regular SQL tables |
| EpisodeStore | SQL + JSON (TEXT column) |
| KnowledgeStore | SQL + JSON, bi-temporal columns |
| VectorIndex | sqlite-vec virtual table + mapping table for user filtering |
| TextIndex | FTS5 virtual table + mapping table for user filtering |

sqlite-vec and FTS5 use virtual tables that don't support `WHERE` clauses directly. memv uses a mapping table pattern to enable per-user filtering — a separate table maps UUIDs to rowids and user IDs, which memv joins at query time.

## Limitations

- **Single-process only** — SQLite uses file-level locking. Concurrent writes from multiple processes will fail.
- **L2 distance only** — sqlite-vec supports Euclidean distance, not cosine similarity.
- **No concurrent connections** — one connection per store instance.

## File location

All tables live in a single `.db` file. To reset, delete it.

```bash
rm memory.db  # fresh start
```
