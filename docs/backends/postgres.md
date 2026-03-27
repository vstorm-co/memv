# PostgreSQL

Production backend using [asyncpg](https://github.com/MagicStack/asyncpg), [pgvector](https://github.com/pgvector/pgvector) for vector search, and PostgreSQL's built-in full-text search.

## When to use

- Multi-process or multi-server deployments
- Production workloads
- When you need connection pooling
- When you already have a Postgres infrastructure

## Setup

Install with the postgres extra:

```bash
uv add memvee[postgres]
# or: pip install memvee[postgres]
```

This installs `asyncpg` and `pgvector`.

Your PostgreSQL server needs the pgvector extension. Most managed providers (Supabase, Neon, AWS RDS) include it. For self-hosted:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

memv runs this automatically on first connection, but the database role must have **superuser** privilege (or `pg_extension_owner` membership on PostgreSQL 15+). Most managed providers (Supabase, Neon) pre-install pgvector and allow non-superuser roles to enable it — no extra steps needed there.

## Usage

Pass a PostgreSQL connection URL as `db_url`:

```python
from memv import Memory
from memv.embeddings import OpenAIEmbedAdapter
from memv.llm import PydanticAIAdapter

memory = Memory(
    db_url="postgresql://user:password@localhost:5432/mydb",
    embedding_client=OpenAIEmbedAdapter(),
    llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
)

async with memory:
    await memory.add_exchange(...)
```

The backend is auto-detected from the URL prefix (`postgresql://` or `postgres://`). You can also set it explicitly:

```python
from memv import Memory, MemoryConfig

config = MemoryConfig(backend="postgres", db_url="postgresql://...")
memory = Memory(config=config, embedding_client=embedder, llm_client=llm)
```

## How it works

| Component | Implementation |
|-----------|---------------|
| MessageStore | Regular SQL tables |
| EpisodeStore | SQL + `jsonb` |
| KnowledgeStore | SQL + `jsonb`, bi-temporal columns |
| VectorIndex | `pgvector` with HNSW index, L2 distance |
| TextIndex | `tsvector` generated column + GIN index |

Unlike SQLite, no mapping tables are needed — pgvector and tsvector support `WHERE` clauses directly, so user filtering happens in the same query as the search.

## Connection pooling

memv creates a single `asyncpg.Pool` shared across all stores. The pool is created when you call `memory.open()` (or enter the `async with` block) and closed on `memory.close()`.

## Schema

Tables are created automatically on first `open()`. All tables use `CREATE TABLE IF NOT EXISTS`, so it's safe to point multiple instances at the same database.

| Table | Purpose |
|-------|---------|
| `messages` | Raw conversation messages |
| `episodes` | Grouped message episodes |
| `semantic_knowledge` | Extracted knowledge with bi-temporal fields |
| `vec_knowledge` | Vector embeddings (pgvector) |
| `fts_knowledge` | Full-text search index (tsvector) |

## Docker quickstart

```bash
docker run -d \
  --name memv-postgres \
  -e POSTGRES_USER=memv \
  -e POSTGRES_PASSWORD=memv \
  -e POSTGRES_DB=memv \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

Then:

```python
memory = Memory(
    db_url="postgresql://memv:memv@localhost:5432/memv",
    embedding_client=embedder,
    llm_client=llm,
)
```
