"""PostgreSQL full-text search index using tsvector/tsquery."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from memv.storage.postgres._base import PgStoreBase, _parse_rowcount

if TYPE_CHECKING:
    import asyncpg


class TextIndex(PgStoreBase):
    """Full-text search index using PostgreSQL tsvector/tsquery with GIN index."""

    async def add(self, uuid: UUID, text: str, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO fts_knowledge (id, user_id, content) VALUES ($1, $2, $3)",
                str(uuid),
                user_id,
                text,
            )

    async def search(self, query: str, top_k: int = 10, user_id: str | None = None) -> list[UUID]:
        if not query.strip():
            return []
        async with self._pool.acquire() as conn:
            if user_id is not None:
                rows = await conn.fetch(
                    """SELECT id FROM fts_knowledge
                    WHERE user_id = $1 AND content_tsvector @@ plainto_tsquery('english', $2)
                    ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', $2)) DESC
                    LIMIT $3""",
                    user_id,
                    query,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id FROM fts_knowledge
                    WHERE content_tsvector @@ plainto_tsquery('english', $1)
                    ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', $1)) DESC
                    LIMIT $2""",
                    query,
                    top_k,
                )
        return [UUID(row["id"]) for row in rows]

    async def delete(self, uuid: UUID) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM fts_knowledge WHERE id = $1", str(uuid))
        return _parse_rowcount(status) > 0

    async def clear_user(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM fts_knowledge WHERE user_id = $1", user_id)
        return _parse_rowcount(status)

    async def _create_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_knowledge (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_fts_knowledge_user ON fts_knowledge(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_fts_knowledge_gin ON fts_knowledge USING gin(content_tsvector)")
