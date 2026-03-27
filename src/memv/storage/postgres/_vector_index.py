"""PostgreSQL vector similarity index using pgvector."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from memv.storage.postgres._base import PgStoreBase, _parse_rowcount

if TYPE_CHECKING:
    import asyncpg


class VectorIndex(PgStoreBase):
    """Vector similarity index using pgvector with HNSW and L2 distance."""

    def __init__(self, pool: asyncpg.Pool, dimensions: int = 1536) -> None:
        super().__init__(pool)
        self.dimensions = dimensions

    async def add(self, uuid: UUID, embedding: list[float], user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO vec_knowledge (id, user_id, embedding) VALUES ($1, $2, $3)",
                str(uuid),
                user_id,
                embedding,
            )

    async def search(self, query_embedding: list[float], top_k: int = 10, user_id: str | None = None) -> list[UUID]:
        async with self._pool.acquire() as conn:
            if user_id is not None:
                rows = await conn.fetch(
                    "SELECT id FROM vec_knowledge WHERE user_id = $1 ORDER BY embedding <-> $2 LIMIT $3",
                    user_id,
                    query_embedding,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    "SELECT id FROM vec_knowledge ORDER BY embedding <-> $1 LIMIT $2",
                    query_embedding,
                    top_k,
                )
        return [UUID(row["id"]) for row in rows]

    async def search_with_scores(
        self, query_embedding: list[float], top_k: int = 10, user_id: str | None = None
    ) -> list[tuple[UUID, float]]:
        """Search and return (uuid, similarity_score) tuples. Uses L2 distance, converted to similarity: 1/(1+distance)."""
        async with self._pool.acquire() as conn:
            if user_id is not None:
                rows = await conn.fetch(
                    "SELECT id, (embedding <-> $1) AS distance FROM vec_knowledge WHERE user_id = $2 ORDER BY distance LIMIT $3",
                    query_embedding,
                    user_id,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    "SELECT id, (embedding <-> $1) AS distance FROM vec_knowledge ORDER BY distance LIMIT $2",
                    query_embedding,
                    top_k,
                )
        return [(UUID(row["id"]), 1.0 / (1.0 + row["distance"])) for row in rows]

    async def has_near_duplicate(self, embedding: list[float], user_id: str, threshold: float) -> tuple[bool, float]:
        similar = await self.search_with_scores(embedding, top_k=1, user_id=user_id)
        if not similar:
            return False, 0.0
        _, score = similar[0]
        return score >= threshold, score

    async def delete(self, uuid: UUID) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM vec_knowledge WHERE id = $1", str(uuid))
        return _parse_rowcount(status) > 0

    async def clear_user(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM vec_knowledge WHERE user_id = $1", user_id)
        return _parse_rowcount(status)

    async def _create_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS vec_knowledge (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                embedding vector({self.dimensions})
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_knowledge_user ON vec_knowledge(user_id)")
        # HNSW index — works on empty tables, no training step needed
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vec_knowledge_hnsw
            ON vec_knowledge USING hnsw (embedding vector_l2_ops)
        """)
