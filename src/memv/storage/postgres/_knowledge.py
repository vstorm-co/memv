"""PostgreSQL semantic knowledge storage."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from memv.models import SemanticKnowledge
from memv.storage.postgres._base import PgStoreBase, _parse_rowcount

if TYPE_CHECKING:
    import asyncpg


class KnowledgeStore(PgStoreBase):
    """PostgreSQL store for semantic knowledge."""

    async def add(self, knowledge: SemanticKnowledge) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO semantic_knowledge
                (id, user_id, statement, source_episode_id, created_at,
                 importance_score, valid_at, invalid_at, expired_at, superseded_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                str(knowledge.id),
                knowledge.user_id,
                knowledge.statement,
                str(knowledge.source_episode_id) if knowledge.source_episode_id else None,
                int(knowledge.created_at.timestamp()),
                knowledge.importance_score,
                int(knowledge.valid_at.timestamp()) if knowledge.valid_at else None,
                int(knowledge.invalid_at.timestamp()) if knowledge.invalid_at else None,
                int(knowledge.expired_at.timestamp()) if knowledge.expired_at else None,
                str(knowledge.superseded_by) if knowledge.superseded_by else None,
            )

    async def get(self, knowledge_id: UUID | str) -> SemanticKnowledge | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM semantic_knowledge WHERE id = $1", str(knowledge_id))
        if row is None:
            return None
        return self._row_to_knowledge(row)

    async def get_by_episode(self, episode_id: UUID | str) -> list[SemanticKnowledge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM semantic_knowledge WHERE source_episode_id = $1 ORDER BY created_at ASC",
                str(episode_id),
            )
        return [self._row_to_knowledge(row) for row in rows]

    async def get_all(self) -> list[SemanticKnowledge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM semantic_knowledge ORDER BY created_at DESC")
        return [self._row_to_knowledge(row) for row in rows]

    async def get_current(self) -> list[SemanticKnowledge]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM semantic_knowledge WHERE expired_at IS NULL ORDER BY created_at DESC")
        return [self._row_to_knowledge(row) for row in rows]

    async def get_valid_at(self, event_time: datetime, include_expired: bool = False) -> list[SemanticKnowledge]:
        event_ts = int(event_time.timestamp())
        async with self._pool.acquire() as conn:
            if include_expired:
                rows = await conn.fetch(
                    """SELECT * FROM semantic_knowledge
                    WHERE (valid_at IS NULL OR valid_at <= $1)
                    AND (invalid_at IS NULL OR invalid_at > $2)
                    ORDER BY created_at DESC""",
                    event_ts,
                    event_ts,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM semantic_knowledge
                    WHERE (valid_at IS NULL OR valid_at <= $1)
                    AND (invalid_at IS NULL OR invalid_at > $2)
                    AND expired_at IS NULL
                    ORDER BY created_at DESC""",
                    event_ts,
                    event_ts,
                )
        return [self._row_to_knowledge(row) for row in rows]

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[SemanticKnowledge]:
        async with self._pool.acquire() as conn:
            if include_expired:
                rows = await conn.fetch(
                    "SELECT * FROM semantic_knowledge WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                    user_id,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM semantic_knowledge WHERE user_id = $1 AND expired_at IS NULL
                    ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
                    user_id,
                    limit,
                    offset,
                )
        return [self._row_to_knowledge(row) for row in rows]

    async def count_by_user(self, user_id: str, include_expired: bool = False) -> int:
        async with self._pool.acquire() as conn:
            if include_expired:
                row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM semantic_knowledge WHERE user_id = $1", user_id)
            else:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) as cnt FROM semantic_knowledge WHERE user_id = $1 AND expired_at IS NULL",
                    user_id,
                )
        return row["cnt"] if row else 0

    async def invalidate(self, knowledge_id: UUID | str) -> bool:
        expired_at = int(datetime.now(timezone.utc).timestamp())
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE semantic_knowledge SET expired_at = $1 WHERE id = $2 AND expired_at IS NULL",
                expired_at,
                str(knowledge_id),
            )
        return _parse_rowcount(status) > 0

    async def invalidate_with_successor(self, knowledge_id: UUID | str, successor_id: UUID | str) -> bool:
        expired_at = int(datetime.now(timezone.utc).timestamp())
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE semantic_knowledge SET expired_at = $1, superseded_by = $2 WHERE id = $3 AND expired_at IS NULL",
                expired_at,
                str(successor_id),
                str(knowledge_id),
            )
        return _parse_rowcount(status) > 0

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM semantic_knowledge")
        return row["cnt"] if row else 0

    async def delete(self, knowledge_id: UUID | str) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM semantic_knowledge WHERE id = $1", str(knowledge_id))
        return _parse_rowcount(status) > 0

    async def clear_by_episodes(self, episode_ids: Sequence[UUID | str]) -> int:
        if not episode_ids:
            return 0
        ids = [str(eid) for eid in episode_ids]
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM semantic_knowledge WHERE source_episode_id = ANY($1::text[])", ids)
        return _parse_rowcount(status)

    async def clear_user(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM semantic_knowledge WHERE user_id = $1", user_id)
        return _parse_rowcount(status)

    def _row_to_knowledge(self, row: asyncpg.Record) -> SemanticKnowledge:
        return SemanticKnowledge(
            id=UUID(row["id"]),
            user_id=row["user_id"],
            statement=row["statement"],
            source_episode_id=UUID(row["source_episode_id"]) if row["source_episode_id"] else None,
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
            importance_score=row["importance_score"],
            embedding=None,
            valid_at=datetime.fromtimestamp(row["valid_at"], tz=timezone.utc) if row["valid_at"] else None,
            invalid_at=datetime.fromtimestamp(row["invalid_at"], tz=timezone.utc) if row["invalid_at"] else None,
            expired_at=datetime.fromtimestamp(row["expired_at"], tz=timezone.utc) if row["expired_at"] else None,
            superseded_by=UUID(row["superseded_by"]) if row["superseded_by"] else None,
        )

    async def _create_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_knowledge (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                source_episode_id TEXT,
                created_at BIGINT NOT NULL,
                importance_score DOUBLE PRECISION,
                valid_at BIGINT,
                invalid_at BIGINT,
                expired_at BIGINT,
                superseded_by TEXT
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_episode ON semantic_knowledge(source_episode_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_valid_at ON semantic_knowledge(valid_at)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_expired_at ON semantic_knowledge(expired_at)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_user_id ON semantic_knowledge(user_id)")
