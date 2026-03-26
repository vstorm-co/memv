"""PostgreSQL episode storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from memv.models import Episode
from memv.storage.postgres._base import PgStoreBase, _parse_rowcount

if TYPE_CHECKING:
    import asyncpg


class EpisodeStore(PgStoreBase):
    """PostgreSQL store for conversation episodes."""

    async def add(self, episode: Episode) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO episodes (id, user_id, title, content, original_messages, start_time, end_time, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)""",
                str(episode.id),
                episode.user_id,
                episode.title,
                episode.content,
                json.dumps(episode.original_messages),
                int(episode.start_time.timestamp()),
                int(episode.end_time.timestamp()),
                int(episode.created_at.timestamp()),
            )

    async def get(self, episode_id: UUID | str) -> Episode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM episodes WHERE id = $1", str(episode_id))
        if row is None:
            return None
        return self._row_to_episode(row)

    async def get_by_user(self, user_id: str) -> list[Episode]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM episodes WHERE user_id = $1 ORDER BY created_at DESC", user_id)
        return [self._row_to_episode(row) for row in rows]

    async def get_by_time_range(self, user_id: str, start_time: datetime, end_time: datetime) -> list[Episode]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM episodes WHERE user_id = $1 AND NOT (end_time < $2 OR start_time > $3) ORDER BY start_time ASC",
                user_id,
                int(start_time.timestamp()),
                int(end_time.timestamp()),
            )
        return [self._row_to_episode(row) for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        async with self._pool.acquire() as conn:
            if user_id:
                row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM episodes WHERE user_id = $1", user_id)
            else:
                row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM episodes")
        return row["cnt"] if row else 0

    async def delete(self, episode_id: UUID | str) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM episodes WHERE id = $1", str(episode_id))
        return _parse_rowcount(status) > 0

    async def clear_user(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM episodes WHERE user_id = $1", user_id)
        return _parse_rowcount(status)

    async def update(self, episode: Episode) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                """UPDATE episodes SET title = $1, content = $2, original_messages = $3::jsonb,
                start_time = $4, end_time = $5 WHERE id = $6""",
                episode.title,
                episode.content,
                json.dumps(episode.original_messages),
                int(episode.start_time.timestamp()),
                int(episode.end_time.timestamp()),
                str(episode.id),
            )
        return _parse_rowcount(status) > 0

    def _row_to_episode(self, row: asyncpg.Record) -> Episode:
        original_messages = row["original_messages"]
        if isinstance(original_messages, str):
            original_messages = json.loads(original_messages)
        return Episode(
            id=UUID(row["id"]),
            user_id=row["user_id"],
            title=row["title"],
            content=row["content"],
            original_messages=original_messages,
            start_time=datetime.fromtimestamp(row["start_time"], tz=timezone.utc),
            end_time=datetime.fromtimestamp(row["end_time"], tz=timezone.utc),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
        )

    async def _create_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                original_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                start_time BIGINT NOT NULL,
                end_time BIGINT NOT NULL,
                created_at BIGINT NOT NULL
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)")
