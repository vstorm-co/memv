"""PostgreSQL message storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from memv.models import Message, MessageRole
from memv.storage.postgres._base import PgStoreBase, _parse_rowcount

if TYPE_CHECKING:
    import asyncpg


class MessageStore(PgStoreBase):
    """PostgreSQL store for conversation messages."""

    async def add(self, message: Message) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (id, user_id, role, content, sent_at) VALUES ($1, $2, $3, $4, $5)",
                str(message.id),
                message.user_id,
                message.role.value,
                message.content,
                int(message.sent_at.timestamp()),
            )

    async def get(self, message_id: UUID | str) -> Message | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM messages WHERE id = $1", str(message_id))
        if row is None:
            return None
        return self._row_to_message(row)

    async def get_by_user(self, user_id: str) -> list[Message]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM messages WHERE user_id = $1 ORDER BY sent_at ASC", user_id)
        return [self._row_to_message(row) for row in rows]

    async def get_by_time_range(self, user_id: str, start: datetime, end: datetime) -> list[Message]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM messages WHERE user_id = $1 AND sent_at BETWEEN $2 AND $3 ORDER BY sent_at ASC",
                user_id,
                int(start.timestamp()),
                int(end.timestamp()),
            )
        return [self._row_to_message(row) for row in rows]

    async def list_users(self) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT user_id FROM messages ORDER BY user_id")
        return [row["user_id"] for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        async with self._pool.acquire() as conn:
            if user_id:
                row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM messages WHERE user_id = $1", user_id)
            else:
                row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM messages")
        return row["cnt"] if row else 0

    async def delete(self, message_id: UUID | str) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM messages WHERE id = $1", str(message_id))
        return _parse_rowcount(status) > 0

    async def clear_user(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)
        return _parse_rowcount(status)

    def _row_to_message(self, row: asyncpg.Record) -> Message:
        return Message(
            id=UUID(row["id"]),
            user_id=row["user_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            sent_at=datetime.fromtimestamp(row["sent_at"], tz=timezone.utc),
        )

    async def _create_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sent_at BIGINT NOT NULL
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_sent_at ON messages(sent_at)")
