"""Semantic knowledge storage."""

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

import aiosqlite

from memv.models import SemanticKnowledge
from memv.storage.sqlite._base import StoreBase


class KnowledgeStore(StoreBase):
    """Store for semantic knowledge."""

    async def add(self, knowledge: SemanticKnowledge) -> None:
        await self._conn.execute(
            """INSERT INTO semantic_knowledge
            (id, user_id, statement, source_episode_id, created_at,
             importance_score, embedding, valid_at, invalid_at, expired_at, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(knowledge.id),
                knowledge.user_id,
                knowledge.statement,
                str(knowledge.source_episode_id) if knowledge.source_episode_id else None,
                int(knowledge.created_at.timestamp()),
                knowledge.importance_score,
                json.dumps(knowledge.embedding),
                int(knowledge.valid_at.timestamp()) if knowledge.valid_at else None,
                int(knowledge.invalid_at.timestamp()) if knowledge.invalid_at else None,
                int(knowledge.expired_at.timestamp()) if knowledge.expired_at else None,
                str(knowledge.superseded_by) if knowledge.superseded_by else None,
            ),
        )
        await self._commit()

    async def get(self, knowledge_id: UUID | str) -> SemanticKnowledge | None:
        cursor = await self._conn.execute("SELECT * FROM semantic_knowledge WHERE id = ?", (str(knowledge_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_knowledge(row)

    async def get_by_episode(self, episode_id: UUID | str) -> list[SemanticKnowledge]:
        cursor = await self._conn.execute(
            "SELECT * FROM semantic_knowledge WHERE source_episode_id = ? ORDER BY created_at ASC", (str(episode_id),)
        )
        rows = await cursor.fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    async def get_all(self) -> list[SemanticKnowledge]:
        """Return all knowledge entries."""
        cursor = await self._conn.execute("SELECT * FROM semantic_knowledge ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    async def get_current(self) -> list[SemanticKnowledge]:
        """Return all non-expired knowledge entries."""
        cursor = await self._conn.execute("SELECT * FROM semantic_knowledge WHERE expired_at IS NULL ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    async def get_valid_at(
        self,
        event_time: datetime,
        include_expired: bool = False,
    ) -> list[SemanticKnowledge]:
        """
        Return knowledge valid at the given event time.

        Args:
            event_time: Point in time to query (event timeline)
            include_expired: If True, include superseded records (transaction timeline)
        """
        event_ts = int(event_time.timestamp())

        if include_expired:
            cursor = await self._conn.execute(
                """SELECT * FROM semantic_knowledge
                WHERE (valid_at IS NULL OR valid_at <= ?)
                AND (invalid_at IS NULL OR invalid_at > ?)
                ORDER BY created_at DESC""",
                (event_ts, event_ts),
            )
        else:
            cursor = await self._conn.execute(
                """SELECT * FROM semantic_knowledge
                WHERE (valid_at IS NULL OR valid_at <= ?)
                AND (invalid_at IS NULL OR invalid_at > ?)
                AND expired_at IS NULL
                ORDER BY created_at DESC""",
                (event_ts, event_ts),
            )

        rows = await cursor.fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[SemanticKnowledge]:
        """List knowledge entries for a user with pagination."""
        if include_expired:
            cursor = await self._conn.execute(
                "SELECT * FROM semantic_knowledge WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM semantic_knowledge WHERE user_id = ? AND expired_at IS NULL ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    async def count_by_user(self, user_id: str, include_expired: bool = False) -> int:
        """Count knowledge entries for a user."""
        if include_expired:
            cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM semantic_knowledge WHERE user_id = ?", (user_id,))
        else:
            cursor = await self._conn.execute(
                "SELECT COUNT(*) as cnt FROM semantic_knowledge WHERE user_id = ? AND expired_at IS NULL", (user_id,)
            )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def invalidate(self, knowledge_id: UUID | str) -> bool:
        """Mark knowledge as expired (superseded). Returns True if updated."""
        expired_at = int(datetime.now(timezone.utc).timestamp())
        cursor = await self._conn.execute(
            "UPDATE semantic_knowledge SET expired_at = ? WHERE id = ? AND expired_at IS NULL",
            (expired_at, str(knowledge_id)),
        )
        await self._commit()
        return cursor.rowcount > 0

    async def invalidate_with_successor(self, knowledge_id: UUID | str, successor_id: UUID | str) -> bool:
        """Mark knowledge as expired and record which entry replaced it. Returns True if updated."""
        expired_at = int(datetime.now(timezone.utc).timestamp())
        cursor = await self._conn.execute(
            "UPDATE semantic_knowledge SET expired_at = ?, superseded_by = ? WHERE id = ? AND expired_at IS NULL",
            (expired_at, str(successor_id), str(knowledge_id)),
        )
        await self._commit()
        return cursor.rowcount > 0

    async def count(self) -> int:
        """Count all knowledge entries."""
        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM semantic_knowledge")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def delete(self, knowledge_id: UUID | str) -> bool:
        """Delete a knowledge entry by ID. Returns True if deleted."""
        cursor = await self._conn.execute("DELETE FROM semantic_knowledge WHERE id = ?", (str(knowledge_id),))
        await self._commit()
        return cursor.rowcount > 0

    async def clear_by_episodes(self, episode_ids: Sequence[UUID | str]) -> int:
        """Delete all knowledge entries for given episodes. Returns count of deleted entries.

        Note: Does not affect injected knowledge (source_episode_id=NULL). Use clear_user() for complete deletion.
        """
        if not episode_ids:
            return 0
        placeholders = ",".join("?" * len(episode_ids))
        cursor = await self._conn.execute(
            f"DELETE FROM semantic_knowledge WHERE source_episode_id IN ({placeholders})",
            [str(eid) for eid in episode_ids],
        )
        await self._commit()
        return cursor.rowcount

    async def clear_user(self, user_id: str) -> int:
        """Delete all knowledge entries for a user. Returns count deleted."""
        cursor = await self._conn.execute("DELETE FROM semantic_knowledge WHERE user_id = ?", (user_id,))
        await self._commit()
        return cursor.rowcount

    def _row_to_knowledge(self, row: aiosqlite.Row) -> SemanticKnowledge:
        return SemanticKnowledge(
            id=UUID(row["id"]),
            user_id=row["user_id"],
            statement=row["statement"],
            source_episode_id=UUID(row["source_episode_id"]) if row["source_episode_id"] else None,
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
            importance_score=row["importance_score"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            valid_at=datetime.fromtimestamp(row["valid_at"], tz=timezone.utc) if row["valid_at"] else None,
            invalid_at=datetime.fromtimestamp(row["invalid_at"], tz=timezone.utc) if row["invalid_at"] else None,
            expired_at=datetime.fromtimestamp(row["expired_at"], tz=timezone.utc) if row["expired_at"] else None,
            superseded_by=UUID(row["superseded_by"]) if row["superseded_by"] else None,
        )

    async def _create_table(self):
        await self._conn.execute(
            """CREATE TABLE IF NOT EXISTS semantic_knowledge (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                statement TEXT,
                source_episode_id TEXT,
                created_at INTEGER,
                importance_score REAL,
                embedding TEXT,
                valid_at INTEGER,
                invalid_at INTEGER,
                expired_at INTEGER,
                superseded_by TEXT
            )"""
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_episode ON semantic_knowledge(source_episode_id)")
        await self._migrate_add_bitemporal_columns()
        await self._migrate_add_user_id_column()
        await self._migrate_add_superseded_by_column()
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_valid_at ON semantic_knowledge(valid_at)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_expired_at ON semantic_knowledge(expired_at)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_user_id ON semantic_knowledge(user_id)")
        await self._commit()

    async def _migrate_add_bitemporal_columns(self):
        """Add valid_at, invalid_at, expired_at columns if they don't exist."""
        cursor = await self._conn.execute("PRAGMA table_info(semantic_knowledge)")
        columns = {row["name"] for row in await cursor.fetchall()}

        for col in ["valid_at", "invalid_at", "expired_at"]:
            if col not in columns:
                await self._conn.execute(f"ALTER TABLE semantic_knowledge ADD COLUMN {col} INTEGER")

    async def _migrate_add_user_id_column(self):
        """Add user_id column and backfill from episodes if it doesn't exist."""
        cursor = await self._conn.execute("PRAGMA table_info(semantic_knowledge)")
        columns = {row["name"] for row in await cursor.fetchall()}

        if "user_id" not in columns:
            await self._conn.execute("ALTER TABLE semantic_knowledge ADD COLUMN user_id TEXT")
            try:
                await self._conn.execute(
                    """UPDATE semantic_knowledge
                    SET user_id = (SELECT user_id FROM episodes WHERE id = semantic_knowledge.source_episode_id)
                    WHERE user_id IS NULL"""
                )
            except aiosqlite.OperationalError as e:
                if "no such table" not in str(e).lower():
                    raise

    async def _migrate_add_superseded_by_column(self):
        """Add superseded_by column if it doesn't exist."""
        cursor = await self._conn.execute("PRAGMA table_info(semantic_knowledge)")
        columns = {row["name"] for row in await cursor.fetchall()}

        if "superseded_by" not in columns:
            await self._conn.execute("ALTER TABLE semantic_knowledge ADD COLUMN superseded_by TEXT")
