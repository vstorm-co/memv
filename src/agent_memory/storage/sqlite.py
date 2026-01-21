import json
import re
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import UUID

import aiosqlite
import sqlite_vec

from agent_memory.models import Episode, Message, MessageRole, SemanticKnowledge


class StoreBase(ABC):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._in_transaction = False

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def open(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_table()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for atomic multi-operation transactions."""
        self._in_transaction = True
        try:
            yield
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise
        finally:
            self._in_transaction = False

    async def _commit(self) -> None:
        """Commit if not inside a transaction."""
        if not self._in_transaction:
            await self._conn.commit()

    @abstractmethod
    async def _create_table(self):
        raise NotImplementedError

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store not opened. Call open() first.")
        return self._db


class MessageStore(StoreBase):
    async def add(self, message: Message) -> None:
        await self._conn.execute(
            "INSERT INTO messages (id, user_id, role, content, sent_at) VALUES (?, ?, ?, ?, ?)",
            (str(message.id), message.user_id, message.role.value, message.content, int(message.sent_at.timestamp())),
        )
        await self._commit()

    async def get(self, message_id: UUID | str) -> Message | None:
        cursor = await self._conn.execute("SELECT * FROM messages WHERE id = ?", (str(message_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_message(row)

    async def get_by_user(self, user_id: str) -> list[Message]:
        cursor = await self._conn.execute("SELECT * FROM messages WHERE user_id = ? ORDER BY sent_at ASC", (user_id,))
        rows = await cursor.fetchall()
        return [self._row_to_message(row) for row in rows]

    async def get_by_time_range(self, user_id: str, start: datetime, end: datetime) -> list[Message]:
        cursor = await self._conn.execute(
            "SELECT * FROM messages WHERE user_id = ? AND sent_at BETWEEN ? AND ? ORDER BY sent_at ASC",
            (user_id, int(start.timestamp()), int(end.timestamp())),
        )
        rows = await cursor.fetchall()
        return [self._row_to_message(row) for row in rows]

    async def list_users(self) -> list[str]:
        """Return distinct user IDs from messages."""
        cursor = await self._conn.execute("SELECT DISTINCT user_id FROM messages ORDER BY user_id")
        rows = await cursor.fetchall()
        return [row["user_id"] for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        """Count messages, optionally filtered by user."""
        if user_id:
            cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM messages WHERE user_id = ?", (user_id,))
        else:
            cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM messages")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def delete(self, message_id: UUID | str) -> bool:
        """Delete a message by ID. Returns True if deleted."""
        cursor = await self._conn.execute("DELETE FROM messages WHERE id = ?", (str(message_id),))
        await self._commit()
        return cursor.rowcount > 0

    async def clear_user(self, user_id: str) -> int:
        """Delete all messages for a user. Returns count of deleted messages."""
        cursor = await self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await self._commit()
        return cursor.rowcount

    def _row_to_message(self, row: aiosqlite.Row) -> Message:
        return Message(
            id=UUID(row["id"]),  # Explicitly convert for correct comparison
            user_id=row["user_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            sent_at=datetime.fromtimestamp(row["sent_at"], tz=timezone.utc),
        )

    async def _create_table(self):
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, user_id TEXT, role TEXT, content TEXT, sent_at INTEGER)"
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_sent_at ON messages(sent_at)")
        await self._commit()


class EpisodeStore(StoreBase):
    async def add(self, episode: Episode) -> None:
        await self._conn.execute(
            "INSERT INTO episodes (id, user_id, title, content, original_messages, start_time, end_time, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",  # noqa E501
            (
                str(episode.id),
                episode.user_id,
                episode.title,
                episode.content,
                json.dumps(episode.original_messages),
                int(episode.start_time.timestamp()),
                int(episode.end_time.timestamp()),
                int(episode.created_at.timestamp()),
            ),
        )
        await self._commit()

    async def get(self, episode_id: UUID | str) -> Episode | None:
        cursor = await self._conn.execute("SELECT * FROM episodes WHERE id = ?", (str(episode_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    async def get_by_user(self, user_id: str) -> list[Episode]:
        cursor = await self._conn.execute("SELECT * FROM episodes WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        rows = await cursor.fetchall()
        return [self._row_to_episode(row) for row in rows]

    async def get_by_time_range(self, user_id: str, start_time: datetime, end_time: datetime) -> list[Episode]:
        """Return episodes that overlap with the given time range."""
        cursor = await self._conn.execute(
            "SELECT * FROM episodes WHERE user_id = ? AND NOT (end_time < ? OR start_time > ?) ORDER BY start_time ASC",
            (user_id, int(start_time.timestamp()), int(end_time.timestamp())),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(row) for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        """Count episodes, optionally filtered by user."""
        if user_id:
            cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM episodes WHERE user_id = ?", (user_id,))
        else:
            cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM episodes")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def delete(self, episode_id: UUID | str) -> bool:
        """Delete an episode by ID. Returns True if deleted."""
        cursor = await self._conn.execute("DELETE FROM episodes WHERE id = ?", (str(episode_id),))
        await self._commit()
        return cursor.rowcount > 0

    async def clear_user(self, user_id: str) -> int:
        """Delete all episodes for a user. Returns count of deleted episodes."""
        cursor = await self._conn.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
        await self._commit()
        return cursor.rowcount

    async def update(self, episode: Episode) -> bool:
        """Update an existing episode. Returns True if updated."""
        cursor = await self._conn.execute(
            """UPDATE episodes SET
                title = ?,
                content = ?,
                original_messages = ?,
                start_time = ?,
                end_time = ?
            WHERE id = ?""",
            (
                episode.title,
                episode.content,
                json.dumps(episode.original_messages),
                int(episode.start_time.timestamp()),
                int(episode.end_time.timestamp()),
                str(episode.id),
            ),
        )
        await self._commit()
        return cursor.rowcount > 0

    def _row_to_episode(self, row: aiosqlite.Row) -> Episode:
        return Episode(
            id=UUID(row["id"]),
            user_id=row["user_id"],
            title=row["title"],
            content=row["content"],
            original_messages=json.loads(row["original_messages"]),
            start_time=datetime.fromtimestamp(row["start_time"], tz=timezone.utc),
            end_time=datetime.fromtimestamp(row["end_time"], tz=timezone.utc),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
        )

    async def _create_table(self):
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, content TEXT, original_messages TEXT, start_time INTEGER, end_time INTEGER, created_at INTEGER)"  # noqa E501
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)")
        # Migrate: add new columns if missing (for older databases)
        await self._migrate_add_columns()
        await self._commit()

    async def _migrate_add_columns(self):
        """Add content and original_messages columns if they don't exist."""
        cursor = await self._conn.execute("PRAGMA table_info(episodes)")
        columns = {row["name"] for row in await cursor.fetchall()}

        if "content" not in columns:
            await self._conn.execute("ALTER TABLE episodes ADD COLUMN content TEXT DEFAULT ''")
        if "original_messages" not in columns:
            await self._conn.execute("ALTER TABLE episodes ADD COLUMN original_messages TEXT DEFAULT '[]'")


class KnowledgeStore(StoreBase):
    async def add(self, knowledge: SemanticKnowledge) -> None:
        await self._conn.execute(
            """INSERT INTO semantic_knowledge
            (id, statement, source_episode_id, created_at, importance_score, embedding, valid_at, invalid_at, expired_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(knowledge.id),
                knowledge.statement,
                str(knowledge.source_episode_id),
                int(knowledge.created_at.timestamp()),
                knowledge.importance_score,
                json.dumps(knowledge.embedding),
                int(knowledge.valid_at.timestamp()) if knowledge.valid_at else None,
                int(knowledge.invalid_at.timestamp()) if knowledge.invalid_at else None,
                int(knowledge.expired_at.timestamp()) if knowledge.expired_at else None,
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
            # Only filter by event time validity
            cursor = await self._conn.execute(
                """SELECT * FROM semantic_knowledge
                WHERE (valid_at IS NULL OR valid_at <= ?)
                AND (invalid_at IS NULL OR invalid_at > ?)
                ORDER BY created_at DESC""",
                (event_ts, event_ts),
            )
        else:
            # Filter by both event time and transaction time (exclude expired)
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

    async def invalidate(self, knowledge_id: UUID | str) -> bool:
        """
        Mark knowledge as expired (superseded).

        Returns True if updated, False if not found.
        """
        expired_at = int(datetime.now(timezone.utc).timestamp())
        cursor = await self._conn.execute(
            "UPDATE semantic_knowledge SET expired_at = ? WHERE id = ? AND expired_at IS NULL",
            (expired_at, str(knowledge_id)),
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
        """Delete all knowledge entries for given episodes. Returns count of deleted entries."""
        if not episode_ids:
            return 0
        placeholders = ",".join("?" * len(episode_ids))
        cursor = await self._conn.execute(
            f"DELETE FROM semantic_knowledge WHERE source_episode_id IN ({placeholders})",
            [str(eid) for eid in episode_ids],
        )
        await self._commit()
        return cursor.rowcount

    def _row_to_knowledge(self, row: aiosqlite.Row) -> SemanticKnowledge:
        return SemanticKnowledge(
            id=UUID(row["id"]),
            statement=row["statement"],
            source_episode_id=UUID(row["source_episode_id"]),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
            importance_score=row["importance_score"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            valid_at=datetime.fromtimestamp(row["valid_at"], tz=timezone.utc) if row["valid_at"] else None,
            invalid_at=datetime.fromtimestamp(row["invalid_at"], tz=timezone.utc) if row["invalid_at"] else None,
            expired_at=datetime.fromtimestamp(row["expired_at"], tz=timezone.utc) if row["expired_at"] else None,
        )

    async def _create_table(self):
        await self._conn.execute(
            """CREATE TABLE IF NOT EXISTS semantic_knowledge (
                id TEXT PRIMARY KEY,
                statement TEXT,
                source_episode_id TEXT,
                created_at INTEGER,
                importance_score REAL,
                embedding TEXT,
                valid_at INTEGER,
                invalid_at INTEGER,
                expired_at INTEGER
            )"""
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_episode ON semantic_knowledge(source_episode_id)")
        # Migrate: add bi-temporal columns if missing (for older databases)
        await self._migrate_add_bitemporal_columns()
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_valid_at ON semantic_knowledge(valid_at)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_expired_at ON semantic_knowledge(expired_at)")
        await self._commit()

    async def _migrate_add_bitemporal_columns(self):
        """Add valid_at, invalid_at, expired_at columns if they don't exist."""
        cursor = await self._conn.execute("PRAGMA table_info(semantic_knowledge)")
        columns = {row["name"] for row in await cursor.fetchall()}

        for col in ["valid_at", "invalid_at", "expired_at"]:
            if col not in columns:
                await self._conn.execute(f"ALTER TABLE semantic_knowledge ADD COLUMN {col} INTEGER")


class VectorIndex(StoreBase):
    """Vector similarity index using sqlite-vec with per-user isolation."""

    def __init__(self, db_path: str, dimensions: int = 1536, name: str = "knowledge"):
        super().__init__(db_path)
        self.dimensions = dimensions
        self.name = name
        self._mapping_table = f"vec_{name}_mapping"
        self._vec_table = f"vec_{name}"

    async def open(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        # Load sqlite-vec extension
        await self._db.enable_load_extension(True)
        await self._db.load_extension(sqlite_vec.loadable_path())
        await self._db.enable_load_extension(False)
        await self._create_table()
        await self._migrate_add_user_id()

    async def _create_table(self):
        # Mapping table with user_id for per-user isolation
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._mapping_table} (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL DEFAULT ''
            )
        """)
        # Index for user_id filtering
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._mapping_table}_user
            ON {self._mapping_table}(user_id)
        """)
        # Vector table
        await self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec_table}
            USING vec0(embedding float[{self.dimensions}])
        """)
        await self._conn.commit()

    async def _migrate_add_user_id(self):
        """Add user_id column if missing (schema migration)."""
        cursor = await self._conn.execute(f"PRAGMA table_info({self._mapping_table})")
        columns = {row[1] for row in await cursor.fetchall()}
        if "user_id" not in columns:
            await self._conn.execute(f"ALTER TABLE {self._mapping_table} ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
            await self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._mapping_table}_user
                ON {self._mapping_table}(user_id)
            """)
            await self._conn.commit()

    async def add(self, uuid: UUID, embedding: list[float], user_id: str) -> None:
        # Insert into mapping to get rowid
        cursor = await self._conn.execute(f"INSERT INTO {self._mapping_table} (uuid, user_id) VALUES (?, ?)", (str(uuid), user_id))
        rowid = cursor.lastrowid

        # Insert into vector table
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        await self._conn.execute(f"INSERT INTO {self._vec_table} (rowid, embedding) VALUES (?, ?)", (rowid, embedding_bytes))
        await self._conn.commit()

    async def search(self, query_embedding: list[float], top_k: int = 10, user_id: str | None = None) -> list[UUID]:
        """Search for similar vectors, optionally filtered by user_id."""
        query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

        if user_id is not None:
            # Request more candidates from vector search, then filter by user_id
            k_candidates = top_k * 10
            cursor = await self._conn.execute(
                f"""
                SELECT {self._mapping_table}.uuid
                FROM {self._vec_table}
                JOIN {self._mapping_table} ON {self._vec_table}.rowid = {self._mapping_table}.rowid
                WHERE {self._vec_table}.embedding MATCH ?
                  AND k = ?
                  AND {self._mapping_table}.user_id = ?
                ORDER BY distance
                LIMIT ?
            """,
                (query_bytes, k_candidates, user_id, top_k),
            )
        else:
            cursor = await self._conn.execute(
                f"""
                SELECT {self._mapping_table}.uuid
                FROM {self._vec_table}
                JOIN {self._mapping_table} ON {self._vec_table}.rowid = {self._mapping_table}.rowid
                WHERE {self._vec_table}.embedding MATCH ?
                  AND k = ?
                ORDER BY distance
            """,
                (query_bytes, top_k),
            )
        rows = await cursor.fetchall()
        return [UUID(row["uuid"]) for row in rows]

    async def search_with_scores(
        self, query_embedding: list[float], top_k: int = 10, user_id: str | None = None
    ) -> list[tuple[UUID, float]]:
        """Search for similar vectors and return (uuid, similarity_score) tuples.

        sqlite-vec returns L2 distance. We convert to similarity: 1 / (1 + distance).
        """
        query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

        if user_id is not None:
            k_candidates = top_k * 10
            cursor = await self._conn.execute(
                f"""
                SELECT {self._mapping_table}.uuid, distance
                FROM {self._vec_table}
                JOIN {self._mapping_table} ON {self._vec_table}.rowid = {self._mapping_table}.rowid
                WHERE {self._vec_table}.embedding MATCH ?
                  AND k = ?
                  AND {self._mapping_table}.user_id = ?
                ORDER BY distance
                LIMIT ?
            """,
                (query_bytes, k_candidates, user_id, top_k),
            )
        else:
            cursor = await self._conn.execute(
                f"""
                SELECT {self._mapping_table}.uuid, distance
                FROM {self._vec_table}
                JOIN {self._mapping_table} ON {self._vec_table}.rowid = {self._mapping_table}.rowid
                WHERE {self._vec_table}.embedding MATCH ?
                  AND k = ?
                ORDER BY distance
            """,
                (query_bytes, top_k),
            )
        rows = await cursor.fetchall()
        # Convert L2 distance to similarity score (higher = more similar)
        return [(UUID(row["uuid"]), 1.0 / (1.0 + row["distance"])) for row in rows]

    async def clear_user(self, user_id: str) -> int:
        """Delete all vector entries for a user. Returns count of deleted entries."""
        # Get rowids to delete from vec table
        cursor = await self._conn.execute(f"SELECT rowid FROM {self._mapping_table} WHERE user_id = ?", (user_id,))
        rows = await cursor.fetchall()
        rowids = [row["rowid"] for row in rows]

        if not rowids:
            return 0

        # Delete from vec table
        placeholders = ",".join("?" * len(rowids))
        await self._conn.execute(f"DELETE FROM {self._vec_table} WHERE rowid IN ({placeholders})", rowids)

        # Delete from mapping table
        cursor = await self._conn.execute(f"DELETE FROM {self._mapping_table} WHERE user_id = ?", (user_id,))
        await self._conn.commit()
        return cursor.rowcount


class TextIndex(StoreBase):
    """Full-text search index using FTS5 with per-user isolation."""

    def __init__(self, db_path: str, name: str = "knowledge"):
        super().__init__(db_path)
        self.name = name
        self._mapping_table = f"fts_{name}_mapping"
        self._fts_table = f"fts_{name}"

    async def open(self):
        await super().open()
        await self._migrate_add_user_id()

    async def _create_table(self):
        # Mapping table with user_id for per-user isolation
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._mapping_table} (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL DEFAULT ''
            )
        """)
        # Index for user_id filtering
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._mapping_table}_user
            ON {self._mapping_table}(user_id)
        """)
        # FTS5 virtual table
        await self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table}
            USING fts5(content)
        """)
        await self._conn.commit()

    async def _migrate_add_user_id(self):
        """Add user_id column if missing (schema migration)."""
        cursor = await self._conn.execute(f"PRAGMA table_info({self._mapping_table})")
        columns = {row[1] for row in await cursor.fetchall()}
        if "user_id" not in columns:
            await self._conn.execute(f"ALTER TABLE {self._mapping_table} ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
            await self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._mapping_table}_user
                ON {self._mapping_table}(user_id)
            """)
            await self._conn.commit()

    async def add(self, uuid: UUID, text: str, user_id: str) -> None:
        # Insert into mapping to get rowid
        cursor = await self._conn.execute(f"INSERT INTO {self._mapping_table} (uuid, user_id) VALUES (?, ?)", (str(uuid), user_id))
        rowid = cursor.lastrowid

        # Insert into FTS5 table
        await self._conn.execute(f"INSERT INTO {self._fts_table} (rowid, content) VALUES (?, ?)", (rowid, text))
        await self._conn.commit()

    async def search(self, query: str, top_k: int = 10, user_id: str | None = None) -> list[UUID]:
        """Search for matching text, optionally filtered by user_id."""
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        if user_id is not None:
            cursor = await self._conn.execute(
                f"""
                SELECT {self._mapping_table}.uuid
                FROM {self._fts_table}
                JOIN {self._mapping_table} ON {self._fts_table}.rowid = {self._mapping_table}.rowid
                WHERE {self._fts_table}.content MATCH ?
                  AND {self._mapping_table}.user_id = ?
                ORDER BY rank
                LIMIT ?
            """,
                (sanitized, user_id, top_k),
            )
        else:
            cursor = await self._conn.execute(
                f"""
                SELECT {self._mapping_table}.uuid
                FROM {self._fts_table}
                JOIN {self._mapping_table} ON {self._fts_table}.rowid = {self._mapping_table}.rowid
                WHERE {self._fts_table}.content MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (sanitized, top_k),
            )
        rows = await cursor.fetchall()
        return [UUID(row["uuid"]) for row in rows]

    def _sanitize_fts_query(self, query: str) -> str:
        """
        Sanitize query for FTS5.

        FTS5 has special syntax for operators like AND, OR, NOT, quotes, etc.
        We escape by quoting each token as a literal phrase.
        """
        # Remove special FTS5 characters and split into words
        # Keep only alphanumeric and spaces
        cleaned = re.sub(r"[^\w\s]", " ", query)
        words = cleaned.split()

        if not words:
            return ""

        # Quote each word to treat as literal, join with space (implicit AND)
        return " ".join(f'"{word}"' for word in words)

    async def clear_user(self, user_id: str) -> int:
        """Delete all text index entries for a user. Returns count of deleted entries."""
        # Get rowids to delete from fts table
        cursor = await self._conn.execute(f"SELECT rowid FROM {self._mapping_table} WHERE user_id = ?", (user_id,))
        rows = await cursor.fetchall()
        rowids = [row["rowid"] for row in rows]

        if not rowids:
            return 0

        # Delete from fts table
        placeholders = ",".join("?" * len(rowids))
        await self._conn.execute(f"DELETE FROM {self._fts_table} WHERE rowid IN ({placeholders})", rowids)

        # Delete from mapping table
        cursor = await self._conn.execute(f"DELETE FROM {self._mapping_table} WHERE user_id = ?", (user_id,))
        await self._conn.commit()
        return cursor.rowcount
