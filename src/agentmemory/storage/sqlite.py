import json
import struct
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import UUID

import aiosqlite
import sqlite_vec

from agentmemory.models import Episode, Message, MessageRole, SemanticKnowledge


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

    def _row_to_message(self, row: aiosqlite.Row) -> Message:
        return Message(
            id=row["id"],
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
            "INSERT INTO episodes (id, user_id, message_ids, title, narrative, start_time, end_time, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",  # noqa E501
            (
                str(episode.id),
                episode.user_id,
                json.dumps([str(msg_id) for msg_id in episode.message_ids]),
                episode.title,
                episode.narrative,
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

    def _row_to_episode(self, row: aiosqlite.Row) -> Episode:
        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            message_ids=[UUID(mid) for mid in json.loads(row["message_ids"])],
            title=row["title"],
            narrative=row["narrative"],
            start_time=datetime.fromtimestamp(row["start_time"], tz=timezone.utc),
            end_time=datetime.fromtimestamp(row["end_time"], tz=timezone.utc),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
        )

    async def _create_table(self):
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, user_id TEXT, message_ids TEXT, title TEXT, narrative TEXT, start_time INTEGER, end_time INTEGER, created_at INTEGER)"  # noqa E501
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)")
        await self._commit()


class KnowledgeStore(StoreBase):
    async def add(self, knowledge: SemanticKnowledge) -> None:
        await self._conn.execute(
            "INSERT INTO semantic_knowledge (id, statement, source_episode_id, created_at, importance_score, embedding) VALUES (?, ?, ?, ?, ?, ?)",  # noqa E501
            (
                str(knowledge.id),
                knowledge.statement,
                str(knowledge.source_episode_id),
                int(knowledge.created_at.timestamp()),
                knowledge.importance_score,
                json.dumps(knowledge.embedding),
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

    def _row_to_knowledge(self, row: aiosqlite.Row) -> SemanticKnowledge:
        return SemanticKnowledge(
            id=row["id"],
            statement=row["statement"],
            source_episode_id=row["source_episode_id"],
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
            importance_score=row["importance_score"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )

    async def _create_table(self):
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS semantic_knowledge (id TEXT PRIMARY KEY, statement TEXT, source_episode_id TEXT, created_at INTEGER, importance_score REAL, embedding TEXT)"  # noqa E501
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_episode ON semantic_knowledge(source_episode_id)")
        await self._commit()


class VectorIndex(StoreBase):
    def __init__(self, db_path: str, dimensions: int = 1536):
        super().__init__(db_path)
        self.dimensions = dimensions

    async def open(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        # Load sqlite-vec extension
        await self._db.enable_load_extension(True)
        await self._db.load_extension(sqlite_vec.loadable_path())
        await self._db.enable_load_extension(False)
        await self._create_table()

    async def _create_table(self):
        # Mapping table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_mapping (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL
            )
        """)
        # Vector table
        await self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_knowledge
            USING vec0(embedding float[{self.dimensions}])
        """)
        await self._conn.commit()

    async def add(self, uuid: UUID, embedding: list[float]) -> None:
        # Insert into mapping to get rowid
        cursor = await self._conn.execute("INSERT INTO vec_mapping (uuid) VALUES (?)", (str(uuid),))
        rowid = cursor.lastrowid

        # Insert into vector table
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        await self._conn.execute("INSERT INTO vec_knowledge (rowid, embedding) VALUES (?, ?)", (rowid, embedding_bytes))
        await self._conn.commit()

    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[UUID]:
        query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        cursor = await self._conn.execute(
            """
            SELECT vec_mapping.uuid
            FROM vec_knowledge
            JOIN vec_mapping ON vec_knowledge.rowid = vec_mapping.rowid
            WHERE vec_knowledge.embedding MATCH ?
              AND k = ?
            ORDER BY distance
        """,
            (query_bytes, top_k),
        )
        rows = await cursor.fetchall()
        return [UUID(row["uuid"]) for row in rows]


class TextIndex(StoreBase):
    """Full-text search index using  FTS5."""

    async def _create_table(self):
        # Mapping table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_mapping (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL
            )
        """)
        # FTS5 virtual table
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_knowledge
            USING fts5(content)
        """)
        await self._conn.commit()

    async def add(self, uuid: UUID, text: str) -> None:
        # Insert into mapping to get rowid
        cursor = await self._conn.execute("INSERT INTO fts_mapping (uuid) VALUES (?)", (str(uuid),))
        rowid = cursor.lastrowid

        # Insert into FTS5 table
        await self._conn.execute("INSERT INTO fts_knowledge (rowid, content) VALUES (?, ?)", (rowid, text))
        await self._conn.commit()

    async def search(self, query: str, top_k: int = 10) -> list[UUID]:
        cursor = await self._conn.execute(
            """
            SELECT fts_mapping.uuid
            FROM fts_knowledge
            JOIN fts_mapping ON fts_knowledge.rowid = fts_mapping.rowid
            WHERE fts_knowledge.content MATCH ?
            ORDER BY rank
            LIMIT ?
        """,
            (query, top_k),
        )
        rows = await cursor.fetchall()
        return [UUID(row["uuid"]) for row in rows]
