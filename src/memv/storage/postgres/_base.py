"""Base class for PostgreSQL stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg


def _parse_rowcount(status: str) -> int:
    """Extract affected row count from asyncpg status string (e.g. 'DELETE 2' → 2)."""
    parts = status.split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


class PgStoreBase(ABC):
    """Base class for PostgreSQL-backed stores.

    All stores share a single asyncpg.Pool managed by LifecycleManager.
    Each operation acquires a connection from the pool and returns it when done.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def open(self) -> None:
        async with self._pool.acquire() as conn:
            await self._create_table(conn)

    async def close(self) -> None:  # noqa: B027
        pass  # Pool lifecycle owned by LifecycleManager

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    @abstractmethod
    async def _create_table(self, conn: asyncpg.Connection) -> None: ...
