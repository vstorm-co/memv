import hashlib
import os
import struct
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from memv.models import Episode, Message, MessageRole, SemanticKnowledge
from memv.storage.sqlite._episodes import EpisodeStore as SqliteEpisodeStore
from memv.storage.sqlite._knowledge import KnowledgeStore as SqliteKnowledgeStore
from memv.storage.sqlite._messages import MessageStore as SqliteMessageStore
from memv.storage.sqlite._text_index import TextIndex as SqliteTextIndex
from memv.storage.sqlite._vector_index import VectorIndex as SqliteVectorIndex

# ---------------------------------------------------------------------------
# Backend parametrization
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--backend", action="append", default=[], help="Storage backends to test (sqlite, postgres)")


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        backends = metafunc.config.getoption("backend") or ["sqlite"]
        metafunc.parametrize("backend", backends, indirect=True)


@pytest.fixture
def backend(request):
    return request.param


# ---------------------------------------------------------------------------
# Postgres pool management
# ---------------------------------------------------------------------------

_pg_extension_created = False


async def _create_pg_pool():
    global _pg_extension_created
    url = os.environ.get("MEMV_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("MEMV_TEST_POSTGRES_URL not set")
    import asyncpg
    from pgvector.asyncpg import register_vector

    if not _pg_extension_created:
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()
        _pg_extension_created = True

    async def _init(conn):
        await register_vector(conn)

    return await asyncpg.create_pool(url, init=_init, min_size=1, max_size=5)


async def _cleanup_pg(pool):
    tables = ["messages", "episodes", "semantic_knowledge", "vec_knowledge", "fts_knowledge"]
    async with pool.acquire() as conn:
        for table in tables:
            try:
                await conn.execute(f"DELETE FROM {table}")  # noqa: S608
            except Exception:
                pass
    await pool.close()


# ---------------------------------------------------------------------------
# Store fixtures (parametrized by backend)
# ---------------------------------------------------------------------------


@pytest.fixture
async def message_store(backend, tmp_path):
    if backend == "postgres":
        from memv.storage.postgres import MessageStore as PgMessageStore

        pool = await _create_pg_pool()
        store = PgMessageStore(pool)
        await store.open()
        yield store
        await _cleanup_pg(pool)
    else:
        store = SqliteMessageStore(str(tmp_path / "test.db"))
        async with store:
            yield store


@pytest.fixture
async def episode_store(backend, tmp_path):
    if backend == "postgres":
        from memv.storage.postgres import EpisodeStore as PgEpisodeStore

        pool = await _create_pg_pool()
        store = PgEpisodeStore(pool)
        await store.open()
        yield store
        await _cleanup_pg(pool)
    else:
        store = SqliteEpisodeStore(str(tmp_path / "test.db"))
        async with store:
            yield store


@pytest.fixture
async def knowledge_store(backend, tmp_path):
    if backend == "postgres":
        from memv.storage.postgres import KnowledgeStore as PgKnowledgeStore

        pool = await _create_pg_pool()
        store = PgKnowledgeStore(pool)
        await store.open()
        yield store
        await _cleanup_pg(pool)
    else:
        store = SqliteKnowledgeStore(str(tmp_path / "test.db"))
        async with store:
            yield store


@pytest.fixture
async def text_index(backend, tmp_path):
    if backend == "postgres":
        from memv.storage.postgres import TextIndex as PgTextIndex

        pool = await _create_pg_pool()
        idx = PgTextIndex(pool)
        await idx.open()
        yield idx
        await _cleanup_pg(pool)
    else:
        idx = SqliteTextIndex(str(tmp_path / "test.db"))
        async with idx:
            yield idx


@pytest.fixture
async def vector_index(backend, tmp_path):
    if backend == "postgres":
        from memv.storage.postgres import VectorIndex as PgVectorIndex

        pool = await _create_pg_pool()
        idx = PgVectorIndex(pool, dimensions=4)
        await idx.open()
        yield idx
        await _cleanup_pg(pool)
    else:
        idx = SqliteVectorIndex(str(tmp_path / "test.db"), dimensions=4)
        try:
            await idx.open()
        except ImportError as e:
            pytest.skip(f"sqlite-vec extension not available: {e}")
        try:
            yield idx
        finally:
            await idx.close()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_message(user_id="user1", role=MessageRole.USER, content="hello", sent_at=None):
    return Message(
        user_id=user_id,
        role=role,
        content=content,
        sent_at=sent_at or datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


def make_episode(user_id="user1", title="Test Episode", content="A test episode.", start_time=None, end_time=None):
    return Episode(
        user_id=user_id,
        title=title,
        content=content,
        original_messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        start_time=start_time or datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        end_time=end_time or datetime(2024, 6, 15, 13, 0, 0, tzinfo=timezone.utc),
        created_at=datetime(2024, 6, 15, 13, 0, 0, tzinfo=timezone.utc),
    )


def make_knowledge(
    episode_id=None,
    user_id="user1",
    statement="User likes Python",
    embedding=None,
    valid_at=None,
    invalid_at=None,
    expired_at=None,
    superseded_by=None,
):
    return SemanticKnowledge(
        user_id=user_id,
        statement=statement,
        source_episode_id=episode_id or uuid4(),
        embedding=embedding,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        superseded_by=superseded_by,
        created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Mock LLM & Embedding clients
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock LLM client with sequential canned responses.

    Usage:
        llm = MockLLM()
        llm.set_responses("generate", ["response1", "response2"])
        llm.set_responses("generate_structured", [obj1, obj2])
    """

    def __init__(self):
        self._responses: dict[str, list] = {"generate": [], "generate_structured": []}
        self._call_index: dict[str, int] = {"generate": 0, "generate_structured": 0}
        self.calls: dict[str, list] = {"generate": [], "generate_structured": []}

    def set_responses(self, method: str, responses: list) -> None:
        self._responses[method] = responses
        self._call_index[method] = 0

    async def generate(self, prompt: str) -> str:
        self.calls["generate"].append(prompt)
        idx = self._call_index["generate"]
        responses = self._responses["generate"]
        if idx >= len(responses):
            raise RuntimeError(f"MockLLM.generate: no response at index {idx} (have {len(responses)})")
        self._call_index["generate"] = idx + 1
        return responses[idx]

    async def generate_structured(self, prompt: str, response_model: type):
        self.calls["generate_structured"].append((prompt, response_model))
        idx = self._call_index["generate_structured"]
        responses = self._responses["generate_structured"]
        if idx >= len(responses):
            raise RuntimeError(f"MockLLM.generate_structured: no response at index {idx} (have {len(responses)})")
        self._call_index["generate_structured"] = idx + 1
        return responses[idx]


class MockEmbedder:
    """Mock embedding client using SHA-256 hash → deterministic unit vector.

    Same text → identical vector (cosine sim = 1.0).
    Different text → near-orthogonal vector (sim ≈ 0).
    """

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self.calls: list[str | list[str]] = []

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Expand hash to fill dimensions (repeat hash bytes as needed)
        needed_bytes = self.dimensions * 4  # 4 bytes per float
        expanded = digest * (needed_bytes // len(digest) + 1)
        expanded = expanded[:needed_bytes]
        raw = struct.unpack(f"{self.dimensions}f", expanded)
        # Normalize to unit vector
        norm = sum(x * x for x in raw) ** 0.5
        if norm == 0:
            return [0.0] * self.dimensions
        return [x / norm for x in raw]

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._hash_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._hash_to_vector(t) for t in texts]


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dimensions=1536)


@pytest.fixture
async def pipeline_stores(tmp_path):
    """All 5 stores on a single temp DB for pipeline/e2e tests (sqlite only)."""
    db_path = str(tmp_path / "pipeline.db")
    messages = SqliteMessageStore(db_path)
    episodes = SqliteEpisodeStore(db_path)
    knowledge = SqliteKnowledgeStore(db_path)
    text_idx = SqliteTextIndex(db_path)
    vec_idx = SqliteVectorIndex(db_path, dimensions=1536)

    await messages.open()
    await episodes.open()
    await knowledge.open()
    await text_idx.open()
    try:
        await vec_idx.open()
    except ImportError:
        await text_idx.close()
        await knowledge.close()
        await episodes.close()
        await messages.close()
        pytest.skip("sqlite-vec extension not available")

    yield {
        "db_path": db_path,
        "messages": messages,
        "episodes": episodes,
        "knowledge": knowledge,
        "text_index": text_idx,
        "vector_index": vec_idx,
    }

    await vec_idx.close()
    await text_idx.close()
    await knowledge.close()
    await episodes.close()
    await messages.close()
