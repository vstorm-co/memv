from datetime import datetime, timezone
from uuid import uuid4

import pytest

from memv.models import Episode, Message, MessageRole, SemanticKnowledge
from memv.storage.sqlite._episodes import EpisodeStore
from memv.storage.sqlite._knowledge import KnowledgeStore
from memv.storage.sqlite._messages import MessageStore
from memv.storage.sqlite._text_index import TextIndex
from memv.storage.sqlite._vector_index import VectorIndex


@pytest.fixture
async def message_store(tmp_path):
    store = MessageStore(str(tmp_path / "test.db"))
    async with store:
        yield store


@pytest.fixture
async def episode_store(tmp_path):
    store = EpisodeStore(str(tmp_path / "test.db"))
    async with store:
        yield store


@pytest.fixture
async def knowledge_store(tmp_path):
    store = KnowledgeStore(str(tmp_path / "test.db"))
    async with store:
        yield store


@pytest.fixture
async def text_index(tmp_path):
    idx = TextIndex(str(tmp_path / "test.db"))
    async with idx:
        yield idx


@pytest.fixture
async def vector_index(tmp_path):
    idx = VectorIndex(str(tmp_path / "test.db"), dimensions=4)
    try:
        await idx.open()
    except ImportError as e:
        pytest.skip(f"sqlite-vec extension not available: {e}")
    try:
        yield idx
    finally:
        await idx.close()


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


def make_knowledge(episode_id=None, statement="User likes Python", embedding=None, valid_at=None, invalid_at=None, expired_at=None):
    return SemanticKnowledge(
        statement=statement,
        source_episode_id=episode_id or uuid4(),
        embedding=embedding,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
