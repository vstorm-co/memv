from datetime import datetime, timedelta, timezone
from sqlite3 import IntegrityError
from uuid import uuid4

import pytest

from agentmemory.models import Episode, Message, MessageRole, SemanticKnowledge
from agentmemory.storage.sqlite import SQLiteEpisodeStore, SQLiteMessageStore, SQLiteSemanticKnowledgeStore

MSG_1 = Message(user_id="user1", role=MessageRole.USER, content="Hello", sent_at=datetime.now(timezone.utc))
MSG_2 = Message(user_id="user2", role=MessageRole.ASSISTANT, content="Hi", sent_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_sqlite_message_storage():
    async with SQLiteMessageStore(":memory:") as storage:
        await storage.add(MSG_1)
        await storage.add(MSG_2)
        message1 = await storage.get(MSG_1.id)
        message2 = await storage.get(MSG_2.id)
    assert message1 is not None
    assert message2 is not None
    assert message1.id == MSG_1.id
    assert message2.id == MSG_2.id


@pytest.mark.asyncio
async def test_sqlite_message_storage_get_by_user_id():
    async with SQLiteMessageStore(":memory:") as storage:
        await storage.add(MSG_1)
        await storage.add(MSG_2)
        messages = await storage.get_by_user("user1")
    assert len(messages) == 1
    assert messages[0].id == MSG_1.id


@pytest.mark.asyncio
async def test_sqlite_message_storage_get_by_time_range():
    async with SQLiteMessageStore(":memory:") as storage:
        await storage.add(MSG_1)
        await storage.add(MSG_2)
        messages = await storage.get_by_time_range("user1", datetime.now(timezone.utc) - timedelta(days=1), datetime.now(timezone.utc))
    assert len(messages) == 1
    assert messages[0].id == MSG_1.id


EP_1 = Episode(
    user_id="user1",
    message_ids=[MSG_1.id],
    title="Episode 1",
    narrative="First episode",
    start_time=datetime.now(timezone.utc) - timedelta(hours=1),
    end_time=datetime.now(timezone.utc),
)
EP_2 = Episode(
    user_id="user2",
    message_ids=[MSG_2.id],
    title="Episode 2",
    narrative="Second episode",
    start_time=datetime.now(timezone.utc) - timedelta(hours=2),
    end_time=datetime.now(timezone.utc) - timedelta(hours=1),
)


@pytest.mark.asyncio
async def test_sqlite_episode_store():
    async with SQLiteEpisodeStore(":memory:") as storage:
        await storage.add(EP_1)
        await storage.add(EP_2)
        episode1 = await storage.get(EP_1.id)
        episode2 = await storage.get(EP_2.id)
    assert episode1 is not None
    assert episode2 is not None
    assert episode1.id == EP_1.id
    assert episode2.id == EP_2.id


@pytest.mark.asyncio
async def test_sqlite_episode_store_get_by_user():
    async with SQLiteEpisodeStore(":memory:") as storage:
        await storage.add(EP_1)
        await storage.add(EP_2)
        episodes = await storage.get_by_user("user1")
    assert len(episodes) == 1
    assert episodes[0].id == EP_1.id


@pytest.mark.asyncio
async def test_sqlite_episode_store_get_by_time_range():
    async with SQLiteEpisodeStore(":memory:") as storage:
        await storage.add(EP_1)
        await storage.add(EP_2)
        episodes = await storage.get_by_time_range("user1", datetime.now(timezone.utc) - timedelta(days=1), datetime.now(timezone.utc))
    assert len(episodes) == 1
    assert episodes[0].id == EP_1.id


SK_1 = SemanticKnowledge(
    statement="User prefers dark mode",
    source_episode_id=EP_1.id,
    importance_score=0.8,
)
SK_2 = SemanticKnowledge(
    statement="User is a software engineer",
    source_episode_id=EP_2.id,
    importance_score=0.9,
)


@pytest.mark.asyncio
async def test_semantic_knowledge_store():
    async with SQLiteSemanticKnowledgeStore(":memory:") as storage:
        await storage.add(SK_1)
        await storage.add(SK_2)
        knowledge1 = await storage.get(SK_1.id)
        knowledge2 = await storage.get(SK_2.id)
    assert knowledge1 is not None
    assert knowledge2 is not None
    assert knowledge1.id == SK_1.id
    assert knowledge2.id == SK_2.id


@pytest.mark.asyncio
async def test_semantic_knowledge_store_get_by_episode():
    async with SQLiteSemanticKnowledgeStore(":memory:") as storage:
        await storage.add(SK_1)
        await storage.add(SK_2)
        knowledge = await storage.get_by_episode(EP_1.id)
    assert len(knowledge) == 1
    assert knowledge[0].id == SK_1.id


# Edge case and error handling tests


@pytest.mark.asyncio
async def test_get_nonexistent_message():
    async with SQLiteMessageStore(":memory:") as storage:
        result = await storage.get(uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_get_nonexistent_episode():
    async with SQLiteEpisodeStore(":memory:") as storage:
        result = await storage.get(uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_get_nonexistent_knowledge():
    async with SQLiteSemanticKnowledgeStore(":memory:") as storage:
        result = await storage.get(uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_duplicate_id_raises_error():
    async with SQLiteMessageStore(":memory:") as storage:
        await storage.add(MSG_1)
        with pytest.raises(IntegrityError):
            await storage.add(MSG_1)


@pytest.mark.asyncio
async def test_empty_time_range_query():
    async with SQLiteMessageStore(":memory:") as storage:
        await storage.add(MSG_1)
        # Query range in the past, before any messages
        messages = await storage.get_by_time_range(
            "user1", datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(2000, 1, 2, tzinfo=timezone.utc)
        )
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_get_by_user_no_results():
    async with SQLiteMessageStore(":memory:") as storage:
        await storage.add(MSG_1)
        messages = await storage.get_by_user("nonexistent_user")
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_store_reopen():
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # First session: add data
        async with SQLiteMessageStore(db_path) as storage:
            await storage.add(MSG_1)

        # Second session: verify data persists
        async with SQLiteMessageStore(db_path) as storage:
            message = await storage.get(MSG_1.id)
        assert message is not None
        assert message.id == MSG_1.id
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_transaction_commits_atomically():
    msg_a = Message(user_id="user1", role=MessageRole.USER, content="A", sent_at=datetime.now(timezone.utc))
    msg_b = Message(user_id="user1", role=MessageRole.USER, content="B", sent_at=datetime.now(timezone.utc))

    async with SQLiteMessageStore(":memory:") as storage:
        async with storage.transaction():
            await storage.add(msg_a)
            await storage.add(msg_b)
        # Both should be committed
        messages = await storage.get_by_user("user1")
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_transaction_rollback_on_error():
    msg_a = Message(user_id="user1", role=MessageRole.USER, content="A", sent_at=datetime.now(timezone.utc))

    async with SQLiteMessageStore(":memory:") as storage:
        try:
            async with storage.transaction():
                await storage.add(msg_a)
                raise ValueError("Simulated error")
        except ValueError:
            pass
        # Should be rolled back
        messages = await storage.get_by_user("user1")
    assert len(messages) == 0
