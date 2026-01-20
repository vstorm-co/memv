from datetime import datetime, timezone

import pytest

from agent_memory.memory import Memory
from agent_memory.models import Message, MessageRole


class FakeEmbedder:
    """Fake embedder for testing."""

    async def embed(self, text: str) -> list[float]:
        hash_val = hash(text) % 1000
        embedding = [float(hash_val + i) for i in range(384)]
        magnitude = sum(x**2 for x in embedding) ** 0.5
        return [x / magnitude for x in embedding]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.mark.asyncio
async def test_memory_open_close():
    """Memory opens and closes cleanly."""
    memory = Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
    )

    await memory.open()
    assert memory._is_open

    await memory.close()
    assert not memory._is_open


@pytest.mark.asyncio
async def test_memory_context_manager():
    """Memory works as async context manager."""
    async with Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
    ) as memory:
        assert memory._is_open

    assert not memory._is_open


@pytest.mark.asyncio
async def test_memory_add_message():
    """Can add a message."""
    async with Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
    ) as memory:
        msg = Message(
            user_id="user-123",
            role=MessageRole.USER,
            content="Hello, world!",
            sent_at=datetime.now(timezone.utc),
        )

        await memory.add_message(msg)

        # Verify it's stored
        stored = await memory._messages.get(msg.id)
        assert stored is not None
        assert stored.content == "Hello, world!"


@pytest.mark.asyncio
async def test_memory_add_exchange():
    """Can add a user/assistant exchange."""
    async with Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
    ) as memory:
        user_msg, assistant_msg = await memory.add_exchange(
            user_id="user-123",
            user_message="What's the weather?",
            assistant_message="It's sunny today.",
        )

        assert user_msg.role == MessageRole.USER
        assert assistant_msg.role == MessageRole.ASSISTANT

        # Verify both stored
        assert await memory._messages.get(user_msg.id) is not None
        assert await memory._messages.get(assistant_msg.id) is not None


@pytest.mark.asyncio
async def test_memory_retrieve_empty():
    """Retrieve on empty memory returns empty results."""
    async with Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
    ) as memory:
        result = await memory.retrieve("test query")

        assert len(result.retrieved_knowledge) == 0


@pytest.mark.asyncio
async def test_memory_not_open_raises():
    """Operations on closed memory raise error."""
    memory = Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
    )

    with pytest.raises(RuntimeError, match="not open"):
        await memory.add_message(
            Message(
                user_id="user-123",
                role=MessageRole.USER,
                content="test",
                sent_at=datetime.now(timezone.utc),
            )
        )


@pytest.mark.asyncio
async def test_memory_process_requires_llm():
    """Process raises if no LLM client provided."""
    async with Memory(
        db_path=":memory:",
        embedding_client=FakeEmbedder(),
        embedding_dimensions=384,
        llm_client=None,
    ) as memory:
        with pytest.raises(RuntimeError, match="LLM client required"):
            await memory.process(user_id="test-user")
