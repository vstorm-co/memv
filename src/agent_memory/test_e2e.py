# test_e2e.py
import asyncio
from datetime import datetime, timezone

from agent_memory.embeddings.openai import OpenAIEmbedAdapter
from agent_memory.llm import PydanticAIAdapter
from agent_memory.memory import Memory
from agent_memory.models import Message, MessageRole


async def main():
    embedder = OpenAIEmbedAdapter()
    llm = PydanticAIAdapter("openai:gpt-4o-mini")

    async with Memory(
        db_path="test_memory.db",
        embedding_client=embedder,
        llm_client=llm,
    ) as memory:
        user_id = "test-user"
        now = datetime.now(timezone.utc)

        messages = [
            # Topic 1: Job
            Message(user_id=user_id, role=MessageRole.USER, content="I just started a new job at Anthropic!", sent_at=now),
            Message(
                user_id=user_id, role=MessageRole.ASSISTANT, content="Congratulations! What will you be doing there?", sent_at=now
            ),
            Message(user_id=user_id, role=MessageRole.USER, content="I'm working on AI safety research.", sent_at=now),
            # Topic 2: Cooking
            Message(
                user_id=user_id,
                role=MessageRole.USER,
                content="By the way, can you help me with a recipe? I want to make pasta carbonara tonight.",
                sent_at=now,
            ),
            Message(
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content="Sure! For carbonara you'll need eggs, pecorino, guanciale, and black pepper.",
                sent_at=now,
            ),
            Message(user_id=user_id, role=MessageRole.USER, content="I don't have guanciale, can I use bacon?", sent_at=now),
            # Topic 3: Travel
            Message(
                user_id=user_id,
                role=MessageRole.USER,
                content="On another note, I'm planning a trip to Japan next month.",
                sent_at=now,
            ),
            Message(
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content="Kyoto is beautiful! The temples and gardens are amazing.",
                sent_at=now,
            ),
        ]

        for msg in messages:
            await memory.add_message(msg)

        print("Processing...")
        count = await memory.process(user_id)
        print(f"Extracted {count} knowledge entries\n")

        # Test retrieval with episodes
        print("=" * 60)
        print("Query: 'What was the user cooking?'")
        print("=" * 60)
        result = await memory.retrieve("What was the user cooking?", top_k=5)
        print(result.to_prompt())

        print("\n" + "=" * 60)
        print("Query: 'Where does the user work?'")
        print("=" * 60)
        result = await memory.retrieve("Where does the user work?", top_k=5)
        print(result.to_prompt())


if __name__ == "__main__":
    asyncio.run(main())
