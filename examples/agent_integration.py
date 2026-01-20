"""
Agent Integration Example
=========================

Shows how to integrate AgentMemory into a simple conversational agent.

The pattern:
1. User sends message
2. Retrieve relevant context from memory
3. Generate response with context
4. Store exchange in memory
5. Periodically process to extract knowledge

Run with:
    uv run python examples/agent_integration.py
"""

import asyncio
from datetime import datetime, timezone

from openai import AsyncOpenAI

from agent_memory import Memory
from agent_memory.embeddings import OpenAIEmbedAdapter
from agent_memory.llm import PydanticAIAdapter


class MemoryAgent:
    """Simple conversational agent with long-term memory."""

    def __init__(self, memory: Memory, openai_client: AsyncOpenAI):
        self.memory = memory
        self.openai = openai_client
        self.user_id = "default-user"
        self.exchange_count = 0
        self.process_every = 5  # Process memory every N exchanges

    async def chat(self, user_message: str) -> str:
        """Process a user message and return a response."""

        # 1. Retrieve relevant context from memory
        context = await self.memory.retrieve(user_message, top_k=5)
        context_prompt = context.to_prompt()

        # 2. Build system prompt with memory context
        system_prompt = f"""You are a helpful assistant with memory of past conversations.

{context_prompt}

Use the above context to personalize your responses. If you remember something about the user \
that's relevant, mention it naturally. Don't make up information that isn't in the context."""

        # 3. Generate response
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
        assistant_message = response.choices[0].message.content or ""

        # 4. Store exchange in memory
        await self.memory.add_exchange(
            user_id=self.user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=datetime.now(timezone.utc),
        )

        # 5. Periodically process to extract knowledge
        self.exchange_count += 1
        if self.exchange_count % self.process_every == 0:
            print(f"\n[Processing memory after {self.exchange_count} exchanges...]")
            count = await self.memory.process(self.user_id)
            print(f"[Extracted {count} knowledge entries]\n")

        return assistant_message


async def main():
    print("=" * 60)
    print("Memory Agent Demo")
    print("=" * 60)
    print("Chat with an agent that remembers your conversations.")
    print("Type 'quit' to exit, 'process' to force memory processing.")
    print("=" * 60)

    # Initialize memory
    memory = Memory(
        db_path=".db/agent_memory.db",
        embedding_client=OpenAIEmbedAdapter(),
        llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
    )

    async with memory:
        # Initialize agent
        openai_client = AsyncOpenAI()
        agent = MemoryAgent(memory, openai_client)

        # Chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower() == "process":
                count = await memory.process(agent.user_id)
                print(f"[Processed memory: {count} knowledge entries extracted]")
                continue

            if user_input.lower() == "debug":
                # Show what's in memory
                result = await memory.retrieve("*", top_k=10)
                print("\n[Memory contents]")
                print(result.to_prompt())
                continue

            response = await agent.chat(user_input)
            print(f"\nAssistant: {response}")

    print("\n[Session ended]")


if __name__ == "__main__":
    asyncio.run(main())
