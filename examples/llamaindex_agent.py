"""
LlamaIndex Agent with memvee Memory
=====================================

Shows how to integrate memvee into a LlamaIndex chat engine for persistent memory.

The pattern:
1. Retrieve relevant context from memory before chat
2. Inject context as system prompt prefix
3. Store exchange after response
4. Processing happens automatically in background

Run with:
    uv run python examples/llamaindex_agent.py

Requires:
    - OPENAI_API_KEY environment variable
    - pip install llama-index llama-index-llms-openai
"""

import asyncio
from datetime import datetime, timezone

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI

from memvee import Memory
from memvee.embeddings import OpenAIEmbedAdapter
from memvee.llm import PydanticAIAdapter

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant with memory of past conversations. "
    "Use the provided context to personalize responses. "
    "Reference specific details you know about the user when relevant. "
    "Don't make up information that isn't in the context."
)


class MemoryAgent:
    """LlamaIndex chat agent with persistent memory via memvee."""

    def __init__(self, memory: Memory, user_id: str = "default-user"):
        self.memory = memory
        self.user_id = user_id
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    async def chat(self, user_message: str) -> str:
        """Process a user message and return a response."""
        # 1. Retrieve relevant context from memory
        result = await self.memory.retrieve(
            user_message,
            user_id=self.user_id,
            top_k=5,
        )

        # 2. Build system prompt with memory context
        system_prompt = BASE_SYSTEM_PROMPT
        context_prompt = result.to_prompt()
        if context_prompt:
            system_prompt += f"\n\n{context_prompt}"

        # 3. Create a fresh chat engine with updated system prompt
        chat_engine = SimpleChatEngine.from_defaults(
            llm=self.llm,
            system_prompt=system_prompt,
        )
        response = await chat_engine.achat(user_message)
        assistant_message = str(response)

        # 4. Store exchange in memory (processing triggers at threshold)
        await self.memory.add_exchange(
            user_id=self.user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=datetime.now(timezone.utc),
        )

        return assistant_message


async def main():
    print("=" * 60)
    print("LlamaIndex Agent with memvee Memory")
    print("=" * 60)
    print("Commands: 'quit', 'flush' (force processing), 'debug' (show memory)")
    print("=" * 60)

    memory = Memory(
        db_path=".db/llamaindex_agent.db",
        embedding_client=OpenAIEmbedAdapter(),
        llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
        auto_process=True,
        batch_threshold=10,
    )

    async with memory:
        agent = MemoryAgent(memory)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                count = await memory.flush(agent.user_id)
                if count > 0:
                    print(f"[Flushed {count} knowledge entries]")
                break

            if user_input.lower() == "flush":
                count = await memory.flush(agent.user_id)
                print(f"[Flushed: {count} knowledge entries extracted]")
                continue

            if user_input.lower() == "debug":
                result = await memory.retrieve("*", user_id=agent.user_id, top_k=10)
                print("\n[Memory contents]")
                print(result.to_prompt())
                continue

            response = await agent.chat(user_input)
            print(f"\nAssistant: {response}")

    print("\n[Session ended]")


if __name__ == "__main__":
    asyncio.run(main())
