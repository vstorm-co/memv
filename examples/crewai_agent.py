"""
CrewAI Agent with memvee Memory
=================================

Shows how to integrate memvee into a CrewAI agent for persistent memory.

The pattern:
1. Retrieve relevant context from memory before each task
2. Inject context into agent backstory
3. Store exchange after task completion
4. Processing happens automatically in background

Run with:
    uv run python examples/crewai_agent.py

Requires:
    - OPENAI_API_KEY environment variable
    - pip install crewai
"""

import asyncio
from datetime import datetime, timezone

from crewai import Agent, Crew, Task

from memvee import Memory
from memvee.embeddings import OpenAIEmbedAdapter
from memvee.llm import PydanticAIAdapter

BASE_BACKSTORY = (
    "You are a helpful assistant with memory of past conversations. "
    "Use the provided context to personalize responses. "
    "Reference specific details you know about the user when relevant. "
    "Don't make up information that isn't in the context."
)


class MemoryAgent:
    """CrewAI agent with persistent memory via memvee."""

    def __init__(self, memory: Memory, user_id: str = "default-user"):
        self.memory = memory
        self.user_id = user_id

    async def chat(self, user_message: str) -> str:
        """Process a user message and return a response."""
        # 1. Retrieve relevant context from memory
        result = await self.memory.retrieve(
            user_message,
            user_id=self.user_id,
            top_k=5,
        )

        # 2. Build backstory with memory context
        backstory = BASE_BACKSTORY
        context_prompt = result.to_prompt()
        if context_prompt:
            backstory += f"\n\n{context_prompt}"

        # 3. Create agent and task with updated backstory
        agent = Agent(
            role="Personal Assistant",
            goal="Help the user with their questions using knowledge from past conversations.",
            backstory=backstory,
            llm="openai/gpt-4o-mini",
            verbose=False,
        )
        task = Task(
            description=user_message,
            expected_output="A helpful response to the user's message.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        # CrewAI's kickoff is synchronous, run in executor
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(None, crew.kickoff)
        assistant_message = str(crew_result)

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
    print("CrewAI Agent with memvee Memory")
    print("=" * 60)
    print("Commands: 'quit', 'flush' (force processing), 'debug' (show memory)")
    print("=" * 60)

    memory = Memory(
        db_path=".db/crewai_agent.db",
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
