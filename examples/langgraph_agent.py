"""
LangGraph Agent with memvee Memory
====================================

Shows how to integrate memvee into a LangGraph StateGraph for persistent memory.

The pattern:
1. Retrieve relevant context from memory before graph invocation
2. Inject context as a system message in the graph state
3. Store exchange after response
4. Processing happens automatically in background

Run with:
    uv run python examples/langgraph_agent.py

Requires:
    - OPENAI_API_KEY environment variable
    - pip install langgraph langchain-openai
"""

import asyncio
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from memvee import Memory
from memvee.embeddings import OpenAIEmbedAdapter
from memvee.llm import PydanticAIAdapter

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant with memory of past conversations. "
    "Use the provided context to personalize responses. "
    "Reference specific details you know about the user when relevant. "
    "Don't make up information that isn't in the context."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(llm: ChatOpenAI) -> StateGraph:
    """Build a minimal chatbot graph: START -> chatbot -> END."""

    def chatbot(state: State) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


class MemoryAgent:
    """LangGraph agent with persistent memory via memvee."""

    def __init__(self, memory: Memory, user_id: str = "default-user"):
        self.memory = memory
        self.user_id = user_id
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.graph = build_graph(self.llm)

    async def chat(self, user_message: str) -> str:
        """Process a user message and return a response."""
        # 1. Retrieve relevant context from memory
        result = await self.memory.retrieve(
            user_message,
            user_id=self.user_id,
            top_k=5,
        )

        # 2. Build messages with memory context as system message
        system_content = BASE_SYSTEM_PROMPT
        context_prompt = result.to_prompt()
        if context_prompt:
            system_content += f"\n\n{context_prompt}"

        messages = [
            ("system", system_content),
            ("user", user_message),
        ]

        # 3. Invoke graph
        response = await self.graph.ainvoke({"messages": messages})
        assistant_message = response["messages"][-1].content

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
    print("LangGraph Agent with memvee Memory")
    print("=" * 60)
    print("Commands: 'quit', 'flush' (force processing), 'debug' (show memory)")
    print("=" * 60)

    memory = Memory(
        db_path=".db/langgraph_agent.db",
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
