"""
AgentMemory - Structured, temporal memory for AI agents.

Example usage:
    from agent_memory import Memory, Message, MessageRole
    from agent_memory.embeddings import OpenAIEmbedAdapter
    from agent_memory.llm import PydanticAIAdapter

    memory = Memory(
        db_path="memory.db",
        embedding_client=OpenAIEmbedAdapter(),
        llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
    )

    async with memory:
        await memory.add_exchange(user_id, user_msg, assistant_msg)
        await memory.process(user_id)
        result = await memory.retrieve("query")
        print(result.to_prompt())
"""

from agent_memory.memory import Memory
from agent_memory.models import (
    Episode,
    ExtractedKnowledge,
    Message,
    MessageRole,
    ProcessStatus,
    ProcessTask,
    RetrievalResult,
    SemanticKnowledge,
)

__all__ = [
    "Memory",
    "Message",
    "MessageRole",
    "Episode",
    "SemanticKnowledge",
    "ExtractedKnowledge",
    "RetrievalResult",
    "ProcessTask",
    "ProcessStatus",
]


def main() -> None:
    print("AgentMemory - Structured, temporal memory for AI agents.")
    print("See examples/ for usage demos.")
