# Getting Started

## Prerequisites

- Python 3.13+
- OpenAI API key (for default adapters)

## Installation

```bash
pip install memvee
```

For development:

```bash
git clone https://github.com/vstorm-co/agentmemory.git
cd agentmemory
uv sync
```

## Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

## First Example

```python
import asyncio
from memvee import Memory
from memvee.embeddings import OpenAIEmbedAdapter
from memvee.llm import PydanticAIAdapter


async def main():
    # Initialize memory with adapters
    memory = Memory(
        db_path="memory.db",
        embedding_client=OpenAIEmbedAdapter(),
        llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
    )

    async with memory:
        user_id = "user-123"

        # Add a conversation exchange
        await memory.add_exchange(
            user_id=user_id,
            user_message="I work at Anthropic as a researcher.",
            assistant_message="That's great! What area do you focus on?",
        )
        await memory.add_exchange(
            user_id=user_id,
            user_message="AI safety, specifically interpretability.",
            assistant_message="Fascinating field!",
        )

        # Process messages into episodes and extract knowledge
        count = await memory.process(user_id)
        print(f"Extracted {count} knowledge entries")

        # Retrieve relevant context for a query
        result = await memory.retrieve("What does the user do for work?", user_id=user_id)
        print(result.to_prompt())


if __name__ == "__main__":
    asyncio.run(main())
```

## Using Different Providers

### Anthropic

```python
from memvee.llm import PydanticAIAdapter

llm = PydanticAIAdapter("anthropic:claude-3-5-sonnet-latest")
```

Requires `ANTHROPIC_API_KEY` environment variable.

### Google

```python
llm = PydanticAIAdapter("google-gla:gemini-2.5-flash")
```

Requires `GOOGLE_API_KEY` environment variable.

### Groq

```python
llm = PydanticAIAdapter("groq:llama-3.3-70b-versatile")
```

Requires `GROQ_API_KEY` environment variable.

See [PydanticAI models](https://ai.pydantic.dev/models/) for the full list.

## Custom Embedding Provider

Implement the `EmbeddingClient` protocol:

```python
from memvee.protocols import EmbeddingClient


class MyEmbedder:
    async def embed(self, text: str) -> list[float]:
        # Return embedding vector
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Return list of embedding vectors
        ...


memory = Memory(
    db_path="memory.db",
    embedding_client=MyEmbedder(),
    llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
)
```

## Custom LLM Provider

Implement the `LLMClient` protocol:

```python
from memvee.protocols import LLMClient
from typing import TypeVar

T = TypeVar("T")


class MyLLM:
    async def generate(self, prompt: str) -> str:
        # Return text response
        ...

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        # Return instance of response_model (Pydantic model)
        ...


memory = Memory(
    db_path="memory.db",
    embedding_client=OpenAIEmbedAdapter(),
    llm_client=MyLLM(),
)
```

## Agent Integration Pattern

Typical pattern for integrating memory into a conversational agent:

```python
class MemoryAgent:
    def __init__(self, memory: Memory):
        self.memory = memory
        self.user_id = "default-user"
        self.exchange_count = 0
        self.process_every = 5  # Process memory every N exchanges

    async def chat(self, user_message: str) -> str:
        # 1. Retrieve relevant context
        context = await self.memory.retrieve(user_message, user_id=self.user_id, top_k=5)

        # 2. Build prompt with memory context
        system_prompt = f"""You are a helpful assistant.

{context.to_prompt()}

Use the context to personalize responses."""

        # 3. Generate response (using your LLM of choice)
        assistant_message = await self.generate_response(system_prompt, user_message)

        # 4. Store the exchange
        await self.memory.add_exchange(
            user_id=self.user_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

        # 5. Periodically process to extract knowledge
        self.exchange_count += 1
        if self.exchange_count % self.process_every == 0:
            await self.memory.process(self.user_id)

        return assistant_message
```

See `examples/agent_integration.py` for a complete working example.

## Non-Blocking Processing

For long conversations, use `process_async()` to avoid blocking:

```python
# Start processing in background
task = memory.process_async(user_id)

# Do other work...
await some_other_operation()

# Wait for result when needed
count = await task.wait()
print(f"Extracted {count} knowledge entries")

# Or check status without waiting
if task.done:
    print(f"Status: {task.status}, Count: {task.knowledge_count}")
```
