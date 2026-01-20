# AgentMemory

Structured, temporal memory for AI agents.

AgentMemory extracts and retrieves knowledge from conversations using a predict-calibrate approach: importance emerges from prediction error, not upfront LLM scoring. This means the system naturally focuses on genuinely novel information rather than extracting everything.

## Installation

```bash
pip install agent-memory
```

## Quick Start

```python
import asyncio
from agent_memory import Memory
from agent_memory.embeddings import OpenAIEmbedAdapter
from agent_memory.llm import PydanticAIAdapter


async def main():
    memory = Memory(
        db_path="memory.db",
        embedding_client=OpenAIEmbedAdapter(),
        llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
    )

    async with memory:
        user_id = "user-123"

        # Add conversation
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

        # Extract knowledge
        await memory.process(user_id)

        # Retrieve context for a new query
        result = await memory.retrieve("What does the user do for work?")
        print(result.to_prompt())


if __name__ == "__main__":
    asyncio.run(main())
```

Requires `OPENAI_API_KEY` environment variable.

## Features

- **Predict-calibrate extraction** - Only extracts what the model failed to predict, focusing on genuinely novel information
- **Episode segmentation** - Automatically groups messages into coherent conversation episodes
- **Hybrid retrieval** - Combines vector similarity and BM25 text search with Reciprocal Rank Fusion
- **Structured output** - `RetrievalResult.to_prompt()` formats context for LLM injection
- **Async processing** - Non-blocking `process_async()` for background knowledge extraction
- **Multi-provider LLM support** - OpenAI, Anthropic, Google, Groq via PydanticAI

## Documentation

- [Getting Started](docs/getting-started.md) - Installation, setup, first example
- [API Reference](docs/api.md) - All public classes and methods
- [Architecture](docs/architecture.md) - How it works internally

## License

MIT
