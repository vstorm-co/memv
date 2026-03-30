<p align="center">
  <img src="assets/banner.png" alt="memv" width="600">
</p>

<h1 align="center">memv</h1>

<p align="center">
  <em>Structured, temporal memory for AI agents</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/memvee/"><img src="https://img.shields.io/pypi/v/memvee.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

---

**memv** (*/mɛm-viː/*) extracts knowledge from conversations using a predict-calibrate approach: importance emerges from prediction error, not upfront LLM scoring.

## Why memv?

| Typical Approach | memv |
|------------------|--------|
| Extract all facts upfront | Extract only what we **failed to predict** |
| Overwrite old facts | **Invalidate** with temporal bounds |
| Retrieve by similarity | **Hybrid** vector + BM25 + RRF |
| Timestamps only | **Bi-temporal**: event time + transaction time |

## Quick Start

```bash
uv add memvee
# or: pip install memvee
```

```python
from memv import Memory
from memv.embeddings import OpenAIEmbedAdapter
from memv.llm import PydanticAIAdapter

memory = Memory(
    db_url="memory.db",
    embedding_client=OpenAIEmbedAdapter(),
    llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
)

async with memory:
    # Store conversation
    await memory.add_exchange(
        user_id="user-123",
        user_message="I just started at Anthropic as a researcher.",
        assistant_message="Congrats! What's your focus area?",
    )

    # Extract knowledge
    await memory.process("user-123")

    # Retrieve context
    result = await memory.retrieve("What does the user do?", user_id="user-123")
    print(result.to_prompt())
```

## Core Capabilities

| Capability | Description |
|------------|-------------|
| **Predict-Calibrate** | Extract only what the model failed to predict ([Nemori](https://arxiv.org/abs/2508.03341)) |
| **Bi-Temporal** | Event time + transaction time for point-in-time queries ([Graphiti](https://github.com/getzep/graphiti)) |
| **Hybrid Retrieval** | Vector similarity + BM25 text search with RRF fusion |
| **Episode Segmentation** | Groups messages into coherent episodes |
| **Contradiction Handling** | New facts invalidate conflicting old facts, full audit trail preserved |
| **Async Processing** | Background processing via `process_async()` with configurable auto-trigger |
| **SQLite + PostgreSQL** | SQLite for local dev, PostgreSQL with pgvector for production |
| **Multiple Embeddings** | OpenAI, Voyage, Cohere, or local via fastembed. Dimensions detected from adapter |

## Architecture

```mermaid
flowchart TD
    M[Messages] --> E[Episodes]
    E --> K[Knowledge]
    K --> VI[Vector Index]
    K --> TI[Text Index]
    VI -.- S1[sqlite-vec / pgvector]
    TI -.- S2[FTS5 / tsvector]
```

See [Core Concepts](concepts/index.md) for details, or [Backends](advanced/backends/sqlite.md) for storage setup.

## Framework Integration

```python
class MyAgent:
    def __init__(self, memory: Memory):
        self.memory = memory

    async def run(self, user_input: str, user_id: str) -> str:
        # 1. Retrieve relevant context
        context = await self.memory.retrieve(user_input, user_id=user_id)

        # 2. Generate response with context
        response = await self.llm.generate(
            f"{context.to_prompt()}\n\nUser: {user_input}"
        )

        # 3. Store the exchange
        await self.memory.add_exchange(user_id, user_input, response)

        return response
```

See [Examples](examples/index.md) for integrations with PydanticAI, LangGraph, LlamaIndex, CrewAI, and AutoGen.

## Next Steps

- [Installation](installation.md) — Requirements and install
- [Getting Started](getting-started.md) — First example and agent pattern
- [Core Concepts](concepts/index.md) — Predict-calibrate, episodes, bi-temporal model
- [Backends](advanced/backends/sqlite.md) — SQLite and PostgreSQL setup
- [API Reference](api.md) — Complete API documentation
- [Examples](examples/index.md) — Framework integrations
