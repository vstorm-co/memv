<p align="center">
  <img src="docs/assets/banner.png" alt="memv" width="600">
</p>

<h1 align="center">memv</h1>

<p align="center">
  <i>/mɛm-viː/</i> · Structured, temporal memory for AI agents
</p>

<p align="center">
  <a href="https://vstorm-co.github.io/memv/">Docs</a> •
  <a href="https://vstorm-co.github.io/memv/getting-started/">Getting Started</a> •
  <a href="https://pypi.org/project/memvee/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/memvee/"><img src="https://img.shields.io/pypi/v/memvee.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

---

Most memory systems extract everything and rely on retrieval to filter it. memv extracts only what the model **failed to predict** — importance emerges from prediction error, not upfront scoring.

| Typical Approach | memv |
|------------------|------|
| Extract all facts upfront | Extract only what we **failed to predict** |
| Overwrite old facts | **Invalidate** with temporal bounds |
| Retrieve by similarity | **Hybrid** vector + BM25 + RRF |
| Timestamps only | **Bi-temporal**: event time + transaction time |

---

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
    db_url="memory.db",  # or "postgresql://user:pass@host/db"
    embedding_client=OpenAIEmbedAdapter(),
    llm_client=PydanticAIAdapter("openai:gpt-4o-mini"),
)

async with memory:
    await memory.add_exchange(
        user_id="user-123",
        user_message="I just started at Anthropic as a researcher.",
        assistant_message="Congrats! What's your focus area?",
    )

    await memory.process("user-123")

    result = await memory.retrieve("What does the user do?", user_id="user-123")
    print(result.to_prompt())
```

---

## Features

**Predict-Calibrate Extraction** · Only extracts what the model failed to predict. Based on [Nemori](https://arxiv.org/abs/2508.03341).

**Bi-Temporal Validity** · Track when facts were true (event time) vs when you learned them (transaction time). Based on [Graphiti](https://github.com/getzep/graphiti).

**Hybrid Retrieval** · Vector similarity + BM25 text search with Reciprocal Rank Fusion.

**Episode Segmentation** · Groups messages into coherent conversation episodes.

**Contradiction Handling** · New facts invalidate conflicting old facts. Full audit trail preserved.

**SQLite + PostgreSQL** · SQLite for local dev, PostgreSQL with pgvector for production. Set `db_url` to choose between them.

**Multiple Embedding Providers** · OpenAI, Voyage, Cohere, or local via fastembed. Dimensions detected from the adapter.

---

## Point-in-Time Queries

memv's bi-temporal model lets you query knowledge as of a specific point in time:

```python
from datetime import datetime

# What did we know about user's job in January 2024?
result = await memory.retrieve(
    "Where does user work?",
    user_id="user-123",
    at_time=datetime(2024, 1, 1),
)

# Show full history including superseded facts
result = await memory.retrieve(
    "Where does user work?",
    user_id="user-123",
    include_expired=True,
)
```

---

## Architecture

```
Messages → Episodes → Knowledge → Vector Index + Text Index
                                   (sqlite-vec / pgvector)  (FTS5 / tsvector)
```

1. Messages buffered until threshold
2. Segmented into coherent episodes
3. Predict what episode should contain (given existing KB)
4. Compare prediction vs actual — extract the gaps
5. Store with embeddings + temporal bounds

---

## Framework Integration

```python
class MyAgent:
    def __init__(self, memory: Memory):
        self.memory = memory

    async def run(self, user_input: str, user_id: str) -> str:
        context = await self.memory.retrieve(user_input, user_id=user_id)
        response = await self.llm.generate(
            f"{context.to_prompt()}\n\nUser: {user_input}"
        )
        await self.memory.add_exchange(user_id, user_input, response)
        return response
```

See [Examples](https://vstorm-co.github.io/memv/examples/) for PydanticAI, LangGraph, LlamaIndex, CrewAI, and AutoGen integrations.

---

## Documentation

- [Getting Started](https://vstorm-co.github.io/memv/getting-started/) — First example and agent pattern
- [Core Concepts](https://vstorm-co.github.io/memv/concepts/) — Predict-calibrate, episodes, bi-temporal, retrieval
- [Backends](https://vstorm-co.github.io/memv/advanced/backends/sqlite/) — SQLite and PostgreSQL setup
- [API Reference](https://vstorm-co.github.io/memv/api/) — All public classes and methods

---

## Contributing

```bash
git clone https://github.com/vstorm-co/memv.git
cd memv
make install
make all
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

MIT — see [LICENSE](LICENSE)

<p align="center">
  <sub>Built by <a href="https://github.com/vstorm-co">vstorm</a></sub>
</p>
