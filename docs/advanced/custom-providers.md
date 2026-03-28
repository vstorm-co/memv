# Custom Providers

memv uses two protocols for external services: `EmbeddingClient` and `LLMClient`. Implement them to use your preferred provider.

## EmbeddingClient

```python
from memv.protocols import EmbeddingClient


class MyEmbedder:
    async def embed(self, text: str) -> list[float]:
        """Embed single text, return vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, return list of vectors."""
        ...
```

!!! tip "Automatic dimensions"
    Built-in adapters declare their output dimensions, and memv reads them directly. For custom adapters, add a `dimensions` attribute to your class or set `embedding_dimensions` in your config.

## LLMClient

```python
from memv.protocols import LLMClient
from typing import TypeVar

T = TypeVar("T")


class MyLLM:
    async def generate(self, prompt: str) -> str:
        """Generate unstructured text response."""
        ...

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate structured response matching Pydantic model."""
        ...
```

`generate_structured` must return an instance of the given Pydantic model. memv calls it during episode generation and knowledge extraction to get structured output.

### Example: Anthropic (direct)

```python
import anthropic
import json


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    async def generate(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"name": "output", "description": "Output", "input_schema": schema}],
            tool_choice={"type": "tool", "name": "output"},
        )
        data = response.content[0].input
        return response_model.model_validate(data)
```

## Built-in Embedding Adapters

### OpenAIEmbedAdapter

```bash
uv add memvee  # included by default
```

```python
from memv.embeddings import OpenAIEmbedAdapter

embedder = OpenAIEmbedAdapter()                              # text-embedding-3-small (1536 dims)
embedder = OpenAIEmbedAdapter(model="text-embedding-3-large")  # 3072 dims
```

### VoyageEmbedAdapter

```bash
uv add memvee[voyage]
```

```python
from memv.embeddings import VoyageEmbedAdapter

embedder = VoyageEmbedAdapter()                        # voyage-3-lite (1024 dims)
embedder = VoyageEmbedAdapter(model="voyage-3")        # voyage-3 (1024 dims)
```

### CohereEmbedAdapter

```bash
uv add memvee[cohere]
```

```python
from memv.embeddings import CohereEmbedAdapter

embedder = CohereEmbedAdapter()                        # embed-v4.0 (1024 dims)
embedder = CohereEmbedAdapter(model="embed-english-light-v3.0")  # 384 dims
```

### FastEmbedAdapter (local, no API key)

```bash
uv add memvee[local]
```

```python
from memv.embeddings import FastEmbedAdapter

embedder = FastEmbedAdapter()                          # BAAI/bge-small-en-v1.5 (384 dims)
embedder = FastEmbedAdapter(model="BAAI/bge-base-en-v1.5")  # 768 dims
```

Runs locally via ONNX runtime. Models download on first use.

## Built-in LLM Adapter

### PydanticAIAdapter

LLM adapter via PydanticAI. Supports OpenAI, Anthropic, Google, and Groq.

```python
from memv.llm import PydanticAIAdapter

llm = PydanticAIAdapter("openai:gpt-4o-mini")
llm = PydanticAIAdapter("anthropic:claude-3-5-sonnet-latest")
llm = PydanticAIAdapter("google-gla:gemini-2.5-flash")
llm = PydanticAIAdapter("groq:llama-3.3-70b-versatile")
```

See [PydanticAI models](https://ai.pydantic.dev/models/) for the full provider list.
