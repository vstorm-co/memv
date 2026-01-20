# API Reference

## Memory

Main entry point for agent_memory. Coordinates storage, processing, and retrieval.

```python
from agent_memory import Memory

memory = Memory(
    db_path: str,                           # SQLite database path
    embedding_client: EmbeddingClient,      # Embedding provider
    llm_client: LLMClient | None = None,    # Required for processing, optional for retrieval-only
    embedding_dimensions: int = 1536,       # OpenAI text-embedding-3-small default
)
```

### Context Manager

```python
async with memory:
    # Memory is open and ready
    ...
# Automatically closed
```

Or manually:

```python
await memory.open()
# ...
await memory.close()
```

### Methods

#### `add_message(message: Message) -> None`

Add a single message to memory. Call `process()` to extract knowledge.

```python
await memory.add_message(Message(
    user_id="user-123",
    role=MessageRole.USER,
    content="Hello",
    sent_at=datetime.now(timezone.utc),
))
```

#### `add_exchange(user_id, user_message, assistant_message, timestamp=None) -> tuple[Message, Message]`

Convenience method to add a user/assistant exchange. Returns the created Message objects.

```python
user_msg, assistant_msg = await memory.add_exchange(
    user_id="user-123",
    user_message="What's the weather?",
    assistant_message="It's sunny today!",
    timestamp=datetime.now(timezone.utc),  # Optional, defaults to now
)
```

#### `process(user_id: str) -> int`

Process unprocessed messages for a user. Extracts knowledge via predict-calibrate. Returns number of knowledge entries extracted.

```python
count = await memory.process("user-123")
```

#### `process_async(user_id: str) -> ProcessTask`

Non-blocking process. Returns handle to monitor/await.

```python
task = memory.process_async("user-123")
# ... do other work ...
count = await task.wait()
```

#### `retrieve(query, top_k=10, vector_weight=0.5, include_episodes=True) -> RetrievalResult`

Retrieve relevant knowledge and episodes for a query.

```python
result = await memory.retrieve(
    query="What does the user do?",
    top_k=10,                  # Results per category
    vector_weight=0.5,         # Balance: 1.0=vector only, 0.0=text only
    include_episodes=True,     # Include episode narratives in results
)
```

#### `process_messages(messages: list[Message], user_id: str) -> int`

Lower-level method for explicit control over what gets processed. Groups messages into one episode.

```python
count = await memory.process_messages(messages, "user-123")
```

---

## Models

### Message

```python
from agent_memory import Message, MessageRole

message = Message(
    id: UUID,              # Auto-generated
    user_id: str,          # User identifier
    role: MessageRole,     # USER, ASSISTANT, or SYSTEM
    content: str,          # Message text
    sent_at: datetime,     # When sent (UTC)
)
```

### MessageRole

```python
from agent_memory import MessageRole

MessageRole.USER
MessageRole.ASSISTANT
MessageRole.SYSTEM
```

### Episode

Conversation segment with title and narrative.

```python
from agent_memory import Episode

episode = Episode(
    id: UUID,                  # Auto-generated
    user_id: str,              # User identifier
    message_ids: list[UUID],   # Messages in this episode
    title: str,                # Episode title
    narrative: str,            # Third-person narrative summary
    start_time: datetime,      # Episode start (UTC)
    end_time: datetime,        # Episode end (UTC)
    created_at: datetime,      # When created (UTC)
)
```

### SemanticKnowledge

Extracted fact with embedding.

```python
from agent_memory import SemanticKnowledge

knowledge = SemanticKnowledge(
    id: UUID,                          # Auto-generated
    statement: str,                    # Declarative statement
    source_episode_id: UUID,           # Source episode
    created_at: datetime,              # When extracted (UTC)
    importance_score: float | None,    # Confidence/importance
    embedding: list[float] | None,     # Vector embedding
)
```

### RetrievalResult

Container for retrieval results.

```python
from agent_memory import RetrievalResult

result = RetrievalResult(
    retrieved_knowledge: list[SemanticKnowledge],
    retrieved_episodes: list[Episode],
)
```

#### `to_prompt() -> str`

Format results for LLM context injection. Groups knowledge by source episode.

```python
context = result.to_prompt()
# Returns formatted markdown with episodes and facts
```

#### `as_text() -> str`

Simple text representation of knowledge statements only.

```python
text = result.as_text()
# Returns newline-separated statements
```

### ProcessTask

Handle for async processing.

```python
from agent_memory import ProcessTask, ProcessStatus

task = memory.process_async(user_id)

task.user_id: str              # User being processed
task.status: ProcessStatus     # PENDING, RUNNING, COMPLETED, FAILED
task.knowledge_count: int      # Extracted count (after completion)
task.error: str | None         # Error message if failed
task.done: bool                # True if COMPLETED or FAILED
```

#### `wait() -> int`

Wait for processing to complete. Returns knowledge count.

```python
count = await task.wait()
```

### ProcessStatus

```python
from agent_memory import ProcessStatus

ProcessStatus.PENDING
ProcessStatus.RUNNING
ProcessStatus.COMPLETED
ProcessStatus.FAILED
```

### ExtractedKnowledge

Output of predict-calibrate extraction (internal but exported).

```python
from agent_memory import ExtractedKnowledge

extracted = ExtractedKnowledge(
    statement: str,                                    # The knowledge statement
    knowledge_type: Literal["new", "update", "contradiction"],
    temporal_info: str | None,                         # e.g., "since January 2024"
    confidence: float = 1.0,
)
```

---

## Protocols

For implementing custom providers.

### EmbeddingClient

```python
from agent_memory.protocols import EmbeddingClient


class EmbeddingClient(Protocol):
    async def embed(self, text: str) -> list[float]:
        """Embed single text, return vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, return list of vectors."""
        ...
```

### LLMClient

```python
from agent_memory.protocols import LLMClient
from typing import TypeVar

T = TypeVar("T")


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str:
        """Generate unstructured text response."""
        ...

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate structured response matching Pydantic model."""
        ...
```

---

## Built-in Adapters

### OpenAIEmbedAdapter

```python
from agent_memory.embeddings import OpenAIEmbedAdapter

embedder = OpenAIEmbedAdapter(
    api_key: str | None = None,              # Uses OPENAI_API_KEY env var if None
    model: str = "text-embedding-3-small",
)
```

### PydanticAIAdapter

Multi-provider LLM adapter via PydanticAI.

```python
from agent_memory.llm import PydanticAIAdapter

llm = PydanticAIAdapter(
    model: str = "openai:gpt-4o-mini",
)
```

Supported model formats:
- `"openai:gpt-4o-mini"` - OpenAI (requires `OPENAI_API_KEY`)
- `"anthropic:claude-3-5-sonnet-latest"` - Anthropic (requires `ANTHROPIC_API_KEY`)
- `"google-gla:gemini-2.5-flash"` - Google (requires `GOOGLE_API_KEY`)
- `"groq:llama-3.3-70b-versatile"` - Groq (requires `GROQ_API_KEY`)

See [PydanticAI models](https://ai.pydantic.dev/models/) for the full list.
