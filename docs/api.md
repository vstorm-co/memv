# API Reference

## Memory

Main entry point for memvee. Coordinates storage, processing, and retrieval.

```python
from memvee import Memory

memory = Memory(
    db_path: str | None = None,                    # SQLite path (default: .db/memory.db)
    embedding_client: EmbeddingClient | None,      # Embedding provider
    llm_client: LLMClient | None = None,           # Required for processing
    embedding_dimensions: int | None = None,       # Vector dimensions (default: 1536)
    config: MemoryConfig | None = None,            # Full config object
    # Auto-processing
    auto_process: bool | None = None,              # Enable auto background processing
    batch_threshold: int | None = None,            # Messages before auto-trigger
    max_retries: int | None = None,                # Retry attempts on failure
    # Segmentation
    segmentation_threshold: int | None = None,     # Max messages per episode
    time_gap_minutes: int | None = None,           # Time gap for episode boundary
    # Deduplication
    enable_knowledge_dedup: bool | None = None,    # Deduplicate similar knowledge
    knowledge_dedup_threshold: float | None = None,# Similarity threshold (0-1)
    # Episode merging
    enable_episode_merging: bool | None = None,    # Merge similar episodes
    merge_similarity_threshold: float | None = None,
    # Predict-calibrate
    max_statements_for_prediction: int | None = None,
    # Embedding cache
    enable_embedding_cache: bool | None = None,
    embedding_cache_size: int | None = None,
    embedding_cache_ttl_seconds: int | None = None,
)
```

Individual params override values in `config`. See `MemoryConfig` for defaults.

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

#### `retrieve(query, user_id, top_k=10, vector_weight=0.5, include_episodes=True) -> RetrievalResult`

Retrieve relevant knowledge and episodes for a query.

```python
result = await memory.retrieve(
    query="What does the user do?",
    user_id="user-123",        # Required: filter to this user only
    top_k=10,                  # Results per category
    vector_weight=0.5,         # Balance: 1.0=vector only, 0.0=text only
    include_episodes=True,     # Include episodes in results
)
```

#### `process_messages(messages: list[Message], user_id: str) -> int`

Lower-level method for explicit control over what gets processed. Groups messages into one episode.

```python
count = await memory.process_messages(messages, "user-123")
```

#### `flush(user_id: str) -> int`

Force processing of buffered messages regardless of threshold. Waits for completion.

```python
count = await memory.flush("user-123")
```

#### `wait_for_processing(user_id: str, timeout: float | None = None) -> int`

Wait for background processing to complete.

```python
count = await memory.wait_for_processing("user-123", timeout=30)
```

#### `clear_user(user_id: str) -> dict[str, int]`

Delete all data for a user: messages, episodes, knowledge, and indices.

```python
counts = await memory.clear_user("user-123")
# Returns: {"messages": 42, "episodes": 3, "knowledge": 15, ...}
```

---

## Models

### Message

```python
from memvee import Message, MessageRole

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
from memvee import MessageRole

MessageRole.USER
MessageRole.ASSISTANT
MessageRole.SYSTEM
```

### Episode

Conversation segment with title and content summary.

```python
from memvee import Episode

episode = Episode(
    id: UUID,                      # Auto-generated
    user_id: str,                  # User identifier
    title: str,                    # Episode title
    content: str,                  # Third-person narrative summary
    original_messages: list[dict], # Raw messages stored on the episode
    start_time: datetime,          # Episode start (UTC)
    end_time: datetime,            # Episode end (UTC)
    created_at: datetime,          # When created (UTC)
)

episode.message_count  # Number of messages in episode
```

### SemanticKnowledge

Extracted fact with embedding and bi-temporal validity.

```python
from memvee import SemanticKnowledge

knowledge = SemanticKnowledge(
    id: UUID,                          # Auto-generated
    statement: str,                    # Declarative statement
    source_episode_id: UUID,           # Source episode
    created_at: datetime,              # When extracted (UTC)
    importance_score: float | None,    # Confidence/importance
    embedding: list[float] | None,     # Vector embedding
    # Bi-temporal validity
    valid_at: datetime | None,         # When fact became true (None = unknown)
    invalid_at: datetime | None,       # When fact stopped being true (None = still true)
    expired_at: datetime | None,       # When record was superseded (None = current)
)

# Methods
knowledge.is_current()               # Check if this is the current record
knowledge.is_valid_at(event_time)    # Check if fact was true at given time
knowledge.invalidate()               # Mark as superseded
```

### RetrievalResult

Container for retrieval results.

```python
from memvee import RetrievalResult

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
from memvee import ProcessTask, ProcessStatus

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
from memvee import ProcessStatus

ProcessStatus.PENDING
ProcessStatus.RUNNING
ProcessStatus.COMPLETED
ProcessStatus.FAILED
```

### ExtractedKnowledge

Output of predict-calibrate extraction (internal but exported).

```python
from memvee import ExtractedKnowledge

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
from memvee.protocols import EmbeddingClient


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
from memvee.protocols import LLMClient
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

## MemoryConfig

Centralized configuration with sensible defaults.

```python
from memvee import MemoryConfig

config = MemoryConfig(
    # Database
    db_path: str = ".db/memory.db",
    embedding_dimensions: int = 1536,
    # Processing triggers
    auto_process: bool = False,
    batch_threshold: int = 10,
    max_retries: int = 1,
    # Segmentation
    segmentation_threshold: int = 20,
    time_gap_minutes: int = 30,
    # Episode merging
    enable_episode_merging: bool = True,
    merge_similarity_threshold: float = 0.9,
    # Knowledge deduplication
    enable_knowledge_dedup: bool = True,
    knowledge_dedup_threshold: float = 0.8,
    # Predict-calibrate
    max_statements_for_prediction: int = 10,
    # Retrieval
    search_top_k_episodes: int = 10,
    search_top_k_knowledge: int = 10,
    # Embedding cache
    enable_embedding_cache: bool = True,
    embedding_cache_size: int = 1000,
    embedding_cache_ttl_seconds: int = 600,
)

memory = Memory(config=config, embedding_client=embedder, llm_client=llm)
```

---

## Built-in Adapters

### OpenAIEmbedAdapter

```python
from memvee.embeddings import OpenAIEmbedAdapter

embedder = OpenAIEmbedAdapter(
    api_key: str | None = None,              # Uses OPENAI_API_KEY env var if None
    model: str = "text-embedding-3-small",
)
```

### PydanticAIAdapter

Multi-provider LLM adapter via PydanticAI.

```python
from memvee.llm import PydanticAIAdapter

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
