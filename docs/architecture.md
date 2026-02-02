# Architecture

## Data Flow

```
User messages → add_exchange() → Message buffer
                                      ↓
                               process()
                                      ↓
                    Episode segmentation (BoundaryDetector)
                                      ↓
                    Episode generation (EpisodeGenerator)
                                      ↓
                    Predict-calibrate extraction
                                      ↓
                         SemanticKnowledge stored
                                      ↓
                    retrieve() → Hybrid search (vector + BM25)
                                      ↓
                    RetrievalResult.to_prompt() → LLM context
```

## Processing Pipeline

### 1. Message Storage

Messages are stored immediately via `add_message()` or `add_exchange()`. They accumulate until `process()` is called.

### 2. Episode Segmentation

`BoundaryDetector` analyzes message sequences to find semantic boundaries:
- Topic shifts
- Intent changes
- Time gaps

Each boundary creates a new episode. This groups related messages for coherent knowledge extraction.

### 3. Episode Generation

`EpisodeGenerator` transforms message groups into structured episodes:
- **Title**: Concise episode label
- **Content**: Third-person narrative summary of what happened

Episodes are indexed for retrieval alongside extracted knowledge.

### 4. Predict-Calibrate Extraction

The core innovation. Traditional extraction asks "what facts are here?" and extracts everything. Predict-calibrate asks "what did I fail to predict?"

**Process:**
1. Retrieve existing knowledge relevant to the episode
2. Given existing knowledge, predict what the episode should contain
3. Compare prediction to actual episode content
4. Extract only what was unpredicted (genuinely novel information)

This naturally focuses on:
- **New facts** - Information not previously known
- **Updates** - Changes to existing knowledge
- **Contradictions** - Conflicts with prior beliefs

Without explicit importance scoring, importance emerges from prediction error.

### 5. Episode Merging (Optional)

When `enable_episode_merging=True`, newly created episodes are compared against recent episodes. If similarity exceeds `merge_similarity_threshold`, episodes are merged to reduce redundancy.

### 6. Knowledge Deduplication (Optional)

When `enable_knowledge_dedup=True`, extracted knowledge is compared against existing knowledge. Semantically similar statements (above `knowledge_dedup_threshold`) are deduplicated.

### 7. Knowledge Storage

Extracted `SemanticKnowledge` entries are:
- Stored in `KnowledgeStore`
- Indexed in `VectorIndex` (sqlite-vec for similarity search)
- Indexed in `TextIndex` (FTS5 for keyword search)

## Storage Layer

All data lives in SQLite with specialized indices.

### Stores

- **MessageStore** - Raw messages with temporal queries
- **EpisodeStore** - Episode metadata and content
- **KnowledgeStore** - Extracted knowledge statements

### Indices

- **VectorIndex** - sqlite-vec for embedding similarity search
- **TextIndex** - FTS5 for BM25 keyword search

Each index exists separately for knowledge and episodes, enabling targeted retrieval.

### Patterns

- UUID primary keys stored as TEXT
- Datetimes as Unix timestamps (INTEGER)
- Complex fields (original_messages, embeddings) as JSON
- Async context managers for connection management
- Transaction support via `async with store.transaction()`

### Bi-Temporal Validity

Knowledge entries track two timelines:
- **Event time** (`valid_at`/`invalid_at`): When the fact was/is true in the world
- **Transaction time** (`created_at`/`expired_at`): When we learned/recorded it

This enables point-in-time queries ("what did we know on date X?") and fact lifecycle tracking.

### Embedding Cache

Optional LRU cache with TTL for query embeddings. Reduces API calls for repeated or similar queries.

## Retrieval

`Retriever` implements hybrid search combining:

1. **Vector similarity** - Semantic search via embeddings
2. **BM25 text search** - Keyword matching via FTS5

Results are merged using Reciprocal Rank Fusion (RRF, k=60):

```
RRF_score = Σ 1 / (k + rank_i)
```

This balances semantic understanding with exact keyword matches.

### Query Flow

```
Query
  ↓
Embed query → Vector search (knowledge + episodes)
  ↓
Query text → BM25 search (knowledge + episodes)
  ↓
RRF fusion → Ranked results
  ↓
RetrievalResult
```

### Output Formatting

`RetrievalResult.to_prompt()` groups knowledge by source episode:

```markdown
## Relevant Context

### User's Work Background
_The user discussed their job at Anthropic during an initial introduction._

Key facts:
- The user works at Anthropic as a researcher
- Their focus area is AI safety, specifically interpretability

### Additional Facts
- The user prefers Python for data analysis
```

## Module Structure

```
src/memvee/
├── __init__.py              # Public exports
├── config.py                # MemoryConfig dataclass
├── models.py                # Data models (Message, Episode, etc.)
├── protocols.py             # EmbeddingClient, LLMClient protocols
├── cache.py                 # Embedding cache (LRU with TTL)
├── memory/                  # Memory class (main API)
│   ├── memory.py            # Memory class
│   ├── _api.py              # Public method implementations
│   ├── _lifecycle.py        # Connection/component lifecycle
│   ├── _pipeline.py         # Processing pipeline
│   └── _task_manager.py     # Async task management
├── embeddings/              # Embedding adapters
│   └── openai.py            # OpenAI embeddings
├── llm/                     # LLM adapters
│   └── pydantic_ai.py       # PydanticAI multi-provider
├── processing/              # Processing components
│   ├── boundary.py          # BoundaryDetector
│   ├── batch_segmenter.py   # Batch message segmentation
│   ├── episodes.py          # EpisodeGenerator
│   ├── episode_merger.py    # Merge similar episodes
│   ├── extraction.py        # PredictCalibrateExtractor
│   └── prompts.py           # LLM prompt templates
├── retrieval/               # Retrieval components
│   └── retriever.py         # Hybrid search with RRF
└── storage/                 # Storage layer
    └── sqlite/              # SQLite implementation
        ├── _base.py         # Base store class
        ├── _messages.py     # MessageStore
        ├── _episodes.py     # EpisodeStore
        ├── _knowledge.py    # KnowledgeStore
        ├── _vector_index.py # VectorIndex (sqlite-vec)
        └── _text_index.py   # TextIndex (FTS5)
```
