# agentmemory

Structured, temporal memory for AI agents.

## Overview

**Core principles:**
1. Framework-agnostic (works with any agent framework)
2. Pydantic-native (models, validation, serialization)
3. Storage-flexible (SQLite -> Postgres -> Neo4j)
4. Incrementally adoptable (simple -> sophisticated)

## Architecture

### Data Model

```
Messages (append-only archive)
    |
    v
Episodes (segmented conversation chunks with narratives)
    |
    v
Entities (resolved, deduplicated) ---- Facts (bi-temporal edges)
```

### Key Features

| Feature | Description |
|---------|-------------|
| Bi-temporal facts | Track when facts were true (event time) vs when we learned them (transaction time) |
| Edge invalidation | New contradicting facts invalidate old ones with temporal bounds |
| Predict-calibrate | Only extract knowledge that fails prediction (importance emerges from error) |
| Episode segmentation | Topic-based chunking, not arbitrary splits |
| Hybrid retrieval | Vector + BM25 + graph traversal |

---

## Key Architectural Insights

### Write-Time Disambiguation

Ablation studies on memory benchmarks show that **56.7% of temporal reasoning performance** comes from disambiguating facts at write-time rather than retrieval-time:

| Configuration | Temporal F1 | Impact |
|---------------|-------------|--------|
| Full system | 58.62% | baseline |
| Without atomization | 25.40% | -56.7% |
| Without consolidation | 55.10% | -6.0% |

This means:
- Temporal normalization: "yesterday" becomes "2023-07-01" at ingestion
- Coreference resolution: "my kids" becomes "Sarah's kids" when we know who's speaking
- Each fact should be self-contained and interpretable without conversation context

### Retrieval Saturation

With high-quality atomic entries, retrieval saturates quickly:

| k (retrieved items) | F1 | % of Peak |
|--------------------|-------|------------|
| k=1 | 35.20% | 81% |
| k=3 | 42.85% | 99% |
| k=10 | 43.45% | 100% |

k=3 achieves 99% of peak performance. Large context windows unnecessary when facts are well-formed.

### Two Valid Philosophies

**Write-time disambiguation approach:**
- Convert everything to atomic, self-contained facts at ingestion
- Pro: Faster retrieval, simpler reasoning, benchmark-proven
- Con: Loses conversational structure

**Preserve episodes approach:**
- Keep episodes intact, extract knowledge via predict-calibrate
- Pro: Retains narrative context, handles nuance better
- Con: More complex, slower

**Our hybrid approach:**
```python
async def process_messages(messages: list[Message]) -> None:
    # 1. Episode boundary detection
    episode = await detect_and_create_episode(messages)

    # 2. Atomization with temporal normalization
    atomic_facts = await atomize(
        episode.raw_messages,
        reference_time=messages[-1].timestamp
    )

    # 3. Predict-calibrate filtering
    novel_facts = await predict_calibrate_filter(
        atomic_facts,
        existing_knowledge
    )

    # 4. Store with provenance linking
    for fact in novel_facts:
        await store_fact(fact, source_episode_id=episode.id)
```

---

## Public API

```python
from agentmemory import Memory, Message

# Initialize
memory = Memory("sqlite:///memory.db")

# Store messages
await memory.add_message(Message(
    role="user",
    content="I just started a new job at Anthropic",
    timestamp=datetime.now(),
))

# Retrieve relevant context
context = await memory.retrieve(
    query="Where does the user work?",
    top_k=10,
)

# Retrieve historical state
context = await memory.retrieve(
    query="Where did the user work?",
    at_time=datetime(2024, 1, 1),
)

# Convenience: add full exchange
await memory.add_exchange(
    user_message="What's the weather?",
    assistant_message="It's sunny in SF.",
)

# Get formatted context for LLM
prompt_context = context.to_prompt()

# Process pending extractions
await memory.process()

# Cleanup
await memory.close()
```

### Framework Integration Pattern

```python
# Works with any agent framework
class MyAgent:
    def __init__(self, memory: Memory):
        self.memory = memory

    async def run(self, user_input: str) -> str:
        # 1. Retrieve
        context = await self.memory.retrieve(user_input)

        # 2. Generate
        response = await self.llm.generate(
            f"{context.to_prompt()}\n\nUser: {user_input}"
        )

        # 3. Store
        await self.memory.add_exchange(user_input, response)

        return response
```

---

## Design Decisions

### LLM Client

Own protocol with adapters (no hard dependency on any framework):

```python
class LLMClient(Protocol):
    async def generate(self, prompt: str, response_model: type[T]) -> T: ...

# Provided adapters
class PydanticAIAdapter(LLMClient): ...
class OpenAIAdapter(LLMClient): ...
class LiteLLMAdapter(LLMClient): ...
```

### Embedding Client

Same pattern:

```python
class EmbeddingClient(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

# Provided adapters
class FastEmbedAdapter(EmbeddingClient): ...  # Local, no API key
class OpenAIEmbedAdapter(EmbeddingClient): ...
class VoyageAdapter(EmbeddingClient): ...
```

### Async

Async-first. Sync wrappers for convenience:

```python
# Async (primary)
await memory.add_message(msg)

# Sync wrapper
memory.add_message_sync(msg)  # Uses asyncio.run internally
```

### Storage

Protocol + implementations:

```python
# Connection string determines backend
Memory("sqlite:///local.db")           # Local dev
Memory("postgresql://user:pass@host/db")  # Production
Memory("neo4j://user:pass@host:7687")     # Graph-heavy workloads
```

---

## Roadmap

Two releases:
- **v0.1.0** - Episodes + predict-calibrate + semantic KB
- **v0.2.0** - + Structured knowledge graph with bi-temporal edges

---

# v0.1.0 - Core Memory

Episodes, predict-calibrate extraction, semantic knowledge base (unstructured statements).
No entity graph, no bi-temporal, no contradiction handling yet.

### Phase 0: Project Setup + Data Models
**Status:** Done

- [x] Project scaffolding (pyproject.toml, ruff, pyright, pytest, CI/CD, pre-commit)
- [x] Core data models
  - [x] `Message`
  - [x] `Episode`
  - [x] `SemanticKnowledge` (unstructured statements)
  - [x] `RetrievalResult`
  - [x] `ExtractedKnowledge` (output of predict-calibrate)
  - [x] `ProcessTask` (async processing handle)

---

### Phase 1: Storage + Basic Retrieval
**Status:** Done

- [x] SQLite storage
  - [x] `SQLiteMessageStore`
  - [x] `SQLiteEpisodeStore`
  - [x] `SQLiteKnowledgeStore`
  - [x] Vector storage via sqlite-vec
  - [x] FTS5 for full-text
- [x] Extract protocols from implementations (`MessageStore`, `EpisodeStore`, `KnowledgeStore`, `EmbeddingClient`)
- [x] Embedding client implementations (`OpenAIEmbedAdapter`, `FastEmbedAdapter`)
- [x] Basic retrieval (vector similarity, top-k)
- [x] `Memory` class (`add_message()`, `add_exchange()`, `retrieve()`, `open()`, `close()`, context manager)

---

### Phase 2: Episodes + Hybrid Retrieval
**Status:** Done

- [x] Hybrid search (BM25 via FTS5, RRF with k=60, configurable `vector_weight`)
- [x] Episode segmentation
  - [x] `BoundaryDetector` (LLM-based)
  - [x] Signals: coherence, topic shift, intent shift, temporal markers
  - [x] Fallback: max buffer size (25 messages)
  - [x] Episode narrative generation via `EpisodeGenerator`
- [x] LLM client implementations (`PydanticAIAdapter`)
- [x] Episode-aware retrieval
  - [x] Search episodes via separate vector + text indices
  - [x] Auto-fetch source episodes for returned knowledge (provenance)
  - [x] `RetrievalResult.to_prompt()` groups knowledge by episode

**Note:** v0.1.0 uses batch segmentation by default (accumulate -> group all at once). Legacy incremental boundary detection available via `use_legacy_segmentation=True` but deprecated.

---

### Phase 3: Predict-Calibrate
**Status:** In progress

- [x] Prediction generation (retrieve relevant KB for episode, generate prediction)
- [x] Calibration (compare prediction vs raw messages, extract gaps)
- [x] Cold start handling (when `existing_knowledge` empty, skip prediction, use dedicated extraction prompt)
- [x] Semantic knowledge extraction (unstructured statements with embeddings, batch embedding)
- [ ] **Atomization step** - HIGH PRIORITY
  - Temporal normalization: resolve relative dates at write-time ("yesterday" -> absolute timestamp)
  - Coreference resolution: resolve pronouns/references at write-time ("my kids" -> "Sarah's kids")
  - Self-contained facts: each knowledge statement should be interpretable without context
  - **Rationale:** Ablation shows 56.7% of temporal reasoning performance comes from write-time disambiguation
- [x] Novelty-based filtering (only extract genuinely novel knowledge, `knowledge_type`: new | update | contradiction)
- [x] `process()` method (get unprocessed messages, segment into episodes, run predict-calibrate, store with provenance)
- [x] `process_async()` method (returns `ProcessTask` handle, non-blocking)

```python
await memory.add_exchange("I just started at Anthropic", "Congrats!")
await memory.process()  # Extracts: "User works at Anthropic"

# Later, redundant info not re-extracted
await memory.add_exchange("Loving my job at Anthropic", "Great to hear!")
await memory.process()  # No new extraction (already known)
```

---

### v0.1.0 Release
**Status:** In progress

- [x] Documentation (API reference, getting started, architecture)
- [ ] Example integrations (PydanticAI example, raw OpenAI example)
- [ ] Package naming resolution
  - `agentmemory` taken on PyPI (dormant elizaOS project)
  - Business constraint requires "Vstorm" or "V" branding
  - Candidates: `vmemo`, `vtrace`, `vgraph`, `agent-memory`
- [ ] Release prep (changelog, version 0.1.0, PyPI publish)

**v0.1.0 provides:**
- Episode-based conversation segmentation
- Batch segmentation with interleaved topic detection
- Time gap segmentation (auto-segment on >30min gap)
- Episode merging (prevents redundant episode accumulation)
- Predict-calibrate extraction (only novel info)
- Semantic knowledge base (unstructured)
- Hybrid retrieval (vector + BM25 with RRF)
- SQLite storage
- Provenance linking (knowledge -> episode -> messages)

**v0.1.0 does NOT provide:**
- Entity resolution / deduplication
- Structured relationships (entity -> relation -> entity)
- Temporal queries ("what was true at time T")
- Contradiction detection / edge invalidation
- Recency decay weighting in retrieval

**v0.1.0 known limitations:**
- `importance_score` exists but not derived from prediction error magnitude (uses LLM confidence)
- `temporal_info` extracted as string but not parsed into structured validity
- No caching layer for embeddings/LLM calls
- No `infer=False` mode for raw storage without extraction

---

# v0.2.0 - Knowledge Graph + Enhancements

Adds structured graph layer on top of v0.1.

### Phase 3.5: Bi-Temporal Validity for SemanticKnowledge

Add bi-temporal validity tracking to `SemanticKnowledge` **without** the full knowledge graph (Entity/Fact models). This enables temporal queries and knowledge invalidation while deferring graph complexity to Phase 4.

**Rationale:** Bi-temporal validity is useful immediately for:
- Point-in-time queries ("what did I know about X as of date Y")
- Knowledge invalidation (mark facts superseded without deletion)
- Contradiction handling foundation (when extraction finds contradictions)
- Recency-aware retrieval (filter/weight by validity)

**Tasks:**
- [ ] Add `BiTemporalValidity` model to `models.py`
  - `valid_at: datetime | None` - when fact became true in world (None = unknown/always)
  - `invalid_at: datetime | None` - when fact stopped being true (None = still true)
  - `expired_at: datetime | None` - when we invalidated this record (None = current)
  - `is_valid_at(event_time)` method for point-in-time checks
  - `is_current()` method for non-expired check
- [ ] Extend `SemanticKnowledge` with validity fields
  - Add `valid_at`, `invalid_at`, `expired_at` fields
  - Add `invalidate()` method to mark as superseded
- [ ] Update `KnowledgeStore` in `storage/sqlite.py`
  - Add columns: `valid_at INTEGER`, `invalid_at INTEGER`, `expired_at INTEGER`
  - Add indices on `valid_at`, `expired_at`
  - Add `get_valid_at(user_id, event_time)` - point-in-time query
  - Add `get_current()` - only non-expired records
  - Add `invalidate(knowledge_id)` - set `expired_at`
  - Handle schema migration for existing DBs
- [ ] Update `Retriever` in `retrieval/retriever.py`
  - Add `at_time: datetime | None` parameter - point-in-time query
  - Add `include_expired: bool = False` parameter
  - Filter by validity in search results
- [ ] Update `Memory` class
  - Pass `at_time` parameter through to retriever
- [ ] Parse `temporal_info` in extraction (optional, can defer)
  - Extract structured validity from strings like "since January 2024", "until next week"
  - Requires reference timestamp handling for relative dates
- [ ] Handle contradictions in extraction
  - When `knowledge_type == "contradiction"`, invalidate conflicting knowledge
  - Link new knowledge to what it supersedes (optional: `supersedes_id` field)

**What this enables:**
- `memory.retrieve("Where does user work?", at_time=datetime(2023, 1, 1))`
- Knowledge marked as superseded, not deleted (full history)
- Foundation for contradiction detection in predict-calibrate
- Recency filtering without decay scoring complexity

**What this defers (to Phase 4):**
- Entity and Fact models
- Entity resolution / deduplication pipeline
- Graph structure and BFS traversal
- Multi-hop reasoning

---

### Phase 3.6: Additional Enhancements (Optional)

**Deferred (optional for v0.2.0):**
- [ ] Episode chaining
  - Add `previous_episode_id: UUID | None` to Episode model
  - Create temporal chain for "what happened after X?" queries
- [ ] Recency weighting (optional)
  - Exponential decay on retrieval scores
  - Configurable half-life parameter
- [ ] True importance scoring
  - Track prediction content vs extracted content
  - Derive importance from prediction error magnitude
  - Larger prediction miss = higher importance
- [ ] Caching layer
  - LRU cache for embeddings (avoid re-embedding same text)
  - Per-user cache with TTL

---

### Phase 4: Knowledge Graph

- [ ] Data models (v0.2 additions)
  - `Entity`: name, entity_type, summary, aliases, embedding, attributes (dict)
  - `Fact` (edge): source_entity, target_entity, relation_type, fact_text, embedding
  - Reuse `BiTemporalValidity` from Phase 3.5 for Fact validity
  - **Multi-episode provenance**: `episode_ids: list[UUID]` on Fact (not single `source_episode_id`)
- [ ] Entity extraction
  - Extract from episode content via LLM
  - Speaker as automatic entity (user, assistant)
  - Entity type inference (person, organization, concept, location, etc.)
- [ ] Entity resolution
  - Hybrid search for candidates (embedding + BM25)
  - LLM verification: "Is NEW ENTITY a duplicate of any EXISTING ENTITIES?"
  - Merge duplicates: combine aliases, update summary
  - Preserve canonical name (most complete/descriptive)
- [ ] Fact extraction
  - Extract relationships between entities via LLM
  - Format: "SOURCE - RELATION_TYPE - TARGET (fact: detailed description)"
  - Temporal info extraction (valid_at, invalid_at from context)
- [ ] Bi-temporal validity on Facts
  - Reuse `BiTemporalValidity` model from Phase 3.5
- [ ] Contradiction handling
  - Compare new facts against existing facts between same entities
  - LLM determines which existing facts are contradicted
  - Set `expired_at` on contradicted facts (don't delete)
  - Set `invalid_at` based on when new fact became valid
- [ ] Deduplication flow
  - Vector search for similar existing facts (limit=5)
  - LLM decides: ADD (new) / UPDATE (merge) / DELETE (superseded) / NONE (duplicate)
  - Prevents fact explosion from repeated mentions
- [ ] Storage implementations
  - `SQLiteEntityStore`: entities with vector index
  - `SQLiteFactStore`: facts with vector index, temporal indices
  - Provenance: facts link to source episode_ids
- [ ] Graph retrieval
  - Search entities and facts via hybrid search
  - BFS traversal from seed entities (1-2 hops)
  - Temporal filtering: only return facts valid at query time
  - Reranking options: RRF (default), node_distance, episode_mentions
- [ ] Integration with predict-calibrate
  - Predict-calibrate gates entity/fact extraction
  - Only extract entities/facts from novel knowledge
  - Reduces extraction volume and LLM costs

```python
results = await memory.retrieve("Where does the user work?")
# Returns facts: "User WORKS_AT Anthropic (since 2024-03)"

results = await memory.retrieve(
    "Where did the user work?",
    at_time=datetime(2023, 1, 1)
)
# Returns facts: "User WORKS_AT Google (2020-01 to 2024-02)"
```

---

### Phase 5: Production Backends + Polish

- [ ] PostgreSQL implementation
  - pgvector for embeddings
  - pg_trgm for text search
  - Recursive CTEs for graph traversal
  - Connection pooling (asyncpg)
- [ ] Neo4j implementation (optional)
  - Native graph traversal
  - Vector index (5.11+)
  - Full-text index
- [ ] Backend parity
  - Same tests pass on all backends
  - Same API, different connection string
- [ ] Migration tooling
  - Export from SQLite
  - Import to Postgres/Neo4j
- [ ] Benchmarking
  - LongMemEval integration
  - Performance profiling
  - Comparison vs baselines
- [ ] CLI tools
  - `agentmemory inspect` - view DB contents
  - `agentmemory benchmark` - run evals
  - `agentmemory migrate` - backend migration

```python
# Same code, different backend
Memory("sqlite:///local.db")
Memory("postgresql://user:pass@host/db")
Memory("neo4j://user:pass@host:7687")
```

---

### v0.2.0 Release

- [ ] Updated documentation
- [ ] LangChain example
- [ ] Changelog
- [ ] PyPI publish

---

## Future Considerations (v0.3+)

### API Enhancements
- `infer=True/False` toggle on `add_message()` / `add_exchange()`
  - `infer=True` (default): LLM extracts facts during `process()`
  - `infer=False`: Raw storage only, skip extraction
  - Useful for bulk imports or when extraction not needed
- `actor_id` + `role` tracking
  - Support multi-agent scenarios (which agent said what)
  - History tracking with actor attribution

### Cross-User Learning
- Knowledge that benefits ALL users, not just one
- Namespace isolation: user-scoped vs global
- Human-in-the-loop gating for quality control (`requires_confirmation=True`)
- Currently: all knowledge is user-scoped

### Operational Maintenance
- Nightly consolidation (merge similar episodes)
- Weekly compression (summarize old episodes)
- Monthly re-indexing (rebuild indices)
- Decay/pruning schedules for stale knowledge

### Tiered Retrieval
- Breadth-first triage before deep dive
- Category selection -> summary check -> item drill-down
- Reduces retrieval latency for large KBs

---

## Dependency Graph

```
Phase 0 (models)
    |
    v
Phase 1 (storage + basic retrieval)
    |
    v
Phase 2 (episodes + hybrid retrieval)
    |
    v
Phase 3 (predict-calibrate)
    |
    v
===============================
    v0.1.0 RELEASE
===============================
    |
    v
Phase 3.5 (bi-temporal validity) <-- temporal queries WITHOUT full graph
    |
    v
Phase 3.6 (optional enhancements) <-- episode chaining, recency, caching
    |
    v
Phase 4 (knowledge graph) <-- Entity/Fact models, entity resolution
    |
    v
Phase 5 (prod backends + polish)
    |
    v
===============================
    v0.2.0 RELEASE
===============================
```

---

## Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Package name | TBD (`vmemo`? `vtrace`?) | `agentmemory` taken on PyPI. Business constraint requires "Vstorm" / "V" branding. |
| Default embedding | Require explicit config | Less magic, clearer dependencies |
| Background processing | Explicit `process()` call | Start simple, add background option later |
| Multi-user | Support from day 1 | Add `user_id` to models now, painful to retrofit |
| Observability | structlog + optional OTEL | Structured logging default, tracing as extra |
| Extraction method | Predict-calibrate | Importance emerges from prediction error, not upfront LLM scoring |
| Disambiguation timing | Write-time | Ablation shows 56.7% of temporal performance from write-time normalization |
| Memory unit | Episode -> atomic facts | Hybrid: preserve episodes for provenance, atomize for retrieval efficiency |

---

## Competitive Position

| Dimension | Typical approach | agentmemory |
|-----------|------------------|-------------|
| Extraction trigger | LLM judges importance upfront | Importance emerges from prediction error |
| Memory unit | Messages or arbitrary chunks | Episodes -> atomic facts (hybrid) |
| Temporal handling | Timestamps + overwrite | Bi-temporal validity (event + transaction time) |
| Disambiguation timing | Retrieval-time | Write-time |
| Contradiction handling | Overwrite old facts | Edge invalidation with full history |
| Retrieval | Store-based or tiered | Hybrid: vector + BM25 + RRF |
| Provenance | Limited | Full: fact -> episode -> messages |
