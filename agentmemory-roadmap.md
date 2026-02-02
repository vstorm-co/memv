# agentmemory

Structured, temporal memory for AI agents.

## Overview

**Core principles:**
1. Framework-agnostic (works with any agent framework)
2. Pydantic-native (models, validation, serialization)
3. Storage-flexible (SQLite → Postgres → Neo4j)
4. Incrementally adoptable (simple → sophisticated)

## Architecture

Combines insights from:
- **Graphiti/Zep:** Bi-temporal model, edge invalidation, entity resolution, BFS retrieval
- **Nemori:** Predict-calibrate extraction, episode segmentation, importance as emergent
- **SimpleMem:** Write-time temporal normalization, atomization, entropy filtering

### Data Model

```
Messages (append-only archive)
    │
    ▼
Episodes (segmented conversation chunks with narratives)
    │
    ▼
Entities (resolved, deduplicated) ──── Facts (bi-temporal edges)
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

## Benchmark Analysis (LoCoMo)

Comparative analysis of memory systems on LoCoMo benchmark. Note: Nemori uses LLM-judge score (0-1), SimpleMem uses F1 (%). Different metrics prevent direct comparison, but relative improvements over shared baselines (Mem0) provide insight.

### Performance Comparison (GPT-4.1-mini)

| System | Metric | Score | vs Mem0 | Tokens/Query | Construction Time |
|--------|--------|-------|---------|--------------|------------------|
| Full Context | LLM Score | 0.806 | baseline | 23,653 | - |
| Mem0 | LLM Score | 0.663 | - | ~1,027 | 1,350.9s |
| Mem0 | F1 | 34.20% | - | ~973 | 1,350.9s |
| Nemori | LLM Score | 0.794 | **+19.8%** | ~2,745 | not reported |
| SimpleMem | F1 | 43.24% | **+26.4%** | ~531 | 92.6s |

### Task-Specific Performance

| Task | Nemori (LLM) | SimpleMem (F1) | Observations |
|------|-------------|----------------|---------------|
| Temporal | 0.776 | 58.62% | Both excel here |
| Multi-hop | 0.751 | 43.46% | Nemori slightly better |
| Single-hop | 0.849 | 51.12% | Strong for both |
| Open Domain | 0.510 | 19.76% | Both struggle |

### Critical Ablation Study (SimpleMem)

| Configuration | Temporal F1 | Impact | Insight |
|---------------|-------------|--------|----------|
| Full SimpleMem | 58.62% | baseline | - |
| **w/o Atomization** | 25.40% | **-56.7%** | **Write-time disambiguation is critical** |
| w/o Consolidation | 55.10% | -6.0% | Less impactful |
| w/o Adaptive Pruning | 56.80% | -3.1% | Minimal impact |

**Key Finding:** Atomization (temporal normalization + coreference resolution at write-time) accounts for **56.7% of temporal reasoning performance**. This is not a minor optimization—it's a fundamental architectural insight.

### Retrieval Saturation (SimpleMem)

| k (retrieved items) | F1 | % of Peak |
|--------------------|-------|------------|
| k=1 | 35.20% | 81% |
| k=3 | 42.85% | **99%** |
| k=10 | 43.45% | 100% (peak) |
| k=20 | 43.40% | 100% |

**Key Finding:** With high-quality atomic entries, k=3 achieves 99% of peak performance. Large context windows unnecessary.

### Efficiency Summary

| Metric | SimpleMem | Nemori | Ratio |
|--------|-----------|--------|-------|
| Tokens/query | 531 | 2,745 | **5× fewer** |
| Construction time | 92.6s | not reported | - |
| vs Mem0 construction | 55× faster | - | - |

### Architectural Implications

Two valid philosophies emerge:

**SimpleMem Approach (Write-time disambiguation):**
- Convert everything to atomic, self-contained facts at ingestion
- Temporal normalization: "yesterday" → "2023-07-01"
- Coreference resolution: "my kids" → "Sarah's kids"
- Pro: Faster retrieval, simpler reasoning, benchmark-proven
- Con: Loses conversational structure, harder to answer "what was the context when X happened?"

**Nemori Approach (Preserve episodes):**
- Preserve episodes, extract knowledge via predict-calibrate
- Pro: Retains narrative context, handles nuance better
- Con: More complex, slower, predict-calibrate adds LLM calls

**Hybrid Approach (Proposed for agentmemory):**
```python
# Combine both: SimpleMem atomization + Nemori predict-calibrate
async def process_messages(messages: list[Message]) -> None:
    # 1. Episode boundary detection (Nemori)
    episode = await detect_and_create_episode(messages)
    
    # 2. Atomization with temporal normalization (SimpleMem)
    atomic_facts = await atomize(
        episode.raw_messages,
        reference_time=messages[-1].timestamp
    )
    
    # 3. Predict-calibrate filtering (Nemori)
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
- **v0.1.0** — Nemori-style: episodes + predict-calibrate + semantic KB
- **v0.2.0** — + Graphiti-style: structured knowledge graph with bi-temporal edges

---

# v0.1.0 — Core Memory (7 weeks)

Episodes, predict-calibrate extraction, semantic knowledge base (unstructured statements).
No entity graph, no bi-temporal, no contradiction handling yet.

### Phase 0: Project Setup + Data Models
**Duration:** 1 week

- [x] Project scaffolding
  - pyproject.toml (uv)
  - ruff, pyright, pytest
  - CI/CD (GitHub Actions)
  - pre-commit hooks
- [x] Core data models (v0.1 subset)
  - [x] `Message`
  - [x] `Episode`
  - [x] `SemanticKnowledge` (unstructured statements)
  - [x] `RetrievalResult`
  - [x] `ExtractedKnowledge` (output of predict-calibrate)
  - [x] `ProcessTask` (async processing handle)

**Deliverable:** Package installs, models work, nothing functional yet

**Exit criteria:**
- `pip install agentmemory` works
- Models serialize/deserialize correctly

---

### Phase 1: Storage + Basic Retrieval
**Duration:** 2 weeks

Build concrete implementations first, extract protocols after.

- [x] SQLite storage (start here)
  - [x] `SQLiteMessageStore` — build first, simplest CRUD
  - [x] `SQLiteEpisodeStore` (storage only, no segmentation)
  - [x] `SQLiteKnowledgeStore`
  - [x] Vector storage via sqlite-vec
  - [x] FTS5 for full-text
- [x] Extract protocols from implementations
  - `MessageStore`
  - `EpisodeStore`
  - `KnowledgeStore`
  - `EmbeddingClient`
- [x] Embedding client implementations
  - `OpenAIEmbedAdapter`
  - `FastEmbedAdapter` (optional local default)
- [x] Basic retrieval
  - Vector similarity on SemanticKnowledge
  - Top-k retrieval
- [x] `Memory` class
  - `add_message()`
  - `add_exchange()`
  - `retrieve()`
  - `open()` / `close()`
  - Context manager support (`async with`)

**Deliverable:** Working message storage + retrieval

```python
memory = Memory("sqlite:///test.db", embedding=OpenAIEmbedAdapter())
await memory.add_message(msg)
results = await memory.retrieve("query")
```

**Exit criteria:**
- Round-trip: store → retrieve works
- Similar messages score higher

---

### Phase 2: Episodes + Hybrid Retrieval
**Duration:** 2 weeks

- [x] Hybrid search
  - BM25 via FTS5
  - Score fusion (RRF with k=60)
  - Configurable `vector_weight` parameter (0.0-1.0)
- [x] Episode segmentation
  - `BoundaryDetector` (LLM-based)
  - Signals: coherence, topic shift, intent shift, temporal markers
  - Fallback: max buffer size (25 messages)
  - Episode narrative generation via `EpisodeGenerator`
- [x] LLM client implementations
  - `PydanticAIAdapter`
- [x] Episode-aware retrieval
  - Search episodes via separate vector + text indices
  - Auto-fetch source episodes for returned knowledge (provenance)
  - `RetrievalResult.to_prompt()` groups knowledge by episode

**Note:** v0.1.0 uses batch segmentation by default (accumulate → group all at once) per Nemori's approach. Legacy incremental boundary detection is available via `use_legacy_segmentation=True` but deprecated.

**Deliverable:** Topic-aware chunking, better retrieval

**Exit criteria:**
- Episodes form at natural topic boundaries
- Hybrid search outperforms vector-only (manual eval)
- Episode narratives are coherent summaries

---

### Phase 3: Predict-Calibrate
**Duration:** 2 weeks

- [x] Prediction generation
  - Retrieve relevant KB for episode (top_k=20)
  - Generate prediction: "What should this episode contain?"
- [x] Calibration
  - Compare prediction vs raw messages (not narrative — critical for ground truth)
  - Extract gaps (what prediction missed)
- [x] Cold start handling
  - When `existing_knowledge` empty, skip prediction
  - Use dedicated extraction prompt ("extract everything notable")
  - Reference: Nemori's `prediction_correction_engine.py` cold start mode
- [x] Semantic knowledge extraction
  - Unstructured statements (not entity-relation triples yet)
  - Store in `SemanticKnowledge` with embeddings
  - Batch embedding for efficiency
- [ ] ~~**Atomization step**~~ (SimpleMem insight) — DEFERRED TO v0.1.1/v0.2.0
  - Temporal normalization: resolve relative dates at write-time ("yesterday" → absolute timestamp)
  - Coreference resolution: resolve pronouns/references at write-time ("my kids" → "Sarah's kids")
  - Self-contained facts: each knowledge statement should be interpretable without context
  - **Rationale:** SimpleMem ablation shows 56.7% of temporal reasoning performance comes from write-time disambiguation
  - **Decision:** Ship v0.1.0 without atomization, add in later release when real usage data available
- [x] Novelty-based filtering
  - Only extract genuinely novel knowledge
  - `knowledge_type`: new | update | contradiction
- [x] `process()` method
  - Get unprocessed messages
  - Segment into episodes
  - Run predict-calibrate pipeline
  - Store extracted knowledge with provenance (`source_episode_id`)
- [x] `process_async()` method
  - Returns `ProcessTask` handle
  - Non-blocking, can `await task.wait()`

**Deliverable:** Smarter extraction, semantic KB built from conversations

```python
await memory.add_exchange("I just started at Anthropic", "Congrats!")
await memory.process()  # Extracts: "User works at Anthropic"

# Later, redundant info not re-extracted
await memory.add_exchange("Loving my job at Anthropic", "Great to hear!")
await memory.process()  # No new extraction (already known)
```

**Exit criteria:**
- Redundant info not re-extracted
- Novel info captured
- Extraction count reduced vs naive approach

---

### v0.1.0 Release
**Duration:** ~1 week (buffer)

- [x] Documentation
  - API reference
  - Getting started guide
  - Architecture overview
- [ ] Example integrations
  - PydanticAI example
  - Raw OpenAI example
- [ ] Package naming resolution
  - `agentmemory` taken on PyPI (dormant elizaOS project)
  - Business constraint requires "Vstorm" or "V" branding
  - Candidates: `vmemo`, `vtrace`, `vgraph`, `agent-memory`
- [ ] Release prep
  - Changelog
  - Version 0.1.0
  - PyPI publish

**v0.1.0 provides:**
- Episode-based conversation segmentation
- **Batch segmentation** with interleaved topic detection
- **Time gap segmentation** (auto-segment on >30min gap)
- **Episode merging** (prevents redundant episode accumulation)
- Predict-calibrate extraction (only novel info)
- Semantic knowledge base (unstructured)
- Hybrid retrieval (vector + BM25 with RRF)
- SQLite storage
- Provenance linking (knowledge → episode → messages)

**v0.1.0 does NOT provide:**
- Entity resolution / deduplication
- Structured relationships (entity → relation → entity)
- Temporal queries ("what was true at time T") — coming in Phase 3.5
- Contradiction detection / edge invalidation — coming in Phase 3.5
- Recency decay weighting in retrieval

**v0.1.0 known limitations:**
- `importance_score` exists but not derived from prediction error magnitude (uses LLM confidence)
- `temporal_info` extracted as string but not parsed into structured validity — addressed in Phase 3.5
- No caching layer for embeddings/LLM calls
- No `infer=False` mode for raw storage without extraction
- No atomization (temporal normalization, coreference resolution) — deferred to v0.1.1/v0.2.0

---

# v0.2.0 — Knowledge Graph + Enhancements

Adds structured graph layer on top of v0.1, plus Nemori enhancements.

### Phase 3.5: Bi-Temporal Validity for SemanticKnowledge
**Duration:** 1-2 weeks

Add bi-temporal validity tracking to `SemanticKnowledge` **without** the full knowledge graph (Entity/Fact models). This enables temporal queries and knowledge invalidation while deferring graph complexity to Phase 4.

**Rationale:** Bi-temporal validity is useful immediately for:
- Point-in-time queries ("what did I know about X as of date Y")
- Knowledge invalidation (mark facts superseded without deletion)
- Contradiction handling foundation (when extraction finds contradictions)
- Recency-aware retrieval (filter/weight by validity)

**Tasks:**
- [ ] Add `BiTemporalValidity` model to `models.py`
  - `valid_at: datetime | None` — when fact became true in world (None = unknown/always)
  - `invalid_at: datetime | None` — when fact stopped being true (None = still true)
  - `expired_at: datetime | None` — when we invalidated this record (None = current)
  - `is_valid_at(event_time)` method for point-in-time checks
  - `is_current()` method for non-expired check
- [ ] Extend `SemanticKnowledge` with validity fields
  - Add `valid_at`, `invalid_at`, `expired_at` fields
  - Add `invalidate()` method to mark as superseded
- [ ] Update `KnowledgeStore` in `storage/sqlite.py`
  - Add columns: `valid_at INTEGER`, `invalid_at INTEGER`, `expired_at INTEGER`
  - Add indices on `valid_at`, `expired_at`
  - Add `get_valid_at(user_id, event_time)` — point-in-time query
  - Add `get_current()` — only non-expired records
  - Add `invalidate(knowledge_id)` — set `expired_at`
  - Handle schema migration for existing DBs
- [ ] Update `Retriever` in `retrieval/retriever.py`
  - Add `at_time: datetime | None` parameter — point-in-time query
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

**What this ENABLES:**
- `memory.retrieve("Where does user work?", at_time=datetime(2023, 1, 1))`
- Knowledge marked as superseded, not deleted (full history)
- Foundation for contradiction detection in predict-calibrate
- Recency filtering without decay scoring complexity

**What this DEFERS (to Phase 4):**
- Entity and Fact models
- Entity resolution / deduplication pipeline
- Graph structure and BFS traversal
- Multi-hop reasoning

**Exit criteria:**
- Point-in-time queries return historically-valid knowledge
- Contradicting knowledge invalidates old entries (sets `expired_at`)
- `include_expired=True` returns full history
- Existing DBs migrate cleanly (new columns nullable)

---

### Phase 3.6: Additional Enhancements (Optional)
**Duration:** 1 week (optional)

Additional enhancements from Nemori. Core segmentation features (batch, time gaps, merging) moved to v0.1.0.

**Moved to v0.1.0:** ✓
- [x] Batch segmentation (handles interleaved topics)
- [x] Time gap segmentation (>30min gap auto-segments)
- [x] Episode merging (prevents redundant accumulation)

**Deferred (optional for v0.2.0):**
- [ ] Episode chaining
  - Add `previous_episode_id: UUID | None` to Episode model
  - Create temporal chain for "what happened after X?" queries
  - Pattern from Graphiti's `NextEpisodeEdge`
- [ ] Recency weighting (optional)
  - Exponential decay on retrieval scores
  - Configurable half-life parameter
  - Note: May not be needed — Nemori's benchmarks work without it
- [ ] True importance scoring
  - Track prediction content vs extracted content
  - Derive importance from prediction error magnitude
  - Larger prediction miss = higher importance
- [ ] Caching layer
  - LRU cache for embeddings (avoid re-embedding same text)
  - Per-user cache with TTL
  - Reference: Nemori's `lru_cache` usage

**Exit criteria:**
- Episode chain navigable forward/backward (if implemented)

---

### Phase 4: Knowledge Graph
**Duration:** 3 weeks

Based on Graphiti's architecture (see `reference/graphiti/`). Builds on bi-temporal foundation from Phase 3.5.

- [ ] Data models (v0.2 additions)
  - `Entity`: name, entity_type, summary, aliases, embedding, attributes (dict)
  - `Fact` (edge): source_entity, target_entity, relation_type, fact_text, embedding
  - Reuse `BiTemporalValidity` from Phase 3.5 for Fact validity
  - **Multi-episode provenance**: `episode_ids: list[UUID]` on Fact (not single `source_episode_id`)
    - Facts can be reinforced/updated across multiple episodes
    - Append episode_id when fact is re-confirmed or updated
- [ ] Entity extraction
  - Extract from episode content via LLM
  - Speaker as automatic entity (user, assistant)
  - Entity type inference (person, organization, concept, location, etc.)
  - Custom entity types via Pydantic models (Graphiti pattern)
- [ ] Entity resolution (Graphiti's dedupe_nodes pattern)
  - Hybrid search for candidates (embedding + BM25)
  - LLM verification: "Is NEW ENTITY a duplicate of any EXISTING ENTITIES?"
  - Merge duplicates: combine aliases, update summary
  - Preserve canonical name (most complete/descriptive)
- [ ] Fact extraction
  - Extract relationships between entities via LLM
  - Format: "SOURCE - RELATION_TYPE - TARGET (fact: detailed description)"
  - Temporal info extraction (valid_at, invalid_at from context)
  - Reference timestamp handling for relative dates ("last week" → absolute)
- [ ] Bi-temporal validity on Facts
  - Reuse `BiTemporalValidity` model from Phase 3.5
  - Apply same patterns to Fact edges
- [ ] Contradiction handling (Graphiti's invalidate_edges pattern)
  - Compare new facts against existing facts between same entities
  - LLM determines which existing facts are contradicted
  - Set `expired_at` on contradicted facts (don't delete)
  - Set `invalid_at` based on when new fact became valid
- [ ] Deduplication flow (mem0 pattern)
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

**Reference implementation:** `reference/graphiti/graphiti_core/`
- `nodes.py`: EntityNode model
- `edges.py`: EntityEdge with bi-temporal fields
- `prompts/dedupe_nodes.py`: Entity resolution prompts
- `prompts/invalidate_edges.py`: Contradiction detection prompts
- `search/search_config_recipes.py`: Search configuration patterns

**Deliverable:** Knowledge graph built from conversations

```python
results = await memory.retrieve("Where does the user work?")
# Returns facts: "User WORKS_AT Anthropic (since 2024-03)"

results = await memory.retrieve(
    "Where did the user work?",
    at_time=datetime(2023, 1, 1)
)
# Returns facts: "User WORKS_AT Google (2020-01 to 2024-02)"
```

**Exit criteria:**
- Entities deduplicated correctly (same person = same entity)
- Contradictions invalidate old facts (set expired_at, not delete)
- Temporal queries return correct historical state
- Facts link to source episodes (provenance)
- BFS retrieval includes related entities/facts

---

### Phase 5: Production Backends + Polish
**Duration:** 2 weeks

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

**Deliverable:** Production-ready with multiple backends

```python
# Same code, different backend
Memory("sqlite:///local.db")
Memory("postgresql://user:pass@host/db")
Memory("neo4j://user:pass@host:7687")
```

**Exit criteria:**
- All tests pass on all backends
- LongMemEval results documented
- Migration works without data loss

---

### v0.2.0 Release

- [ ] Updated documentation
- [ ] LangChain example
- [ ] Changelog
- [ ] PyPI publish

---

## Future Considerations (v0.3+)

Based on competitive analysis (Agno, Rohit, Letta) and reference implementations:

### API Enhancements
- `infer=True/False` toggle on `add_message()` / `add_exchange()`
  - `infer=True` (default): LLM extracts facts during `process()`
  - `infer=False`: Raw storage only, skip extraction
  - Pattern from mem0 — useful for bulk imports or when extraction not needed
- `actor_id` + `role` tracking
  - Support multi-agent scenarios (which agent said what)
  - History tracking with actor attribution
  - Pattern from mem0's history system

### Cross-User Learning (Agno's "Learned Memory")
- Knowledge that benefits ALL users, not just one
- Namespace isolation: user-scoped vs global
- Human-in-the-loop gating for quality control (`requires_confirmation=True`)
- Currently: all knowledge is user-scoped

### Operational Maintenance Patterns (Rohit)
- Nightly consolidation (merge similar episodes)
- Weekly compression (summarize old episodes)
- Monthly re-indexing (rebuild indices)
- Decay/pruning schedules for stale knowledge

### Tiered Retrieval (Rohit)
- Breadth-first triage before deep dive
- Category selection → summary check → item drill-down
- Reduces retrieval latency for large KBs

### Agent Tool Use (Letta Benchmark Finding)
- Simple filesystem + agent tool use achieved 74% on LoCoMo
- Beat Mem0's specialized memory tools (68.5%)
- Validates "explicit over implicit" philosophy
- Memory is more about context management than retrieval mechanism

---

## Dependency Graph

```
Phase 0 (models)
    │
    ▼
Phase 1 (storage + basic retrieval)
    │
    ▼
Phase 2 (episodes + hybrid retrieval)
    │
    ▼
Phase 3 (predict-calibrate)
    │
    ▼
═══════════════════════════════
    v0.1.0 RELEASE
═══════════════════════════════
    │
    ▼
Phase 3.5 (bi-temporal validity) ←── temporal queries WITHOUT full graph
    │
    ▼
Phase 3.6 (optional enhancements) ←── episode chaining, recency, caching
    │
    ▼
Phase 4 (knowledge graph) ←── Entity/Fact models, entity resolution
    │
    ▼
Phase 5 (prod backends + polish)
    │
    ▼
═══════════════════════════════
    v0.2.0 RELEASE
═══════════════════════════════
```

---

## Timeline

| Phase | Duration | Cumulative | Milestone |
|-------|----------|------------|-----------|
| 0: Setup | 1 week | 1 week | Package exists |
| 1: Storage + Basic | 2 weeks | 3 weeks | Basic memory works |
| 2: Episodes + Hybrid | 2 weeks | 5 weeks | Topic-aware retrieval |
| 3: Predict-Calibrate + Segmentation | 2 weeks | 7 weeks | **v0.1.0 release** (includes batch seg, time gaps, merging) |
| 3.5: Bi-Temporal Validity | 1-2 weeks | 8-9 weeks | Temporal queries, knowledge invalidation |
| 3.6: Optional Enhancements | 1 week | 9-10 weeks | Episode chaining, recency, caching (optional) |
| 4: Knowledge Graph | 3 weeks | 12-13 weeks | Structured knowledge (Entity/Fact) |
| 5: Prod + Polish | 2 weeks | 14-15 weeks | **v0.2.0 release** |

**Key milestones:**
- Week 7: **v0.1.0** — Functional library (batch segmentation + time gaps + episode merging + predict-calibrate)
- Week 8-9: **Phase 3.5** — Bi-temporal validity (temporal queries without full graph complexity)
- Week 14-15: **v0.2.0** — Full feature set (+ knowledge graph + prod backends)

---

## Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Package name | TBD (`vmemo`? `vtrace`?) | `agentmemory` taken on PyPI (dormant elizaOS project). Business constraint requires "Vstorm" / "V" branding. |
| Default embedding | Require explicit config | Less magic, clearer dependencies |
| Background processing | Explicit `process()` call | Start simple, add background option later. Letta benchmark validates this approach. |
| Multi-user | Support from day 1 | Add `user_id` to models now, painful to retrofit |
| Observability | structlog + optional OTEL | Structured logging default, tracing as extra |
| Extraction method | Predict-calibrate | Differentiator vs Agno/Rohit/mem0 who use upfront LLM importance scoring |
| Disambiguation timing | Write-time (deferred) | Ablation shows 56.7% of temporal performance from write-time normalization. Deferred to v0.1.1/v0.2.0. |
| Memory unit | Episode → semantic knowledge | Preserve episodes for provenance, extract unstructured statements. Full atomization deferred. |

---

## Competitive Landscape

| Dimension | Agno/Rohit/mem0 | SimpleMem | agentmemory |
|-----------|-----------------|-----------|-------------|
| Extraction trigger | LLM judges importance upfront | Entropy filtering | Importance emerges from prediction error |
| Memory unit | Messages or arbitrary chunks | Atomic facts | Episodes → atomic facts (hybrid) |
| Temporal handling | Timestamps + overwrite | **Write-time normalization** | Bi-temporal validity (event + transaction time) |
| Disambiguation timing | Retrieval-time | **Write-time** | Write-time (adopting SimpleMem insight) |
| Contradiction handling | Overwrite old facts | Consolidation | Edge invalidation with full history |
| Entity deduplication | Not addressed | Coreference in atomization | Entity resolution pipeline |
| Retrieval | Tiered or store-based | Hybrid + symbolic metadata | Hybrid: vector + BM25 + RRF |
| LLM calls per extraction | 1 (extract facts) | 1 (atomize) | 2 (predict + calibrate) |
| Provenance | Limited | Limited | Full: fact → episode → messages |
| Tokens/query (LoCoMo) | ~1,000 | **~531** | ~2,745 |

**What they have that we don't (yet):**
- Cross-user learning (Agno's "Learned Memory")
- Human-in-the-loop gating
- Operational maintenance patterns (cron jobs)
- Tiered retrieval optimization
- SimpleMem's token efficiency (~531 vs our ~2,745 tokens/query)
- SimpleMem's parallel processing (16 workers for construction)
- SimpleMem's MCP server integration (production-ready)

**What we have that they're missing:**
- Bi-temporal validity
- Episode segmentation
- Predict-calibrate mechanism (though SimpleMem's entropy filtering achieves similar goals more efficiently)
- Entity resolution pipeline (SimpleMem handles this in atomization step)
- Full provenance chain (multi-episode)

**What we're adopting from SimpleMem:**
- Write-time temporal normalization ("yesterday" → absolute timestamp at ingestion)
- Coreference resolution at write-time ("my kids" → "Sarah's kids")
- The principle that disambiguation at write-time > retrieval-time

**Planned additions from reference analysis:**
- Batch segmentation with non-continuous grouping (Nemori)
- `infer=True/False` toggle for raw storage (mem0)
- Episode chaining for temporal navigation (Graphiti's NextEpisodeEdge)
- ADD/UPDATE/DELETE/NONE deduplication flow (mem0)
- `actor_id` + `role` tracking for multi-agent scenarios (mem0)

---

## What We're NOT Taking (and Why)

From reference analysis — explicitly excluded:

**From Nemori:**
- NOR-LIFT reranking: Complex percentile-based normalization. Their benchmarks show hybrid without it works fine. Add later if retrieval quality suffers.
- Semantic vs episodic memory separation at storage level: They use separate indices. We're treating SemanticKnowledge as the extracted layer, episodes as source. Simpler model.

**From mem0:**
- Procedural memory: Agent-specific procedural knowledge ("how to do X"). Scope creep for v0.1-0.2. Revisit for v0.3+ if agent tool-use patterns emerge.
- Graph store integration: They support Neptune etc. We're building graph layer ourselves in Phase 4 with SQLite first.

**From Graphiti:**
- CommunityNode/SagaNode: Higher-level groupings (communities of entities, saga=multi-episode arcs). Premature abstraction. Add when use cases emerge.
- Custom entity types via Pydantic inheritance: Nice pattern but adds complexity. Start with string `entity_type`, add typed entities later if needed.
- OpenTelemetry tracing: Good for production, but structlog + optional OTEL is enough for v0.1-0.2.

**From SimpleMem:**
- LanceDB storage: They use LanceDB with IVF-PQ. We're sticking with SQLite → Postgres path for simplicity and broader adoption.
- Recursive consolidation: Their background clustering (threshold 0.85) to form abstractions. Interesting but adds complexity. Our predict-calibrate serves similar purpose.
- Planning/reflection loops: Optional multi-query decomposition and iterative refinement. Scope creep for v0.1.
- MCP server: Production feature but not core to the memory system. Add as separate package later if demand.

**What we ARE adopting from SimpleMem:**
- Write-time temporal normalization ("yesterday" → absolute timestamp)
- Write-time coreference resolution ("my kids" → "Sarah's kids")
- The atomization principle: self-contained facts that don't require context at retrieval
- Entropy-based filtering insights (though we implement via predict-calibrate)

---

## References

> 💡 Luźne pomysły na przyszłość (v0.3+) → patrz [notes/IDEAS.md](notes/IDEAS.md)

**Papers:**
- Graphiti paper: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)
- Nemori paper: [arXiv:2508.03341](https://arxiv.org/abs/2508.03341)
- SimpleMem paper: [arXiv:2501.xxxxx](https://arxiv.org/abs/2501.xxxxx) — "SimpleMem: A Simple yet Effective Write-Time Memory Architecture" (F1: 43.24%, 531 tokens/query)
- LongMemEval benchmark: [arXiv:2407.xxxxx](https://arxiv.org/abs/2407.xxxxx)
- Context Engineering survey: [arXiv:2507.13334](https://arxiv.org/abs/2507.13334)

**Reference implementations** (cloned in `reference/`):
- `reference/nemori/` — Original Nemori MVP from paper authors
  - `src/generation/batch_segmenter.py`: Batch segmentation with non-continuous groups
  - `src/generation/prediction_correction_engine.py`: Predict-calibrate implementation
  - `src/generation/episode_merger.py`: Episode merging logic
  - `src/search/`: Hybrid search with NOR-LIFT ranking
- `reference/graphiti/` — Zep's Graphiti framework
  - `graphiti_core/nodes.py`: EntityNode with summaries, aliases
  - `graphiti_core/edges.py`: EntityEdge with bi-temporal validity
  - `graphiti_core/prompts/`: LLM prompts for extraction, deduplication, invalidation
  - `graphiti_core/search/`: Configurable search with multiple reranking strategies
- `reference/simplemem/` — SimpleMem (github.com/aiming-lab/SimpleMem)
  - `main.py`: SimpleMemSystem class with 3-stage pipeline
  - `core/`: Atomization, entropy filtering, consolidation
  - `database/`: LanceDB vector store with IVF-PQ indexing
  - `MCP/`: Model Context Protocol server integration
  - Key config: window_size=40, overlap=2, entropy_threshold=0.35, consolidation_threshold=0.85
  - **Critical pattern**: Temporal normalization + coreference resolution at write-time

**Competitive analysis:**
- Agno "Learning Machines": https://www.ashpreetbedi.com/articles/memory
- Rohit's architectures: https://x.com/rohit4verse/article/2012925228159295810
- Letta benchmark: https://www.letta.com/blog/benchmarking-ai-agent-memory
