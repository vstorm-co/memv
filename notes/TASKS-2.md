# Tasks (Post v0.1.0)

Work for v0.2.0 release. Requires v0.1.0 completion.

---

## 1. Bi-Temporal Validity (Phase 3.5)

**Goal:** Enable temporal queries and knowledge invalidation without full graph complexity.

### 1.1 BiTemporalValidity Model

- [ ] Add to `src/agent_memory/models.py`:
  ```python
  class BiTemporalValidity(BaseModel):
      valid_at: datetime | None = None    # When fact became true in world
      invalid_at: datetime | None = None  # When fact stopped being true
      expired_at: datetime | None = None  # When we invalidated this record

      def is_valid_at(self, event_time: datetime) -> bool: ...
      def is_current(self) -> bool: ...
  ```
- [ ] Extend `SemanticKnowledge` with validity fields
- [ ] Add `invalidate()` method to mark knowledge as superseded
- [ ] Add optional `supersedes_id: UUID | None` for linking replacements

### 1.2 Storage Updates

- [ ] Update `KnowledgeStore` in `src/agent_memory/storage/sqlite/`:
  - Add columns: `valid_at INTEGER`, `invalid_at INTEGER`, `expired_at INTEGER`
  - Add indices on `valid_at`, `expired_at` for query performance
  - Add `get_valid_at(user_id, event_time)` - point-in-time query
  - Add `get_current(user_id)` - only non-expired records
  - Add `invalidate(knowledge_id)` - set `expired_at`
- [ ] Schema migration for existing databases (new columns nullable)

### 1.3 Retriever Updates

- [ ] Update `Retriever` in `src/agent_memory/retrieval/retriever.py`:
  - Add `at_time: datetime | None` parameter for point-in-time queries
  - Add `include_expired: bool = False` parameter
  - Filter results by validity before returning
- [ ] Update `Memory.retrieve()` to pass through `at_time`

### 1.4 Contradiction Handling

- [ ] Update `PredictCalibrateExtractor`:
  - When `knowledge_type == "contradiction"`, identify conflicting knowledge
  - Call `invalidate()` on conflicting entries
  - Link new knowledge via `supersedes_id`
- [ ] Add prompt guidance for detecting contradictions

### 1.5 Temporal Info Parsing (Optional)

- [ ] Parse `temporal_info` strings into structured validity:
  - "since January 2024" → `valid_at=2024-01-01`
  - "until next week" → `invalid_at=<computed>`
  - "from 2020 to 2023" → `valid_at`, `invalid_at`
- [ ] Requires reference timestamp for relative expressions

---

## 2. Optional Enhancements (Phase 3.6)

### 2.1 Episode Chaining

- [ ] Add `previous_episode_id: UUID | None` to `Episode` model
- [ ] Update `EpisodeStore` to maintain chain links
- [ ] Enable queries: "what happened after X?"
- [ ] Pattern from Graphiti's `NextEpisodeEdge`

### 2.2 Recency Weighting

- [ ] Add exponential decay scoring in retrieval
- [ ] Configurable half-life parameter
- [ ] Optional: may not be needed per Nemori benchmarks

### 2.3 True Importance Scoring

- [ ] Track prediction content vs extracted content
- [ ] Derive importance from prediction error magnitude
- [ ] Larger prediction miss = higher importance score
- [ ] Replace current LLM-confidence-based scoring

### 2.4 Caching Layer

- [ ] LRU cache for embeddings (avoid re-embedding same text)
- [ ] Per-user cache with TTL
- [ ] Cache invalidation on knowledge updates

---

## 3. Knowledge Graph (Phase 4)

**Goal:** Structured entity-relationship layer with deduplication.

### 3.1 Data Models

- [ ] Create `Entity` model:
  ```python
  class Entity(BaseModel):
      id: UUID
      name: str
      entity_type: str  # person, organization, concept, location
      summary: str | None
      aliases: list[str] = []
      embedding: list[float]
      attributes: dict[str, Any] = {}
  ```
- [ ] Create `Fact` model (edge):
  ```python
  class Fact(BaseModel):
      id: UUID
      source_entity_id: UUID
      target_entity_id: UUID
      relation_type: str
      fact_text: str
      embedding: list[float]
      episode_ids: list[UUID]  # Multi-episode provenance
      # Reuse BiTemporalValidity fields
  ```

### 3.2 Entity Extraction

- [ ] Extract entities from episode content via LLM
- [ ] Auto-create speaker entities (user, assistant)
- [ ] Infer entity types from context

### 3.3 Entity Resolution (Deduplication)

- [ ] Hybrid search for candidate duplicates (embedding + BM25)
- [ ] LLM verification: "Is NEW_ENTITY a duplicate of EXISTING?"
- [ ] Merge duplicates: combine aliases, update summary
- [ ] Preserve canonical name (most complete/descriptive)
- [ ] Reference: `reference/graphiti/graphiti_core/prompts/dedupe_nodes.py`

### 3.4 Fact Extraction

- [ ] Extract relationships via LLM
- [ ] Format: SOURCE - RELATION_TYPE - TARGET
- [ ] Extract temporal validity from context
- [ ] Reference: `reference/graphiti/graphiti_core/prompts/extract_edges.py`

### 3.5 Fact Deduplication

- [ ] Vector search for similar existing facts
- [ ] LLM decides: ADD / UPDATE / DELETE / NONE
- [ ] Prevents fact explosion from repeated mentions
- [ ] Reference: mem0's deduplication flow

### 3.6 Storage

- [ ] Create `SQLiteEntityStore` with vector index
- [ ] Create `SQLiteFactStore` with vector index + temporal indices
- [ ] Provenance: facts link to source `episode_ids`

### 3.7 Graph Retrieval

- [ ] Search entities and facts via hybrid search
- [ ] BFS traversal from seed entities (1-2 hops)
- [ ] Temporal filtering: only facts valid at query time
- [ ] Reranking: RRF (default), node_distance, episode_mentions

### 3.8 Integration

- [ ] Gate entity/fact extraction behind predict-calibrate
- [ ] Only extract from novel knowledge
- [ ] Reduces extraction volume and LLM costs

---

## 4. Production Backends (Phase 5)

### 4.1 PostgreSQL

- [ ] Implement stores using `asyncpg`
- [ ] `pgvector` for embeddings
- [ ] `pg_trgm` for text search
- [ ] Recursive CTEs for graph traversal
- [ ] Connection pooling

### 4.2 Neo4j (Optional)

- [ ] Native graph traversal
- [ ] Vector index (5.11+)
- [ ] Full-text index

### 4.3 Backend Parity

- [ ] Same tests pass on all backends
- [ ] Same API, different connection string
- [ ] Protocol-based storage abstraction

### 4.4 Migration Tooling

- [ ] Export from SQLite
- [ ] Import to Postgres/Neo4j
- [ ] Data validation

### 4.5 CLI Tools

- [ ] `agentmemory inspect` - view DB contents
- [ ] `agentmemory benchmark` - run evals
- [ ] `agentmemory migrate` - backend migration

---

## 5. v0.2.0 Release

- [ ] Updated documentation for new features
- [ ] LangChain integration example
- [ ] Changelog
- [ ] PyPI publish

---

## Delegation Guide

| Task | Complexity | Delegatable | Dependencies |
|------|------------|-------------|--------------|
| 1.1-1.3 Bi-temporal basics | Medium | Yes | None |
| 1.4 Contradiction handling | Medium | Yes | 1.1-1.3 |
| 1.5 Temporal parsing | Medium | Yes | 1.1 |
| 2.1 Episode chaining | Low | Yes | None |
| 2.2-2.4 Enhancements | Low-Medium | Yes | None |
| 3.1-3.2 Entity models | Medium | Yes | None |
| 3.3 Entity resolution | High | Partial | 3.1-3.2, needs design review |
| 3.4-3.5 Fact extraction | High | Partial | 3.1-3.3, needs design review |
| 3.6-3.7 Graph storage/retrieval | High | Yes | 3.1-3.5 |
| 4.1 PostgreSQL | High | Yes | All Phase 3-4 |
| 4.2 Neo4j | High | Yes | All Phase 3-4 |
| 5. Release | Low | Partial | All above |

---

## Priority Recommendation

1. **Phase 3.5** (Bi-Temporal) - Highest value, enables temporal queries
2. **Phase 3.6** (Enhancements) - Low effort, incremental improvements
3. **Phase 4** (Knowledge Graph) - High effort, major feature
4. **Phase 5** (Backends) - Production readiness

---

## References

- Graphiti entity resolution: `reference/graphiti/graphiti_core/prompts/dedupe_nodes.py`
- Graphiti edge invalidation: `reference/graphiti/graphiti_core/prompts/invalidate_edges.py`
- Graphiti search config: `reference/graphiti/graphiti_core/search/search_config_recipes.py`
- Full roadmap: `ROADMAP.md`
