# Next Steps Plan
Last updated: 2026-03-25

**Mission:** Make memv the best "remember what users said" library. Solid semantic memory first, then procedural/agent memory as the differentiator.

**Current state:** v0.1.1 ready to ship. Core pipeline works end-to-end. 229 tests for ~4.8K LOC. Predict-calibrate extraction, write-time temporal normalization, bi-temporal validity, episode segmentation — no single competitor has all four. user_id denormalization, knowledge CRUD (PR #16), contradiction handling, direct injection (PR #18), duplicate embedding column removal (#21) all complete. LongMemEval harness built, full run deferred.

**What's unique about memv:**
- Predict-calibrate extraction (only Nemori shares this — importance from prediction error)
- Write-time temporal normalization (only SimpleMem shares this — accounts for 56.7% of temporal reasoning per their ablation)
- Bi-temporal validity (only Graphiti shares this — event time vs transaction time)
- Episode segmentation (only Nemori shares this — topic-coherent grouping before extraction)

**What's table-stakes that memv is still missing:**
- ~~Knowledge CRUD through public API~~ (done — PR #16)
- ~~Score/relevance threshold on retrieval~~ (skipped — `top_k` sufficient, see §5)
- ~~Direct knowledge injection (bootstrapping without fake conversations)~~ (done — PR #18)
- A second storage backend (Mem0 has 6+, most have 2+)

**Related docs:**
- `notes/PLAN_v1_backup.md` — previous plan with full competitive-analysis-driven roadmap
- `notes/COMPETITIVE_ANALYSIS.md` — competitor deep dives + API comparison
- `notes/RESOURCES.md` — reading list + library analysis notes

---

## Completed: Tests

183 tests covering models, storage, pipeline, retrieval, and e2e. All passing in CI.

<details>
<summary>Completed checkboxes (click to expand)</summary>

### Storage integration tests
- [x] MessageStore: add, get, get_by_user, get_by_time_range, list_users, count, delete, clear_user, user_isolation
- [x] EpisodeStore: add, get, get_by_user, get_by_time_range (overlap/contained/no-match), count, delete, clear_user, update, JSON roundtrip
- [x] KnowledgeStore: add, get, get_by_episode, get_all, get_current, get_valid_at (bi-temporal), invalidate, count, delete, clear_by_episodes
- [x] VectorIndex: add, search (ordering, top_k, scores, user filter), clear_user
- [x] TextIndex: add, search (match, no-match, top_k, user filter, special chars), clear_user, sanitize_fts_query

### Pipeline integration tests (mock LLM/embeddings)
- [x] Messages → BatchSegmenter → episodes (topic detection, time gap splitting)
- [x] Episodes → EpisodeGenerator → narrative content
- [x] Episodes → PredictCalibrateExtractor → knowledge (cold start + warm)
- [x] Episode merging (similar episodes deduplicated)
- [x] Knowledge deduplication (similar statements filtered)

### Retrieval integration tests
- [x] Hybrid search: vector + BM25 fusion via RRF
- [x] User isolation: user A's knowledge not returned for user B
- [x] Bi-temporal filtering: at_time, include_expired

### End-to-end test
- [x] `Memory` class through full cycle: add_exchange → process → retrieve → verify results
- [x] Auto-processing: buffer threshold → background processing → retrieval

### Acceptance criteria
- [x] All stores have round-trip tests
- [x] Pipeline tested with mock LLM that returns deterministic outputs
- [x] Retrieval tested for correctness, not just no-crash
- [x] CI passes on all tests

</details>

---

## Completed: Atomization

Prompt-level rules, confidence filter, temporal parsing module. Code-level regex filters removed (English-only, moved to prompt-level enforcement for language agnosticism).

<details>
<summary>Completed checkboxes (click to expand)</summary>

### Temporal normalization in extraction prompts
- [x] "yesterday" → absolute date (requires reference timestamp)
- [x] "next Monday" → absolute date
- [x] "last week" → date range
- [x] Thread `reference_time` (from last message timestamp) through extraction pipeline

### Coreference resolution
- [x] "my kids" → "Sarah's kids" (when user_id maps to known name) — prompt-level only
- [x] "he/she/they" → named entity from conversation context — prompt-level only
- [x] "this place" → specific location from context — prompt-level only

### Self-contained statement constraint
- [x] Each extracted knowledge statement interpretable without episode context
- [x] Add explicit prompt instructions: prohibit pronouns, relative time, ambiguous references

### Extraction robustness
- [x] Confidence >= 0.7 threshold. Regex filters removed — English-only, moved to prompt-level.

### Temporal parsing
- [x] `src/memv/processing/temporal.py` — `parse_temporal_expression`, `contains_relative_time`, `backfill_temporal_fields`

### Acceptance criteria
- [x] Self-contained statements enforced via prompts
- [x] Temporal expressions resolved to absolute dates
- [x] Regression tests pass
- [ ] Before/after atomization comparison via LongMemEval (blocked on first benchmark run)

</details>

---

## Completed: Benchmark Harness

LongMemEval harness built. Full run pending.

<details>
<summary>Completed checkboxes (click to expand)</summary>

- [x] Dataset loader, ingestion, search, evaluation, config presets, runner, pipeline parallelization, `make benchmark`

</details>

---

## v0.1.1 — "Make it usable"

**Goal:** A developer can pip install memv, feed conversations, inspect what was learned, fix mistakes, seed known facts, and trust that retrieval doesn't return garbage. Table-stakes for any memory library.

**Internal order:** user_id denorm → CRUD → Contradiction → Injection → ~~Score threshold~~ → ~~Benchmarks~~

**Status:** All items complete. Ready to tag and publish.

### 1. Add user_id to SemanticKnowledge

Knowledge is only linked to users through episodes (join required). Unblocks CRUD, injection, simpler queries.

**Model** (`src/memv/models.py`):
- [x] Add `user_id: str | None = None` to `SemanticKnowledge` (None for backwards compat)

**Schema** (`src/memv/storage/sqlite/_knowledge.py`):
- [x] Migration: `ALTER TABLE semantic_knowledge ADD COLUMN user_id TEXT`
- [x] Index: `CREATE INDEX idx_sk_user_id ON semantic_knowledge(user_id)`
- [x] Backfill: `UPDATE semantic_knowledge SET user_id = (SELECT user_id FROM episodes WHERE id = source_episode_id)`
- [x] Update `add()`, `_row_to_knowledge()`

**Pipeline** (`src/memv/memory/_pipeline.py`):
- [x] Set `user_id` when creating `SemanticKnowledge`

### 2. Knowledge CRUD

Can't use a system you can't inspect. Every competitor has at least add/search/delete.

**New KnowledgeStore methods** (`src/memv/storage/sqlite/_knowledge.py`):
- [x] `list_by_user(user_id, limit=50, offset=0, include_expired=False)`
- [x] `count_by_user(user_id, include_expired=False)`

**New VectorIndex/TextIndex methods**:
- [x] `delete(uuid)` — per-entry delete (currently only `clear_user` exists)

**New Memory API** (`src/memv/memory/_api.py`, `memory.py`):
- [x] `list_knowledge(user_id, limit, offset, include_expired)`
- [x] `get_knowledge(knowledge_id)`
- [x] `invalidate_knowledge(knowledge_id)` — soft-delete (set expired_at)
- [x] `delete_knowledge(knowledge_id)` — hard-delete (DB + vector index + text index)

### 3. Contradiction Handling

Index-based supersedes: existing knowledge passed as numbered list to extraction prompt, LLM outputs index of entry being replaced.

**Model** (`src/memv/models.py`):
- [x] Add `supersedes: int | None = None` to `ExtractedKnowledge` (index into numbered list)
- [x] Add `superseded_by: UUID | None = None` to `SemanticKnowledge`

**Schema** (`src/memv/storage/sqlite/_knowledge.py`):
- [x] Migration: `ALTER TABLE semantic_knowledge ADD COLUMN superseded_by TEXT`
- [x] `invalidate_with_successor(knowledge_id, successor_id)` — atomic expired_at + superseded_by

**Prompts** (`src/memv/processing/prompts.py`):
- [x] `extraction_prompt_with_prediction` accepts `existing_knowledge_numbered` param, inserts numbered list, adds `supersedes` to output format

**Extractor** (`src/memv/processing/extraction.py`):
- [x] `_extract_gaps` passes existing knowledge as numbered list to warm prompt
- [x] `_format_numbered_knowledge` helper

**Pipeline** (`src/memv/memory/_pipeline.py`):
- [x] Index-based: `supersedes` in bounds → `invalidate_with_successor(old_id, new_id)`
- [x] Fallback: `supersedes` is None or out of bounds → vector-based matching
- [x] Handle `knowledge_type == "update"` same as "contradiction"
- [x] Store new entry first, then invalidate old (audit trail needs successor ID)

### 4. Direct Knowledge Injection

Bootstrapping requmires fake conversations without this. Every competitor with a managed API has an inject/add endpoint.

**API** (`src/memv/memory/_api.py`, `memory.py`):
- [x] `add_knowledge(user_id, statement, valid_at=None, invalid_at=None) → SemanticKnowledge`
- [x] `add_knowledge_batch(user_id, items: list[...])` with `embed_batch`

**Implementation**:
- [x] Embed statement, optional dedup check, index in vector + text
- [x] Make `source_episode_id` nullable (None = injected)
- [x] Return the created entry

### 5. Score Threshold Filtering — SKIPPED

PR #19 closed. `top_k` is sufficient for now — with per-user knowledge bases (typically 50-200 facts), retrieval quality isn't a problem. If users report "retrieval returns irrelevant junk," revisit with cosine similarity pre-filter approach (filter vector search candidates before RRF fusion, not normalized RRF scores post-fusion). See PROGRESS.md 2026-03-25 for full rationale.

### 6. Benchmark Runs — DEFERRED to v0.2.0+

Moved out of v0.1.1. Harness is built and smoke-tested (3 questions, 66.7% on fast config). Full run deferred — not blocking adoption. Run when marketing needs numbers.

### v0.1.1 Verification

- All 229 tests pass
- New tests for: CRUD (list, get, invalidate, delete), contradiction (supersedes flow, audit trail), injection (single, batch, dedup check)
- Duplicate embedding column removed (#21) — ~50% DB size reduction
- `make all` passes

---

## v0.2.0 — "Production-ready"

**Goal:** memv can be deployed in production with Postgres. Pluggable backends via protocols, PostgreSQL as first alternative. This unblocks real-world testing and makes the library marketable.

**Internal order:** Protocol cleanup → PostgreSQL backend

### 7. Protocol Cleanup

Current protocols are incomplete — they define read interfaces but omit mutation methods the codebase actually calls. `VectorIndex` and `TextIndex` have no protocol at all. `LifecycleManager` imports concrete SQLite classes directly. This blocks any alternative backend.

- [x] Complete store protocols — add all user-scoped methods (KnowledgeStore: `invalidate`, `invalidate_with_successor`, `delete`, `clear_by_episodes`, `list_by_user`, `count_by_user`; MessageStore: `list_users`, `count`, `delete`, `clear_user`; EpisodeStore: `count`, `delete`, `clear_user`, `update`). Unscoped methods (`get_all`, `get_current`, `get_valid_at`, `count`) intentionally excluded — violate user isolation. SQLite keeps them for dashboard; scoped versions will be added to protocol when needed.
- [x] Add `VectorIndex` protocol (`open`, `close`, `add`, `search`, `search_with_scores`, `has_near_duplicate`, `delete`, `clear_user`)
- [x] Add `TextIndex` protocol (`open`, `close`, `add`, `search`, `delete`, `clear_user`)
- [x] Add `open`/`close` to all store protocols
- [x] Backend factory in `LifecycleManager` — config-driven creation via `MemoryConfig.backend`, lazy imports
- [x] Fix Retriever imports — import from `memv.protocols` instead of `memv.storage`

### 8. PostgreSQL Backend

Production-grade alternative. SQLite is fine for dev/single-process, but anything multi-process or deployed needs Postgres.

| Store | SQLite | PostgreSQL |
|-------|--------|------------|
| MessageStore | Regular SQL | Regular SQL (trivial port) |
| EpisodeStore | SQL + JSON | SQL + `jsonb` |
| KnowledgeStore | SQL + JSON | SQL + `jsonb` |
| VectorIndex | `sqlite-vec` | `pgvector` (`vector` type, `<=>` cosine) |
| TextIndex | FTS5 | `tsvector`/`tsquery` + GIN index |

- [ ] All 5 stores implemented for Postgres (asyncpg + pgvector)
- [ ] `db_url` parameter on `Memory` (`postgresql://...`), mutually exclusive with `db_path`
- [ ] Optional dependency: `pip install memvee[postgres]`
- [ ] Parametrized tests: `@pytest.mark.parametrize("backend", ["sqlite", "postgres"])`
- [ ] CI service container for Postgres

### v0.2.0 Verification

- Parametrized test suite passes on both SQLite and Postgres
- All protocols have complete method coverage matching actual usage
- A new backend can be implemented purely from protocols (no need to read SQLite source)
- `make all` passes

### 9. Embedding Adapters

Only OpenAI today. Users want provider choice and a no-API-key local option. The `EmbeddingClient` protocol is 2 methods — each adapter is ~15 lines.

**Adapters:**
- [ ] Voyage AI (`memvee[voyage]`) — `voyageai` SDK, `voyage-3-lite` default
- [ ] Cohere (`memvee[cohere]`) — `cohere` SDK, `embed-v4.0` default
- [ ] Local/fastembed (`memvee[local]`) — `fastembed` (ONNX, no GPU required), `BAAI/bge-small-en-v1.5` default (384 dims)
- [ ] Update `MemoryConfig.embedding_dimensions` handling — adapters should declare their dimensions so users don't have to set it manually

**Not adding:**
- Sentence-transformers — heavier dependency (PyTorch). `fastembed` covers the local use case with ONNX.
- More vector DB backends (Qdrant, Pinecone, Milvus, Chroma) — they only replace VectorIndex, not TextIndex. memv's hybrid retrieval (vector + BM25 via RRF) is a strength. Splitting storage across two systems adds operational complexity without clear demand. Revisit if users request it.
- More LLM adapters — PydanticAI already supports OpenAI, Anthropic, Google, Groq, Ollama, and others. No work needed.

---

## After v0.2.0: Agent/Procedural Memory

Semantic memory is solid. The unsolved problem — and memv's long-term differentiator — is **procedural memory for agents**. Agents don't just have conversations; they have runs with actions, tool calls, decisions, and outcomes. Learning *how to do things better* from past runs is what no one has solved well.

This is big enough to be its own planning effort. Key questions to answer before committing:

1. **Same library or separate package?** `memvee[agent]` extra vs `agentmemory` as a separate library that depends on memv for semantic storage.
2. **What's the minimum viable procedural memory?** The old plan had 3 levels (tool patterns → workflows → strategies). Maybe level 1 alone is enough to ship and learn.
3. **Which agent framework to target first?** PydanticAI is the obvious choice given existing adapter work.
4. **What does the API look like?** `start_run()` → `add_action()` → `end_run()` → `process_runs()` → `retrieve_for_tool()` is the sketch, but needs validation against real agent architectures.

Defer detailed planning until v0.2.0 ships and there's real usage data for semantic memory.

---

## Ideas Parking Lot

Not committed to. Revisit based on usage data, benchmark results, and user feedback.

| Idea | Source | Notes |
|------|--------|-------|
| Knowledge relationships (`extends` + cascade) | Supermemory | Deferred from v0.2.0 — open design questions on cascade aggression and undo. See PROGRESS.md 2026-03-25. |
| User profiles (static/dynamic split) | Supermemory | Deferred from v0.2.0 — open question on how to classify static vs dynamic (age-based is wrong). |
| DX: improved `to_prompt()` | — | Temporal annotations, source info, configurable format. |
| DX: token-budgeted retrieval | — | `max_tokens` as alternative to `top_k`, accumulate by descending RRF rank. |
| DX: stats API | — | `count_knowledge/messages/episodes(user_id)` through public API. |
| DX: idempotent writes | Supermemory | `custom_id` on add methods for upsert semantics. Prevents duplicates on retry/replay. |
| DX: simplify constructor | — | `MemoryConfig` only, remove 16 duplicate kwargs from `Memory.__init__`. |
| Benchmark runs (LongMemEval) | — | Harness built, smoke-tested. Full run (500 questions) + ablation deferred. Run when marketing needs numbers. |
| Score threshold filtering | PR #19 | If revisited, use cosine similarity pre-filter, not normalized RRF. See PROGRESS.md 2026-03-25. |
| Knowledge categorization | Multiple competitors | Deferred — unclear value without a concrete consumer. Revisit if smart formatting or filtered retrieval becomes needed. |
| Retrieval trigger field (`when_to_use`) | ReMe | Interesting but adds LLM output field + extra embedding per fact. Revisit after benchmarks show retrieval is the bottleneck. |
| Retrieval reinforcement | OpenMemory | Boost frequently-retrieved facts. Adds complexity to scoring. Need data showing it helps. |
| Knowledge compaction | MemMachine | Cluster and merge related facts. Solve when knowledge growth is actually a problem. |
| Simhash pre-dedup | ReMe, OpenMemory | Cheap Hamming distance check before embedding comparison. Optimization — solve when dedup cost is measured. |
| Conversation-aware retrieval | cognee, ReMe | LLM query expansion from conversation context. Adds LLM call to retrieval path. |
| Feedback loop | cognee | Confidence adjustment from user corrections. Requires agent framework integration. |
| Extraction cost tracking | — | `ProcessingResult` from `process()`. Nice-to-have observability. |
| Hooks/Events | — | `EventBus` for composability. Useful for framework integration. |
| Memory scoping | — | Namespaces for different memory spaces. No concrete request yet. |
| MCP server | — | Separate package if demand materializes. |
| Search results with graph context | Supermemory | Return `context.parents[]` + `context.children[]` with relationship types in retrieval results. Depends on §7 (extends). Retrieval-time join. |
| Memory Router (proxy pattern) | Supermemory | Reverse proxy between app and LLM provider, auto-injects memories. Great adoption UX but wrong layer for a library. Revisit as separate package. |
| `derives` relationship | Supermemory | Inferred knowledge from combining facts, marked `isInference=true`. No validated use case yet. |
| `forgetAfter` / scheduled expiration | Supermemory | Auto-expire memories after a date. We have `invalid_at` (event time) but no automatic pruning at retrieval. |

### Not doing
| Item | Why |
|------|-----|
| Full Knowledge Graph | `extends` + cascade invalidation adopted (§7). Full graph (Neo4j, entity-relation triples, graph traversal queries) is still overkill. `derives` relationship deferred — no validated use case yet. |
| Neo4j backend | Niche. Postgres covers production needs. |
| Background consolidation | Premature without knowledge growth data. |
| `reflect()`-style generation | Wrong layer — memory retrieves, agent generates. |
| Entropy-based pre-filtering | Predict-calibrate already handles extraction quality. |
| Cognitive sector classification | Over-engineered (OpenMemory's 5-sector taxonomy with 5x5 resonance matrix). |
| Multi-phase chain-of-thought retrieval | Adds latency and LLM calls to retrieval. |
