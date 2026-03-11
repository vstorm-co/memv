# Next Steps Plan
Last updated: 2026-03-11

**Mission:** Make memv the best "remember what users said" library. Solid semantic memory first, then procedural/agent memory as the differentiator.

**Current state:** v0.1.0 shipped (alpha, Feb 2026). Core pipeline works end-to-end. 215 tests for ~4.8K LOC. Predict-calibrate extraction, write-time temporal normalization, bi-temporal validity, episode segmentation — no single competitor has all four. LongMemEval benchmark harness built, full run pending. user_id denormalization and knowledge CRUD complete (PR #16). Contradiction handling with index-based supersedes implemented.

**What's unique about memv:**
- Predict-calibrate extraction (only Nemori shares this — importance from prediction error)
- Write-time temporal normalization (only SimpleMem shares this — accounts for 56.7% of temporal reasoning per their ablation)
- Bi-temporal validity (only Graphiti shares this — event time vs transaction time)
- Episode segmentation (only Nemori shares this — topic-coherent grouping before extraction)

**What's table-stakes that memv is still missing:**
- ~~Knowledge CRUD through public API~~ (done — PR #16)
- Score/relevance threshold on retrieval
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

**Internal order:** user_id denorm → CRUD → Contradiction → Injection → Score threshold → Benchmarks

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

### 5. Score Threshold Filtering

`retrieve()` always returns `top_k` results regardless of relevance. Low-quality results waste context tokens.

**Retrieval** (`src/memv/retrieval/retriever.py`):
- [ ] Add `min_score: float | None = None` to `retrieve()`
- [ ] Post-RRF: filter results below threshold
- [ ] `allow_empty: bool = False` — always return at least 1 result unless explicitly allowed

**Config** (`src/memv/config.py`):
- [ ] `default_min_score: float | None = None` — global default, overridable per-call

**API**: Pass `min_score` through `Memory.retrieve()` → `Retriever.retrieve()`

### 6. Benchmark Runs

Run the harness, get numbers. Can't market predict-calibrate without data.

- [ ] First full run (500 questions) — overall + per-type accuracy
- [ ] Ablation: with/without predict-calibrate, with/without episode segmentation
- [ ] Compare against Nemori/Zep published baselines
- [ ] Published LongMemEval numbers for memv

### v0.1.1 Verification

- All 206 existing tests pass
- New tests for: CRUD (list, get, invalidate, delete), contradiction (supersedes flow, audit trail), injection (single, batch, dedup check), score threshold (filtering, allow_empty)
- `make all` passes

---

## v0.2.0 — "Production-ready"

**Goal:** memv can be used in production. Pluggable backends, a Postgres option, cleaner DX. After this release, semantic memory is solid enough to pivot focus to agent/procedural memory.

### 7. Protocol Cleanup

Current protocols are incomplete — they define read interfaces but omit mutation methods the codebase actually calls. `VectorIndex` and `TextIndex` have no protocol at all. `LifecycleManager` imports concrete SQLite classes directly. This blocks any alternative backend.

- [ ] Complete store protocols — add all methods actually used (KnowledgeStore: `get_all`, `get_current`, `get_valid_at`, `invalidate`, `delete`, `clear_by_episodes`, `count`, `list_by_user`, `count_by_user`; MessageStore: `list_users`, `count`, `delete`, `clear_user`; EpisodeStore: `count`, `delete`, `clear_user`, `update`)
- [ ] Add `VectorIndex` protocol (`open`, `close`, `add`, `search`, `search_with_scores`, `delete`, `clear_user`)
- [ ] Add `TextIndex` protocol (`open`, `close`, `add`, `search`, `delete`, `clear_user`)
- [ ] Add `open`/`close` to all store protocols
- [ ] Backend factory in `LifecycleManager` — config-driven creation instead of hardcoded SQLite imports
- [ ] Fix Retriever imports — import from `memv.protocols` instead of `memv.storage`

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

### 9. DX Improvements

Friction points identified from API analysis.

**Retrieval output:**
- [ ] Expose RRF scores on `RetrievalResult` — `list[tuple[SemanticKnowledge, float]]` or scores dict
- [ ] Improve `to_prompt()` — temporal annotations ("learned 3 weeks ago"), source info, configurable format
- [ ] Token-budgeted retrieval — `max_tokens` as alternative to `top_k`, accumulate by descending score until budget exhausted

**API surface:**
- [ ] `count_knowledge(user_id)` / `count_messages(user_id)` / `count_episodes(user_id)` — stats through public API
- [ ] Make `embedding_client` optional at construction — only required when calling `process()` or `retrieve()`, not for storage-only use
- [ ] `get_episode(episode_id)` — navigate from knowledge back to source conversation context

**Config:**
- [ ] Simplify constructor — `MemoryConfig` only, remove 16 duplicate kwargs from `Memory.__init__`

### v0.2.0 Verification

- Parametrized test suite passes on both SQLite and Postgres
- All protocols have complete method coverage matching actual usage
- A new backend can be implemented purely from protocols (no need to read SQLite source)
- `make all` passes

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

### Not doing
| Item | Why |
|------|-----|
| Knowledge Graph | High effort, low urgency. Flat semantic knowledge works. |
| Neo4j backend | Niche. Postgres covers production needs. |
| Background consolidation | Premature without knowledge growth data. |
| `reflect()`-style generation | Wrong layer — memory retrieves, agent generates. |
| Entropy-based pre-filtering | Predict-calibrate already handles extraction quality. |
| Cognitive sector classification | Over-engineered (OpenMemory's 5-sector taxonomy with 5x5 resonance matrix). |
| Multi-phase chain-of-thought retrieval | Adds latency and LLM calls to retrieval. |
