# Progress

Append-only log of plan changes. Each entry records what was updated and why.

## 2026-02-20

- Added: checkboxes to all PLAN.md items — every deliverable, sub-task, and acceptance criterion is now trackable as `[ ]`/`[x]`
- Added: CLAUDE.md sections for MemoryConfig, public API, protocols incompleteness note, CI workflows, ruff rules
- No items checked off — no project feature work done this session

## 2026-02-20 (session 2)

- No items checked off — commit f472635 was notes consolidation + CLAUDE.md updates, no feature work
- Added: "Plan Tracking" section to CLAUDE.md — instructs Claude to update plan after committing code changes
- Decision: session notes moved from `logs/claude/sessions/` to `notes/claude/worklog/YYYY/MM/DD/<id>.md` (zettelkasten-style IDs, date-based directory structure)
- Fixed: `plan-status.sh` timezone bug — `git log --since` with bare date uses UTC midnight, missing same-day commits in CET. Changed to `--after` with previous day.

## 2026-02-20 (session 3)

- Removed: "Episode retrieval: include_episodes flag" from retrieval tests — episodes removed from retrieval surface entirely (commit df88793, PR #8, closes #6)
- Removed: "Apply same logic to episodes if include_episodes=True" from token-budgeted retrieval — no longer applicable
- Added: "Code-level filter to enforce user-only extraction" under Atomization — for local/smaller model support (#9)
- Added: "Minimum relevance threshold" under Token-Budgeted Retrieval — filter low-scoring results (#10)
- Decision: episodes are an internal processing artifact, not a retrieval concept. Removed from retrieval API, RetrievalResult model, and indices. Orphaned SQLite tables in existing DBs accepted (inert, no migration needed).
- Decision: removed `RetrievalResult.as_text()` — near-duplicate of `to_prompt()` after episode removal, no production callers
- Decision: removed dead config fields `search_top_k_episodes` and `search_top_k_knowledge` — never wired to any call site

## 2026-02-21

- Checked off: all 5 storage integration test items (MessageStore, EpisodeStore, KnowledgeStore, VectorIndex, TextIndex) — 68 async tests in PR #11 (commit 7a228e4)
- Checked off: "All stores have round-trip tests" acceptance criterion — every store has add+get round-trip coverage
- Updated: storage test item descriptions to reflect actual test coverage (broader than originally planned, e.g. time range queries, bi-temporal filtering, JSON roundtrip)
- Added: `asyncio_mode = "auto"` to pytest config, shared fixtures/factories in `tests/conftest.py`

## 2026-02-21 (session 2)

- Checked off: all 5 pipeline integration test items — 10 BatchSegmenter + 7 EpisodeGenerator + 7 Extractor + 8 EpisodeMerger + dedup via e2e (PR #12, commit 40cc5c8)
- Checked off: all 3 retrieval integration test items — 10 Retriever tests covering RRF fusion, user isolation, bi-temporal filtering (PR #12)
- Checked off: both e2e test items — 10 Memory lifecycle tests including full cycle, auto-processing, flush, dedup (PR #12)
- Checked off: "Pipeline tested with mock LLM" and "Retrieval tested for correctness" acceptance criteria
- Added: MockLLM (sequential canned responses) and MockEmbedder (SHA-256 hash → deterministic unit vector) to conftest.py
- Added: `pipeline_stores` fixture (all 5 stores on single temp DB) for pipeline/e2e tests
- PR #11 review fixes committed on `test/storage-integration-tests` (9c94f8f): narrowed exception to `ImportError`, module-level uuid4, isolation test, sanitize_fts_query tmp_path, special chars assertion
- PR #12 review fixes committed on `test/pipeline-e2e-tests` (fce5540): moved hashlib/struct imports to top (E402 fix), narrowed exception guards to `vec_idx.open()` only, fixed resource leak on skip, replaced `__new__` hack
- Resolved merge conflicts after PR #11 merged to main (4212e57)
- Deleted `develop` branch (local + remote) — unused
- Updated: test count in plan header from "10 model unit tests" to "131 tests"

## 2026-02-21 (session 3)

- Checked off: "CI passes on all tests" — CI workflow green on main (commit dbf9080, both CI and Deploy Documentation succeeded)
- Section 1 (Tests) is now fully complete — all 19/19 checkboxes done
- Decision: next work item is Section 2 (Atomization) — self-contained statement constraint + temporal normalization/parsing identified as starting points

## 2026-02-21 (session 4)

- Checked off: all 4 temporal normalization items — `ATOMIZATION_RULES` in prompts, strengthened `reference_timestamp` wording, `temporal.py` module handles parsing
- Checked off: all 3 coreference resolution items — prompt-level via `ATOMIZATION_RULES` (code-level deferred per plan)
- Checked off: both self-contained statement constraint items — `ATOMIZATION_RULES` constant + injected into both extraction prompts
- Checked off: code-level extraction filter (#9) — `_validate_extraction` expanded with 4 checks: third-person, first-person regex, relative time, assistant-sourced patterns
- Checked off: all 4 temporal parsing items — `src/memv/processing/temporal.py` created with `contains_relative_time`, `parse_temporal_expression`, `backfill_temporal_fields`
- Checked off: 3 of 4 acceptance criteria — self-contained statements enforced, temporal resolution working, regression tests pass. "Before/after comparison on sample conversations" remains open (needs benchmarks).
- Added: `python-dateutil>=2.9.0` to `pyproject.toml` — explicit dep for temporal parsing (was transitive)
- Added: `tests/test_temporal.py` (32 tests), `tests/test_atomization.py` (12 tests), 3 new tests in `test_extractor.py`, 1 new e2e test in `test_memory_e2e.py`
- Fixed: 4 existing e2e test fixtures updated to use "User"-prefixed statements (required by new third-person validation check)
- Updated: test count 131 → 179, LOC ~4.4K → ~4.8K
- Decision: coreference resolution is prompt-only — code-level NLP deps or extra LLM calls deferred. SimpleMem proves prompt-only works.
- Decision: no pronoun rejection in code — too many false positives ("User uses Python because **it** is readable"). Third-person + first-person checks are high-precision.
- Decision: `user_name` config field deferred — "User" as subject is sufficient for self-containedness
- Section 2 (Atomization) is now nearly complete — 17/18 checkboxes done, only "before/after comparison" remains (blocked on Section 3 benchmarks)

## 2026-02-21 (session 5)

- Fixed: `startswith("User")` → `^User\b` word-boundary regex — PR #13 review feedback (commit 09de2b3)
- Fixed: `_FIRST_PERSON_RE` I/O false positive — added negative lookahead/lookbehind for "/" around "I" (commit ee0408e)
- Fixed: `rstrip("s")` → `removesuffix("s")` in temporal parser — more precise intent (commit 09de2b3)
- Fixed: `_ASSISTANT_SOURCE_RE` expanded to include told/instructed/encouraged/shown/given (commit 09de2b3)
- Fixed: `_FIRST_PERSON_RE` case-insensitive for my/me/we/our but case-sensitive for I (commit 09de2b3)
- Added: `_validate_extraction` docstring documenting `^User\b` scope constraint — third-party facts intentionally dropped (commit ee0408e)
- Added: tests for same-weekday edge case, backfill with relative date, assistant-sourced "told" variant (commit 09de2b3)
- Fixed: `test_user_isolation_e2e` fixtures — "User1"/"User2" → "User" to comply with word-boundary check (commit 09de2b3)
- Updated: test count 179 → 182
- Decision: code-level regexes are English-only and intentionally non-exhaustive — they're a safety net for high-frequency failures, not a complete solution. If benchmarks show leakage, the right fix is an LLM-based validation pass, not more regexes.
- Decision: hardcoded dates in few-shot prompt examples kept as-is — placeholders would reduce example utility. ATOMIZATION_RULES (instructions) already uses placeholder syntax.

## 2026-02-21 (session 6)

- Fixed: missing `_ASSISTANT_SOURCE_RE` code-level check — was referenced in test but never implemented; added with infinitive requirement (`was advised to`) to avoid false positives on "User was given a promotion" (commit 5f46be1)
- Fixed: `_THIRD_PERSON_RE` `r"^User\b"` → `r"^[Uu]ser\b"` — LLMs occasionally output lowercase "user" (commit 5f46be1)
- Fixed: `began?` regex typo in `_SINCE_PATTERN` → `began` — the `?` made "a" optional, matching "beg" (commit 5f46be1)
- Fixed: removed `later`/`earlier` from `_RELATIVE_PATTERNS` — caused false positives on adjective usage like "the earlier version" (commit 5f46be1)
- Fixed: removed bare `to` from `_UNTIL_PATTERN` — too broad, matched non-temporal uses like "related to data pipelines" (commit 5f46be1)
- Refactored: extracted `_WEEKDAY_NAMES` module-level constant in `temporal.py`, replaced duplicate inline list + `.index()` with `enumerate` (commit 5f46be1)
- Fixed: `test_rejects_first_person` was testing `_THIRD_PERSON_RE`, not `_FIRST_PERSON_RE` — changed statement to "User mentioned that my team uses React" so it passes third-person check and actually exercises first-person filter (commit a572e2b)
- Added: 7 new tests — assistant-source rejection, passive without infinitive accepted, lowercase "user" accepted, earlier/later not flagged, bare "to" not matched, "began" prefix, first-person in User statement
- Updated: test count 182 → 187
- Decision: `_ASSISTANT_SOURCE_RE` requires infinitive (`to`) after passive verb — scopes to assistant advice patterns ("was advised to try X") without catching legitimate user facts ("was given a promotion")

## 2026-02-21 (session 7)

- Removed: all code-level regex filters from `_pipeline.py` (`_THIRD_PERSON_RE`, `_FIRST_PERSON_RE`, `_ASSISTANT_SOURCE_RE`, `contains_relative_time`) — English-only, moved to prompt-level enforcement for language agnosticism. `_validate_extraction` now only checks confidence >= 0.7.
- Added: nuanced assistant-contamination rules in `prompts.py` — "DO extract factual info from assistant", "Treat assistant suggestions as speculative", "Attribute preferences only to explicit user claims" (replaces blanket "IGNORE all ASSISTANT lines")
- Checked off: 5 LongMemEval evaluation harness items — dataset.py, add.py, search.py, evaluate.py, config.py all created in `benchmarks/longmemeval/`
- Checked off: `make benchmark` target — Makefile updated with 3-stage pipeline
- Updated: Section 3 header from "LoCoMo Benchmarks" to "Benchmarks (LongMemEval)" — LongMemEval uses user-assistant paradigm (matches memv), LoCoMo uses two-human conversations (mismatch)
- Updated: test count 187 → 183 — removed 5 regex-specific tests, added 1 confidence boundary test
- Decision: regex filters are an English-only dead end. Quality enforcement belongs in prompts (language-agnostic) not code (regex). Confidence threshold (0.7) is the only code-level gate.
- Decision: LongMemEval over LoCoMo — ICLR 2025, user-assistant chat histories, tests 5 memory abilities that map to memv features. Dataset: `xiaowu0162/longmemeval-cleaned` (500 questions)

## 2026-02-21 (session 8)

- Created `feat/benchmarks` branch from `main` — dedicated branch for benchmark harness work
- Popped stash `feat(benchmark): LongMemEval harness + gitignore + Makefile` (from `feat/atomization`) onto new branch
- No new items checked off — benchmark harness was already tracked as complete in session 7, this session is branch setup only

## 2026-02-22

- Checked off: 3 new harness items — `run.py` runner, concurrent processing in add/search, pipeline parallelization (commits 0d55bcf, 698fa6e on `feat/benchmarks`)
- Added: `fast` config preset to config.py — skips predict-calibrate + dedup for quick iteration
- Added: concurrent question processing in add.py/search.py — semaphore + gather + JSONL checkpoint/resume + per-question timeout
- Added: `run.py` end-to-end runner — single entry point with `--model`/`--stages`/`--max-concurrent` flags
- Added: pipeline parallelization in `src/memv/` — concurrent episode processing (`_pipeline.py`) + concurrent segmentation batches (`batch_segmenter.py`), both with `Semaphore(10)`
- Fixed: extraction quality — scoped prompts to user-specific knowledge. 2035→441 facts (78% reduction) at same accuracy (2/3 smoke). Root cause: "Extract ALL concrete facts" directive extracted general knowledge (radiation therapy, Bitcoin, cooking tips). Changed to quality-over-quantity + explicit exclusions for general/topical knowledge and assistant educational content.
- Decision: pipeline parallelization is safe — Nemori does the same, dedup handles overlap between concurrent episodes
- Decision: extraction prompts are the quality gate, not code-level regex — English-only regexes were already removed, now prompts explicitly exclude general knowledge with concrete bad examples
- Updated: Section 3 header items to reflect actual harness capabilities (concurrency, checkpoint, runner)
- Updated: dependency notes — harness is smoke-tested, full run pending

## 2026-02-25

- Fixed: PR #15 CI lint failure — wrapped long prompt strings in evaluate.py to fit 135 char limit (94d697c)
- Fixed: PR #15 review feedback — extracted JSONL checkpoint helpers to `_checkpoint.py`, added checkpoint/resume to evaluate.py, excluded errored items from accuracy scoring, removed hardcoded `enable_episode_merging=False` from add.py, added `--no-resume` to run.py, added type annotations, fixed UTC timestamp (1eaaa47)
- Fixed: PR #15 review — clarified concurrent episode comment re: aiosqlite write serialization (c67e372)
- Fixed: deslop pass — removed 20 lines of obvious section-label comments, trimmed pipeline concurrency comment from 6→2 lines (4eb190d)
- Decision: concurrent episode processing tradeoff documented — episodes see stale KB, predict-calibrate can't suppress intra-batch dupes, dedup handles it. Matches Nemori's approach.

## 2026-03-03

Major plan expansion — no code changes, pure planning session. Analyzed 5 new reference libraries (claude-mem, cognee, MemMachine, OpenMemory, ReMe), extracted actionable ideas, redesigned v0.4/v0.5.

- Added: "Beyond v0.1" section to PLAN.md — reorganizes remaining work into v0.1.1 through v0.5.0 releases
- Added: section 9 (Knowledge Categorization) — `KnowledgeCategory` StrEnum, schema migration, prompt/pipeline/retrieval changes
- Added: section 10 (user_id denormalization on SemanticKnowledge) — unblocks CRUD and injection
- Added: section 11 (Knowledge CRUD) — list/get/invalidate/delete through Memory API
- Added: section 12 (Contradiction Handling) — `supersedes` field on extraction, `superseded_by` audit trail, improved invalidation precision
- Added: section 13 (Direct Knowledge Injection) — bypass pipeline, `add_knowledge`/`add_knowledge_batch`, nullable `source_episode_id`
- Added: section 14 (Extraction Cost Tracking) — `ProcessingResult` dataclass returned from `process()`
- Added: section 15 (Simhash Pre-Dedup) — 64-bit simhash, Hamming distance ≤ 3 skips embedding comparison (inspired by ReMe/OpenMemory)
- Added: section 16 (Score Threshold Filtering) — `min_score` param on `retrieve()`, post-RRF filtering (inspired by MemMachine/OpenMemory)
- Added: section 17 (Token-Budgeted Retrieval) — extends existing section 5 with RRF score passthrough
- Added: section 18 (Retrieval Trigger Field) — `retrieval_hint` embedded instead of statement for vector search (inspired by ReMe's `when_to_use`)
- Added: section 19 (Retrieval Reinforcement) — `retrieval_count`/`last_retrieved_at`, score boost for useful facts (inspired by OpenMemory)
- Added: section 20 (Conversation-Aware Retrieval) — `QueryExpander` with LLM-generated sub-queries
- Added: section 21 (Smart Retrieval Formatting) — group by category, temporal annotations, budget-aware
- Added: section 22 (Retrieval Feedback Loop) — `record_feedback()` API, asymmetric confidence adjustment (inspired by cognee)
- Added: section 23 (Hooks/Events) — `EventBus` with typed events, pipeline integration
- Added: section 24 (Protocol Cleanup) — extends existing section 6a
- Added: section 25 (PostgreSQL Backend) — extends existing section 6b-d
- Added: section 26 (Memory Scoping/Namespaces) — `scope` field, session lifecycle
- Added: section 27 (PydanticAI Middleware Example) — extends existing section 7
- Added: section 28 (Knowledge Compaction) — cluster similar facts, LLM merge, soft-delete originals
- Added: sections 29-35 (v0.4.0 — Episodic memory + tool patterns) — `AgentRun`/`RunAction` model, run storage, narrative generation, episodic retrieval, tool pattern extraction (Level 1), generic SDK, PydanticAI adapter
- Added: sections 36-39 (v0.5.0 — Full procedural intelligence) — workflow patterns (Level 2), strategy patterns (Level 3), cross-run analysis engine, blended retrieval v2
- Added: packaging structure — `memvee[chat]`, `memvee[agent]` (includes chat), `memvee[pydantic-ai]`, `memvee[postgres]`, `memvee[all]`
- Added: schema migration summary for all releases (v0.1.1 through v0.5.0 tables)
- Added: dependency graph organized by release
- Added: verification criteria for v0.4.0 and v0.5.0
- Added: 5 library analyses to notes/RESOURCES.md (claude-mem, cognee, MemMachine, OpenMemory, ReMe)
- Decision: memv differentiates on episodic + procedural memory for agents, not just semantic memory for chatbots
- Decision: agents have "runs" (goal-directed execution sessions), not just conversations — this is the fundamental unit for episodic/procedural memory
- Decision: procedural memory has 3 levels — tool patterns (v0.4), workflow patterns (v0.5), strategy patterns (v0.5)
- Decision: SDK-first integration — generic SDK as contract, PydanticAI adapter as first convenience layer
- Decision: `memvee[agent]` includes `memvee[chat]` — agents need semantic memory too
- Decision: packaging split happens at v0.4.0 when agent modules land, single package until then
- Decision: v0.4 = episodic + tool patterns (foundation), v0.5 = workflows + strategies + cross-run analysis (full vision)

## 2026-03-08

Major plan restructure — cut scope from 35 sections across 5 releases down to 9 sections across 2 releases. Backed up old plan to `notes/PLAN_v1_backup.md`.

- Decision: "Become the best remember-what-users-said library first." Semantic memory must be a very solid option before expanding to agent/procedural memory.
- Decision: procedural/agent memory is the long-term differentiator (unsolved problem) but deferred to after v0.2.0. May be same library or separate package — TBD after usage data.
- Decision: knowledge categorization deferred — no concrete consumer yet, premature to add schema/prompt complexity.
- Removed from v0.1.1: categorization, simhash pre-dedup, extraction cost tracking — not blocking adoption.
- Removed: v0.3.0 (memory scoping), v0.4.0 (episodic memory), v0.5.0 (procedural intelligence) as committed releases — moved to "After v0.2.0" section and "Ideas Parking Lot".
- Added: v0.2.0 "Production-ready" — protocol cleanup, PostgreSQL backend, DX improvements (RRF scores on results, improved to_prompt, token-budgeted retrieval, stats API, simplified constructor).
- Added: "After v0.2.0" section framing agent memory as its own planning effort with open questions.
- Added: "Ideas Parking Lot" — all speculative items in one table with source and notes. Not committed to.
- Restructured: competitive position summary at top of plan — what's unique (4 features), what's table-stakes missing (4 gaps).
- Kept v0.1.1 items: user_id denorm, CRUD, contradiction, injection, score threshold, benchmark runs.

## 2026-03-08 (session 2)

- Checked off: all 4 items under "1. Add user_id to SemanticKnowledge" — model field, migration, index, backfill, add/row_to_knowledge updates (commit a635bd7)
- Checked off: all 4 items under "2. Knowledge CRUD" — `list_by_user`/`count_by_user` on KnowledgeStore, `delete(uuid)` on VectorIndex/TextIndex, 4 CRUD methods on Memory API (commit 7390a3c)
- Checked off: "Set user_id when creating SemanticKnowledge" in pipeline (commit a635bd7)
- Updated: test count 183 → 206 — added 23 tests (KnowledgeStore: 5, VectorIndex: 3, TextIndex: 3, e2e: 11, conftest: 1)
- Updated: "table-stakes missing" list — struck through Knowledge CRUD (done)
- PR #16 opened: `feat/knowledge-crud` → `main` — combines user_id denorm + CRUD in one PR
- Decision: combined user_id denorm and CRUD into single branch/PR — CRUD depends on user_id, shipping separately adds no value
- Decision: `delete_knowledge` does three-store cleanup (DB + vector + text) — caller shouldn't need to know about index implementation
- Decision: bare `except Exception` narrowed to `aiosqlite.OperationalError` in migration backfill — catch only what's expected

## 2026-03-08 (session 3)

- PR #16 merged to main (squash, commit 780b2d0) — user_id denormalization + knowledge CRUD
- Fixed: PR review feedback — removed incorrect `# noqa: F401` on runtime-used `UUID`, moved `SemanticKnowledge` to `TYPE_CHECKING`, narrowed migration `except` to reraise unless "no such table" (commit 2caf4f7)
- Skipped: 4 review items — non-atomic delete comment (obvious from code), NULL user_id startup warning (transient edge case), `count_by_user` on Memory (v0.2.0 scope), limit/offset bounds validation (trusted callers)
- Decision: `supersedes` field on `ExtractedKnowledge` will use verbatim existing statement (not semantic description) — existing knowledge is already in prediction prompt context, so LLM can copy exact text for precise matching

## 2026-03-08 (session 4)

- Checked off: all 10 items under "3. Contradiction Handling" — index-based supersedes implemented end-to-end
- Changed: `supersedes` field type from `str` to `int | None` — index into numbered existing knowledge list (more precise than verbatim text matching)
- Added: `superseded_by: UUID | None` to `SemanticKnowledge` model — audit trail linking old → new facts
- Added: `superseded_by TEXT` column to schema + migration (`_migrate_add_superseded_by_column`)
- Added: `invalidate_with_successor(knowledge_id, successor_id)` to KnowledgeStore — atomic update of expired_at + superseded_by
- Added: `existing_knowledge_numbered` param to `extraction_prompt_with_prediction` — numbered `[0] statement` list
- Added: `_format_numbered_knowledge()` helper in extraction.py
- Added: `_handle_supersedes()` in pipeline — three paths: valid index → invalidate_with_successor, out-of-bounds → vector fallback, None → vector fallback
- Changed: pipeline processes store-then-invalidate (not invalidate-then-store) — successor ID needed for audit trail
- Changed: `knowledge_type == "update"` now triggers invalidation (was only "contradiction" before)
- Updated: conftest `make_knowledge` factory to accept `superseded_by` param
- Added: 9 new tests — 3 knowledge store (invalidate_with_successor, already expired, roundtrip), 2 extractor (supersedes preserved, numbered knowledge in prompt), 4 e2e (contradiction with supersedes, update type, out-of-bounds, fallback without supersedes)
- Updated: test count 206 → 215

## 2026-03-11

- Checked off: `add_knowledge` / `add_knowledge_batch` API — implemented in `_api.py` + `memory.py`, commit 3a09cdd
- Checked off: `source_episode_id` nullable — `UUID | None`, `None = injected`, storage handles both cases
- Checked off: dedup check on injection — reuses `VectorIndex.has_near_duplicate` (extracted from `Pipeline._is_duplicate_knowledge`)
- Checked off: batch injection with `embed_batch` — `KnowledgeInput` dataclass added, exported from `memv`
- Changed: `clear_user` in `_api.py` and `dashboard/app.py` simplified — `KnowledgeStore.clear_user(user_id)` added, no longer fetches episodes first
- Added: `VectorIndex.has_near_duplicate()` helper — dedup logic shared between Pipeline and injection API
- Updated: test count 215 → 248 (5 e2e injection tests, 4 vector index tests)

## 2026-03-25

- Closed: PR #19 (score threshold filtering) — approach was wrong. RRF scores are rank-based, not relevance-based. Normalizing them to [0,1] produces unintuitive thresholds (single-source results cap at 0.5 regardless of actual relevance). Competitor analysis: Supermemory/Graphiti threshold on cosine similarity, Letta uses RRF but doesn't threshold on it.
- Skipped: §5 (Score Threshold Filtering) entirely — `top_k` is sufficient for per-user knowledge bases. If revisited later, correct approach is cosine similarity pre-filter (not normalized RRF post-filter).
- Added: open questions to §7 (Knowledge Relationships) — cascade invalidation may be too aggressive (not all children are context-dependent on parent), no undo mechanism for cascade
- Redesigned: §8 (User Profiles) — age-based static/dynamic classification replaced with open question. Age doesn't correlate with stability ("User's name is X" is static from day 1). Need to decide classification approach before implementing. Removed `profile_static_age_days` config.
- Removed: "Expose RRF scores on RetrievalResult" from §11 (DX) — RRF scores aren't meaningful to callers
- Updated: §11 token-budgeted retrieval wording — "by descending RRF rank" not "by descending score" (rank order matters, score magnitude doesn't)
- Updated: §9 VectorIndex protocol — removed `search_with_scores` (unnecessary now that §5 is skipped)
- Reverted: 4 local GSD planning commits (`.planning/` directory) — added to `.git/info/exclude`
- Committed: `b6fe578` — drop duplicate embedding TEXT column from semantic_knowledge (#21). ~50% DB size reduction.
- Restructured: v0.1.1 is now complete — benchmarks (§6) deferred to v0.2.0+, all other items done.
- Restructured: v0.2.0 narrowed to Protocol cleanup (§7, was §9) + PostgreSQL backend (§8, was §10) only. Knowledge relationships, user profiles, DX improvements, and benchmarks moved to Ideas Parking Lot. Driver: CEO pressure to ship marketable release + friend needs Postgres for real-world testing.
- Decision: ship v0.1.1 now, then focus v0.2.0 entirely on Postgres to unblock first external user.

## 2026-03-27

- Checked off: all 6 items under §7 (Protocol Cleanup) — complete store/index protocols, VectorIndex/TextIndex protocols, open/close, backend factory, Retriever imports. Merged as PR #22.
- Checked off: all 5 items under §8 (PostgreSQL Backend) — 5 stores in `src/memv/storage/postgres/`, `db_url` param, `memvee[postgres]` optional dep, parametrized test fixtures via `--backend`, CI service container with `pgvector/pgvector:pg17`. PR #23.
- Added: §9 (Embedding Adapters) — Voyage, Cohere, local/fastembed. Not adding: more vector DBs (hybrid retrieval is a strength), more LLM adapters (PydanticAI covers it).
- Decision: unified `db_path`+`db_url` into single `db_url` parameter. File path for SQLite, `postgresql://...` for Postgres. Backend auto-detected from URL prefix or set explicitly via `MemoryConfig.backend` ("auto"/"sqlite"/"postgres"). Breaking change from v0.1.1.
- Decision: unscoped methods (`get_all`, `get_current`, `get_valid_at`, `count`) excluded from protocols — violate user isolation. SQLite keeps them for dashboard. Scoped versions to be added when needed.
- Decision: L2 distance in pgvector (not cosine) to match sqlite-vec behavior — keeps `has_near_duplicate` thresholds consistent across backends.
- Decision: no mapping tables in Postgres — pgvector and tsvector/tsquery support WHERE clauses directly (unlike sqlite-vec/FTS5).
- Decision: `plainto_tsquery` over `to_tsquery` in Postgres TextIndex — handles stop words gracefully without manual sanitization.
- Decision: per-fixture pool creation in tests (not shared) — shared pool breaks across pytest-asyncio event loops.
- Decision: `user_id NOT NULL` in Postgres semantic_knowledge schema (SQLite keeps nullable for backwards compat with legacy data).
- Verified: demo.py works end-to-end against both SQLite and Postgres.
- Updated: current state summary — v0.2.0 nearly complete, embedding adapters (§9) remaining.
