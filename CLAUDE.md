# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

memv is a structured, temporal memory system for AI agents. It extracts and retrieves knowledge from conversations using a predict-calibrate approach (inspired by Nemori): importance emerges from prediction error, not upfront LLM scoring.

## Development Setup

Uses `uv` for dependency management. Requires Python 3.13.

```bash
make install                # uv sync + pre-commit hooks
make sync                   # update deps
```

## Commands

```bash
# Quality
make format                 # ruff format + fix
make lint                   # ruff check + format check
make typecheck              # ty check src/
make all                    # format + lint + typecheck + test

# Testing
uv run pytest                                    # all tests
uv run pytest tests/test_models.py::test_name    # single test

# Docs
make docs                   # mkdocs build --strict
make docs-serve             # local preview

# Pre-commit
uv run pre-commit run --all-files
```

Pre-commit hooks (ruff-check, ruff-format, ty) run on `src/` only.

## Architecture

### Data Flow

```
Messages → BatchSegmenter → Episodes → EpisodeGenerator → PredictCalibrateExtractor → SemanticKnowledge
                                                                    ↓
                                                    VectorIndex + TextIndex (per type)
                                                              ↓
                                                    Retriever (RRF fusion)
```

### Key Modules (`src/memv/`)

**`memory/`** — `Memory` class is the public API. Internally split into:
- `_api.py`: add_message, add_exchange, retrieve, clear_user
- `_lifecycle.py`: LifecycleManager (init/open/close, wires all components)
- `_pipeline.py`: Pipeline (orchestrates processing stages)
- `_task_manager.py`: TaskManager (auto-processing, buffering, background tasks)

**`processing/`** — All extraction logic:
- `batch_segmenter.py`: Groups messages into episodes via single LLM call. Handles interleaved topics, splits on time gaps (30 min default).
- `episodes.py`: EpisodeGenerator — converts message sequences to episodes with third-person narratives.
- `extraction.py`: PredictCalibrateExtractor — the core innovation. Predicts what episode should contain given existing KB, extracts only what was unpredicted.
- `episode_merger.py`: Deduplicates semantically similar episodes (cosine similarity threshold).
- `boundary.py`: Legacy incremental boundary detector (replaced by BatchSegmenter, kept behind `use_legacy_segmentation` flag).
- `prompts.py`: All LLM prompt templates.

**`retrieval/retriever.py`** — Hybrid search: vector similarity (sqlite-vec) + BM25 text (FTS5), merged via Reciprocal Rank Fusion (k=60). Temporal filtering via `at_time` and `include_expired`. All queries scoped by `user_id`.

**`storage/sqlite/`** — All stores inherit `StoreBase` (async context manager + transactions). Convention: UUIDs as TEXT, datetimes as Unix timestamps (INTEGER), complex fields as JSON.

**`protocols.py`** — `EmbeddingClient` (`embed`, `embed_batch`) and `LLMClient` (`generate`, `generate_structured`). Implement these for custom providers.

**`embeddings/openai.py`** — `OpenAIEmbedAdapter` (text-embedding-3-small, 1536 dims).

**`llm/pydantic_ai.py`** — `PydanticAIAdapter` (multi-provider via PydanticAI).

**`dashboard/`** — Textual TUI for browsing memory state. Run via `uv run python -m memv.dashboard`.

### Critical Design Decisions

**Episode.original_messages is ground truth for extraction**, not Episode.content. The narrative content is for retrieval display. The extractor works against the raw messages to find novel knowledge.

**Bi-temporal model**: Every `SemanticKnowledge` has event time (`valid_at`/`invalid_at` — when fact is true in world) and transaction time (`created_at`/`expired_at` — when we recorded it). `is_valid_at(time)` checks event time, `is_current()` checks transaction time.

**User isolation is mandatory**: All retrieval and storage operations require and filter by `user_id`. No cross-user queries possible by design.

## Code Style

- Line length: 135 characters
- Ruff for imports/linting, ty for types
- All datetimes in UTC (`datetime.now(timezone.utc)`)
- Async everywhere: stores, embedding calls, LLM calls, processing
