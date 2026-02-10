# Tasks

Next steps for v0.1.0 release.

---

## 1. Atomization Step (HIGH PRIORITY)

**Goal:** Resolve temporal expressions and coreferences at write-time to improve retrieval quality by ~56%.

### 1.1 Extend SemanticKnowledge model

- [ ] Add structured metadata fields to `SemanticKnowledge` in `src/agent_memory/models.py`:
  ```python
  timestamp: datetime | None = None  # Parsed absolute time
  location: str | None = None
  persons: list[str] = []
  entities: list[str] = []
  topic: str | None = None
  ```
- [ ] Keep `temporal_info: str | None` for raw extraction, add `timestamp` for parsed value
- [ ] Update `KnowledgeStore` schema to include new columns

### 1.2 Modify extraction prompts

- [ ] Update `PredictCalibrateExtractor` prompts in `src/agent_memory/processing/extractor.py`:
  - Add explicit instruction: "PROHIBIT pronouns (he, she, it, they) and relative time (yesterday, tomorrow)"
  - Add instruction: "Each statement must be self-contained and understandable without context"
  - Include reference timestamp in prompt for temporal resolution
- [ ] Update cold-start extraction prompt with same constraints
- [ ] Update `ExtractedKnowledge` model to include structured metadata

### 1.3 Add temporal parsing

- [ ] Create `src/agent_memory/processing/temporal.py`:
  - `parse_relative_time(text: str, reference: datetime) -> datetime | None`
  - Handle: "yesterday", "tomorrow", "last week", "next Monday", etc.
  - Use `dateutil.parser` or similar for parsing
  - LLM fallback for complex expressions
- [ ] Integrate into extraction pipeline (post-process extracted knowledge)

### 1.4 Testing

- [ ] Add tests for temporal parsing edge cases
- [ ] Add tests verifying extracted knowledge is self-contained
- [ ] Regression test: ensure predict-calibrate still filters redundant info

---

## 2. Example Integrations

### 2.1 PydanticAI Example

- [ ] Create `examples/pydantic_ai_example.py`:
  - Simple chatbot with memory
  - Show `add_exchange()` → `process()` → `retrieve()` flow
  - Include `.env.example` for API keys

### 2.2 Raw OpenAI Example

- [ ] Create `examples/openai_example.py`:
  - Same flow without PydanticAI dependency
  - Use `OpenAIEmbedAdapter` directly
  - Show manual LLM integration pattern

---

## 3. Package Naming (Decision Required)

**Constraint:** `agentmemory` taken on PyPI, business requires "V" branding.

Options:
- [ ] `vmemo` - short, memorable
- [ ] `vtrace` - implies history/tracking
- [ ] `vgraph` - implies knowledge graph (v0.2 feature)
- [ ] `agent-memory` - hyphenated variant

**Action:** Pick name, update `pyproject.toml`, verify PyPI availability.

---

## 4. Release Prep

- [ ] Write CHANGELOG.md for v0.1.0
- [ ] Bump version in `pyproject.toml` to 0.1.0
- [ ] Final `uv run pre-commit run --all-files`
- [ ] Final `uv run pytest`
- [ ] Tag release: `git tag v0.1.0`
- [ ] Publish to PyPI: `uv publish`

---

## Delegation Guide

| Task | Complexity | Delegatable | Notes |
|------|------------|-------------|-------|
| 1.1 Model changes | Low | Yes | Straightforward schema update |
| 1.2 Prompt updates | Medium | Yes | Needs careful prompt engineering |
| 1.3 Temporal parsing | Medium | Yes | Well-defined scope |
| 1.4 Testing | Low | Yes | Standard test patterns |
| 2.1 PydanticAI example | Low | Yes | Follow existing patterns |
| 2.2 OpenAI example | Low | Yes | Follow existing patterns |
| 3. Package naming | N/A | No | Business decision |
| 4. Release prep | Low | Partial | PyPI credentials needed |

---

## References

- SimpleMem atomization: `reference/SimpleMem/core/memory_builder.py`
- SimpleMem prompt: `_build_extraction_prompt()` method (lines 221-298)
- SimpleMem model: `reference/SimpleMem/models/memory_entry.py`
- Ablation study: 56.7% of temporal reasoning from write-time disambiguation
