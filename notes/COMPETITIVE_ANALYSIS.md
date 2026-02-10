# Competitive Analysis: Agent Memory Systems

Comprehensive analysis of agent memory implementations and agentmemory's positioning.

---

## Executive Summary

The agent memory landscape has converged on several distinct approaches:

| Approach | Representative | Core Idea |
|----------|----------------|-----------|
| Knowledge Graph | Graphiti | Bi-temporal entity-relationship modeling |
| Multi-Level Memory | Mem0 | Plugin architecture with tiered storage |
| Stateful Agents | Letta | Persistent agent state with memory blocks |
| Episodic Memory | Nemori | Predict-calibrate extraction from episodes |
| Semantic Compression | SimpleMem | Write-time atomization with entropy filtering |

**agentmemory's position**: Hybrid approach combining Nemori's predict-calibrate extraction with SimpleMem's write-time atomization, Graphiti's bi-temporal validity, and SQLite-first storage for simplicity.

---

## Competitor Deep Dives

### 1. Graphiti (by Zep)

**Overview**: Real-time knowledge graph for AI agents. Designed for dynamic, frequently-updated datasets where temporal semantics matter.

**Architecture**:
```
Messages → Entity Extraction → Entity Resolution → Fact Extraction → Knowledge Graph
                                      ↓
              Bi-temporal edges (event time + transaction time)
```

**Data Model**:
- `EntityNode`: name, type, summary, aliases, embedding, attributes
- `EntityEdge`: source → relation → target with bi-temporal validity
- `EpisodicNode`, `CommunityNode`, `SagaNode` for higher-level structures

**Key Features**:
- Bi-temporal validity: tracks when facts were true vs when we learned them
- Edge invalidation: contradictions set `expired_at` rather than delete
- Real-time incremental updates (no batch recomputation)
- BFS graph traversal for multi-hop reasoning
- Custom entity types via Pydantic models

**Strengths**:
- Sub-second query latency at scale
- Explicit relationship modeling
- Multiple graph backends (Neo4j, FalkorDB, Kuzu, Neptune)
- Production-ready with REST/MCP servers

**Limitations**:
- Graph complexity adds overhead for simple use cases
- Requires graph database infrastructure
- Entity resolution can be noisy with LLM-only approach

**Performance**: Sub-second latency, designed for enterprise scale

---

### 2. Mem0

**Overview**: Multi-level memory system with extensive provider ecosystem. Business-focused with hosted platform option.

**Architecture**:
```
Messages → LLM Extraction → Multi-Level Storage → Hybrid Search
                                 ↓
              User-level / Session-level / Agent-level
```

**Data Model**:
- Flat memory entries with metadata
- Optional graph layer via Neptune
- Multi-level hierarchy: user → session → agent

**Key Features**:
- 15+ LLM providers (OpenAI, Anthropic, Google, Groq, etc.)
- 6+ vector store backends (Qdrant, Pinecone, Chroma, Weaviate, etc.)
- Memory operations: add, search, update, delete
- Hosted platform for turnkey deployment
- `infer=True/False` toggle for raw vs extracted storage

**Strengths**:
- Extensive provider ecosystem
- Deployment flexibility (local/cloud)
- Business-ready with enterprise features
- Active development and community

**Limitations**:
- Upfront LLM importance scoring (no predict-calibrate)
- Limited temporal handling
- No episode segmentation

**Performance (LoCoMo)**:
- F1: 34.20%
- 91% faster than full-context methods
- 90% lower token usage

---

### 3. Letta (formerly MemGPT)

**Overview**: Stateful agent framework with persistent memory. Focus on long-running agents that learn from interactions.

**Architecture**:
```
Conversation → Agent State → Tool Invocation → Memory Update
                   ↓
           Memory Blocks (structured context windows)
```

**Data Model**:
- Agent state with memory blocks
- Message history with role-based semantics
- Tool/skill definitions per agent
- Multi-tenant isolation (users/groups)

**Key Features**:
- Persistent agent state across conversations
- Local LLM support (LM Studio, Ollama, vLLM, llama.cpp)
- Skill/plugin system for extensibility
- Self-improving agents
- WebSocket API for real-time communication

**Strengths**:
- Agent-centric design (not just memory)
- Extensive customization points
- Local LLM support for privacy
- Enterprise-grade multi-tenancy

**Limitations**:
- Scope beyond memory (full agent framework)
- PostgreSQL required
- Heavy codebase (~358K LOC)
- Less focused on memory optimization

**Performance**: Agent-dependent; LoCoMo benchmark showed 74% with simple filesystem + tool use (beat Mem0's 68.5%)

---

### 4. Nemori

**Overview**: Cognitive science-grounded memory system. Inspired by event segmentation theory and predictive processing.

**Architecture**:
```
Messages → Batch Segmenter → Episode Generator → Predict-Calibrate → Semantic Memory
                                      ↓
                    Episode narrative + knowledge provenance
```

**Data Model**:
- `Episode`: message groups with narrative, temporal bounds
- `SemanticMemory`: extracted facts with importance, provenance
- Dual-memory: episodic (narrative) + semantic (declarative)

**Key Features**:
- Predict-calibrate extraction: predict from existing knowledge, extract gaps
- Batch segmentation with topic coherence
- Episode merging to reduce redundancy
- Time gap auto-segmentation (>30min)
- Per-user isolation with directory structure

**Strengths**:
- Token efficient (3K context for 0.81 alignment)
- Cognitive science foundation
- Self-organizing knowledge structure
- Minimal context budget requirements

**Limitations**:
- Research/beta maturity
- JSONL file storage (not production-ready)
- No bi-temporal validity
- OpenAI-only for LLM

**Performance (LoCoMo)**:
- LLM Score: 0.794 (+19.8% vs Mem0)
- Temporal F1: 0.5874
- Multi-Hop F1: 0.4312
- Context: ~2,745 tokens/query

---

### 5. SimpleMem

**Overview**: Write-time semantic compression for maximum token efficiency. Focus on atomic, self-contained facts.

**Architecture**:
```
Dialogue → Entropy Filter → Atomization → Multi-View Index → Adaptive Retrieval
                                 ↓
              Temporal normalization + coreference resolution
```

**Data Model**:
- `MemoryEntry`: atomic, self-contained facts
- Multi-view indexing: semantic (vector) + lexical (BM25) + symbolic (metadata)
- No episodes; direct message → atomic fact mapping

**Key Features**:
- Write-time disambiguation: "yesterday" → absolute timestamp at ingestion
- Coreference resolution: "my kids" → "Sarah's kids"
- Entropy-based filtering (threshold 0.35)
- Adaptive retrieval depth: `k_dyn = k_base * (1 + δ * C_q)`
- Parallel batch processing (16 workers)
- MCP server for production deployment

**Strengths**:
- Extreme token efficiency (30× fewer tokens)
- Fast construction (92.6s vs Mem0's 1,350.9s)
- Self-contained facts need no context at retrieval
- Production-ready with MCP integration

**Limitations**:
- Loses conversational structure
- No episode preservation
- Harder to answer "what was the context when X happened?"
- No predict-calibrate (uses entropy instead)

**Performance (LoCoMo)**:
- F1: 43.24% (highest)
- Temporal F1: 58.62%
- Construction: 92.6s (55× faster than Mem0)
- Tokens/query: ~531

**Critical Ablation**:
| Configuration | Temporal F1 | Impact |
|---------------|-------------|--------|
| Full SimpleMem | 58.62% | baseline |
| w/o Atomization | 25.40% | **-56.7%** |
| w/o Consolidation | 55.10% | -6.0% |

Write-time atomization accounts for **56.7%** of temporal reasoning performance.

---

## Feature Comparison Matrix

### Core Architecture

| Feature | Graphiti | Mem0 | Letta | Nemori | SimpleMem | agentmemory |
|---------|----------|------|-------|--------|-----------|-------------|
| Primary Structure | Knowledge graph | Flat memories | Agent state | Episodes + semantic | Atomic entries | Episodes + atomic facts |
| Data Model | Entity-Relation | Key-value | Memory blocks | Dual-memory | Atomic facts | Hybrid |
| Storage | Neo4j/FalkorDB | 6+ backends | PostgreSQL | JSONL | LanceDB | SQLite → Postgres |
| Vector DB | Graph-native | Pluggable | TurboBuffer | ChromaDB | Built-in | sqlite-vec |

### Temporal Handling

| Feature | Graphiti | Mem0 | Letta | Nemori | SimpleMem | agentmemory |
|---------|----------|------|-------|--------|-----------|-------------|
| Temporal Model | Bi-temporal | Implicit | Implicit | Event time | Write-time normalized | Bi-temporal |
| Contradiction Handling | Edge invalidation | Overwrite | Manual | None | Consolidation | Edge invalidation |
| Point-in-time Queries | Yes | No | No | No | No | Yes (planned) |
| Temporal Normalization | No | No | No | No | **Yes** | **Yes** (adopted) |

### Extraction & Processing

| Feature | Graphiti | Mem0 | Letta | Nemori | SimpleMem | agentmemory |
|---------|----------|------|-------|--------|-----------|-------------|
| Extraction Trigger | Real-time | Upfront LLM | Conversation | Predict-calibrate | Entropy filter | Predict-calibrate |
| Episode Segmentation | Episodic edges | No | No | **Yes** | No | **Yes** |
| Entity Resolution | LLM-based | No | No | No | In atomization | LLM-based (planned) |
| LLM Calls/Extraction | 1-2 | 1 | 1 | 2 (predict+calibrate) | 1 | 2 |

### Retrieval

| Feature | Graphiti | Mem0 | Letta | Nemori | SimpleMem | agentmemory |
|---------|----------|------|-------|--------|-----------|-------------|
| Search Methods | Vector + BM25 + Graph | Vector + metadata | Native RAG | Vector + BM25 | Vector + BM25 + symbolic | Vector + BM25 + RRF |
| Reranking | RRF/cross-encoder | Basic | None | Optional NOR-LIFT | Adaptive depth | RRF (k=60) |
| Tokens/Query | Good | ~1,000 | Moderate | ~2,745 | **~531** | ~2,745 |

### Provider Support

| Feature | Graphiti | Mem0 | Letta | Nemori | SimpleMem | agentmemory |
|---------|----------|------|-------|--------|-----------|-------------|
| LLM Providers | 5+ | 15+ | 10+ | OpenAI | OpenAI-compat | Via PydanticAI |
| Embedding Providers | 4+ | Multiple | Multiple | OpenAI | Qwen3 | OpenAI + local |
| Local LLM | Ollama | No | Yes (LM Studio, vLLM) | No | No | No |

### Deployment & Operations

| Feature | Graphiti | Mem0 | Letta | Nemori | SimpleMem | agentmemory |
|---------|----------|------|-------|--------|-----------|-------------|
| Deployment | Local + REST/MCP | Local + Cloud | Local + Cloud | Local only | Local + MCP | Local |
| Multi-tenancy | Per-group | Platform-native | User/group | Per-user dirs | Token-based | user_id from day 1 |
| Caching | Disk | N/A | N/A | LRU + TTL | N/A | None (planned) |
| Code Maturity | Production | Production | Enterprise | Research | Research | v0.1 |

---

## Architectural Patterns

### Processing Models

| Pattern | Systems | Trade-offs |
|---------|---------|------------|
| Real-time incremental | Graphiti | Low latency, higher complexity |
| Batch processing | Nemori, SimpleMem | Higher latency, simpler implementation |
| Conversation-driven | Letta, Mem0 | Agent-dependent, less predictable |

### Memory Organization

| Pattern | Systems | Trade-offs |
|---------|---------|------------|
| Knowledge graph | Graphiti | Rich relationships, graph DB required |
| Flat memories | Mem0 | Simple, limited relationship modeling |
| Agent state blocks | Letta | Agent-centric, not reusable |
| Episodic + semantic | Nemori, agentmemory | Preserves context, more complex |
| Atomic facts | SimpleMem | Token efficient, loses context |

### Extraction Approaches

| Pattern | Systems | Trade-offs |
|---------|---------|------------|
| Upfront LLM scoring | Mem0, Letta | Simple, may miss importance |
| Predict-calibrate | Nemori, agentmemory | Importance emerges from error, 2 LLM calls |
| Entropy filtering | SimpleMem | No LLM for filtering, less semantic |

### Disambiguation Timing

| Pattern | Systems | Trade-offs |
|---------|---------|------------|
| Write-time | SimpleMem, agentmemory | Fast retrieval, requires reference context |
| Retrieval-time | Most others | Flexible, slower retrieval |

---

## agentmemory Position

### What We Adopt

| From | What | Rationale |
|------|------|-----------|
| **SimpleMem** | Write-time temporal normalization | 56.7% of temporal reasoning from ablation |
| **SimpleMem** | Coreference resolution at write | Self-contained facts for efficient retrieval |
| **Nemori** | Predict-calibrate extraction | Importance emerges from prediction error |
| **Nemori** | Batch episode segmentation | Topic-coherent chunks |
| **Nemori** | Episode merging | Reduce redundancy |
| **Graphiti** | Bi-temporal validity | Track event time + transaction time |
| **Graphiti** | Edge invalidation | Contradictions preserve history |
| **Mem0** | `infer=True/False` toggle | Raw storage when extraction not needed |
| **Mem0** | ADD/UPDATE/DELETE/NONE flow | Prevent fact explosion |

### What We Reject

| From | What | Why |
|------|------|-----|
| **Graphiti** | CommunityNode/SagaNode | Premature abstraction |
| **Graphiti** | Custom Pydantic entity types | Adds complexity; start with string types |
| **Nemori** | NOR-LIFT reranking | Benchmarks work without it |
| **Nemori** | Separate episodic/semantic indices | Simpler unified model |
| **SimpleMem** | LanceDB storage | SQLite → Postgres path has broader adoption |
| **SimpleMem** | MCP server | Separate package if demand |
| **Letta** | Full agent framework | We're memory-focused |
| **Mem0** | 15+ provider integrations | Via PydanticAI adapter is enough |

### Our Differentiators

1. **Hybrid approach**: SimpleMem atomization + Nemori predict-calibrate
   - Write-time disambiguation for retrieval efficiency
   - Predict-calibrate for importance-based extraction

2. **Episode preservation with provenance**
   - Facts link back to source episodes
   - Multi-episode provenance (`episode_ids: list[UUID]`)
   - Answer "what was the context when X happened?"

3. **Bi-temporal without full graph**
   - Phase 3.5 adds validity tracking to `SemanticKnowledge`
   - Point-in-time queries without Entity/Fact complexity
   - Graph layer optional (Phase 4)

4. **SQLite-first, production-ready path**
   - sqlite-vec for vectors, FTS5 for text
   - Same API scales to Postgres/Neo4j

5. **Framework-agnostic**
   - Works with any agent framework
   - LLM/embedding via protocol adapters

---

## Decision Matrix: When to Use Each

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Enterprise knowledge graph** | Graphiti | Native graph, bi-temporal, production-ready |
| **Quick integration, many providers** | Mem0 | Plugin ecosystem, hosted option |
| **Long-running stateful agents** | Letta | Agent-centric design, skill system |
| **Research, minimal tokens** | Nemori or SimpleMem | Token efficiency, cognitive grounding |
| **Temporal reasoning priority** | SimpleMem or agentmemory | Write-time normalization proven critical |
| **Episode context preservation** | Nemori or agentmemory | Episodic memory with provenance |
| **SQLite → Postgres path** | agentmemory | Simple storage progression |
| **Predict-calibrate + atomization** | agentmemory | Hybrid approach (unique) |

---

## Benchmark Summary (LoCoMo)

| System | F1 | LLM Score | Tokens/Query | Construction |
|--------|-----|-----------|--------------|--------------|
| SimpleMem | **43.24%** | - | **531** | **92.6s** |
| Nemori | - | **0.794** | 2,745 | not reported |
| Mem0 | 34.20% | 0.663 | ~1,000 | 1,350.9s |
| Full Context | - | 0.806 | 23,653 | - |

**Key insight**: SimpleMem's write-time atomization achieves highest F1 with lowest tokens. Nemori's predict-calibrate approaches full-context quality with 8× fewer tokens.

---

## References

**Papers**:
- Graphiti: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)
- Nemori: [arXiv:2508.03341](https://arxiv.org/abs/2508.03341)
- SimpleMem: Write-Time Memory Architecture
- Letta benchmark: [letta.com/blog/benchmarking-ai-agent-memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)

**Repositories**:
- Graphiti: github.com/getzep/graphiti
- Mem0: github.com/mem0ai/mem0
- Letta: github.com/letta-ai/letta
- Nemori: reference/nemori/
- SimpleMem: github.com/aiming-lab/SimpleMem
