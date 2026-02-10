# Reading List

Papery, artykuły, repozytoria do przeczytania i notatki z przeczytanych.

---

## Do przeczytania

- [ ] Context Engineering survey — [arXiv:2507.13334](https://arxiv.org/abs/2507.13334)
- [ ] LongMemEval benchmark — [arXiv:2407.xxxxx](https://arxiv.org/abs/2407.xxxxx)
- [ ] ...

---

## Przeczytane

### Graphiti paper — [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)

**TL;DR:** Bi-temporal knowledge graph dla agentów. Edge invalidation, entity resolution, BFS retrieval.

**Key insights:**
- Bi-temporal: event time (kiedy fakt był prawdą) vs transaction time (kiedy zapisaliśmy)
- Entity resolution przez LLM similarity
- Episodic → Entity → Fact hierarchy

**Co bierzemy:** Bi-temporal model, edge invalidation concept

---

### Nemori paper — [arXiv:2508.03341](https://arxiv.org/abs/2508.03341)

**TL;DR:** Predict-calibrate extraction. Importance emerges from prediction error.

**Key insights:**
- Zamiast "oceń czy ważne" → "przewiduj co będzie, wyciągnij czego nie przewidziałeś"
- Episode segmentation przez topic detection
- Hybrid retrieval: vector + BM25

**Co bierzemy:** Predict-calibrate jako core mechanism, batch segmentation

---

### SimpleMem paper — [arXiv:2501.xxxxx](https://arxiv.org/abs/2501.xxxxx)

**TL;DR:** Write-time atomization = 56.7% temporal reasoning. Simple > complex.

**Key insights:**
- "yesterday" → absolute date AT WRITE TIME, nie retrieval time
- Coreference resolution at write time
- 531 tokens/query vs 2745 (Nemori)
- k=3 gives 99% of peak performance

**Co bierzemy:** Write-time disambiguation principle

---

### Letta benchmark — [blog post](https://www.letta.com/blog/benchmarking-ai-agent-memory)

**TL;DR:** Simple filesystem + agent tool use = 74% LoCoMo. Beat Mem0 (68.5%).

**Key insight:** Memory is about context management, not retrieval mechanism. Explicit > implicit.

---

(dodawaj notatki z kolejnych lektur)
