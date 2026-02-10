# Decisions Log

Architektoniczne i projektowe decyzje z rationale. Żeby za 3 miesiące pamiętać *dlaczego*.

Format:
```
## [Data] Tytuł decyzji
**Kontekst:** Co było do zdecydowania?
**Decyzja:** Co zdecydowaliśmy?
**Alternatywy:** Co odrzuciliśmy?
**Rationale:** Dlaczego tak?
```

---

## [2026-01] Predict-calibrate zamiast importance scoring

**Kontekst:** Jak decydować co jest warte ekstrakcji z konwersacji?

**Decyzja:** Predict-calibrate — najpierw przewiduj co powinno być, potem wyciągaj tylko to czego nie przewidziałeś.

**Alternatywy:** 
- LLM importance scoring (mem0, Agno) — "oceń 1-10 czy to ważne"
- Entropy filtering (SimpleMem) — filtruj po entropii tekstu

**Rationale:** 
- Importance scoring jest subiektywny i inconsistent
- Predict-calibrate: importance emerguje z prediction error, nie z explicit scoringu
- Differentiator vs konkurencja

---

## [2026-01] Write-time disambiguation (z SimpleMem)

**Kontekst:** Kiedy rozwiązywać temporal references ("yesterday") i coreferences ("my kids")?

**Decyzja:** Write-time, nie retrieval-time.

**Alternatywy:** Retrieval-time resolution

**Rationale:** 
- Ablation study SimpleMem: 56.7% temporal reasoning performance from write-time normalization
- "yesterday" w momencie zapisu → konkretna data
- Przy retrieval "yesterday" nie ma kontekstu kiedy było powiedziane

---

## [2026-01] SQLite first, Postgres later

**Kontekst:** Jaki storage backend na start?

**Decyzja:** SQLite jako default, Postgres jako upgrade path.

**Alternatywy:**
- LanceDB (SimpleMem używa)
- Postgres from day 1
- Qdrant/Pinecone (vector-only)

**Rationale:**
- Zero-config local development
- Szersza adopcja (nie każdy ma Postgres)
- SQLite vec extension dla embeddings
- Postgres gdy potrzebna skalowalność

---

## [2026-01] Episode-based, nie message-based

**Kontekst:** Jaka podstawowa jednostka pamięci?

**Decyzja:** Episodes (segmentowane konwersacje) → atomic facts

**Alternatywy:**
- Raw messages (jak większość)
- Tylko atomic facts (bez provenance)

**Rationale:**
- Episodes dają kontekst i provenance
- Atomic facts dają retrieval precision
- Hybrid: zachowaj oba warstwy

---

(dodawaj kolejne decyzje)
