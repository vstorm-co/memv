# Glossary

Terminy używane w projekcie. Definicje dla spójności.

---

## Core Concepts

**Episode**
: Segmentowany fragment konwersacji o spójnym temacie. Zawiera tytuł, narrację (content), i oryginalne wiadomości. Podstawowa jednostka kontekstu.

**SemanticKnowledge** (Fact)
: Atomowy fakt wyekstrahowany z konwersacji. Self-contained statement o userze/świecie. Np. "User works at Vstorm as AI Engineer".

**Bi-temporal validity**
: Śledzenie dwóch osi czasowych: *event time* (kiedy fakt był prawdą w świecie) i *transaction time* (kiedy go zapisaliśmy).

**Predict-calibrate**
: Mechanizm ekstrakcji: najpierw przewiduj co episode powinien zawierać (na podstawie istniejącej wiedzy), potem wyciągnij tylko to czego NIE przewidziałeś. Importance emerges from prediction error.

**Atomization**
: Proces zamiany raw text na self-contained facts. Includes temporal normalization ("yesterday" → absolute date) i coreference resolution ("my kids" → "User's kids Sarah and Tom").

---

## Storage

**Message**
: Raw wiadomość z konwersacji. Append-only archive.

**Episode**
: Przetworzony segment konwersacji z narracją.

**Entity** *(v0.2+)*
: Resolved, deduplicated entity (osoba, miejsce, organizacja).

**Fact/Edge** *(v0.2+)*
: Relacja między entities z bi-temporal validity.

---

## Retrieval

**Hybrid retrieval**
: Kombinacja vector similarity + BM25 text search + RRF (Reciprocal Rank Fusion).

**RRF (Reciprocal Rank Fusion)**
: Algorytm łączenia wyników z różnych retrieverów. Score = Σ 1/(k + rank).

---

## Metrics

**LoCoMo**
: Long Context Memory benchmark. Testuje temporal reasoning, multi-hop, single-hop, open domain.

**F1**
: Harmonic mean of precision and recall. SimpleMem metric.

**LLM Score**
: Judge-based score (0-1). Nemori metric.

---

## Konkurencja

**mem0**
: Memory layer for AI. Importance scoring upfront.

**Nemori**
: Predict-calibrate, episode segmentation. ~2745 tokens/query.

**SimpleMem**
: Write-time atomization. ~531 tokens/query. F1: 43.24%.

**Graphiti (Zep)**
: Knowledge graph approach. Bi-temporal, entity resolution.

---

(dodawaj nowe terminy gdy się pojawiają)
