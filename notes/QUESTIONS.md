# Open Questions

Pytania do rozwiązania. Rzeczy których jeszcze nie wiemy.

---

## Architektura

- [ ] **Contradiction handling** — jak wykrywać i obsługiwać sprzeczne fakty? Edge invalidation vs versioning vs explicit conflict resolution?

- [ ] **Cross-user knowledge** — czy/jak dzielić wiedzę między userami? Namespace isolation? Human-in-the-loop gating?

- [ ] **Procedural memory** — jak przechowywać "lessons learned" z tool execution? Osobna warstwa czy część semantic knowledge?

---

## Implementacja

- [ ] **Package name** — `agentmemory` zajęte na PyPI. Opcje: `vmemo`, `vtrace`, `vgraph`, `agent-memory`?

- [ ] **Embedding model default** — wymagać explicit config czy mieć sensowny default? Który?

- [ ] **Background processing** — explicit `process()` vs automatic? Threshold-based?

---

## Performance

- [ ] **Token efficiency** — SimpleMem: 531 tokens/query, my: ~2745. Jak zejść niżej bez utraty jakości?

- [ ] **Retrieval k** — SimpleMem pokazuje k=3 daje 99% peak. Czy to generalizuje?

---

## Product

- [ ] **Target users** — coding agents? personal assistants? enterprise? Fokus na jeden segment czy generalnie?

- [ ] **Monetization** — OSS core + hosted? Dual license? Pure OSS?

---

(dodawaj pytania gdy się pojawiają, usuwaj gdy rozwiązane → przenieś do DECISIONS.md)
