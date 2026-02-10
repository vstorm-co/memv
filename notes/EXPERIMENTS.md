# Experiments Log

Wyniki eksperymentów, benchmarków, ablacji. Dane > opinie.

---

## Benchmark: LoCoMo (z literatury)

**Data:** 2026-01 (przegląd literatury)

**Setup:** GPT-4.1-mini na LoCoMo benchmark

| System | Metric | Score | vs Mem0 | Tokens/Query |
|--------|--------|-------|---------|--------------|
| Full Context | LLM Score | 0.806 | baseline | 23,653 |
| Mem0 | LLM Score | 0.663 | - | ~1,027 |
| Nemori | LLM Score | 0.794 | +19.8% | ~2,745 |
| SimpleMem | F1 | 43.24% | +26.4% | ~531 |

**Wnioski:**
- SimpleMem osiąga świetne wyniki przy 5x mniejszym token usage
- Write-time atomization to klucz

---

## Ablation: SimpleMem atomization (z literatury)

**Data:** 2026-01 (z paperu SimpleMem)

| Configuration | Temporal F1 | Impact |
|---------------|-------------|--------|
| Full SimpleMem | 58.62% | baseline |
| w/o Atomization | 25.40% | **-56.7%** |
| w/o Consolidation | 55.10% | -6.0% |
| w/o Adaptive Pruning | 56.80% | -3.1% |

**Wniosek:** Atomization (write-time disambiguation) = 56.7% performance. Krytyczne.

---

## (dodawaj własne eksperymenty)

### Template:
```
## [Nazwa eksperymentu]

**Data:** YYYY-MM-DD
**Cel:** Co testujemy?
**Setup:** Jak testujemy? (model, dataset, params)
**Wyniki:** Tabela/liczby
**Wnioski:** Co z tego wynika?
**Next steps:** Co dalej?
```
