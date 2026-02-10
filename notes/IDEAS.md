# Ideas & Future Considerations

Pomysły na przyszłość, które nie są jeszcze na roadmapie. Luźne notatki, do rozważenia w v0.3+.

---

## Procedural Memory (Tool Execution Lessons)

**Problem:** Agenci często wywołują toole i napotykają błędy, by za N-tym razem zrobić coś poprawnie. Ta wiedza trial-and-error jest tracona między sesjami.

**Przykład:**
```
Agent: mmctl post create channel:name → fails
Agent: mmctl post create team:channel → fails  
Agent: mmctl post create vstorm:channel-name → works!
```

**Koncept:** Warstwa "lessons learned" z wykonywania tooli:
- Wzorce wywołań które zadziałały
- Typowe błędy i ich rozwiązania
- Konfiguracje specyficzne dla środowiska odkryte przez próby

**Potencjalna architektura:**
```
ToolExecution (append-only log)
    │
    ▼
ProceduralKnowledge
    - tool_name: str
    - pattern: str ("use vstorm:channel format for mmctl")
    - context: str (when this applies)
    - success_count: int
    - failure_examples: list[str]
    - discovered_at: datetime
```

**Trigger retrieval:** Gdy agent ma wywołać tool, pobierz relevantną wiedzę proceduralną dla tego toola/kontekstu.

**Podejście do ekstrakcji:** 
- Śledź wywołania tooli (success/failure)
- Po rozwiązaniu powtarzających się błędów, wyciągnij "lekcję"
- Można użyć predict-calibrate: przewiduj że tool zadziała → fail → extract co sprawiło że zadziałał

**Otwarte pytania:**
- Jak generalizować bez overfittingu do konkretnych przypadków?
- Jak obsługiwać zmiany w API tooli?
- Per-user czy shared między userami?
- Jak wykryć że pattern jest przestarzały?

**Powiązane:** mem0 ma concept "procedural memory" ale nie rozbudowany.

---

## (dodawaj kolejne pomysły tutaj)
