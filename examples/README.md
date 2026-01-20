# Agent Memory Examples

## Prerequisites

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

## Examples

### Quickstart (`quickstart.py`)

Minimal example showing core API - add messages, process, retrieve.

```bash
uv run python examples/quickstart.py
```

### Full Demo (`demo.py`)

Comprehensive demo showing:
- Multiple conversation topics
- Episode segmentation
- Knowledge extraction via predict-calibrate
- Context retrieval for different queries
- How to use retrieved context in LLM prompts

```bash
uv run python examples/demo.py
```

### Agent Integration (`agent_integration.py`)

Interactive chat agent with long-term memory. Shows the pattern for integrating Agent Memory into a real agent:

1. User sends message
2. Retrieve relevant context from memory
3. Generate response with context
4. Store exchange in memory
5. Periodically process to extract knowledge

```bash
uv run python examples/agent_integration.py
```

Commands in the chat:
- `quit` - exit
- `process` - force memory processing
- `debug` - show current memory contents

## Core Concepts

### Memory Flow

```
User messages → add_exchange() → Message buffer
                                      ↓
                               process()
                                      ↓
                    Episode segmentation (BoundaryDetector)
                                      ↓
                    Episode generation (EpisodeGenerator)
                                      ↓
                    Predict-calibrate extraction
                                      ↓
                         SemanticKnowledge stored
                                      ↓
                    retrieve() → Hybrid search (vector + BM25)
                                      ↓
                    RetrievalResult.to_prompt() → LLM context
```

### Why Predict-Calibrate?

Traditional extraction asks: "What facts are in this conversation?"
This extracts everything, including redundant information.

Predict-calibrate asks: "What did I fail to predict?"
1. Given what I already know, predict what this conversation contains
2. Compare prediction to actual content
3. Only extract what was unpredicted (genuinely novel)

This naturally focuses on new, important information without explicit importance scoring.
