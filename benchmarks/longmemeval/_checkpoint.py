"""JSONL checkpoint helpers for crash-safe benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_completed(jsonl_path: Path) -> set[str]:
    """Load completed question IDs from checkpoint JSONL file."""
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            completed.add(obj["question_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def append_jsonl(jsonl_path: Path, result: dict) -> None:
    """Append a single result as a JSONL line (atomic append)."""
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_all_results(jsonl_path: Path) -> list[dict]:
    """Load all results from checkpoint JSONL file."""
    results: list[dict] = []
    if not jsonl_path.exists():
        return results
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results
