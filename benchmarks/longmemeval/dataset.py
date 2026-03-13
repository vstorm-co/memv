"""LongMemEval dataset loader and Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, field_validator


class LongMemEvalQuestion(BaseModel):
    question_id: str
    question_type: str
    question: str
    answer: str

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, v: object) -> str:
        return str(v)

    question_date: str  # "2023/05/20 (Sat) 02:21"
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[dict]]  # list of sessions, each is list of {role, content}
    answer_session_ids: list[str]


DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "longmemeval_s_cleaned.json"


def load_dataset(path: Path | str | None = None) -> list[LongMemEvalQuestion]:
    """Load LongMemEval dataset from JSON file.

    Args:
        path: Path to longmemeval_s_cleaned.json. Defaults to benchmarks/data/.

    Returns:
        List of parsed questions.
    """
    path = Path(path) if path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it with:\n"
            f"  wget -P benchmarks/data/ "
            f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LongMemEvalQuestion.model_validate(item) for item in raw]
