"""Stage 1: Ingest LongMemEval conversation histories into memv."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from memv.memory.memory import Memory
from memv.models import Message, MessageRole

from ._checkpoint import RESULTS_DIR, append_jsonl, load_all_results, load_completed
from .config import get_config
from .dataset import LongMemEvalQuestion, load_dataset

logger = logging.getLogger(__name__)


def parse_longmemeval_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/05/20 (Sat) 02:21' → datetime (UTC)."""
    try:
        dt = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("Failed to parse date '%s', using epoch", date_str)
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


async def process_question(
    question_idx: int,
    question_data: LongMemEvalQuestion,
    db_dir: Path,
    config_name: str,
    embedding_client,
    llm_client,
) -> dict:
    """Process a single LongMemEval question: ingest all sessions, extract knowledge."""
    question_id = question_data.question_id
    user_id = f"question_{question_id}"
    db_path = str(db_dir / f"{question_id}.db")

    config = get_config(config_name)

    memory = Memory(
        db_path=db_path,
        config=config,
        embedding_client=embedding_client,
        llm_client=llm_client,
        enable_embedding_cache=True,
    )

    start_time = time.monotonic()
    total_messages = 0

    async with memory:
        for session, date_str in zip(question_data.haystack_sessions, question_data.haystack_dates, strict=True):
            timestamp = parse_longmemeval_date(date_str)
            for turn in session:
                role = MessageRole.USER if turn["role"] == "user" else MessageRole.ASSISTANT
                msg = Message(
                    user_id=user_id,
                    role=role,
                    content=turn["content"],
                    sent_at=timestamp,
                )
                await memory.add_message(msg)
                total_messages += 1

        knowledge_count = await memory.process(user_id)

    elapsed = time.monotonic() - start_time

    return {
        "question_id": question_id,
        "question_type": question_data.question_type,
        "messages_count": total_messages,
        "knowledge_count": knowledge_count,
        "sessions_count": len(question_data.haystack_sessions),
        "construction_time_s": round(elapsed, 2),
    }


async def run(
    run_name: str = "baseline",
    config_name: str = "default",
    data_path: str | None = None,
    num_questions: int | None = None,
    max_concurrent: int = 5,
    timeout: int = 1200,
    resume: bool = True,
    embedding_client=None,
    llm_client=None,
):
    """Run ingestion stage for all questions.

    Args:
        run_name: Name for this benchmark run.
        config_name: Config preset name from config.py.
        data_path: Path to dataset JSON (None = default location).
        num_questions: Limit number of questions (None = all).
        max_concurrent: Max concurrent question processing tasks.
        timeout: Per-question timeout in seconds.
        resume: Resume from checkpoint if prior results exist.
        embedding_client: EmbeddingClient instance.
        llm_client: LLMClient instance.
    """
    if embedding_client is None or llm_client is None:
        raise RuntimeError("embedding_client and llm_client are required. Pass them directly or set up default clients.")

    dataset = load_dataset(data_path)
    if num_questions is not None:
        dataset = dataset[:num_questions]

    run_dir = RESULTS_DIR / run_name
    db_dir = run_dir / "dbs"
    db_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "add.jsonl"

    completed_ids = load_completed(jsonl_path) if resume else set()
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    remaining = [q for q in dataset if q.question_id not in completed_ids]

    print(
        f"LongMemEval Add | run={run_name} config={config_name} "
        f"questions={len(dataset)} remaining={len(remaining)} concurrent={max_concurrent}"
    )
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already completed")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = len(completed_ids)
    total_count = len(dataset)

    async def process_with_guard(idx: int, question: LongMemEvalQuestion) -> dict | None:
        nonlocal completed_count
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    process_question(idx, question, db_dir, config_name, embedding_client, llm_client),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                result = {
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "error": "timeout",
                    "construction_time_s": timeout,
                }
            except Exception as e:
                logger.exception("Failed to process question %s", question.question_id)
                result = {
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "error": str(e),
                    "construction_time_s": 0,
                }

            append_jsonl(jsonl_path, result)
            completed_count += 1
            error = result.get("error")
            if error:
                print(f"  [{completed_count}/{total_count}] {question.question_id} ERROR: {error}")
            else:
                print(
                    f"  [{completed_count}/{total_count}] {question.question_id} "
                    f"→ {result['knowledge_count']} facts in {result['construction_time_s']}s"
                )
            return result

    tasks = [process_with_guard(idx, q) for idx, q in enumerate(remaining)]
    await asyncio.gather(*tasks)

    # Write compatibility JSON from all JSONL results
    all_results = load_all_results(jsonl_path)
    output_path = run_dir / "add.json"
    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    total_knowledge = sum(r.get("knowledge_count", 0) for r in all_results)
    total_time = sum(r.get("construction_time_s", 0) for r in all_results)
    print(f"Total: {total_knowledge} facts extracted in {total_time:.1f}s")

    return all_results


def _make_clients():
    from memv.embeddings.openai import OpenAIEmbedAdapter
    from memv.llm.pydantic_ai import PydanticAIAdapter

    return OpenAIEmbedAdapter(), PydanticAIAdapter()


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Stage 1: Ingestion")
    parser.add_argument("--run-name", default="baseline", help="Name for this run")
    parser.add_argument("--config", default="default", help="Config preset name")
    parser.add_argument("--data-path", default=None, help="Path to dataset JSON")
    parser.add_argument("--num-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent question processing")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-question timeout in seconds")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore prior checkpoint")
    args = parser.parse_args()

    embedding_client, llm_client = _make_clients()
    asyncio.run(
        run(
            run_name=args.run_name,
            config_name=args.config,
            data_path=args.data_path,
            num_questions=args.num_questions,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            resume=not args.no_resume,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )
    )


if __name__ == "__main__":
    main()
