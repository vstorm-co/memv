"""Stage 2: Retrieve memories and generate answers for LongMemEval questions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from memv.memory.memory import Memory

from ._checkpoint import RESULTS_DIR, append_jsonl, load_all_results, load_completed
from .config import get_config
from .dataset import LongMemEvalQuestion, load_dataset

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a memory assistant that retrieves accurate information from conversation memories.

## Instructions
1. Carefully analyze all provided memories
2. Pay special attention to timestamps to determine the correct answer
3. If memories contain contradictory information, prioritize the most recent memory
4. Convert relative time references to specific dates using the question date as reference
5. The answer should be concise (less than 5-6 words)

## Memories
{memories}

## Question Date
{question_date}

## Question
{question}

Answer:"""


async def process_question(
    question_data: LongMemEvalQuestion,
    db_dir: Path,
    config_name: str,
    embedding_client,
    llm_client,
    top_k: int = 10,
) -> dict:
    """Retrieve and answer a single question."""
    question_id = question_data.question_id
    user_id = f"question_{question_id}"
    db_path = str(db_dir / f"{question_id}.db")

    if not Path(db_path).exists():
        return {
            "question_id": question_id,
            "question": question_data.question,
            "question_type": question_data.question_type,
            "answer": question_data.answer,
            "question_date": question_data.question_date,
            "response": "",
            "retrieval_time_s": 0,
            "error": f"DB not found: {db_path}",
        }

    config = get_config(config_name)

    memory = Memory(
        db_path=db_path,
        config=config,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )

    start_time = time.monotonic()

    async with memory:
        result = await memory.retrieve(question_data.question, user_id=user_id, top_k=top_k)
        retrieval_time = time.monotonic() - start_time

        memory_lines = []
        for k in result.retrieved_knowledge:
            validity = ""
            if k.valid_at:
                validity = f" [valid from {k.valid_at.strftime('%Y-%m-%d')}]"
            if k.invalid_at:
                validity += f" [invalid after {k.invalid_at.strftime('%Y-%m-%d')}]"
            memory_lines.append(f"- {k.statement}{validity}")

        memories_text = "\n".join(memory_lines) if memory_lines else "No relevant memories found."

        prompt = ANSWER_PROMPT.format(
            memories=memories_text,
            question_date=question_data.question_date,
            question=question_data.question,
        )
        response = await llm_client.generate(prompt)

    return {
        "question_id": question_id,
        "question": question_data.question,
        "question_type": question_data.question_type,
        "answer": question_data.answer,
        "question_date": question_data.question_date,
        "response": response.strip(),
        "retrieved_count": len(result.retrieved_knowledge),
        "retrieval_time_s": round(retrieval_time, 3),
    }


async def run(
    run_name: str = "baseline",
    config_name: str = "default",
    data_path: str | None = None,
    num_questions: int | None = None,
    top_k: int = 10,
    max_concurrent: int = 10,
    timeout: int = 1200,
    resume: bool = True,
    embedding_client=None,
    llm_client=None,
):
    """Run search stage for all questions.

    Args:
        run_name: Name for this benchmark run (must match add stage).
        config_name: Config preset name.
        data_path: Path to dataset JSON.
        num_questions: Limit number of questions.
        top_k: Number of memories to retrieve per question.
        max_concurrent: Max concurrent question processing tasks.
        timeout: Per-question timeout in seconds.
        resume: Resume from checkpoint if prior results exist.
        embedding_client: EmbeddingClient instance.
        llm_client: LLMClient instance.
    """
    if embedding_client is None or llm_client is None:
        raise RuntimeError("embedding_client and llm_client are required.")

    dataset = load_dataset(data_path)
    if num_questions is not None:
        dataset = dataset[:num_questions]

    run_dir = RESULTS_DIR / run_name
    db_dir = run_dir / "dbs"
    if not db_dir.exists():
        raise FileNotFoundError(f"No DBs found at {db_dir}. Run add stage first.")

    jsonl_path = run_dir / "search.jsonl"

    completed_ids = load_completed(jsonl_path) if resume else set()
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    remaining = [q for q in dataset if q.question_id not in completed_ids]

    print(
        f"LongMemEval Search | run={run_name} config={config_name} "
        f"questions={len(dataset)} remaining={len(remaining)} top_k={top_k} concurrent={max_concurrent}"
    )
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already completed")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = len(completed_ids)
    total_count = len(dataset)

    async def process_with_guard(question: LongMemEvalQuestion) -> dict | None:
        nonlocal completed_count
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    process_question(question, db_dir, config_name, embedding_client, llm_client, top_k),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                result = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "question_type": question.question_type,
                    "answer": question.answer,
                    "question_date": question.question_date,
                    "response": "",
                    "error": "timeout",
                    "retrieval_time_s": timeout,
                }
            except Exception as e:
                logger.exception("Failed to process question %s", question.question_id)
                result = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "question_type": question.question_type,
                    "answer": question.answer,
                    "question_date": question.question_date,
                    "response": "",
                    "error": str(e),
                    "retrieval_time_s": 0,
                }

            append_jsonl(jsonl_path, result)
            completed_count += 1
            error = result.get("error")
            if error:
                print(f"  [{completed_count}/{total_count}] {question.question_id} ERROR: {error}")
            else:
                print(
                    f"  [{completed_count}/{total_count}] {question.question_id} "
                    f"→ {result['retrieved_count']} memories, {result['retrieval_time_s']}s"
                )
            return result

    tasks = [process_with_guard(q) for q in remaining]
    await asyncio.gather(*tasks)

    # Write compatibility JSON from all JSONL results
    all_results = load_all_results(jsonl_path)
    output_path = run_dir / "search.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    return all_results


def _make_clients():
    from memv.embeddings.openai import OpenAIEmbedAdapter
    from memv.llm.pydantic_ai import PydanticAIAdapter

    return OpenAIEmbedAdapter(), PydanticAIAdapter()


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Stage 2: Search + Answer")
    parser.add_argument("--run-name", default="baseline", help="Name for this run")
    parser.add_argument("--config", default="default", help="Config preset name")
    parser.add_argument("--data-path", default=None, help="Path to dataset JSON")
    parser.add_argument("--num-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--top-k", type=int, default=10, help="Number of memories to retrieve")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent question processing")
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
            top_k=args.top_k,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            resume=not args.no_resume,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )
    )


if __name__ == "__main__":
    main()
