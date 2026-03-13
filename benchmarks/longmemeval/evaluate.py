"""Stage 3: LLM-judge evaluation of LongMemEval search results."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone

from ._checkpoint import RESULTS_DIR, append_jsonl, load_all_results, load_completed

logger = logging.getLogger(__name__)

# --- Type-specific judge prompts (adapted from Nemori/Zep LongMemEval evals) ---

TEMPORAL_REASONING_PROMPT = """I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, \
you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \
In addition, do not penalize off-by-one errors for the number of days. \
If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors \
(e.g., predicting 19 days when the answer is 18), the model's response is still correct.

<QUESTION>
{question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
{response}
</RESPONSE>"""

KNOWLEDGE_UPDATE_PROMPT = """I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
If the response contains some previous information along with an updated answer, \
the response should be considered as correct as long as the updated answer is the required answer.

<QUESTION>
{question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
{response}
</RESPONSE>"""

SINGLE_SESSION_PREFERENCE_PROMPT = """I will give you a question, a rubric for desired personalized response, \
and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. \
The model does not need to reflect all the points in the rubric. \
The response is correct as long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
{question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
{response}
</RESPONSE>"""

DEFAULT_PROMPT = """I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, answer no. \
If the response is equivalent to the correct answer or contains all the intermediate steps \
to get the correct answer, you should also answer yes. \
If the response only contains a subset of the information required by the answer, answer no.

<QUESTION>
{question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
{response}
</RESPONSE>"""

SYSTEM_PROMPT = "You are an expert grader. Respond with ONLY 'yes' or 'no'."

PROMPTS_BY_TYPE = {
    "temporal-reasoning": TEMPORAL_REASONING_PROMPT,
    "knowledge-update": KNOWLEDGE_UPDATE_PROMPT,
    "single-session-preference": SINGLE_SESSION_PREFERENCE_PROMPT,
}


async def evaluate_single(
    llm_client,
    question: str,
    gold_answer: str,
    response: str,
    question_type: str,
) -> bool:
    """Evaluate a single question-response pair using LLM judge."""
    template = PROMPTS_BY_TYPE.get(question_type, DEFAULT_PROMPT)
    prompt = template.format(question=question, gold_answer=gold_answer, response=response)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    result = await llm_client.generate(full_prompt)
    return result.strip().lower().startswith("yes")


async def run(
    run_name: str = "baseline",
    llm_client=None,
    max_concurrent: int = 10,
    resume: bool = True,
):
    """Run evaluation on search results.

    Args:
        run_name: Name for this benchmark run (must match search stage).
        llm_client: LLMClient instance for LLM-judge.
        max_concurrent: Max concurrent LLM calls.
        resume: Resume from checkpoint if prior results exist.
    """
    if llm_client is None:
        raise RuntimeError("llm_client is required.")

    run_dir = RESULTS_DIR / run_name
    search_path = run_dir / "search.json"
    if not search_path.exists():
        raise FileNotFoundError(f"No search results at {search_path}. Run search stage first.")

    data = json.loads(search_path.read_text(encoding="utf-8"))

    jsonl_path = run_dir / "evaluate.jsonl"

    completed_ids = load_completed(jsonl_path) if resume else set()
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    remaining = [item for item in data if item["question_id"] not in completed_ids]

    print(f"LongMemEval Evaluate | run={run_name} questions={len(data)} remaining={len(remaining)}")
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already completed")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_with_semaphore(item: dict) -> None:
        async with semaphore:
            # Skip items that errored in search stage
            if item.get("error"):
                scored = {
                    "question_id": item["question_id"],
                    "question_type": item.get("question_type"),
                    "is_correct": None,
                    "error": item["error"],
                    "question": item.get("question", ""),
                    "gold_answer": item.get("answer", ""),
                    "response": item.get("response", ""),
                }
            else:
                try:
                    is_correct = await evaluate_single(
                        llm_client,
                        item["question"],
                        item["answer"],
                        item["response"],
                        item.get("question_type", "default"),
                    )
                    scored = {
                        "question_id": item["question_id"],
                        "question_type": item.get("question_type"),
                        "is_correct": is_correct,
                        "question": item["question"],
                        "gold_answer": item["answer"],
                        "response": item["response"],
                    }
                except Exception as e:
                    logger.error("Evaluation failed for %s: %s", item["question_id"], e)
                    scored = {
                        "question_id": item["question_id"],
                        "question_type": item.get("question_type"),
                        "is_correct": None,
                        "error": f"eval_failed: {e}",
                        "question": item.get("question", ""),
                        "gold_answer": item.get("answer", ""),
                        "response": item.get("response", ""),
                    }
            append_jsonl(jsonl_path, scored)

    tasks = [eval_with_semaphore(item) for item in remaining]
    await asyncio.gather(*tasks)

    all_scored = load_all_results(jsonl_path)

    type_stats: dict[str, dict[str, int]] = {}
    total_correct = 0
    total_scored = 0
    total_errors = 0

    for scored in all_scored:
        qtype = scored.get("question_type", "unknown")
        if qtype not in type_stats:
            type_stats[qtype] = {"correct": 0, "total": 0}

        if scored.get("is_correct") is None:
            total_errors += 1
            continue

        type_stats[qtype]["total"] += 1
        total_scored += 1

        if scored["is_correct"]:
            type_stats[qtype]["correct"] += 1
            total_correct += 1

    overall_accuracy = total_correct / total_scored if total_scored > 0 else 0
    accuracy_by_type = {}
    for qtype, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        accuracy_by_type[qtype] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(acc, 4),
        }

    scores = {
        "run_name": run_name,
        "total_questions": len(all_scored),
        "scored_questions": total_scored,
        "errors": total_errors,
        "correct_answers": total_correct,
        "overall_accuracy": round(overall_accuracy, 4),
        "accuracy_by_type": accuracy_by_type,
        "evaluation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "scored_items": all_scored,
    }

    print(f"\n{'=' * 50}")
    print(f"Overall: {total_correct}/{total_scored} = {overall_accuracy:.1%}")
    if total_errors:
        print(f"Errors (excluded from scoring): {total_errors}")
    print(f"{'=' * 50}")
    for qtype, stats in sorted(accuracy_by_type.items()):
        print(f"  {qtype}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")
    print(f"{'=' * 50}")

    output_path = run_dir / "scores.json"
    output_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nScores saved to {output_path}")

    return scores


def _make_llm_client():
    from memv.llm.pydantic_ai import PydanticAIAdapter

    return PydanticAIAdapter()


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Stage 3: Evaluation")
    parser.add_argument("--run-name", default="baseline", help="Name for this run")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore prior checkpoint")
    args = parser.parse_args()

    llm_client = _make_llm_client()
    asyncio.run(run(run_name=args.run_name, llm_client=llm_client, max_concurrent=args.max_concurrent, resume=not args.no_resume))


if __name__ == "__main__":
    main()
