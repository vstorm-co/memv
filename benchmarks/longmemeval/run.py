"""End-to-end runner for LongMemEval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import time

from . import add, evaluate, search


def _make_clients(model: str = "openai:gpt-4.1-mini"):
    from memv.embeddings.openai import OpenAIEmbedAdapter
    from memv.llm.pydantic_ai import PydanticAIAdapter

    return OpenAIEmbedAdapter(), PydanticAIAdapter(model=model)


async def run(
    run_name: str = "baseline",
    config_name: str = "default",
    data_path: str | None = None,
    num_questions: int | None = None,
    max_concurrent: int = 5,
    timeout: int = 1200,
    top_k: int = 10,
    model: str = "openai:gpt-4.1-mini",
    stages: list[str] | None = None,
    resume: bool = True,
):
    stages = stages or ["add", "search", "evaluate"]
    embedding_client, llm_client = _make_clients(model=model)
    print(f"Model: {model}")

    total_start = time.monotonic()

    if "add" in stages:
        print(f"\n{'=' * 60}")
        print("STAGE 1: ADD")
        print(f"{'=' * 60}\n")
        await add.run(
            run_name=run_name,
            config_name=config_name,
            data_path=data_path,
            num_questions=num_questions,
            max_concurrent=max_concurrent,
            timeout=timeout,
            resume=resume,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )

    if "search" in stages:
        print(f"\n{'=' * 60}")
        print("STAGE 2: SEARCH")
        print(f"{'=' * 60}\n")
        await search.run(
            run_name=run_name,
            config_name=config_name,
            data_path=data_path,
            num_questions=num_questions,
            top_k=top_k,
            max_concurrent=max_concurrent * 2,  # search is lighter than add
            timeout=timeout,
            resume=resume,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )

    if "evaluate" in stages:
        print(f"\n{'=' * 60}")
        print("STAGE 3: EVALUATE")
        print(f"{'=' * 60}\n")
        await evaluate.run(
            run_name=run_name,
            llm_client=llm_client,
            resume=resume,
        )

    total_elapsed = time.monotonic() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done. Total time: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark Runner")
    parser.add_argument("--run-name", default="baseline", help="Name for this run")
    parser.add_argument("--config", default="default", help="Config preset name")
    parser.add_argument("--data-path", default=None, help="Path to dataset JSON")
    parser.add_argument("--num-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent question processing")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-question timeout in seconds")
    parser.add_argument("--top-k", type=int, default=10, help="Number of memories to retrieve")
    parser.add_argument(
        "--model",
        default="openai:gpt-4.1-mini",
        help="PydanticAI model string (e.g. google-gla:gemini-2.5-flash, groq:llama-3.3-70b-versatile)",
    )
    parser.add_argument("--stages", default="add,search,evaluate", help="Comma-separated stages to run")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore prior checkpoints")
    args = parser.parse_args()

    asyncio.run(
        run(
            run_name=args.run_name,
            config_name=args.config,
            data_path=args.data_path,
            num_questions=args.num_questions,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            top_k=args.top_k,
            model=args.model,
            stages=args.stages.split(","),
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
