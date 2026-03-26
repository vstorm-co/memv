"""
Test whether supersession actually fires with a real LLM.

Feeds contradicting information across two process() calls and checks
if superseded_by gets set on the old knowledge entry.

Run with:
    uv run python examples/test_supersession.py
"""

import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from memv import Memory
from memv.embeddings import OpenAIEmbedAdapter
from memv.llm import PydanticAIAdapter

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def main():
    memory = Memory(
        db_url=".db/test_supersession.db",
        embedding_client=OpenAIEmbedAdapter(),
        llm_client=PydanticAIAdapter("openai:gpt-4.1-mini"),
        enable_knowledge_dedup=False,  # don't dedup — we want to see contradictions
    )

    user_id = "supersession-test"

    async with memory:
        # Clear any previous state
        await memory.clear_user(user_id)

        # --- Round 1: Establish baseline facts ---
        await memory.add_exchange(
            user_id=user_id,
            user_message="I work at Google as a software engineer. I live in San Francisco.",
            assistant_message="Nice! What do you work on there?",
        )
        await memory.add_exchange(
            user_id=user_id,
            user_message="I'm on the Search team. I use Python for most of my work.",
            assistant_message="Cool, Python is great for that kind of work.",
        )

        count1 = await memory.process(user_id)
        logger.info("Round 1: extracted %d knowledge entries", count1)

        # Dump current knowledge
        knowledge1 = await memory._lifecycle.knowledge.list_by_user(user_id, limit=100)
        print("\n=== AFTER ROUND 1 ===")
        for k in knowledge1:
            print(f"  [{k.id}] {k.statement}")
            print(f"    expired_at={k.expired_at}, superseded_by={k.superseded_by}")

        # --- Round 2: Contradict the facts ---
        await memory.add_exchange(
            user_id=user_id,
            user_message="I just left Google. I started a new job at Anthropic this week.",
            assistant_message="Congratulations on the new role! What will you be doing?",
        )
        await memory.add_exchange(
            user_id=user_id,
            user_message="I'm a researcher now, focusing on AI safety. Also I moved to New York.",
            assistant_message="That's a big change!",
        )

        count2 = await memory.process(user_id)
        logger.info("Round 2: extracted %d knowledge entries", count2)

        # Dump all knowledge including expired
        knowledge2 = await memory._lifecycle.knowledge.list_by_user(user_id, limit=100, include_expired=True)
        print("\n=== AFTER ROUND 2 (all entries, including expired) ===")
        superseded_count = 0
        for k in knowledge2:
            status = "CURRENT" if k.is_current() else "EXPIRED"
            print(f"  [{status}] {k.statement}")
            print(f"    expired_at={k.expired_at}, superseded_by={k.superseded_by}")
            if k.superseded_by:
                superseded_count += 1

        # Current-only
        current = await memory._lifecycle.knowledge.list_by_user(user_id, limit=100, include_expired=False)
        print(f"\n=== SUMMARY ===")
        print(f"  Total entries (all):     {len(knowledge2)}")
        print(f"  Current entries:         {len(current)}")
        print(f"  Expired entries:         {len(knowledge2) - len(current)}")
        print(f"  With superseded_by set:  {superseded_count}")

        if superseded_count > 0:
            print("\n  SUPERSESSION WORKS.")
        elif len(knowledge2) - len(current) > 0:
            print("\n  Entries expired but superseded_by not set — vector fallback fired, not LLM index.")
        else:
            print("\n  NO SUPERSESSION DETECTED. The LLM did not produce contradiction/update types.")


if __name__ == "__main__":
    asyncio.run(main())
