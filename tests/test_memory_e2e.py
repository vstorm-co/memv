"""End-to-end tests for the Memory class."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from memv import ExtractedKnowledge, KnowledgeInput
from memv.memory.memory import Memory
from memv.processing.extraction import ExtractionResponse

from .conftest import MockEmbedder, MockLLM


def _ts(minutes: int = 0) -> datetime:
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def _episode_json(title="Test Episode", content="A test narrative."):
    return json.dumps({"title": title, "content": content})


def _extraction(statements: list[str]) -> ExtractionResponse:
    return ExtractionResponse(extracted=[ExtractedKnowledge(statement=s, knowledge_type="new", confidence=0.9) for s in statements])


def _make_memory(tmp_path, llm, embedder, **kwargs):
    db_path = str(tmp_path / "e2e.db")
    return Memory(
        db_path=db_path,
        embedding_client=embedder,
        llm_client=llm,
        embedding_dimensions=1536,
        enable_episode_merging=False,
        enable_embedding_cache=False,
        **kwargs,
    )


async def test_full_cycle(tmp_path):
    """add_exchange -> process -> retrieve -> verify statement found."""
    llm = MockLLM()
    embedder = MockEmbedder()

    # add_exchange creates 2 msgs (<=2, segmenter skips LLM)
    # Pipeline: 1 generate (episode gen) + 1 structured (cold start extraction)
    llm.set_responses("generate", [_episode_json("Python Chat", "User discussed Python")])
    llm.set_responses("generate_structured", [_extraction(["User likes Python"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I like Python", "Great choice!", timestamp=_ts())
        count = await memory.process("user1")

        assert count == 1

        result = await memory.retrieve("User likes Python", user_id="user1")
        assert len(result.retrieved_knowledge) >= 1
        statements = [k.statement for k in result.retrieved_knowledge]
        assert "User likes Python" in statements


async def test_process_returns_count(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User knows Fact A", "User knows Fact B"])])

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "hi", "hello", timestamp=_ts())
        count = await memory.process("user1")
        assert count == 2


async def test_process_no_messages(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        count = await memory.process("user1")
        assert count == 0


async def test_multiple_exchanges(tmp_path):
    """4 messages → segmentation + episode gen + extraction."""
    llm = MockLLM()
    embedder = MockEmbedder()

    # 4 msgs with same timestamp → 1 time batch, >2 msgs → LLM segmentation
    # Segmenter groups all 4 together
    llm.set_responses(
        "generate",
        [
            json.dumps([[0, 1, 2, 3]]),  # segmentation: all in one group
            _episode_json("Multi Exchange", "Multiple exchanges"),  # episode gen
        ],
    )
    llm.set_responses("generate_structured", [_extraction(["User studies AI"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I study AI", "Interesting!", timestamp=_ts())
        await memory.add_exchange("user1", "It's fun", "Indeed!", timestamp=_ts(1))
        count = await memory.process("user1")

        assert count == 1
        # Verify: 1 segmentation + 1 episode gen = 2 generate calls
        assert len(llm.calls["generate"]) == 2


async def test_user_isolation_e2e(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    # Two users, each gets episode gen + extraction
    llm.set_responses(
        "generate",
        [
            _episode_json("User1 Ep", "User1 narrative"),
            _episode_json("User2 Ep", "User2 narrative"),
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User likes cats"]),
            _extraction(["User likes dogs"]),
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I like cats", "Nice!", timestamp=_ts())
        await memory.process("user1")
        await memory.add_exchange("user2", "I like dogs", "Cool!", timestamp=_ts())
        await memory.process("user2")

        r1 = await memory.retrieve("cats", user_id="user1")
        r2 = await memory.retrieve("dogs", user_id="user2")

        s1 = [k.statement for k in r1.retrieved_knowledge]
        s2 = [k.statement for k in r2.retrieved_knowledge]
        assert "User likes cats" in s1
        assert "User likes dogs" not in s1
        assert "User likes dogs" in s2
        assert "User likes cats" not in s2


async def test_clear_user_after_extraction(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User has a fact to delete"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "hi", "hello", timestamp=_ts())
        await memory.process("user1")

        counts = await memory.clear_user("user1")
        assert counts["messages"] >= 2
        assert counts["episodes"] >= 1

        result = await memory.retrieve("User has a fact to delete", user_id="user1")
        assert len(result.retrieved_knowledge) == 0


async def test_clear_user_after_injection(tmp_path):
    """clear_user removes injected knowledge (source_episode_id=None)."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_knowledge("user1", KnowledgeInput(statement="User works at Anthropic"))
        await memory.add_knowledge("user1", KnowledgeInput(statement="User likes coffee"))

        counts = await memory.clear_user("user1")
        assert counts["knowledge"] == 2

        assert await memory.list_knowledge("user1") == []
        result = await memory.retrieve("Anthropic", user_id="user1")
        assert len(result.retrieved_knowledge) == 0


async def test_to_prompt_integration(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User is an engineer"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I'm an engineer", "Cool!", timestamp=_ts())
        await memory.process("user1")

        result = await memory.retrieve("User is an engineer", user_id="user1")
        prompt = result.to_prompt()
        assert "## Relevant Context" in prompt
        assert "User is an engineer" in prompt


async def test_auto_process_at_threshold(tmp_path):
    """auto_process=True, batch_threshold=4 → 2 exchanges (4 msgs) trigger background processing."""
    llm = MockLLM()
    embedder = MockEmbedder()

    # 4 msgs with same timestamp → 1 time batch, >2 → LLM segmentation
    llm.set_responses(
        "generate",
        [
            json.dumps([[0, 1, 2, 3]]),  # segmentation
            _episode_json("Auto", "Auto processed"),  # episode gen
        ],
    )
    llm.set_responses("generate_structured", [_extraction(["User has auto fact"])])

    memory = _make_memory(tmp_path, llm, embedder, auto_process=True, batch_threshold=4)
    async with memory:
        await memory.add_exchange("user1", "msg1", "resp1", timestamp=_ts())
        # 2 msgs buffered, below threshold
        await memory.add_exchange("user1", "msg2", "resp2", timestamp=_ts(1))
        # 4 msgs buffered, at threshold → triggers background processing

        count = await memory.wait_for_processing("user1", timeout=10)
        assert count == 1

        result = await memory.retrieve("Auto fact", user_id="user1")
        assert len(result.retrieved_knowledge) >= 1


async def test_flush_forces_processing(tmp_path):
    """flush() processes below threshold."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json("Flushed", "Flushed episode")])
    llm.set_responses("generate_structured", [_extraction(["User has flushed fact"])])

    memory = _make_memory(tmp_path, llm, embedder, auto_process=True, batch_threshold=100)
    async with memory:
        await memory.add_exchange("user1", "hi", "hello", timestamp=_ts())
        # Below threshold (2 < 100), but flush forces processing
        count = await memory.flush("user1")
        assert count == 1


async def test_confidence_filters_low_quality_statements(tmp_path):
    """Only statements with confidence >= 0.7 survive the pipeline filter."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json("Mixed", "Mixed quality")])
    llm.set_responses(
        "generate_structured",
        [
            ExtractionResponse(
                extracted=[
                    ExtractedKnowledge(statement="User prefers Vim", knowledge_type="new", confidence=0.9),
                    ExtractedKnowledge(statement="User likes Emacs", knowledge_type="new", confidence=0.5),
                    ExtractedKnowledge(statement="User uses Neovim", knowledge_type="new", confidence=0.3),
                ]
            )
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I prefer Vim", "Nice!", timestamp=_ts())
        count = await memory.process("user1")
        assert count == 1

        result = await memory.retrieve("Vim", user_id="user1")
        assert result.retrieved_knowledge[0].statement == "User prefers Vim"


async def test_dedup_skips_duplicate(tmp_path):
    """enable_knowledge_dedup=True → identical statement not stored twice."""
    llm = MockLLM()
    embedder = MockEmbedder()

    # First process: episode gen (cold start, no prediction)
    # Second process: episode gen + prediction (existing knowledge exists)
    llm.set_responses(
        "generate",
        [
            _episode_json("Ep1", "First episode"),
            _episode_json("Ep2", "Second episode"),
            "I predict user likes Python.",  # prediction for second process
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User likes Python"]),
            _extraction(["User likes Python"]),  # duplicate
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=True, knowledge_dedup_threshold=0.8)
    async with memory:
        await memory.add_exchange("user1", "I like Python", "Cool!", timestamp=_ts())
        count1 = await memory.process("user1")
        assert count1 == 1

        await memory.add_exchange("user1", "Python is great", "Indeed!", timestamp=_ts(60))
        count2 = await memory.process("user1")
        # Second extraction should be skipped as duplicate (identical embedding)
        assert count2 == 0


# ---------------------------------------------------------------------------
# Knowledge CRUD e2e
# ---------------------------------------------------------------------------


async def test_list_knowledge(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User likes cats", "User likes dogs"])])

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I like cats and dogs", "Nice!", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        assert len(entries) == 2
        assert all(k.user_id == "user1" for k in entries)

        # User isolation
        assert await memory.list_knowledge("user2") == []


async def test_list_knowledge_pagination(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["Fact A", "Fact B", "Fact C"])])

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "stuff", "ok", timestamp=_ts())
        await memory.process("user1")

        page1 = await memory.list_knowledge("user1", limit=2, offset=0)
        page2 = await memory.list_knowledge("user1", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 1
        all_ids = {k.id for k in page1 + page2}
        assert len(all_ids) == 3


async def test_list_knowledge_include_expired(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User lives in NYC"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I live in NYC", "Nice!", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        await memory.invalidate_knowledge(entries[0].id)

        assert await memory.list_knowledge("user1") == []
        assert len(await memory.list_knowledge("user1", include_expired=True)) == 1


async def test_get_knowledge(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User is an engineer"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I'm an engineer", "Cool!", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        got = await memory.get_knowledge(entries[0].id)
        assert got is not None
        assert got.statement == "User is an engineer"


async def test_get_knowledge_nonexistent(tmp_path):
    from uuid import uuid4

    llm = MockLLM()
    embedder = MockEmbedder()

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        assert await memory.get_knowledge(uuid4()) is None


async def test_invalidate_knowledge(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User lives in NYC"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I live in NYC", "Nice!", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        kid = entries[0].id

        assert await memory.invalidate_knowledge(kid) is True

        # No longer in default list
        assert await memory.list_knowledge("user1") == []
        # Visible with include_expired
        expired = await memory.list_knowledge("user1", include_expired=True)
        assert len(expired) == 1
        assert expired[0].expired_at is not None

        # Excluded from retrieve()
        result = await memory.retrieve("User lives in NYC", user_id="user1")
        assert len(result.retrieved_knowledge) == 0

        # But visible when including expired in retrieval
        result_expired = await memory.retrieve("User lives in NYC", user_id="user1", include_expired=True)
        assert len(result_expired.retrieved_knowledge) == 1


async def test_invalidate_knowledge_nonexistent(tmp_path):
    from uuid import uuid4

    llm = MockLLM()
    embedder = MockEmbedder()

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        assert await memory.invalidate_knowledge(uuid4()) is False


async def test_invalidate_knowledge_already_expired(tmp_path):
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User likes tea"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "I like tea", "Nice!", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        kid = entries[0].id

        assert await memory.invalidate_knowledge(kid) is True
        # Second invalidation returns False (already expired)
        assert await memory.invalidate_knowledge(kid) is False


async def test_delete_knowledge(tmp_path):
    """Hard-delete removes from DB, vector index, and text index."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User has a secret"])])

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_exchange("user1", "secret info", "ok", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        kid = entries[0].id

        assert await memory.delete_knowledge(kid) is True
        # Gone from store
        assert await memory.get_knowledge(kid) is None
        # Gone from list
        assert await memory.list_knowledge("user1") == []
        # Gone from retrieval
        result = await memory.retrieve("User has a secret", user_id="user1")
        assert len(result.retrieved_knowledge) == 0

        # Double-delete returns False
        assert await memory.delete_knowledge(kid) is False


async def test_delete_knowledge_nonexistent(tmp_path):
    from uuid import uuid4

    llm = MockLLM()
    embedder = MockEmbedder()

    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        assert await memory.delete_knowledge(uuid4()) is False


async def test_add_knowledge(tmp_path):
    """Inject a single statement and retrieve it."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        k = await memory.add_knowledge("user1", KnowledgeInput(statement="User works at Anthropic"))
        assert k is not None
        assert k.statement == "User works at Anthropic"
        assert k.source_episode_id is None

        result = await memory.retrieve("Anthropic", user_id="user1")
        assert any(r.statement == "User works at Anthropic" for r in result.retrieved_knowledge)


async def test_add_knowledge_temporal(tmp_path):
    """Injected knowledge respects valid_at/invalid_at in temporal retrieval."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_knowledge(
            "user1",
            KnowledgeInput(
                statement="User visited Tokyo",
                valid_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
                invalid_at=datetime(2024, 3, 31, tzinfo=timezone.utc),
            ),
        )

        result = await memory.retrieve("Tokyo", user_id="user1", at_time=datetime(2024, 3, 15, tzinfo=timezone.utc))
        assert len(result.retrieved_knowledge) == 1

        result = await memory.retrieve("Tokyo", user_id="user1", at_time=datetime(2024, 5, 1, tzinfo=timezone.utc))
        assert len(result.retrieved_knowledge) == 0


async def test_add_knowledge_batch(tmp_path):
    """Batch inject multiple entries."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        items = [KnowledgeInput(statement="Fact A"), KnowledgeInput(statement="Fact B"), KnowledgeInput(statement="Fact C")]
        created = await memory.add_knowledge_batch("user1", items)
        assert len(created) == 3

        entries = await memory.list_knowledge("user1")
        assert {k.statement for k in entries} == {"Fact A", "Fact B", "Fact C"}


async def test_add_knowledge_dedup(tmp_path):
    """Duplicate injection returns None when dedup is enabled."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=True, knowledge_dedup_threshold=0.8)
    async with memory:
        k1 = await memory.add_knowledge("user1", KnowledgeInput(statement="User likes Python"))
        k2 = await memory.add_knowledge("user1", KnowledgeInput(statement="User likes Python"))
        assert k1 is not None
        assert k2 is None

        assert len(await memory.list_knowledge("user1")) == 1


async def test_add_knowledge_batch_dedup(tmp_path):
    """Batch dedup skips intra-batch and cross-existing duplicates."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=True, knowledge_dedup_threshold=0.8)
    async with memory:
        # Pre-existing entry
        await memory.add_knowledge("user1", KnowledgeInput(statement="User likes Python"))

        # Batch: one duplicate of existing, one duplicate within batch, one new
        items = [
            KnowledgeInput(statement="User likes Python"),
            KnowledgeInput(statement="User likes cats"),
            KnowledgeInput(statement="User likes cats"),
        ]
        created = await memory.add_knowledge_batch("user1", items)

        assert len(created) == 1
        assert created[0].statement == "User likes cats"
        assert len(await memory.list_knowledge("user1")) == 2


async def test_add_knowledge_empty_statement(tmp_path):
    """Empty or whitespace-only statements are rejected."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        with pytest.raises(ValidationError, match="non-empty"):
            await memory.add_knowledge("user1", KnowledgeInput(statement=""))
        with pytest.raises(ValidationError, match="non-empty"):
            await memory.add_knowledge("user1", KnowledgeInput(statement="   "))
        with pytest.raises(ValidationError, match="non-empty"):
            await memory.add_knowledge_batch("user1", [KnowledgeInput(statement="")])


async def test_add_knowledge_invalid_temporal_range(tmp_path):
    """invalid_at before valid_at is rejected."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        with pytest.raises(ValidationError, match="invalid_at must be after"):
            await memory.add_knowledge("user1", KnowledgeInput(statement="Fact", valid_at=_ts(60), invalid_at=_ts(0)))
        with pytest.raises(ValidationError, match="invalid_at must be after"):
            await memory.add_knowledge_batch("user1", [KnowledgeInput(statement="Fact", valid_at=_ts(60), invalid_at=_ts(0))])


async def test_add_knowledge_user_isolation(tmp_path):
    """Injected knowledge is isolated per user."""
    llm = MockLLM()
    embedder = MockEmbedder()
    memory = _make_memory(tmp_path, llm, embedder)
    async with memory:
        await memory.add_knowledge("user1", KnowledgeInput(statement="User likes cats"))
        await memory.add_knowledge("user2", KnowledgeInput(statement="User likes dogs"))

        r1 = await memory.retrieve("cats", user_id="user1")
        r2 = await memory.retrieve("dogs", user_id="user2")
        assert any(k.statement == "User likes cats" for k in r1.retrieved_knowledge)
        assert not any(k.statement == "User likes dogs" for k in r1.retrieved_knowledge)
        assert any(k.statement == "User likes dogs" for k in r2.retrieved_knowledge)
        assert not any(k.statement == "User likes cats" for k in r2.retrieved_knowledge)


# ---------------------------------------------------------------------------
# Contradiction / supersedes e2e
# ---------------------------------------------------------------------------


async def test_contradiction_with_supersedes_invalidates_old(tmp_path):
    """Contradiction with supersedes index invalidates correct entry + sets audit trail."""
    llm = MockLLM()
    embedder = MockEmbedder()

    # First process: cold start
    llm.set_responses(
        "generate",
        [
            _episode_json("Setup", "First ep"),
            _episode_json("Update", "Second ep"),
            "I predict user likes Python.",  # prediction for second process
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User lives in NYC"]),
            ExtractionResponse(
                extracted=[
                    ExtractedKnowledge(
                        statement="User lives in Berlin",
                        knowledge_type="contradiction",
                        confidence=0.95,
                        supersedes=0,
                    )
                ]
            ),
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I live in NYC", "Cool!", timestamp=_ts())
        await memory.process("user1")

        await memory.add_exchange("user1", "I moved to Berlin", "Nice!", timestamp=_ts(60))
        await memory.process("user1")

        # Old entry should be expired with superseded_by set
        all_entries = await memory.list_knowledge("user1", include_expired=True)
        current = [k for k in all_entries if k.expired_at is None]
        expired = [k for k in all_entries if k.expired_at is not None]

        assert len(current) == 1
        assert current[0].statement == "User lives in Berlin"
        assert len(expired) == 1
        assert expired[0].statement == "User lives in NYC"
        assert expired[0].superseded_by == current[0].id


async def test_update_type_also_invalidates(tmp_path):
    """knowledge_type='update' triggers invalidation same as 'contradiction', with audit trail."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses(
        "generate",
        [
            _episode_json("Setup", "First ep"),
            _episode_json("Refine", "Second ep"),
            "I predict user likes Python.",
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User works at Acme"]),
            ExtractionResponse(
                extracted=[
                    ExtractedKnowledge(
                        statement="User works at Acme as a senior engineer",
                        knowledge_type="update",
                        confidence=0.9,
                        supersedes=0,
                    )
                ]
            ),
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I work at Acme", "Cool!", timestamp=_ts())
        await memory.process("user1")

        await memory.add_exchange("user1", "I'm a senior engineer at Acme", "Nice!", timestamp=_ts(60))
        await memory.process("user1")

        all_entries = await memory.list_knowledge("user1", include_expired=True)
        current = [k for k in all_entries if k.expired_at is None]
        expired = [k for k in all_entries if k.expired_at is not None]

        assert len(current) == 1
        assert current[0].statement == "User works at Acme as a senior engineer"
        # Verify audit trail — proves index-based path was used, not vector fallback
        assert len(expired) == 1
        assert expired[0].superseded_by == current[0].id


async def test_out_of_bounds_supersedes_falls_back(tmp_path):
    """Out-of-bounds supersedes index stores new entry, old entry unchanged (no superseded_by)."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses(
        "generate",
        [
            _episode_json("Setup", "First ep"),
            _episode_json("Bad idx", "Second ep"),
            "prediction",
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User likes tea"]),
            ExtractionResponse(
                extracted=[
                    ExtractedKnowledge(
                        statement="User likes coffee",
                        knowledge_type="contradiction",
                        confidence=0.9,
                        supersedes=999,  # out of bounds
                    )
                ]
            ),
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I like tea", "Ok!", timestamp=_ts())
        await memory.process("user1")

        await memory.add_exchange("user1", "Actually coffee", "Sure!", timestamp=_ts(60))
        count = await memory.process("user1")

        assert count == 1
        all_entries = await memory.list_knowledge("user1", include_expired=True)
        coffee = next(k for k in all_entries if k.statement == "User likes coffee")
        # New entry must not have been self-invalidated
        assert coffee.expired_at is None
        for entry in all_entries:
            assert entry.superseded_by is None


async def test_contradiction_without_supersedes_no_audit_trail(tmp_path):
    """contradiction with supersedes=None uses vector fallback — no superseded_by set."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses(
        "generate",
        [
            _episode_json("Setup", "First ep"),
            _episode_json("Change", "Second ep"),
            "prediction",
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User likes tea"]),
            ExtractionResponse(
                extracted=[
                    ExtractedKnowledge(
                        statement="User likes coffee",
                        knowledge_type="contradiction",
                        confidence=0.9,
                    )
                ]
            ),
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I like tea", "Ok!", timestamp=_ts())
        await memory.process("user1")

        await memory.add_exchange("user1", "Actually coffee", "Sure!", timestamp=_ts(60))
        count = await memory.process("user1")

        assert count == 1
        all_entries = await memory.list_knowledge("user1", include_expired=True)
        coffee = next(k for k in all_entries if k.statement == "User likes coffee")
        # New entry must not have been self-invalidated
        assert coffee.expired_at is None
        for entry in all_entries:
            assert entry.superseded_by is None


async def test_multiple_extractions_superseding_same_index(tmp_path):
    """Two extractions pointing at same index: first invalidates, second is no-op."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses(
        "generate",
        [
            _episode_json("Setup", "First ep"),
            _episode_json("Double", "Second ep"),
            "prediction",
        ],
    )
    llm.set_responses(
        "generate_structured",
        [
            _extraction(["User lives in NYC"]),
            ExtractionResponse(
                extracted=[
                    ExtractedKnowledge(
                        statement="User lives in Berlin",
                        knowledge_type="contradiction",
                        confidence=0.9,
                        supersedes=0,
                    ),
                    ExtractedKnowledge(
                        statement="User moved to Berlin in 2025",
                        knowledge_type="update",
                        confidence=0.85,
                        supersedes=0,  # same index
                    ),
                ]
            ),
        ],
    )

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I live in NYC", "Cool!", timestamp=_ts())
        await memory.process("user1")

        await memory.add_exchange("user1", "I moved to Berlin in 2025", "Nice!", timestamp=_ts(60))
        count = await memory.process("user1")

        assert count == 2
        all_entries = await memory.list_knowledge("user1", include_expired=True)
        expired = [k for k in all_entries if k.expired_at is not None]
        assert len(expired) == 1
        # First supersedes call won, superseded_by points to "User lives in Berlin"
        berlin_entry = next(k for k in all_entries if k.statement == "User lives in Berlin")
        assert expired[0].superseded_by == berlin_entry.id


async def test_delete_knowledge_preserves_others(tmp_path):
    """Deleting one entry leaves other entries intact in all stores."""
    llm = MockLLM()
    embedder = MockEmbedder()

    llm.set_responses("generate", [_episode_json()])
    llm.set_responses("generate_structured", [_extraction(["User likes Python", "User likes Rust"])])

    memory = _make_memory(tmp_path, llm, embedder, enable_knowledge_dedup=False)
    async with memory:
        await memory.add_exchange("user1", "I like Python and Rust", "Cool!", timestamp=_ts())
        await memory.process("user1")

        entries = await memory.list_knowledge("user1")
        assert len(entries) == 2

        # Delete one
        to_delete = next(k for k in entries if k.statement == "User likes Python")
        to_keep = next(k for k in entries if k.statement == "User likes Rust")
        await memory.delete_knowledge(to_delete.id)

        # Other entry survives in store
        remaining = await memory.list_knowledge("user1")
        assert len(remaining) == 1
        assert remaining[0].id == to_keep.id

        # Other entry survives in retrieval
        result = await memory.retrieve("Rust", user_id="user1")
        assert any(k.statement == "User likes Rust" for k in result.retrieved_knowledge)
