"""Tests for the memv MCP server tool logic."""

import json
import re

import pytest
from mcp.server.fastmcp import FastMCP

from memv import ExtractedKnowledge, Memory
from memv.mcp.server import (
    create_server,
    do_add_conversation,
    do_add_memory,
    do_delete_memory,
    do_list_memories,
    do_search_memory,
)
from memv.processing.extraction import ExtractionResponse


@pytest.fixture
async def memory(tmp_path, mock_embedder):
    mem = Memory(
        db_url=str(tmp_path / "mcp_test.db"),
        embedding_client=mock_embedder,
        embedding_dimensions=1536,
        enable_embedding_cache=False,
    )
    async with mem:
        yield mem


USER_ID = "test-user"


# ── search_memory ────────────────────────────────────────────────────


async def test_search_empty(memory):
    result = await do_search_memory(memory, USER_ID, "anything")
    assert result == "No relevant memories found."


async def test_search_finds_added_memory(memory):
    await do_add_memory(memory, USER_ID, "User's favorite language is Python")
    result = await do_search_memory(memory, USER_ID, "User's favorite language is Python")
    assert "Python" in result


async def test_search_respects_top_k(memory):
    for i in range(5):
        await do_add_memory(memory, USER_ID, f"Fact number {i} about unique topic {i}")
    result = await do_search_memory(memory, USER_ID, "unique topic", top_k=2)
    lines = [line for line in result.splitlines() if line.startswith("- ")]
    assert 1 <= len(lines) <= 2


# ── add_memory ───────────────────────────────────────────────────────


async def test_add_memory_returns_confirmation(memory):
    result = await do_add_memory(memory, USER_ID, "User prefers dark mode")
    assert "Remembered" in result
    assert "dark mode" in result
    assert "(id:" in result


async def test_add_memory_dedup(memory):
    await do_add_memory(memory, USER_ID, "User likes cats")
    result = await do_add_memory(memory, USER_ID, "User likes cats")
    assert "duplicate" in result.lower()


# ── add_conversation ─────────────────────────────────────────────────


async def test_add_conversation_without_llm(memory):
    result = await do_add_conversation(memory, USER_ID, "Hi there", "Hello!", has_llm=False)
    assert "Stored exchange" in result
    assert "--llm-model" in result


async def test_add_conversation_with_llm_extracts_knowledge(tmp_path, mock_embedder, mock_llm):
    mock_llm.set_responses("generate", [json.dumps({"title": "Intro", "content": "User shared favorite language."})])
    mock_llm.set_responses(
        "generate_structured",
        [ExtractionResponse(extracted=[ExtractedKnowledge(statement="User likes Python", knowledge_type="new", confidence=0.9)])],
    )
    mem = Memory(
        db_url=str(tmp_path / "mcp_llm.db"),
        embedding_client=mock_embedder,
        llm_client=mock_llm,
        embedding_dimensions=1536,
        enable_episode_merging=False,
        enable_embedding_cache=False,
    )
    async with mem:
        result = await do_add_conversation(mem, USER_ID, "I like Python", "Great choice!", has_llm=True)
        assert "Extracted 1 knowledge entry" in result

        listing = await do_list_memories(mem, USER_ID)
        assert "User likes Python" in listing


# ── list_memories ────────────────────────────────────────────────────


async def test_list_empty(memory):
    result = await do_list_memories(memory, USER_ID)
    assert result == "No memories stored."


async def test_list_after_add(memory):
    await do_add_memory(memory, USER_ID, "User is an AI engineer")
    result = await do_list_memories(memory, USER_ID)
    assert "AI engineer" in result
    assert "(id:" in result


async def test_list_pagination(memory):
    for i in range(5):
        await do_add_memory(memory, USER_ID, f"Distinct fact {i} with unique content {i}")
    result = await do_list_memories(memory, USER_ID, limit=2, offset=0)
    lines = [line for line in result.split("\n") if line.startswith("- ")]
    assert len(lines) == 2


# ── delete_memory ────────────────────────────────────────────────────


async def test_delete_existing(memory):
    add_result = await do_add_memory(memory, USER_ID, "Temporary fact")
    knowledge_id = re.search(r"\(id: ([^)]+)\)", add_result).group(1)

    result = await do_delete_memory(memory, USER_ID, knowledge_id)
    assert "Deleted" in result

    list_result = await do_list_memories(memory, USER_ID)
    assert list_result == "No memories stored."


async def test_delete_nonexistent(memory):
    result = await do_delete_memory(memory, USER_ID, "00000000-0000-0000-0000-000000000000")
    assert "not found" in result


async def test_delete_rejects_cross_user(memory):
    add_result = await do_add_memory(memory, "alice", "Alice's secret fact")
    knowledge_id = re.search(r"\(id: ([^)]+)\)", add_result).group(1)

    result = await do_delete_memory(memory, "bob", knowledge_id)
    assert "not found" in result

    listing = await do_list_memories(memory, "alice")
    assert "Alice's secret fact" in listing


# ── full cycle ───────────────────────────────────────────────────────


async def test_add_search_delete_cycle(memory):
    await do_add_memory(memory, USER_ID, "User lives in Warsaw")
    await do_add_memory(memory, USER_ID, "User works at a startup")

    search = await do_search_memory(memory, USER_ID, "User lives in Warsaw")
    assert "Warsaw" in search

    listing = await do_list_memories(memory, USER_ID)
    assert "Warsaw" in listing
    assert "startup" in listing

    knowledge_id = re.search(r"\(id: ([^)]+)\)", listing).group(1)
    await do_delete_memory(memory, USER_ID, knowledge_id)

    listing = await do_list_memories(memory, USER_ID)
    lines = [line for line in listing.split("\n") if line.startswith("- ")]
    assert len(lines) == 1


# ── create_server smoke test ─────────────────────────────────────────


def test_create_server_returns_fastmcp(tmp_path, mock_embedder):
    server = create_server(
        db_url=str(tmp_path / "smoke.db"),
        default_user_id="test",
        embedding_client=mock_embedder,
    )
    assert isinstance(server, FastMCP)
