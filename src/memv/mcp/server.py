"""memv MCP server — exposes memory operations as MCP tools."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context, FastMCP

from memv import KnowledgeInput, Memory

if TYPE_CHECKING:
    from memv.protocols import EmbeddingClient, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    memory: Memory
    default_user_id: str
    has_llm: bool


# ── Tool logic (testable without MCP) ───────────────────────────────


async def do_search_memory(memory: Memory, user_id: str, query: str, top_k: int = 10) -> str:
    result = await memory.retrieve(query, user_id=user_id, top_k=top_k)
    if not result.retrieved_knowledge:
        return "No relevant memories found."
    return result.to_prompt()


async def do_add_memory(memory: Memory, user_id: str, statement: str) -> str:
    entry = await memory.add_knowledge(user_id, KnowledgeInput(statement=statement))
    if entry is None:
        return "Already stored — duplicate detected."
    return f"Remembered: {entry.statement} (id: {entry.id})"


async def do_add_conversation(memory: Memory, user_id: str, user_message: str, assistant_message: str, *, has_llm: bool) -> str:
    await memory.add_exchange(user_id, user_message, assistant_message)
    if not has_llm:
        return "Stored exchange. Configure --llm-model to enable knowledge extraction."
    count = await memory.flush(user_id)
    if count > 0:
        return f"Stored exchange. Extracted {count} knowledge {'entry' if count == 1 else 'entries'} from all pending messages."
    return "Stored exchange. No new knowledge extracted from pending messages."


async def do_list_memories(memory: Memory, user_id: str, limit: int = 20, offset: int = 0, include_expired: bool = False) -> str:
    entries = await memory.list_knowledge(user_id, limit=limit, offset=offset, include_expired=include_expired)
    if not entries:
        return "No memories stored."
    lines = []
    for entry in entries:
        status = " [expired]" if entry.expired_at else ""
        lines.append(f"- {entry.statement} (id: {entry.id}){status}")
    return "\n".join(lines)


async def do_delete_memory(memory: Memory, user_id: str, knowledge_id: str) -> str:
    entry = await memory.get_knowledge(knowledge_id)
    if entry is None or entry.user_id != user_id:
        return f"Memory {knowledge_id} not found."
    deleted = await memory.delete_knowledge(knowledge_id)
    if deleted:
        return f"Deleted memory {knowledge_id}."
    return f"Memory {knowledge_id} not found."


# ── Client builders ──────────────────────────────────────────────────


def _build_embedding_client(provider: str, model: str | None) -> EmbeddingClient:
    if provider == "openai":
        from memv.embeddings.openai import OpenAIEmbedAdapter

        return OpenAIEmbedAdapter(model=model) if model else OpenAIEmbedAdapter()
    if provider == "voyage":
        from memv.embeddings.voyage import VoyageEmbedAdapter

        return VoyageEmbedAdapter(model=model) if model else VoyageEmbedAdapter()
    if provider == "cohere":
        from memv.embeddings.cohere import CohereEmbedAdapter

        return CohereEmbedAdapter(model=model) if model else CohereEmbedAdapter()
    if provider == "local":
        from memv.embeddings.fastembed import FastEmbedAdapter

        return FastEmbedAdapter(model=model) if model else FastEmbedAdapter()
    raise ValueError(f"Unknown embedding provider: {provider!r}. Options: openai, voyage, cohere, local")


def _build_llm_client(model: str) -> LLMClient:
    from memv.llm.pydantic_ai import PydanticAIAdapter

    return PydanticAIAdapter(model)


# ── Server factory ───────────────────────────────────────────────────


def create_server(
    *,
    db_url: str,
    default_user_id: str,
    embedding_provider: str = "openai",
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    llm_model: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    llm_client: LLMClient | None = None,
) -> FastMCP:
    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
        embedder = embedding_client or _build_embedding_client(embedding_provider, embedding_model)
        dims = embedding_dimensions or getattr(embedder, "dimensions", None)
        llm = llm_client or (_build_llm_client(llm_model) if llm_model else None)

        memory = Memory(
            db_url=db_url,
            embedding_client=embedder,
            llm_client=llm,
            embedding_dimensions=dims,
        )
        await memory.open()
        try:
            yield AppContext(memory=memory, default_user_id=default_user_id, has_llm=llm is not None)
        finally:
            await memory.close()

    mcp = FastMCP(
        "memv",
        instructions="Memory system for AI agents. Use search_memory to recall stored knowledge, add_memory to remember facts.",
        lifespan=lifespan,
    )

    def _app(ctx: Context) -> AppContext:
        return ctx.request_context.lifespan_context

    def _user_id(ctx: Context, user_id: str | None) -> str:
        return user_id or _app(ctx).default_user_id

    # ── MCP tool wrappers ────────────────────────────────────────────

    @mcp.tool()
    async def search_memory(query: str, ctx: Context, user_id: str | None = None, top_k: int = 10) -> str:
        """Search memory for relevant knowledge.

        Args:
            query: What to search for (natural language)
            user_id: Override default user ID
            top_k: Maximum number of results
        """
        return await do_search_memory(_app(ctx).memory, _user_id(ctx, user_id), query, top_k)

    @mcp.tool()
    async def add_memory(statement: str, ctx: Context, user_id: str | None = None) -> str:
        """Store a fact in memory.

        Args:
            statement: The fact to remember (e.g. "User prefers dark mode")
            user_id: Override default user ID
        """
        return await do_add_memory(_app(ctx).memory, _user_id(ctx, user_id), statement)

    @mcp.tool()
    async def add_conversation(user_message: str, assistant_message: str, ctx: Context, user_id: str | None = None) -> str:
        """Store a conversation exchange and extract knowledge from it.

        Requires LLM to be configured for knowledge extraction.
        Without LLM, messages are stored but no knowledge is extracted.

        Note: extraction runs a full LLM round-trip (segmentation + predict-calibrate) inline,
        which can take 10-30+ seconds on long histories. Configure your MCP client timeout accordingly.

        Args:
            user_message: What the user said
            assistant_message: What the assistant replied
            user_id: Override default user ID
        """
        app = _app(ctx)
        return await do_add_conversation(app.memory, _user_id(ctx, user_id), user_message, assistant_message, has_llm=app.has_llm)

    @mcp.tool()
    async def list_memories(
        ctx: Context, user_id: str | None = None, limit: int = 20, offset: int = 0, include_expired: bool = False
    ) -> str:
        """List stored knowledge for a user.

        Args:
            user_id: Override default user ID
            limit: Maximum entries to return
            offset: Skip this many entries (for pagination)
            include_expired: If True, also surface superseded entries (marked [expired])
        """
        return await do_list_memories(_app(ctx).memory, _user_id(ctx, user_id), limit, offset, include_expired)

    @mcp.tool()
    async def delete_memory(knowledge_id: str, ctx: Context, user_id: str | None = None) -> str:
        """Permanently delete a memory entry owned by the caller.

        Returns "not found" both when the UUID is unknown and when it belongs to another user —
        no information leak about which UUIDs exist for other users.

        Args:
            knowledge_id: UUID of the knowledge entry to delete
            user_id: Override default user ID
        """
        return await do_delete_memory(_app(ctx).memory, _user_id(ctx, user_id), knowledge_id)

    return mcp
