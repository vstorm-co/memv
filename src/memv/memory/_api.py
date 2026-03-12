"""Public API implementations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from memv.models import (
    KnowledgeInput,
    Message,
    MessageRole,
    RetrievalResult,
    SemanticKnowledge,
)

if TYPE_CHECKING:
    from memv.memory._lifecycle import LifecycleManager
    from memv.memory._task_manager import TaskManager

logger = logging.getLogger(__name__)


async def add_message(lifecycle: LifecycleManager, message: Message) -> None:
    """Add a message to memory."""
    lifecycle.ensure_open()
    await lifecycle.messages.add(message)


async def add_exchange(
    lifecycle: LifecycleManager,
    task_manager: TaskManager,
    user_id: str,
    user_message: str,
    assistant_message: str,
    timestamp: datetime | None = None,
) -> tuple[Message, Message]:
    """
    Add a user/assistant exchange.

    If auto_process is enabled, buffers messages and triggers background
    processing when batch_threshold is reached.

    Returns the created Message objects.
    """
    lifecycle.ensure_open()
    ts = timestamp or datetime.now(timezone.utc)

    user_msg = Message(
        user_id=user_id,
        role=MessageRole.USER,
        content=user_message,
        sent_at=ts,
    )
    assistant_msg = Message(
        user_id=user_id,
        role=MessageRole.ASSISTANT,
        content=assistant_message,
        sent_at=ts,
    )

    await lifecycle.messages.add(user_msg)
    await lifecycle.messages.add(assistant_msg)

    # Track buffered messages for auto-processing
    if lifecycle.auto_process and lifecycle.llm is not None:
        task_manager.increment_buffer(user_id, 2)

        if task_manager.should_process(user_id):
            task_manager.schedule_processing(user_id)

    return user_msg, assistant_msg


async def retrieve(
    lifecycle: LifecycleManager,
    query: str,
    user_id: str,
    top_k: int = 10,
    vector_weight: float = 0.5,
    at_time: datetime | None = None,
    include_expired: bool = False,
) -> RetrievalResult:
    """
    Retrieve relevant knowledge for a query.

    Args:
        lifecycle: LifecycleManager instance
        query: Search query
        user_id: Filter results to this user only (required for privacy)
        top_k: Number of results to return
        vector_weight: Balance between vector (1.0) and text (0.0) search
        at_time: If provided, filter knowledge by validity at this event time
        include_expired: If True, include superseded (expired) records

    Returns:
        RetrievalResult containing knowledge statements.
    """
    lifecycle.ensure_open()
    if lifecycle.retriever is None:
        raise RuntimeError("Retriever not initialized")

    return await lifecycle.retriever.retrieve(
        query=query,
        user_id=user_id,
        top_k=top_k,
        vector_weight=vector_weight,
        at_time=at_time,
        include_expired=include_expired,
    )


async def clear_user(
    lifecycle: LifecycleManager,
    task_manager: TaskManager,
    user_id: str,
) -> dict[str, int]:
    """
    Delete all data for a user: messages, episodes, knowledge, and indices.

    Returns:
        Dict with counts of deleted items per category
    """
    lifecycle.ensure_open()

    # Cancel any pending processing for this user
    await task_manager.cancel_user_tasks(user_id)

    # Delete in order: indices first, then stores
    counts: dict[str, int] = {}

    # Clear knowledge indices
    counts["knowledge_vectors"] = await lifecycle.vector_index.clear_user(user_id)
    counts["knowledge_text"] = await lifecycle.text_index.clear_user(user_id)

    # Clear knowledge
    counts["knowledge"] = await lifecycle.knowledge.clear_user(user_id)

    # Clear episodes
    counts["episodes"] = await lifecycle.episodes.clear_user(user_id)

    # Clear messages
    counts["messages"] = await lifecycle.messages.clear_user(user_id)

    return counts


# -------------------------------------------------------------------------
# Knowledge CRUD
# -------------------------------------------------------------------------


async def list_knowledge(
    lifecycle: LifecycleManager,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    include_expired: bool = False,
) -> list[SemanticKnowledge]:
    """List knowledge entries for a user with pagination."""
    lifecycle.ensure_open()
    return await lifecycle.knowledge.list_by_user(user_id, limit=limit, offset=offset, include_expired=include_expired)


async def get_knowledge(lifecycle: LifecycleManager, knowledge_id: UUID | str) -> SemanticKnowledge | None:
    """Get a single knowledge entry by ID."""
    lifecycle.ensure_open()
    return await lifecycle.knowledge.get(knowledge_id)


async def invalidate_knowledge(lifecycle: LifecycleManager, knowledge_id: UUID | str) -> bool:
    """Mark knowledge as expired. Returns True if updated."""
    lifecycle.ensure_open()
    return await lifecycle.knowledge.invalidate(knowledge_id)


async def delete_knowledge(lifecycle: LifecycleManager, knowledge_id: UUID | str) -> bool:
    """Delete knowledge from all stores and indices."""
    lifecycle.ensure_open()
    deleted = await lifecycle.knowledge.delete(knowledge_id)
    if not deleted:
        return False
    await lifecycle.vector_index.delete(UUID(str(knowledge_id)))
    await lifecycle.text_index.delete(UUID(str(knowledge_id)))
    return True


async def add_knowledge(
    lifecycle: LifecycleManager,
    user_id: str,
    item: KnowledgeInput,
) -> SemanticKnowledge | None:
    """Inject knowledge directly.

    Embeds the statement, optionally checks for duplicates, and indexes
    in both vector and text indices. Injected knowledge is assigned
    importance_score=1.0 (maximum), representing explicit user intent.

    Returns the created entry, or None if deduplicated.
    """
    lifecycle.ensure_open()

    embedding = await lifecycle.embedder.embed(item.statement)

    if lifecycle.enable_knowledge_dedup:
        is_duplicate, score = await lifecycle.vector_index.has_near_duplicate(embedding, user_id, lifecycle.knowledge_dedup_threshold)
        if is_duplicate:
            logger.info("Skipping duplicate injection: '%s...' (score=%.3f)", item.statement[:50], score)
            return None

    knowledge = SemanticKnowledge(
        user_id=user_id,
        statement=item.statement,
        source_episode_id=None,
        importance_score=1.0,
        embedding=embedding,
        valid_at=item.valid_at,
        invalid_at=item.invalid_at,
    )

    await lifecycle.knowledge.add(knowledge)
    await lifecycle.vector_index.add(knowledge.id, embedding, user_id)
    await lifecycle.text_index.add(knowledge.id, knowledge.statement, user_id)

    return knowledge


async def add_knowledge_batch(
    lifecycle: LifecycleManager,
    user_id: str,
    items: list[KnowledgeInput],
) -> list[SemanticKnowledge]:
    """Batch inject multiple knowledge entries.

    Uses batch embedding for efficiency. Each injected entry is assigned
    importance_score=1.0 (maximum), representing explicit user intent.

    Args:
        lifecycle: LifecycleManager instance
        user_id: User this knowledge belongs to
        items: List of knowledge entries (statement, valid_at, invalid_at)

    Returns:
        List of created SemanticKnowledge entries (excludes duplicates).
    """
    lifecycle.ensure_open()

    if not items:
        return []

    statements = [item.statement for item in items]
    embeddings = await lifecycle.embedder.embed_batch(statements)

    created: list[SemanticKnowledge] = []
    for item, embedding in zip(items, embeddings, strict=True):
        if lifecycle.enable_knowledge_dedup:
            is_duplicate, score = await lifecycle.vector_index.has_near_duplicate(
                embedding, user_id, lifecycle.knowledge_dedup_threshold
            )
            if is_duplicate:
                logger.info("Skipping duplicate in batch: '%s...' (score=%.3f)", item.statement[:50], score)
                continue

        knowledge = SemanticKnowledge(
            user_id=user_id,
            statement=item.statement,
            source_episode_id=None,
            importance_score=1.0,
            embedding=embedding,
            valid_at=item.valid_at,
            invalid_at=item.invalid_at,
        )

        await lifecycle.knowledge.add(knowledge)
        await lifecycle.vector_index.add(knowledge.id, embedding, user_id)
        await lifecycle.text_index.add(knowledge.id, knowledge.statement, user_id)
        created.append(knowledge)

    return created
