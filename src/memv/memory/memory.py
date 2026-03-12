"""Memory class - main entry point for memv."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from memv.config import MemoryConfig
from memv.memory._api import (
    add_exchange,
    add_knowledge,
    add_knowledge_batch,
    add_message,
    clear_user,
    delete_knowledge,
    get_knowledge,
    invalidate_knowledge,
    list_knowledge,
    retrieve,
)
from memv.memory._lifecycle import LifecycleManager
from memv.memory._pipeline import Pipeline
from memv.memory._task_manager import TaskManager
from memv.models import KnowledgeInput, Message, ProcessTask, RetrievalResult

if TYPE_CHECKING:
    from uuid import UUID

    from memv.models import SemanticKnowledge
    from memv.protocols import EmbeddingClient, LLMClient


class Memory:
    """Main entry point for memv.

    Example:
        ```python
        memory = Memory(db_path="memory.db", embedding_client=embedder, llm_client=llm)
        await memory.open()

        await memory.add_message(Message(...))
        await memory.process(user_id="user123")  # Extract knowledge

        results = await memory.retrieve("query", user_id="user123")
        print(results.to_prompt())  # Formatted for LLM context

        await memory.close()
        ```

    Example: Auto-processing:
        ```python
        memory = Memory(
            db_path="memory.db",
            embedding_client=embedder,
            llm_client=llm,
            auto_process=True,      # Enable automatic processing
            batch_threshold=10,     # Messages before trigger
        )
        async with memory:
            # Messages buffer automatically, processing triggers at threshold
            await memory.add_exchange(user_id, user_msg, assistant_msg)

            # Force processing of buffered messages
            await memory.flush(user_id)

            # Wait for background processing to complete
            await memory.wait_for_processing(user_id, timeout=30)
        ```
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedding_client: EmbeddingClient | None = None,
        llm_client: LLMClient | None = None,
        embedding_dimensions: int | None = None,
        config: MemoryConfig | None = None,
        # Auto-processing config
        auto_process: bool | None = None,
        batch_threshold: int | None = None,
        max_retries: int | None = None,
        # Segmentation config
        segmentation_threshold: int | None = None,
        time_gap_minutes: int | None = None,
        use_legacy_segmentation: bool | None = None,
        # Deduplication config
        enable_knowledge_dedup: bool | None = None,
        knowledge_dedup_threshold: float | None = None,
        # Episode merging config
        enable_episode_merging: bool | None = None,
        merge_similarity_threshold: float | None = None,
        # Prediction-calibrate config
        max_statements_for_prediction: int | None = None,
        # Cache config
        enable_embedding_cache: bool | None = None,
        embedding_cache_size: int | None = None,
        embedding_cache_ttl_seconds: int | None = None,
    ):
        self._lifecycle = LifecycleManager(
            db_path=db_path,
            embedding_client=embedding_client,
            llm_client=llm_client,
            embedding_dimensions=embedding_dimensions,
            config=config,
            auto_process=auto_process,
            batch_threshold=batch_threshold,
            max_retries=max_retries,
            segmentation_threshold=segmentation_threshold,
            time_gap_minutes=time_gap_minutes,
            use_legacy_segmentation=use_legacy_segmentation,
            enable_knowledge_dedup=enable_knowledge_dedup,
            knowledge_dedup_threshold=knowledge_dedup_threshold,
            enable_episode_merging=enable_episode_merging,
            merge_similarity_threshold=merge_similarity_threshold,
            max_statements_for_prediction=max_statements_for_prediction,
            enable_embedding_cache=enable_embedding_cache,
            embedding_cache_size=embedding_cache_size,
            embedding_cache_ttl_seconds=embedding_cache_ttl_seconds,
        )
        self._pipeline = Pipeline(self._lifecycle)
        self._task_manager = TaskManager(self._lifecycle, self._pipeline)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def open(self) -> None:
        """Open all database connections and initialize components."""
        await self._lifecycle.open()

        # Recover buffer state from DB (for auto-processing after restart)
        if self._lifecycle.auto_process:
            await self._task_manager.recover_buffer_state()

    async def close(self, cancel_pending: bool = True) -> None:
        """
        Close all database connections.

        Args:
            cancel_pending: If True, cancel any running processing tasks.
                           If False, wait for them to complete first.
        """
        await self._task_manager.cancel_pending_tasks(wait=not cancel_pending)
        await self._lifecycle.close()

    async def __aenter__(self) -> "Memory":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def add_message(self, message: Message) -> None:
        """
        Add a message to memory.

        Messages are stored immediately. Call process() to extract knowledge.
        """
        await add_message(self._lifecycle, message)

    async def add_exchange(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        timestamp: datetime | None = None,
    ) -> tuple[Message, Message]:
        """
        Convenience method to add a user/assistant exchange.

        If auto_process is enabled, buffers messages and triggers background
        processing when batch_threshold is reached.

        Returns the created Message objects.
        """
        return await add_exchange(
            self._lifecycle,
            self._task_manager,
            user_id,
            user_message,
            assistant_message,
            timestamp,
        )

    async def retrieve(
        self,
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
            query: Search query
            user_id: Filter results to this user only (required for privacy)
            top_k: Number of results to return
            vector_weight: Balance between vector (1.0) and text (0.0) search
            at_time: If provided, filter knowledge by validity at this event time.
                     Returns only knowledge that was valid at that point in time.
            include_expired: If True, include superseded (expired) records.
                            Useful for viewing full history of a fact.

        Returns:
            RetrievalResult containing knowledge statements.
            Use result.to_prompt() to get formatted context for LLM.
        """
        return await retrieve(
            self._lifecycle,
            query,
            user_id,
            top_k,
            vector_weight,
            at_time,
            include_expired,
        )

    async def process(self, user_id: str) -> int:
        """
        Process unprocessed messages for a user into episodes and extract knowledge.

        Flow:
        1. Get messages not yet assigned to episodes
        2. Segment into episodes (boundary detection)
        3. Generate episode title/narrative
        4. Index episode for retrieval
        5. Retrieve existing knowledge for context
        6. Run predict-calibrate extraction
        7. Store extracted knowledge with embeddings

        Args:
            user_id: Process messages for this user

        Returns:
            Number of knowledge entries extracted
        """
        return await self._pipeline.process(user_id)

    def process_async(self, user_id: str) -> ProcessTask:
        """
        Non-blocking process. Returns handle to monitor/await.

        Args:
            user_id: Process messages for this user

        Returns:
            ProcessTask handle to monitor progress or await completion

        Example:
            task = memory.process_async(user_id)
            # ... do other work ...
            count = await task.wait()
        """
        return self._task_manager.process_async(user_id)

    async def process_messages(self, messages: list[Message], user_id: str) -> int:
        """
        Process explicit messages into an episode and extract knowledge.

        Lower-level method for when you want direct control over what gets processed.

        Args:
            messages: Messages to process (will be grouped into one episode)
            user_id: User ID for the episode

        Returns:
            Number of knowledge entries extracted
        """
        return await self._pipeline.process_messages(messages, user_id)

    async def wait_for_processing(self, user_id: str, timeout: float | None = None) -> int:
        """
        Wait for background processing to complete.

        Args:
            user_id: User whose processing to wait for
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            Number of knowledge entries extracted, or 0 if no task running

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        return await self._task_manager.wait_for_processing(user_id, timeout)

    async def flush(self, user_id: str) -> int:
        """
        Force processing of buffered messages regardless of threshold.

        Schedules processing if there are unprocessed messages and waits
        for completion.

        Args:
            user_id: User whose messages to process

        Returns:
            Number of knowledge entries extracted
        """
        return await self._task_manager.flush(user_id)

    async def clear_user(self, user_id: str) -> dict[str, int]:
        """
        Delete all data for a user: messages, episodes, knowledge, and indices.

        This is a destructive operation. Use with caution.

        Args:
            user_id: User whose data to delete

        Returns:
            Dict with counts of deleted items per category
        """
        return await clear_user(self._lifecycle, self._task_manager, user_id)

    # -------------------------------------------------------------------------
    # Knowledge CRUD
    # -------------------------------------------------------------------------

    async def list_knowledge(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[SemanticKnowledge]:
        """List knowledge entries for a user with pagination."""
        return await list_knowledge(self._lifecycle, user_id, limit=limit, offset=offset, include_expired=include_expired)

    async def get_knowledge(self, knowledge_id: UUID | str) -> SemanticKnowledge | None:
        """Get a single knowledge entry by ID."""
        return await get_knowledge(self._lifecycle, knowledge_id)

    async def invalidate_knowledge(self, knowledge_id: UUID | str) -> bool:
        """Mark knowledge as expired. Returns True if updated."""
        return await invalidate_knowledge(self._lifecycle, knowledge_id)

    async def delete_knowledge(self, knowledge_id: UUID | str) -> bool:
        """Delete knowledge from all stores and indices. Returns True if deleted."""
        return await delete_knowledge(self._lifecycle, knowledge_id)

    async def add_knowledge(
        self,
        user_id: str,
        item: KnowledgeInput,
    ) -> SemanticKnowledge | None:
        """Inject knowledge directly.

        The statement is embedded, optionally deduplicated, and indexed immediately.
        Injected knowledge is assigned importance_score=1.0 (maximum).

        Returns the created entry, or None if deduplicated.
        """
        return await add_knowledge(self._lifecycle, user_id, item)

    async def add_knowledge_batch(
        self,
        user_id: str,
        items: list[KnowledgeInput],
    ) -> list[SemanticKnowledge]:
        """Batch inject multiple knowledge entries with batch embedding.

        Each injected entry is assigned importance_score=1.0 (maximum).

        Args:
            user_id: User this knowledge belongs to
            items: List of knowledge entries (statement, valid_at, invalid_at)

        Returns:
            List of created entries (excludes duplicates).
        """
        return await add_knowledge_batch(self._lifecycle, user_id, items)
