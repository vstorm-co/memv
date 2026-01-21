from __future__ import annotations

import asyncio
import warnings
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.cache import EmbeddingCache
from agent_memory.config import MemoryConfig
from agent_memory.models import (
    Episode,
    ExtractedKnowledge,
    Message,
    MessageRole,
    ProcessStatus,
    ProcessTask,
    RetrievalResult,
    SemanticKnowledge,
)
from agent_memory.processing import BatchSegmenter, BoundaryDetector, EpisodeGenerator, EpisodeMerger, PredictCalibrateExtractor
from agent_memory.protocols import EmbeddingClient, LLMClient
from agent_memory.retrieval import Retriever
from agent_memory.storage.sqlite import (
    EpisodeStore,
    KnowledgeStore,
    MessageStore,
    TextIndex,
    VectorIndex,
)


class Memory:
    """
    Main entry point for agent_memory.

    Usage:
        memory = Memory(db_path="memory.db", embedding_client=embedder, llm_client=llm)
        await memory.open()

        await memory.add_message(Message(...))
        await memory.process(user_id="user123")  # Extract knowledge

        results = await memory.retrieve("query", user_id="user123")
        print(results.to_prompt())  # Formatted for LLM context

        await memory.close()

    Auto-processing (Nemori-style):
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
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedding_client: EmbeddingClient | None = None,
        llm_client: LLMClient | None = None,  # Required for extraction, optional for retrieval-only
        embedding_dimensions: int | None = None,
        # Config object (alternative to individual params)
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
        # Use config or defaults
        cfg = config or MemoryConfig()

        # Determine database path from argument or config
        if db_path is None and embedding_client is None:
            # Using config for both
            self.db_path = cfg.db_path
        elif db_path is not None:
            self.db_path = db_path
        else:
            self.db_path = cfg.db_path

        if embedding_client is None:
            raise ValueError("embedding_client is required")
        self.embedder = embedding_client
        self.llm = llm_client

        # Use param if provided, else use config value
        self.dimensions = embedding_dimensions if embedding_dimensions is not None else cfg.embedding_dimensions

        # Auto-processing config
        self.auto_process = auto_process if auto_process is not None else cfg.auto_process
        self.batch_threshold = batch_threshold if batch_threshold is not None else cfg.batch_threshold
        self.max_retries = max_retries if max_retries is not None else cfg.max_retries

        # Segmentation config
        self.segmentation_threshold = segmentation_threshold if segmentation_threshold is not None else cfg.segmentation_threshold
        self.time_gap_minutes = time_gap_minutes if time_gap_minutes is not None else cfg.time_gap_minutes
        self.use_legacy_segmentation = use_legacy_segmentation if use_legacy_segmentation is not None else cfg.use_legacy_segmentation

        # Episode merging config
        self.enable_episode_merging = enable_episode_merging if enable_episode_merging is not None else cfg.enable_episode_merging
        self.merge_similarity_threshold = (
            merge_similarity_threshold if merge_similarity_threshold is not None else cfg.merge_similarity_threshold
        )

        # Knowledge deduplication config
        self.enable_knowledge_dedup = enable_knowledge_dedup if enable_knowledge_dedup is not None else cfg.enable_knowledge_dedup
        self.knowledge_dedup_threshold = (
            knowledge_dedup_threshold if knowledge_dedup_threshold is not None else cfg.knowledge_dedup_threshold
        )

        # Prediction-calibrate config
        self.max_statements_for_prediction = (
            max_statements_for_prediction if max_statements_for_prediction is not None else cfg.max_statements_for_prediction
        )

        # Cache config
        self.enable_embedding_cache = enable_embedding_cache if enable_embedding_cache is not None else cfg.enable_embedding_cache
        self.embedding_cache_size = embedding_cache_size if embedding_cache_size is not None else cfg.embedding_cache_size
        self.embedding_cache_ttl_seconds = (
            embedding_cache_ttl_seconds if embedding_cache_ttl_seconds is not None else cfg.embedding_cache_ttl_seconds
        )

        # Per-user message buffers (tracks count since last processing)
        self._buffers: dict[str, int] = {}
        # Per-user processing tasks
        self._processing_tasks: dict[str, ProcessTask] = {}

        # Ensure parent directory exists
        db_dir = Path(self.db_path).parent
        if db_dir != Path("."):
            db_dir.mkdir(parents=True, exist_ok=True)

        # Stores
        self._messages = MessageStore(self.db_path)
        self._episodes = EpisodeStore(self.db_path)
        self._knowledge = KnowledgeStore(self.db_path)

        # Knowledge indices
        self._vector_index = VectorIndex(self.db_path, dimensions=self.dimensions, name="knowledge")
        self._text_index = TextIndex(self.db_path, name="knowledge")

        # Episode indices
        self._episode_vector_index = VectorIndex(self.db_path, dimensions=self.dimensions, name="episode")
        self._episode_text_index = TextIndex(self.db_path, name="episode")

        self._retriever: Retriever | None = None
        self._segmenter: BatchSegmenter | None = None
        self._legacy_boundary_detector: BoundaryDetector | None = None  # Deprecated
        self._episode_generator: EpisodeGenerator | None = None
        self._episode_merger: EpisodeMerger | None = None
        self._extractor: PredictCalibrateExtractor | None = None

        # State
        self._is_open = False

    async def open(self) -> None:
        """Open all database connections and initialize components."""
        if self._is_open:
            return

        await self._messages.open()
        await self._episodes.open()
        await self._knowledge.open()
        await self._vector_index.open()
        await self._text_index.open()
        await self._episode_vector_index.open()
        await self._episode_text_index.open()

        # Create embedding cache if enabled
        embedding_cache = None
        if self.enable_embedding_cache:
            embedding_cache = EmbeddingCache(
                max_size=self.embedding_cache_size,
                ttl_seconds=self.embedding_cache_ttl_seconds,
            )

        self._retriever = Retriever(
            knowledge_store=self._knowledge,
            episode_store=self._episodes,
            vector_index=self._vector_index,
            text_index=self._text_index,
            episode_vector_index=self._episode_vector_index,
            episode_text_index=self._episode_text_index,
            embedding_client=self.embedder,
            embedding_cache=embedding_cache,
        )

        if self.llm is not None:
            if self.use_legacy_segmentation:
                warnings.warn(
                    "BoundaryDetector is deprecated and will be removed in v0.2.0. Use BatchSegmenter instead (default).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._legacy_boundary_detector = BoundaryDetector(self.llm)
            else:
                self._segmenter = BatchSegmenter(
                    llm_client=self.llm,
                    batch_threshold=self.segmentation_threshold,
                    time_gap_minutes=self.time_gap_minutes,
                )
            self._episode_generator = EpisodeGenerator(self.llm)
            if self.enable_episode_merging:
                self._episode_merger = EpisodeMerger(
                    llm_client=self.llm,
                    embedding_client=self.embedder,
                    similarity_threshold=self.merge_similarity_threshold,
                )
            self._extractor = PredictCalibrateExtractor(self.llm)

        # Recover buffer state from DB (for auto-processing after restart)
        if self.auto_process:
            await self._recover_buffer_state()

        self._is_open = True

    async def close(self, cancel_pending: bool = True) -> None:
        """
        Close all database connections.

        Args:
            cancel_pending: If True, cancel any running processing tasks.
                           If False, wait for them to complete first.
        """
        if not self._is_open:
            return

        # Handle pending processing tasks
        for _user_id, task in list(self._processing_tasks.items()):
            if not task.done:
                if cancel_pending and task._task is not None:
                    task._task.cancel()
                    try:
                        await task._task
                    except asyncio.CancelledError:
                        pass
                else:
                    await task.wait()

        self._processing_tasks.clear()
        self._buffers.clear()

        await self._messages.close()
        await self._episodes.close()
        await self._knowledge.close()
        await self._vector_index.close()
        await self._text_index.close()
        await self._episode_vector_index.close()
        await self._episode_text_index.close()

        self._is_open = False

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
        self._ensure_open()
        await self._messages.add(message)

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
        self._ensure_open()
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

        await self._messages.add(user_msg)
        await self._messages.add(assistant_msg)

        # Track buffered messages for auto-processing
        if self.auto_process and self.llm is not None:
            self._buffers[user_id] = self._buffers.get(user_id, 0) + 2

            if self._buffers[user_id] >= self.batch_threshold:
                self._schedule_processing(user_id)

        return user_msg, assistant_msg

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        include_episodes: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge and episodes for a query.

        Args:
            query: Search query
            user_id: Filter results to this user only (required for privacy)
            top_k: Number of results to return per category
            vector_weight: Balance between vector (1.0) and text (0.0) search
            include_episodes: Whether to search and return episodes

        Returns:
            RetrievalResult containing knowledge and episodes.
            Use result.to_prompt() to get formatted context for LLM.
        """
        self._ensure_open()
        if self._retriever is None:
            raise RuntimeError("Retriever not initialized")

        return await self._retriever.retrieve(
            query=query,
            user_id=user_id,
            top_k=top_k,
            vector_weight=vector_weight,
            include_episodes=include_episodes,
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
        return await self._process_internal(user_id)

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
        self._ensure_open()
        if self.llm is None:
            raise RuntimeError("LLM client required for processing. Pass llm_client to Memory().")

        task = ProcessTask(user_id=user_id, status=ProcessStatus.RUNNING)

        async def _run() -> int:
            try:
                result = await self._process_internal(user_id)
                task.knowledge_count = result
                task.status = ProcessStatus.COMPLETED
                return result
            except Exception as e:
                task.status = ProcessStatus.FAILED
                task.error = str(e)
                raise

        task._task = asyncio.create_task(_run())
        return task

    async def _process_internal(self, user_id: str) -> int:
        """Internal processing implementation."""
        self._ensure_open()
        if self.llm is None:
            raise RuntimeError("LLM client required for processing. Pass llm_client to Memory().")

        # Get unprocessed messages (those after the last processed episode)
        all_messages = await self._messages.get_by_user(user_id)
        episodes = await self._episodes.get_by_user(user_id)

        if episodes:
            # Find the latest end_time among all episodes
            latest_end = max(ep.end_time for ep in episodes)
            # Only process messages sent after the last episode
            unprocessed = [m for m in all_messages if m.sent_at > latest_end]
        else:
            # No episodes yet, process all messages
            unprocessed = all_messages

        if not unprocessed:
            return 0

        # Segment into episodes
        episodes_messages = await self._segment_messages(unprocessed)

        # Process episodes sequentially to ensure each sees prior extractions
        total_extracted = 0
        for messages in episodes_messages:
            count = await self._process_episode(messages, user_id)
            total_extracted += count

        return total_extracted

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
        self._ensure_open()
        if self.llm is None:
            raise RuntimeError("LLM client required for processing.")

        if not messages:
            return 0

        return await self._process_episode(messages, user_id)

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
        task = self._processing_tasks.get(user_id)
        if task is None or task.done:
            return task.knowledge_count if task else 0

        if timeout is not None:
            return await asyncio.wait_for(task.wait(), timeout)
        return await task.wait()

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
        self._ensure_open()
        if self.llm is None:
            raise RuntimeError("LLM client required for processing.")

        # Schedule processing (will handle checking for unprocessed messages)
        self._schedule_processing(user_id)

        # Wait for completion
        return await self.wait_for_processing(user_id)

    async def clear_user(self, user_id: str) -> dict[str, int]:
        """
        Delete all data for a user: messages, episodes, knowledge, and indices.

        This is a destructive operation. Use with caution.

        Args:
            user_id: User whose data to delete

        Returns:
            Dict with counts of deleted items per category
        """
        self._ensure_open()

        # Cancel any pending processing for this user
        existing_task = self._processing_tasks.get(user_id)
        if existing_task and not existing_task.done and existing_task._task:
            existing_task._task.cancel()
            try:
                await existing_task._task
            except asyncio.CancelledError:
                pass
        self._processing_tasks.pop(user_id, None)
        self._buffers.pop(user_id, None)

        # Get episode IDs for knowledge deletion
        episodes = await self._episodes.get_by_user(user_id)
        episode_ids = [ep.id for ep in episodes]

        # Delete in order: indices first, then stores
        counts = {}

        # Clear knowledge indices
        counts["knowledge_vectors"] = await self._vector_index.clear_user(user_id)
        counts["knowledge_text"] = await self._text_index.clear_user(user_id)

        # Clear episode indices
        counts["episode_vectors"] = await self._episode_vector_index.clear_user(user_id)
        counts["episode_text"] = await self._episode_text_index.clear_user(user_id)

        # Clear knowledge (by episode IDs since knowledge doesn't have user_id)
        counts["knowledge"] = await self._knowledge.clear_by_episodes(episode_ids)

        # Clear episodes
        counts["episodes"] = await self._episodes.clear_user(user_id)

        # Clear messages
        counts["messages"] = await self._messages.clear_user(user_id)

        return counts

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("Memory not open. Call await memory.open() first.")

    def _schedule_processing(self, user_id: str) -> None:
        """Schedule background processing if not already running."""
        # Don't schedule if already processing
        existing = self._processing_tasks.get(user_id)
        if existing is not None and not existing.done:
            return

        # Start background task
        task = self.process_async(user_id)
        self._processing_tasks[user_id] = task

        # Reset buffer (messages now being processed)
        self._buffers[user_id] = 0

    async def _recover_buffer_state(self) -> None:
        """Recover buffer counts from DB state after restart."""
        users = await self._messages.list_users()
        for user_id in users:
            all_messages = await self._messages.get_by_user(user_id)
            episodes = await self._episodes.get_by_user(user_id)

            if episodes:
                latest_end = max(ep.end_time for ep in episodes)
                unprocessed_count = sum(1 for m in all_messages if m.sent_at > latest_end)
            else:
                unprocessed_count = len(all_messages)

            if unprocessed_count > 0:
                self._buffers[user_id] = unprocessed_count

    async def _segment_messages(self, messages: list[Message]) -> list[list[Message]]:
        """
        Segment messages into episode-sized chunks.

        Uses BatchSegmenter by default (handles interleaved topics, time gaps).
        Falls back to legacy BoundaryDetector if use_legacy_segmentation=True.
        """
        if not messages:
            return []

        # Use new batch segmenter (default)
        if self._segmenter is not None:
            return await self._segmenter.segment(messages)

        # Legacy: incremental boundary detection (deprecated)
        if self._legacy_boundary_detector is not None:
            return await self._segment_messages_legacy(messages)

        # No segmenter available, return all as one episode
        return [messages]

    async def _segment_messages_legacy(self, messages: list[Message]) -> list[list[Message]]:
        """
        Legacy segmentation using BoundaryDetector (deprecated).

        Walks through messages and uses BoundaryDetector to find semantic
        boundaries (topic shifts, intent changes, etc.).
        """
        if not messages or not self._legacy_boundary_detector:
            return [messages] if messages else []

        episodes = []
        buffer: list[Message] = []

        for message in messages:
            should_split, _signal = await self._legacy_boundary_detector.should_segment(message, buffer)

            if should_split and buffer:
                episodes.append(buffer)
                buffer = []

            buffer.append(message)

        # Don't forget the last buffer
        if buffer:
            episodes.append(buffer)

        return episodes

    def _validate_extraction(self, item: ExtractedKnowledge) -> bool:
        """
        Filter low-confidence extractions.

        Primary filtering happens at the prompt level (SOURCE RULE).
        This is a safety net for low-confidence items.
        """
        if item.confidence < 0.7:
            return False

        return True

    async def _is_duplicate_knowledge(self, embedding: list[float], user_id: str) -> tuple[bool, float]:
        """
        Check if knowledge with this embedding already exists for this user.

        Uses vector similarity search to find near-duplicates within the user's knowledge.
        Returns (is_duplicate, similarity_score).
        """
        # Search for similar existing knowledge with scores (filtered by user_id)
        similar = await self._vector_index.search_with_scores(embedding, top_k=1, user_id=user_id)

        if not similar:
            return False, 0.0

        # Check if top result exceeds threshold
        _top_id, top_score = similar[0]
        return top_score >= self.knowledge_dedup_threshold, top_score

    async def _process_episode(self, messages: list[Message], user_id: str) -> int:
        """Process a single episode: generate, merge if needed, index, extract, store."""
        if not self._episode_generator or not self._extractor or not self._retriever:
            raise RuntimeError("Processing components not initialized")

        # 1. Generate episode
        episode = await self._episode_generator.generate(messages, user_id)

        # 2. Check for episode merging
        merged_with: Episode | None = None
        if self._episode_merger is not None:
            existing_episodes = await self._episodes.get_by_user(user_id)
            episode, merged_with = await self._episode_merger.merge_if_appropriate(episode, existing_episodes)

            if merged_with is not None:
                # Delete the old episode and its indices
                await self._episodes.delete(merged_with.id)
                # Note: We don't delete from vector/text indices as they don't support deletion
                # The old entries will become orphaned but won't affect results significantly

        # 3. Store episode (new or merged)
        await self._episodes.add(episode)

        # 4. Index episode for retrieval
        await self._index_episode(episode)

        # 5. Retrieve existing knowledge for predict-calibrate
        existing = await self._retriever.retrieve(
            query=f"{episode.title} {episode.content}",
            user_id=user_id,
            top_k=self.max_statements_for_prediction,
            include_episodes=False,  # Only need knowledge for prediction
        )

        # 6. Extract novel knowledge (episode contains original_messages)
        extracted = await self._extractor.extract(
            episode=episode,
            existing_knowledge=existing.retrieved_knowledge,
        )

        # 7. Filter out disguised general knowledge and low-quality extractions
        extracted = [item for item in extracted if self._validate_extraction(item)]

        # 8. Convert to SemanticKnowledge and store with embeddings
        if not extracted:
            return 0

        # Batch embed all statements in one call
        statements = [item.statement for item in extracted]
        embeddings = await self.embedder.embed_batch(statements)

        # 9. Process each extracted item (handle contradictions, dedup, store)
        stored_count = 0
        for item, embedding in zip(extracted, embeddings, strict=True):
            # Handle contradictions: invalidate conflicting existing knowledge
            if item.knowledge_type == "contradiction":
                await self._invalidate_contradicted_knowledge(embedding, user_id)

            # Check for duplicates if enabled
            if self.enable_knowledge_dedup:
                is_duplicate, score = await self._is_duplicate_knowledge(embedding, user_id)
                if is_duplicate:
                    import logging

                    logging.getLogger(__name__).info(f"Skipping duplicate: '{item.statement[:50]}...' (score={score:.3f})")
                    continue

            knowledge = SemanticKnowledge(
                statement=item.statement,
                source_episode_id=episode.id,
                importance_score=item.confidence,
                embedding=embedding,
            )

            # Store in all indices (with user_id for isolation)
            await self._knowledge.add(knowledge)
            await self._vector_index.add(knowledge.id, embedding, user_id)
            await self._text_index.add(knowledge.id, knowledge.statement, user_id)
            stored_count += 1

        return stored_count

    async def _invalidate_contradicted_knowledge(self, new_embedding: list[float], user_id: str) -> None:
        """
        Find and invalidate existing knowledge that contradicts the new statement.

        Uses semantic similarity to find the most similar existing knowledge
        and marks it as expired (superseded by the new contradicting knowledge).
        """
        # Find similar existing knowledge
        similar = await self._vector_index.search_with_scores(new_embedding, top_k=1, user_id=user_id)

        if not similar:
            return

        top_id, top_score = similar[0]

        # Only invalidate if similarity is high enough (same topic, different facts)
        # Use a moderate threshold - contradictions should be semantically related
        contradiction_threshold = 0.7
        if top_score >= contradiction_threshold:
            await self._knowledge.invalidate(top_id)

    async def _index_episode(self, episode: Episode) -> None:
        """Index episode title and content for retrieval."""
        # Combine title and content for indexing
        text_content = f"{episode.title} {episode.content}"

        # Generate embedding
        embedding = await self.embedder.embed(text_content)

        # Index in both vector and text indices (with user_id for isolation)
        await self._episode_vector_index.add(episode.id, embedding, episode.user_id)
        await self._episode_text_index.add(episode.id, text_content, episode.user_id)
