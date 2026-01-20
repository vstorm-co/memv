import asyncio
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.models import Episode, Message, MessageRole, ProcessStatus, ProcessTask, RetrievalResult, SemanticKnowledge
from agent_memory.processing import BoundaryDetector, EpisodeGenerator, PredictCalibrateExtractor
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

        results = await memory.retrieve("query")
        print(results.to_prompt())  # Formatted for LLM context

        await memory.close()
    """

    def __init__(
        self,
        db_path: str,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient | None = None,  # Required for extraction, optional for retrieval-only
        embedding_dimensions: int = 1536,  # OpenAI text-embedding-3-small default
    ):
        self.db_path = db_path
        self.embedder = embedding_client
        self.llm = llm_client
        self.dimensions = embedding_dimensions

        # Ensure parent directory exists
        db_dir = Path(db_path).parent
        if db_dir != Path("."):
            db_dir.mkdir(parents=True, exist_ok=True)

        # Stores
        self._messages = MessageStore(db_path)
        self._episodes = EpisodeStore(db_path)
        self._knowledge = KnowledgeStore(db_path)

        # Knowledge indices
        self._vector_index = VectorIndex(db_path, dimensions=embedding_dimensions, name="knowledge")
        self._text_index = TextIndex(db_path, name="knowledge")

        # Episode indices
        self._episode_vector_index = VectorIndex(db_path, dimensions=embedding_dimensions, name="episode")
        self._episode_text_index = TextIndex(db_path, name="episode")

        self._retriever: Retriever | None = None
        self._boundary_detector: BoundaryDetector | None = None
        self._episode_generator: EpisodeGenerator | None = None
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

        self._retriever = Retriever(
            knowledge_store=self._knowledge,
            episode_store=self._episodes,
            vector_index=self._vector_index,
            text_index=self._text_index,
            episode_vector_index=self._episode_vector_index,
            episode_text_index=self._episode_text_index,
            embedding_client=self.embedder,
        )

        if self.llm is not None:
            self._boundary_detector = BoundaryDetector(self.llm)
            self._episode_generator = EpisodeGenerator(self.llm)
            self._extractor = PredictCalibrateExtractor(self.llm)

        self._is_open = True

    async def close(self) -> None:
        """Close all database connections."""
        if not self._is_open:
            return

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

        return user_msg, assistant_msg

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        include_episodes: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge and episodes for a query.

        Args:
            query: Search query
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

        # Get unprocessed messages
        all_messages = await self._messages.get_by_user(user_id)
        processed_ids = await self._get_processed_message_ids(user_id)
        unprocessed = [m for m in all_messages if m.id not in processed_ids]

        if not unprocessed:
            return 0

        # Segment into episodes
        episodes_messages = await self._segment_messages(unprocessed)

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

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("Memory not open. Call await memory.open() first.")

    async def _get_processed_message_ids(self, user_id: str) -> set:
        """Get IDs of messages already assigned to episodes."""
        episodes = await self._episodes.get_by_user(user_id)
        processed = set()
        for ep in episodes:
            processed.update(ep.message_ids)
        return processed

    async def _segment_messages(self, messages: list[Message]) -> list[list[Message]]:
        """
        Segment messages into episode-sized chunks using boundary detection.

        Walks through messages and uses BoundaryDetector to find semantic
        boundaries (topic shifts, intent changes, etc.).
        """
        if not messages or not self._boundary_detector:
            return [messages] if messages else []

        episodes = []
        buffer: list[Message] = []

        for message in messages:
            should_split, signal = await self._boundary_detector.should_segment(message, buffer)

            if should_split and buffer:
                episodes.append(buffer)
                buffer = []

            buffer.append(message)

        # Don't forget the last buffer
        if buffer:
            episodes.append(buffer)

        return episodes

    async def _process_episode(self, messages: list[Message], user_id: str) -> int:
        """Process a single episode: generate, index, extract, store."""
        if not self._episode_generator or not self._extractor or not self._retriever:
            raise RuntimeError("Processing components not initialized")

        # 1. Generate episode
        episode = await self._episode_generator.generate(messages, user_id)
        await self._episodes.add(episode)

        # 2. Index episode for retrieval
        await self._index_episode(episode)

        # 3. Retrieve existing knowledge for predict-calibrate
        existing = await self._retriever.retrieve(
            query=f"{episode.title} {episode.narrative}",
            top_k=20,
            include_episodes=False,  # Only need knowledge for prediction
        )

        # 4. Extract novel knowledge
        extracted = await self._extractor.extract(
            episode=episode,
            raw_messages=messages,
            existing_knowledge=existing.retrieved_knowledge,
        )

        # 5. Convert to SemanticKnowledge and store with embeddings
        if not extracted:
            return 0

        # Batch embed all statements in one call
        statements = [item.statement for item in extracted]
        embeddings = await self.embedder.embed_batch(statements)

        for item, embedding in zip(extracted, embeddings, strict=True):
            knowledge = SemanticKnowledge(
                statement=item.statement,
                source_episode_id=episode.id,
                importance_score=item.confidence,
                embedding=embedding,
            )

            # Store in all indices
            await self._knowledge.add(knowledge)
            await self._vector_index.add(knowledge.id, embedding)
            await self._text_index.add(knowledge.id, knowledge.statement)

        return len(extracted)

    async def _index_episode(self, episode: Episode) -> None:
        """Index episode title and narrative for retrieval."""
        # Combine title and narrative for indexing
        text_content = f"{episode.title} {episode.narrative}"

        # Generate embedding
        embedding = await self.embedder.embed(text_content)

        # Index in both vector and text indices
        await self._episode_vector_index.add(episode.id, embedding)
        await self._episode_text_index.add(episode.id, text_content)
