"""Hybrid retrieval combining vector and text search."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from agent_memory.models import Episode, RetrievalResult, SemanticKnowledge
from agent_memory.protocols import EmbeddingClient, EpisodeStore, KnowledgeStore
from agent_memory.storage import TextIndex, VectorIndex

if TYPE_CHECKING:
    from agent_memory.cache import EmbeddingCache


class Retriever:
    """
    Hybrid retriever combining vector similarity and text search.

    Searches both knowledge statements and episodes, returning
    unified results with RRF fusion.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        episode_store: EpisodeStore,
        vector_index: VectorIndex,
        text_index: TextIndex,
        episode_vector_index: VectorIndex | None = None,
        episode_text_index: TextIndex | None = None,
        embedding_client: EmbeddingClient | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ):
        self.knowledge = knowledge_store
        self.episodes = episode_store
        self.vector_index = vector_index
        self.text_index = text_index
        self.episode_vector_index = episode_vector_index
        self.episode_text_index = episode_text_index
        self.embedder = embedding_client
        self._embedding_cache = embedding_cache

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        include_episodes: bool = True,
        at_time: datetime | None = None,
        include_expired: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge and episodes for a query.

        Args:
            query: Search query text
            user_id: Filter results to this user only (required for privacy)
            top_k: Number of results to return per category
            vector_weight: Weight for vector vs text (0-1, where 0.5 is balanced)
            include_episodes: Whether to search and return episodes
            at_time: If provided, filter knowledge by validity at this event time
            include_expired: If True, include superseded (expired) records

        Returns:
            RetrievalResult containing knowledge statements and episodes
        """
        if self.embedder is None:
            raise RuntimeError("Embedding client required for retrieval")

        # 1. Embed query (with caching)
        query_embedding = None
        if self._embedding_cache is not None:
            query_embedding = self._embedding_cache.get(query)

        if query_embedding is None:
            query_embedding = await self.embedder.embed(query)
            if self._embedding_cache is not None:
                self._embedding_cache.set(query, query_embedding)

        # 2. Search knowledge (filtered by user_id)
        knowledge = await self._search_knowledge(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            vector_weight=vector_weight,
            user_id=user_id,
            at_time=at_time,
            include_expired=include_expired,
        )

        # 3. Search episodes (if enabled and indices exist)
        episodes: list[Episode] = []
        if include_episodes and self.episode_vector_index and self.episode_text_index:
            episodes = await self._search_episodes(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                vector_weight=vector_weight,
                user_id=user_id,
            )

        # 4. Also fetch episodes for the returned knowledge (for context)
        episode_ids_from_knowledge = {k.source_episode_id for k in knowledge}
        existing_episode_ids = {ep.id for ep in episodes}
        missing_episode_ids = episode_ids_from_knowledge - existing_episode_ids

        for ep_id in missing_episode_ids:
            ep = await self.episodes.get(ep_id)
            if ep:
                episodes.append(ep)

        return RetrievalResult(
            retrieved_knowledge=knowledge,
            retrieved_episodes=episodes,
        )

    async def _search_knowledge(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        vector_weight: float,
        user_id: str,
        at_time: datetime | None = None,
        include_expired: bool = False,
    ) -> list[SemanticKnowledge]:
        """Search knowledge using hybrid vector + text search, filtered by user_id."""
        # Vector search (filtered by user_id)
        vector_ids = await self.vector_index.search(query_embedding, top_k=top_k * 3, user_id=user_id)

        # Text search (BM25) (filtered by user_id)
        text_ids = await self.text_index.search(query, top_k=top_k * 3, user_id=user_id)

        # RRF fusion
        fused_ids = self._rrf_fusion(vector_ids, text_ids, vector_weight=vector_weight)

        # Fetch full objects (deduplicated) with temporal filtering
        knowledge = []
        seen = set()
        for kid in fused_ids:
            if kid in seen:
                continue
            if len(knowledge) >= top_k:
                break

            k = await self.knowledge.get(kid)
            if k:
                # Apply temporal filtering
                if not self._passes_temporal_filter(k, at_time, include_expired):
                    continue
                knowledge.append(k)
                seen.add(kid)

        return knowledge

    def _passes_temporal_filter(
        self,
        knowledge: SemanticKnowledge,
        at_time: datetime | None,
        include_expired: bool,
    ) -> bool:
        """Check if knowledge passes temporal filtering criteria."""
        # Filter by transaction time (expired_at)
        if not include_expired and not knowledge.is_current():
            return False

        # Filter by event time (valid_at/invalid_at)
        if at_time is not None and not knowledge.is_valid_at(at_time):
            return False

        return True

    async def _search_episodes(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        vector_weight: float,
        user_id: str,
    ) -> list[Episode]:
        """Search episodes using hybrid vector + text search, filtered by user_id."""
        if not self.episode_vector_index or not self.episode_text_index:
            return []

        # Vector search on episode embeddings (filtered by user_id)
        vector_ids = await self.episode_vector_index.search(query_embedding, top_k=top_k * 3, user_id=user_id)

        # Text search on episode title/content (filtered by user_id)
        text_ids = await self.episode_text_index.search(query, top_k=top_k * 3, user_id=user_id)

        # RRF fusion
        fused_ids = self._rrf_fusion(vector_ids, text_ids, vector_weight=vector_weight)

        # Fetch full objects
        episodes = []
        seen = set()
        for eid in fused_ids[:top_k]:
            if eid in seen:
                continue
            ep = await self.episodes.get(eid)
            if ep:
                episodes.append(ep)
                seen.add(eid)

        return episodes

    def _rrf_fusion(
        self,
        vector_ids: list[UUID],
        text_ids: list[UUID],
        vector_weight: float = 0.5,
        k: int = 60,  # RRF constant
    ) -> list[UUID]:
        """
        Reciprocal Rank Fusion.

        RRF score = vector_weight * (1/(k + rank_vector)) +
                    (1 - vector_weight) * (1/(k + rank_text))

        k=60 is standard from literature.
        """
        scores: dict[UUID, float] = {}

        # Vector contributions
        for rank, uid in enumerate(vector_ids):
            scores[uid] = scores.get(uid, 0.0) + vector_weight * (1 / (k + rank + 1))

        # Text contributions
        text_weight = 1.0 - vector_weight
        for rank, uid in enumerate(text_ids):
            scores[uid] = scores.get(uid, 0.0) + text_weight * (1 / (k + rank + 1))

        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda uid: scores[uid], reverse=True)
        return sorted_ids
