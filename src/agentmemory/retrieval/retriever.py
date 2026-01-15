"""Hybrid retrieval combining vector and text search."""

from uuid import UUID

from agentmemory.models import RetrievalResult, SemanticKnowledge
from agentmemory.protocols import EmbeddingClient, KnowledgeStore
from agentmemory.storage.sqlite import TextIndex, VectorIndex


class Retriever:
    """
    Hybrid retriever combining vector similarity and text search.

    For v0.1, uses simple union of results. Phase 2 will add RRF fusion.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        vector_index: VectorIndex,
        text_index: TextIndex,
        embedding_client: EmbeddingClient,
    ):
        self.knowledge = knowledge_store
        self.vector_index = vector_index
        self.text_index = text_index
        self.embedder = embedding_client

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query: Search query text
            top_k: Number of results to return
            vector_weight: Weight for vector vs text (0-1, where 0.5 is balanced)

        Returns:
            RetrievalResult containing retrieved knowledge statements
        """
        # 1. Embed query
        query_embedding = await self.embedder.embed(query)

        # 2. Vector search
        vector_ids = await self.vector_index.search(query_embedding, top_k=top_k * 3)

        # 3. Text search (BM25)
        text_ids = await self.text_index.search(query, top_k=top_k * 3)

        # 4. RRF fusion
        fused_ids = self._rrf_fusion(vector_ids, text_ids, vector_weight=vector_weight)

        # 5. Fetch full objects (deduplicated)
        knowledge = []
        seen = set()
        for kid in fused_ids[:top_k]:
            if kid in seen:
                continue
            k = await self.knowledge.get(kid)
            if k:
                knowledge.append(k)
                seen.add(kid)

        return RetrievalResult(retrieved_knowledge=knowledge)

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
