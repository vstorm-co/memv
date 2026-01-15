from agentmemory.models import RetrievalResult
from agentmemory.protocols import EmbeddingClient, KnowledgeStore
from agentmemory.storage.sqlite import VectorIndex


class Retriever:
    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        vector_index: VectorIndex,
        embedding_client: EmbeddingClient,
    ):
        self.knowledge = knowledge_store
        self.vector = vector_index
        self.embedder = embedding_client

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        recency_weight: float = 0.01,  # Exponential decay factor
    ) -> RetrievalResult:
        # 1. Embed query
        query_embedding = await self.embedder.embed(query)

        # 2. Vector search (get more than top_k for reranking)
        candidate_ids = await self.vector.search(query_embedding, top_k=top_k * 3)

        # 3. Fetch full objects
        candidates = []
        for kid in candidate_ids:
            k = await self.knowledge.get(kid)
            if k:
                candidates.append(k)

        # 4. Return top_k
        return RetrievalResult(retrieved_knowledge=candidates[:top_k])
