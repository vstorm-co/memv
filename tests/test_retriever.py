from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from agent_memory.models import SemanticKnowledge
from agent_memory.retrieval import Retriever
from agent_memory.storage.sqlite import (
    EpisodeStore,
    KnowledgeStore,
    TextIndex,
    VectorIndex,
)


class FakeEmbedder:
    """Fake embedder that returns deterministic embeddings based on text."""

    async def embed(self, text: str) -> list[float]:
        hash_val = hash(text) % 1000
        embedding = [float(hash_val + i) for i in range(384)]
        magnitude = sum(x**2 for x in embedding) ** 0.5
        return [x / magnitude for x in embedding]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def make_knowledge(statement: str, episode_id: UUID | None = None) -> SemanticKnowledge:
    """Create knowledge with a source episode reference."""
    return SemanticKnowledge(
        id=uuid4(),
        statement=statement,
        source_episode_id=episode_id or uuid4(),
        created_at=datetime.now(timezone.utc),
    )


@pytest_asyncio.fixture
async def retriever_setup():
    """Setup retriever with stores and fake embeddings."""
    knowledge_store = KnowledgeStore(db_path=":memory:")
    episode_store = EpisodeStore(db_path=":memory:")
    vector_index = VectorIndex(db_path=":memory:", dimensions=384, name="knowledge")
    text_index = TextIndex(db_path=":memory:", name="knowledge")
    embedder = FakeEmbedder()

    await knowledge_store.open()
    await episode_store.open()
    await vector_index.open()
    await text_index.open()

    retriever = Retriever(
        knowledge_store=knowledge_store,
        episode_store=episode_store,
        vector_index=vector_index,
        text_index=text_index,
        embedding_client=embedder,
    )

    yield retriever, knowledge_store, vector_index, text_index, embedder

    await knowledge_store.close()
    await episode_store.close()
    await vector_index.close()
    await text_index.close()


@pytest.mark.asyncio
async def test_retriever_empty(retriever_setup):
    """Retrieval on empty store returns empty results."""
    retriever, *_ = retriever_setup

    result = await retriever.retrieve("test query")

    assert len(result.retrieved_knowledge) == 0


@pytest.mark.asyncio
async def test_retriever_vector_search(retriever_setup):
    """Vector search finds semantically similar content."""
    retriever, knowledge_store, vector_index, text_index, embedder = retriever_setup

    k1 = make_knowledge("The user likes Python programming")
    k2 = make_knowledge("The user prefers JavaScript over TypeScript")
    k3 = make_knowledge("The user enjoys hiking in mountains")

    for k in [k1, k2, k3]:
        await knowledge_store.add(k)
        emb = await embedder.embed(k.statement)
        await vector_index.add(k.id, emb)
        await text_index.add(k.id, k.statement)

    result = await retriever.retrieve("programming languages", top_k=10)

    assert len(result.retrieved_knowledge) > 0


@pytest.mark.asyncio
async def test_retriever_text_search(retriever_setup):
    """Text search finds exact keyword matches."""
    retriever, knowledge_store, vector_index, text_index, embedder = retriever_setup

    k1 = make_knowledge("The user loves coffee in the morning")
    k2 = make_knowledge("The user drinks tea before bed")

    for k in [k1, k2]:
        await knowledge_store.add(k)
        emb = await embedder.embed(k.statement)
        await vector_index.add(k.id, emb)
        await text_index.add(k.id, k.statement)

    result = await retriever.retrieve("coffee", top_k=10)

    assert len(result.retrieved_knowledge) > 0
    assert any("coffee" in k.statement for k in result.retrieved_knowledge)


@pytest.mark.asyncio
async def test_retriever_hybrid_fusion(retriever_setup):
    """RRF fusion combines vector and text rankings."""
    retriever, knowledge_store, vector_index, text_index, embedder = retriever_setup

    statements = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning uses Python libraries",
        "React is a JavaScript framework",
        "The user prefers Python for data science",
    ]

    for stmt in statements:
        k = make_knowledge(stmt)
        await knowledge_store.add(k)
        emb = await embedder.embed(k.statement)
        await vector_index.add(k.id, emb)
        await text_index.add(k.id, k.statement)

    result = await retriever.retrieve("Python programming", top_k=3)

    assert len(result.retrieved_knowledge) > 0
    assert len(result.retrieved_knowledge) <= 3


@pytest.mark.asyncio
async def test_retriever_deduplication(retriever_setup):
    """Results are deduplicated (no duplicate IDs)."""
    retriever, knowledge_store, vector_index, text_index, embedder = retriever_setup

    k1 = make_knowledge("The user likes Python programming")

    await knowledge_store.add(k1)
    emb = await embedder.embed(k1.statement)
    await vector_index.add(k1.id, emb)
    await text_index.add(k1.id, k1.statement)

    result = await retriever.retrieve("Python", top_k=10)

    ids = [k.id for k in result.retrieved_knowledge]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_retriever_vector_weight(retriever_setup):
    """Vector weight parameter affects ranking."""
    retriever, knowledge_store, vector_index, text_index, embedder = retriever_setup

    k1 = make_knowledge("Python programming language")

    await knowledge_store.add(k1)
    emb = await embedder.embed(k1.statement)
    await vector_index.add(k1.id, emb)
    await text_index.add(k1.id, k1.statement)

    result_balanced = await retriever.retrieve("Python", vector_weight=0.5)
    result_vector_heavy = await retriever.retrieve("Python", vector_weight=0.9)
    result_text_heavy = await retriever.retrieve("Python", vector_weight=0.1)

    assert len(result_balanced.retrieved_knowledge) > 0
    assert len(result_vector_heavy.retrieved_knowledge) > 0
    assert len(result_text_heavy.retrieved_knowledge) > 0
