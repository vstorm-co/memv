import asyncio

from fastembed import TextEmbedding

_MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastEmbedAdapter:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        if model not in _MODEL_DIMENSIONS:
            raise ValueError(f"Unknown model {model!r}. Known: {list(_MODEL_DIMENSIONS)}. Use a custom adapter for other models.")
        self.model_name = model
        self._model = TextEmbedding(model)
        self.dimensions = _MODEL_DIMENSIONS[model]

    async def embed(self, text: str) -> list[float]:
        embeddings = await asyncio.to_thread(lambda: list(self._model.embed([text])))
        return embeddings[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(lambda: list(self._model.embed(texts)))
        return [e.tolist() for e in embeddings]
