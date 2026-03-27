import cohere

_MODEL_DIMENSIONS = {
    "embed-v4.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbedAdapter:
    def __init__(self, api_key: str | None = None, model: str = "embed-v4.0"):
        self.client = cohere.AsyncClientV2(api_key=api_key)
        self.model = model
        if model not in _MODEL_DIMENSIONS:
            raise ValueError(f"Unknown model {model!r}. Known: {list(_MODEL_DIMENSIONS)}. Use a custom adapter for other models.")
        self.dimensions = _MODEL_DIMENSIONS[model]

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embed(texts=[text], model=self.model, input_type="search_query", embedding_types=["float"])
        return response.embeddings.float_[0]  # type: ignore[index]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embed(texts=texts, model=self.model, input_type="search_document", embedding_types=["float"])
        return response.embeddings.float_  # type: ignore[return-value]
