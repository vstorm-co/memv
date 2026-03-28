from voyageai import AsyncClient

_MODEL_DIMENSIONS = {
    "voyage-3-lite": 1024,
    "voyage-3": 1024,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
}


class VoyageEmbedAdapter:
    def __init__(self, api_key: str | None = None, model: str = "voyage-3-lite"):
        self.client = AsyncClient(api_key=api_key)
        self.model = model
        if model not in _MODEL_DIMENSIONS:
            raise ValueError(f"Unknown model {model!r}. Known: {list(_MODEL_DIMENSIONS)}. Use a custom adapter for other models.")
        self.dimensions = _MODEL_DIMENSIONS[model]

    async def embed(self, text: str) -> list[float]:
        result = await self.client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]  # type: ignore[index]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self.client.embed(texts, model=self.model, input_type="document")
        return result.embeddings  # type: ignore[return-value]
