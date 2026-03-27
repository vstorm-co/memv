from openai import AsyncOpenAI

_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedAdapter:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        if model not in _MODEL_DIMENSIONS:
            raise ValueError(f"Unknown model {model!r}. Known: {list(_MODEL_DIMENSIONS)}. Use a custom adapter for other models.")
        self.dimensions = _MODEL_DIMENSIONS[model]

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
