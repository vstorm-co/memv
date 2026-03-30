from memv.embeddings.openai import OpenAIEmbedAdapter

_LAZY_IMPORTS = {
    "VoyageEmbedAdapter": "memv.embeddings.voyage",
    "CohereEmbedAdapter": "memv.embeddings.cohere",
    "FastEmbedAdapter": "memv.embeddings.fastembed",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'memv.embeddings' has no attribute {name!r}")


__all__ = ["OpenAIEmbedAdapter", "VoyageEmbedAdapter", "CohereEmbedAdapter", "FastEmbedAdapter"]
