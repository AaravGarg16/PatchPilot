import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

# Trained on code, runs locally through ONNX, so indexing costs nothing.
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
# fastembed defaults to the system temp dir, which macOS clears; keep the download somewhere stable.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "patchpilot" / "fastembed"


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, cache_dir: Path = DEFAULT_CACHE_DIR):
        from fastembed import TextEmbedding

        supported = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"fastembed does not support embedding model {model_name!r}")
        self.model_name = model_name
        self.dim = supported[model_name]
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


def load_embedder() -> FastEmbedEmbedder:
    load_dotenv()
    model = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    cache_dir = Path(os.getenv("EMBEDDING_CACHE_DIR") or DEFAULT_CACHE_DIR).expanduser()
    return FastEmbedEmbedder(model, cache_dir)
