import logging
import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

logger = logging.getLogger("patchpilot.embeddings")

# Trained on code, runs locally through ONNX, so indexing costs nothing.
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
# fastembed defaults to the system temp dir, which macOS clears; keep the download somewhere stable.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "patchpilot" / "fastembed"
# fastembed's default batch of 256 long sequences needs tens of GB (attention grows with length squared); this
# keeps a batch of capped chunks to a few hundred MB.
BATCH_SIZE = 16
DEVICES = ("cpu", "cuda")
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _active_providers(model) -> list[str] | None:
    """The providers the ONNX session really runs on (it can fall back to CPU on its own when CUDA fails to load)."""
    session = getattr(getattr(model, "model", None), "model", None)
    get_providers = getattr(session, "get_providers", None)
    return list(get_providers()) if callable(get_providers) else None


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, cache_dir: Path = DEFAULT_CACHE_DIR,
                 device: str = "cpu"):
        import onnxruntime
        from fastembed import TextEmbedding

        if device not in DEVICES:
            raise ValueError(f"EMBEDDING_DEVICE must be one of {', '.join(DEVICES)}, not {device!r}")
        supported = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"fastembed does not support embedding model {model_name!r}")
        self.model_name = model_name
        self.dim = supported[model_name]
        self.requested_device = device

        def load(providers: list[str]):
            # Always pass providers explicitly: fastembed's default picks CUDA whenever it exists, so "cpu" would
            # silently mean GPU on a GPU machine.
            return TextEmbedding(model_name, cache_dir=str(cache_dir), providers=providers)

        fallback_reason = None
        if device == "cuda":
            available = onnxruntime.get_available_providers()
            if CUDA_PROVIDER not in available:
                fallback_reason = (
                    f"onnxruntime has no CUDA support (available: {', '.join(available)}); "
                    "install fastembed-gpu instead of fastembed, see docs/vm_runbook.md"
                )
            else:
                try:
                    self._model = load([CUDA_PROVIDER, CPU_PROVIDER])
                    active = _active_providers(self._model)
                    if active is not None and CUDA_PROVIDER not in active:
                        fallback_reason = f"the CUDA provider failed to load (session is on {', '.join(active)})"
                except Exception as e:  # noqa: BLE001 - any CUDA init failure falls back to CPU, with the reason logged
                    fallback_reason = f"loading on CUDA failed: {type(e).__name__}: {e}"

        if device == "cpu" or fallback_reason:
            self._model = load([CPU_PROVIDER])
        self.device = "cpu" if device == "cpu" or fallback_reason else "cuda"
        self.fallback_reason = fallback_reason
        if fallback_reason:
            logger.warning("Embedding device: cpu (EMBEDDING_DEVICE=cuda requested; %s)", fallback_reason)
        else:
            logger.info("Embedding device: %s (%s)", self.device, model_name)

    def describe(self) -> str:
        """One line for run output, so it is obvious which device did the work."""
        note = f", fell back from cuda: {self.fallback_reason}" if self.fallback_reason else ""
        return f"{self.model_name} on {self.device}{note}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=BATCH_SIZE)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


def load_embedder() -> FastEmbedEmbedder:
    load_dotenv()
    model = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    cache_dir = Path(os.getenv("EMBEDDING_CACHE_DIR") or DEFAULT_CACHE_DIR).expanduser()
    device = (os.getenv("EMBEDDING_DEVICE") or "cpu").strip().lower()
    return FastEmbedEmbedder(model, cache_dir, device)
