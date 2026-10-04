"""Device selection for embeddings. fastembed and onnxruntime are faked: no GPU or model download needed."""
import logging
from typing import ClassVar

import fastembed
import onnxruntime
import pytest

from retrieval import embeddings
from retrieval.embeddings import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    FastEmbedEmbedder,
    load_embedder,
)


class FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class FakeTextEmbedding:
    loads: ClassVar[list[list[str]]] = []
    # What the ONNX session ends up on when CUDA is requested (it can fall back to CPU by itself).
    cuda_session_providers: ClassVar[list[str]] = [CUDA_PROVIDER, CPU_PROVIDER]
    fail_on_cuda = False

    def __init__(self, model_name, cache_dir=None, providers=None):
        FakeTextEmbedding.loads.append(list(providers))
        if CUDA_PROVIDER in providers and FakeTextEmbedding.fail_on_cuda:
            raise RuntimeError("libcudnn.so.9: cannot open shared object file")
        active = FakeTextEmbedding.cuda_session_providers if CUDA_PROVIDER in providers else [CPU_PROVIDER]
        self.model = type("Inner", (), {"model": FakeSession(active)})()

    @staticmethod
    def list_supported_models():
        return [{"model": "fake/model", "dim": 4}]


@pytest.fixture
def fake_fastembed(monkeypatch):
    FakeTextEmbedding.loads = []
    FakeTextEmbedding.cuda_session_providers = [CUDA_PROVIDER, CPU_PROVIDER]
    FakeTextEmbedding.fail_on_cuda = False
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    def set_available(*providers):
        monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: list(providers))

    set_available(CPU_PROVIDER)
    return set_available


def make(device: str) -> FastEmbedEmbedder:
    return FastEmbedEmbedder("fake/model", device=device)


def test_cpu_is_pinned_even_when_cuda_exists(fake_fastembed):
    fake_fastembed(CUDA_PROVIDER, CPU_PROVIDER)
    embedder = make("cpu")
    assert embedder.device == "cpu" and embedder.fallback_reason is None
    assert FakeTextEmbedding.loads == [[CPU_PROVIDER]]  # never fastembed's "auto", which would pick the GPU


def test_cuda_when_available_and_active(fake_fastembed, caplog):
    fake_fastembed(CUDA_PROVIDER, CPU_PROVIDER)
    with caplog.at_level(logging.INFO, logger="patchpilot.embeddings"):
        embedder = make("cuda")
    assert embedder.device == "cuda"
    assert FakeTextEmbedding.loads == [[CUDA_PROVIDER, CPU_PROVIDER]]
    assert "Embedding device: cuda" in caplog.text
    assert embedder.describe() == "fake/model on cuda"


def test_cuda_without_gpu_build_falls_back_with_a_clear_reason(fake_fastembed, caplog):
    fake_fastembed(CPU_PROVIDER)  # plain fastembed / onnxruntime: no CUDA provider at all
    with caplog.at_level(logging.WARNING, logger="patchpilot.embeddings"):
        embedder = make("cuda")
    assert embedder.device == "cpu"
    assert FakeTextEmbedding.loads == [[CPU_PROVIDER]]
    assert "Embedding device: cpu (EMBEDDING_DEVICE=cuda requested; onnxruntime has no CUDA support" in caplog.text
    assert "fastembed-gpu" in caplog.text
    assert "fell back from cuda" in embedder.describe()


def test_cuda_provider_that_silently_lands_on_cpu_is_reported_as_cpu(fake_fastembed, caplog):
    fake_fastembed(CUDA_PROVIDER, CPU_PROVIDER)
    FakeTextEmbedding.cuda_session_providers = [CPU_PROVIDER]
    with caplog.at_level(logging.WARNING, logger="patchpilot.embeddings"):
        embedder = make("cuda")
    assert embedder.device == "cpu"
    assert "CUDA provider failed to load" in caplog.text
    assert FakeTextEmbedding.loads[-1] == [CPU_PROVIDER]


def test_cuda_load_error_falls_back_to_cpu(fake_fastembed, caplog):
    fake_fastembed(CUDA_PROVIDER, CPU_PROVIDER)
    FakeTextEmbedding.fail_on_cuda = True
    with caplog.at_level(logging.WARNING, logger="patchpilot.embeddings"):
        embedder = make("cuda")
    assert embedder.device == "cpu"
    assert "libcudnn.so.9" in caplog.text
    assert FakeTextEmbedding.loads == [[CUDA_PROVIDER, CPU_PROVIDER], [CPU_PROVIDER]]


def test_unknown_device_is_rejected(fake_fastembed):
    with pytest.raises(ValueError, match="EMBEDDING_DEVICE must be one of cpu, cuda"):
        make("gpu")


def test_setting_comes_from_env_and_defaults_to_cpu(fake_fastembed, monkeypatch):
    fake_fastembed(CUDA_PROVIDER, CPU_PROVIDER)
    monkeypatch.setattr(embeddings, "load_dotenv", lambda: None)
    monkeypatch.setenv("EMBEDDING_MODEL", "fake/model")
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    assert load_embedder().device == "cpu"
    monkeypatch.setenv("EMBEDDING_DEVICE", " CUDA ")
    assert load_embedder().device == "cuda"
