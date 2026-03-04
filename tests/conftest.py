"""Shared test fixtures for llm-semantic-cache tests."""
import pytest

from llm_semantic_cache.embeddings import _l2_normalize
from llm_semantic_cache.storage.base import CacheEntry
from llm_semantic_cache.storage.memory import InMemoryStorage


def make_entry(
    embedding: list[float],
    prompt_text: str = "test prompt",
    context_hash: str = "abc123",
    namespace: str = "test",
    embedding_model_id: str = "test-model",
    response: dict | None = None,
    ttl: float | None = None,
) -> CacheEntry:
    return CacheEntry(
        embedding=embedding,
        prompt_text=prompt_text,
        context_hash=context_hash,
        namespace=namespace,
        embedding_model_id=embedding_model_id,
        response=response or {"content": "cached response"},
        ttl=ttl,
    )


@pytest.fixture
def memory_storage() -> InMemoryStorage:
    return InMemoryStorage()


class FakeEmbedder:
    """Deterministic embedder for testing — no ML dependencies."""

    def __init__(self, model_name: str = "fake-model") -> None:
        self._model_name = model_name

    @property
    def model_id(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        dim = 4
        raw = [float(len(text) % (i + 2)) for i in range(dim)]
        return _l2_normalize(raw)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
