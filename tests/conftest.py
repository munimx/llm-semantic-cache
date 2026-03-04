"""Shared test fixtures for llm-semantic-cache tests."""
import pytest

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
