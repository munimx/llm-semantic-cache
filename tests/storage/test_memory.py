import time
from unittest.mock import patch

import pytest

from llm_semantic_cache.storage.memory import InMemoryStorage
from tests.conftest import make_entry


def test_store_and_retrieve_by_exact_match(memory_storage: InMemoryStorage) -> None:
    entry = make_entry(embedding=[1.0, 0.0, 0.0])
    memory_storage.store(entry)
    result = memory_storage.search(
        embedding=[1.0, 0.0, 0.0],
        namespace="test",
        embedding_model_id="test-model",
        context_hash="abc123",
        threshold=0.92,
    )
    assert result is not None
    assert result.entry is not None
    assert result.entry.id == entry.id
    assert isinstance(result.best_score, float)


def test_search_returns_none_when_namespace_empty(memory_storage: InMemoryStorage) -> None:
    result = memory_storage.search([1.0, 0.0, 0.0], "empty", "test-model", "abc123", 0.92)
    assert result is None


def test_search_returns_none_when_model_id_differs(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry(embedding=[1.0, 0.0, 0.0], embedding_model_id="model-a"))
    result = memory_storage.search([1.0, 0.0, 0.0], "test", "model-b", "abc123", 0.92)
    assert result is None


def test_search_returns_none_when_context_hash_differs(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry(embedding=[1.0, 0.0, 0.0], context_hash="hash-a"))
    result = memory_storage.search([1.0, 0.0, 0.0], "test", "test-model", "hash-b", 0.92)
    assert result is None


def test_search_returns_none_below_threshold(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry(embedding=[1.0, 0.0, 0.0]))
    result = memory_storage.search([0.0, 1.0, 0.0], "test", "test-model", "abc123", 0.5)
    assert result is not None
    assert result.entry is None
    assert isinstance(result.best_score, float)


def test_search_returns_best_match_above_threshold(memory_storage: InMemoryStorage) -> None:
    weaker = make_entry(embedding=[0.70710678, 0.70710678, 0.0], prompt_text="weaker")
    stronger = make_entry(embedding=[1.0, 0.0, 0.0], prompt_text="stronger")
    memory_storage.store(weaker)
    memory_storage.store(stronger)

    result = memory_storage.search([1.0, 0.0, 0.0], "test", "test-model", "abc123", 0.7)
    assert result is not None
    assert result.entry is not None
    assert result.entry.prompt_text == "stronger"


def test_search_ignores_expired_entries(memory_storage: InMemoryStorage) -> None:
    expired = make_entry(embedding=[1.0, 0.0, 0.0], ttl=0.01)
    memory_storage.store(expired)
    time.sleep(0.02)
    result = memory_storage.search([1.0, 0.0, 0.0], "test", "test-model", "abc123", 0.9)
    assert result is None


def test_search_evicts_expired_entries_from_store(memory_storage: InMemoryStorage) -> None:
    expired = make_entry(embedding=[1.0, 0.0, 0.0], ttl=0.001)
    memory_storage.store(expired)
    time.sleep(0.01)
    memory_storage.search([1.0, 0.0, 0.0], "test", "test-model", "abc123", 0.9)
    assert memory_storage.namespace_size("test") == 0


def test_invalidate_namespace_removes_all_entries(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry([1.0, 0.0, 0.0], namespace="ns"))
    memory_storage.store(make_entry([0.0, 1.0, 0.0], namespace="ns"))
    memory_storage.invalidate_namespace("ns")
    assert memory_storage.namespace_size("ns") == 0


def test_invalidate_namespace_returns_correct_count(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry([1.0, 0.0, 0.0], namespace="ns"))
    memory_storage.store(make_entry([0.0, 1.0, 0.0], namespace="ns"))
    deleted = memory_storage.invalidate_namespace("ns")
    assert deleted == 2


def test_invalidate_nonexistent_namespace_returns_zero(memory_storage: InMemoryStorage) -> None:
    assert memory_storage.invalidate_namespace("missing") == 0


def test_clear_removes_all_namespaces(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry([1.0, 0.0, 0.0], namespace="ns_a"))
    memory_storage.store(make_entry([0.0, 1.0, 0.0], namespace="ns_b"))
    memory_storage.clear()
    assert memory_storage.namespace_size("ns_a") == 0
    assert memory_storage.namespace_size("ns_b") == 0


def test_namespace_isolation(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry([1.0, 0.0, 0.0], namespace="ns_a"))
    result = memory_storage.search([1.0, 0.0, 0.0], "ns_b", "test-model", "abc123", 0.9)
    assert result is None


def test_store_multiple_entries_same_namespace(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry([1.0, 0.0, 0.0], namespace="ns"))
    memory_storage.store(make_entry([0.0, 1.0, 0.0], namespace="ns"))
    memory_storage.store(make_entry([0.0, 0.0, 1.0], namespace="ns"))
    assert memory_storage.namespace_size("ns") == 3


def test_namespace_size_returns_correct_count(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(make_entry([1.0, 0.0, 0.0], namespace="ns"))
    memory_storage.store(make_entry([0.0, 1.0, 0.0], namespace="ns"))
    assert memory_storage.namespace_size("ns") == 2


def test_embedding_model_id_filtering(memory_storage: InMemoryStorage) -> None:
    memory_storage.store(
        make_entry(
            [1.0, 0.0, 0.0],
            namespace="ns",
            context_hash="ctx",
            embedding_model_id="model-a",
        )
    )
    memory_storage.store(
        make_entry(
            [1.0, 0.0, 0.0],
            namespace="ns",
            context_hash="ctx",
            embedding_model_id="model-b",
        )
    )
    result = memory_storage.search([1.0, 0.0, 0.0], "ns", "model-b", "ctx", 0.9)
    assert result is not None
    assert result.entry is not None
    assert result.entry.embedding_model_id == "model-b"


@pytest.mark.asyncio
async def test_async_methods_do_not_use_thread_pool(memory_storage: InMemoryStorage) -> None:
    entry = make_entry(embedding=[1.0, 0.0, 0.0], namespace="ns", context_hash="ctx")
    with patch("asyncio.to_thread") as to_thread:
        await memory_storage.astore(entry)
        result = await memory_storage.asearch([1.0, 0.0, 0.0], "ns", "test-model", "ctx", 0.9)
    assert result is not None
    assert result.entry is not None
    to_thread.assert_not_called()
