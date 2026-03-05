"""Tests for SemanticCache wrap(), hit/miss paths, and behavioral invariants."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from llm_semantic_cache.cache import SemanticCache
from llm_semantic_cache.config import CacheConfig
from llm_semantic_cache.storage.base import CacheEntry
from llm_semantic_cache.storage.memory import InMemoryStorage

MESSAGES = [{"role": "user", "content": "What is Python?"}]
RESPONSE = {"id": "resp-1", "choices": [{"message": {"content": "A language."}}]}


def make_cache(fake_embedder: Any, timeout: float = 1.0) -> SemanticCache:
    return SemanticCache(
        storage=InMemoryStorage(),
        config=CacheConfig(threshold=0.85, cache_timeout_seconds=timeout),
        embedder=fake_embedder,
    )


def test_sync_cache_miss_calls_original_function(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    response = wrapped(messages=MESSAGES, cache_context={})

    assert response == RESPONSE
    assert calls["count"] == 1


def test_embedding_called_once_on_miss(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    embed_calls = {"count": 0}
    original_embed = fake_embedder.embed

    def counted_embed(text: str) -> list[float]:
        embed_calls["count"] += 1
        return original_embed(text)

    fake_embedder.embed = counted_embed  # type: ignore[method-assign]

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=MESSAGES, cache_context={})

    assert embed_calls["count"] == 1


def test_sync_cache_hit_does_not_call_original_function(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    first = wrapped(messages=MESSAGES, cache_context={})
    second = wrapped(messages=MESSAGES, cache_context={})

    assert first == RESPONSE
    assert second == RESPONSE
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_async_cache_miss_calls_original_function(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    async def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    response = await wrapped(messages=MESSAGES, cache_context={})

    assert response == RESPONSE
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_async_cache_hit_returns_cached_response(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    async def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    first = await wrapped(messages=MESSAGES, cache_context={})
    second = await wrapped(messages=MESSAGES, cache_context={})

    assert first == RESPONSE
    assert second == RESPONSE
    assert calls["count"] == 1


def test_cache_context_required_raises_value_error(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create)
    with pytest.raises(ValueError):
        wrapped(messages=MESSAGES)


def test_cache_context_empty_dict_is_valid(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create)
    assert wrapped(messages=MESSAGES, cache_context={}) == RESPONSE


def test_bad_cache_context_type_fails_open(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    response = wrapped(messages=MESSAGES, cache_context={"key": object()})

    assert response == RESPONSE
    assert calls["count"] == 1


def test_stream_true_bypasses_cache_sync(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=MESSAGES, cache_context={}, stream=True)
    wrapped(messages=MESSAGES, cache_context={}, stream=True)

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_stream_true_bypasses_cache_async(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    async def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    await wrapped(messages=MESSAGES, cache_context={}, stream=True)
    await wrapped(messages=MESSAGES, cache_context={}, stream=True)

    assert calls["count"] == 2


def test_no_user_message_bypasses_cache(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}
    messages = [{"role": "system", "content": "You are helpful"}]

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=messages, cache_context={})
    wrapped(messages=messages, cache_context={})

    assert calls["count"] == 2


def test_different_context_hash_is_cache_miss(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=MESSAGES, cache_context={"user_id": "a"})
    wrapped(messages=MESSAGES, cache_context={"user_id": "b"})

    assert calls["count"] == 2


def test_different_namespace_is_cache_miss(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)
    calls = {"count": 0}

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=MESSAGES, cache_context={}, cache_namespace="one")
    wrapped(messages=MESSAGES, cache_context={}, cache_namespace="two")

    assert calls["count"] == 2


def test_invalidate_namespace_clears_entries(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=MESSAGES, cache_context={}, cache_namespace="team-a")

    deleted = cache.invalidate_namespace("team-a")
    assert deleted == 1


def test_sync_lookup_failure_fails_open(fake_embedder: Any) -> None:
    class FailingSearchStorage(InMemoryStorage):
        def search(
            self,
            embedding: list[float],
            namespace: str,
            embedding_model_id: str,
            context_hash: str,
            threshold: float,
        ) -> CacheEntry | None:
            raise RuntimeError("search failed")

    calls = {"count": 0}
    cache = SemanticCache(
        storage=FailingSearchStorage(),
        config=CacheConfig(threshold=0.85, cache_timeout_seconds=1.0),
        embedder=fake_embedder,
    )

    def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    response = wrapped(messages=MESSAGES, cache_context={})

    assert response == RESPONSE
    assert calls["count"] == 1


def test_sync_store_failure_does_not_raise(fake_embedder: Any) -> None:
    class FailingStoreStorage(InMemoryStorage):
        def store(self, entry: CacheEntry) -> None:
            raise RuntimeError("store failed")

    cache = SemanticCache(
        storage=FailingStoreStorage(),
        config=CacheConfig(threshold=0.85, cache_timeout_seconds=1.0),
        embedder=fake_embedder,
    )

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create)
    assert wrapped(messages=MESSAGES, cache_context={}) == RESPONSE


def test_namespace_size_warning_logged(
    fake_embedder: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LargeNamespaceStorage(InMemoryStorage):
        def namespace_size(self, namespace: str) -> int:
            return 5001

    warnings: list[tuple[str, dict[str, Any]]] = []

    def capture_warning(event: str, **kwargs: Any) -> None:
        warnings.append((event, kwargs))

    monkeypatch.setattr("llm_semantic_cache.cache.log.warning", capture_warning)

    cache = SemanticCache(
        storage=LargeNamespaceStorage(),
        config=CacheConfig(threshold=0.85, cache_timeout_seconds=1.0),
        embedder=fake_embedder,
    )

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create)
    wrapped(messages=MESSAGES, cache_context={}, cache_namespace="ns")

    assert warnings
    assert warnings[0][0] == "cache.namespace_too_large"
    assert warnings[0][1]["namespace"] == "ns"
    assert warnings[0][1]["size"] == 5001


@pytest.mark.asyncio
async def test_async_lookup_timeout_fails_open(fake_embedder: Any) -> None:
    class HangingSearchStorage(InMemoryStorage):
        async def asearch(
            self,
            embedding: list[float],
            namespace: str,
            embedding_model_id: str,
            context_hash: str,
            threshold: float,
        ) -> CacheEntry | None:
            await asyncio.sleep(0.05)
            return None

    calls = {"count": 0}
    cache = SemanticCache(
        storage=HangingSearchStorage(),
        config=CacheConfig(threshold=0.85, cache_timeout_seconds=0.01),
        embedder=fake_embedder,
    )

    async def create(**_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return RESPONSE

    wrapped = cache.wrap(create)
    response = await wrapped(messages=MESSAGES, cache_context={})

    assert response == RESPONSE
    assert calls["count"] == 1


def test_wrap_auto_detects_async_function(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    async def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create, mode="auto")
    assert inspect.iscoroutinefunction(wrapped)


def test_wrap_auto_detects_sync_function(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create, mode="auto")
    assert not inspect.iscoroutinefunction(wrapped)


def test_wrap_mode_sync_forces_sync_wrapper(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    async def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create, mode="sync")
    assert not inspect.iscoroutinefunction(wrapped)


def test_wrap_mode_async_forces_async_wrapper(fake_embedder: Any) -> None:
    cache = make_cache(fake_embedder)

    def create(**_: Any) -> dict[str, Any]:
        return RESPONSE

    wrapped = cache.wrap(create, mode="async")
    assert inspect.iscoroutinefunction(wrapped)
