"""Tests for RedisStorage using fakeredis (no real Redis required)."""
import asyncio

import fakeredis
import fakeredis.aioredis
import pytest
import pytest_asyncio

from llm_semantic_cache.storage.redis import RedisStorage
from tests.conftest import make_entry


@pytest_asyncio.fixture
async def redis_client():
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def redis_storage(redis_client):
    return RedisStorage(redis_client)


@pytest.mark.asyncio
async def test_astore_and_asearch_returns_match(redis_storage):
    entry = make_entry([1.0, 0.0])
    await redis_storage.astore(entry)

    match = await redis_storage.asearch([1.0, 0.0], "test", "test-model", "abc123", 0.9)

    assert match is not None
    assert match.entry is not None
    assert match.entry.id == entry.id


@pytest.mark.asyncio
async def test_asearch_returns_none_when_empty(redis_storage):
    match = await redis_storage.asearch([1.0, 0.0], "test", "test-model", "abc123", 0.9)

    assert match is None


@pytest.mark.asyncio
async def test_asearch_returns_none_when_model_id_differs(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0], embedding_model_id="other-model"))

    match = await redis_storage.asearch([1.0, 0.0], "test", "test-model", "abc123", 0.9)

    assert match is None


@pytest.mark.asyncio
async def test_asearch_returns_none_when_context_hash_differs(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0], context_hash="different"))

    match = await redis_storage.asearch([1.0, 0.0], "test", "test-model", "abc123", 0.9)

    assert match is None


@pytest.mark.asyncio
async def test_asearch_returns_none_below_threshold(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0]))

    match = await redis_storage.asearch([0.0, 1.0], "test", "test-model", "abc123", 0.5)

    assert match is not None
    assert match.entry is None


@pytest.mark.asyncio
async def test_asearch_returns_best_match_above_threshold(redis_storage):
    worse = make_entry([0.8, 0.2])
    better = make_entry([1.0, 0.0])
    await redis_storage.astore(worse)
    await redis_storage.astore(better)

    match = await redis_storage.asearch([1.0, 0.0], "test", "test-model", "abc123", 0.7)

    assert match is not None
    assert match.entry is not None
    assert match.entry.id == better.id


@pytest.mark.asyncio
async def test_asearch_pipelines_hgetall(redis_storage):
    worse = make_entry([0.7, 0.3], namespace="ns")
    better = make_entry([1.0, 0.0], namespace="ns")
    await redis_storage.astore(worse)
    await redis_storage.astore(better)

    match = await redis_storage.asearch([1.0, 0.0], "ns", "test-model", "abc123", 0.6)

    assert match is not None
    assert match.entry is not None
    assert match.entry.id == better.id


@pytest.mark.asyncio
async def test_asearch_does_not_fetch_response_before_filtering(redis_storage):
    winner = make_entry([1.0, 0.0], namespace="ns", context_hash="target")
    filtered = make_entry([0.9, 0.1], namespace="ns", context_hash="other")
    await redis_storage.astore(winner)
    await redis_storage.astore(filtered)

    original_hgetall = redis_storage._client.hgetall
    call_count = 0

    async def counted_hgetall(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await original_hgetall(*args, **kwargs)

    redis_storage._client.hgetall = counted_hgetall
    try:
        match = await redis_storage.asearch(
            [1.0, 0.0], "ns", "test-model", "target", 0.6
        )
    finally:
        redis_storage._client.hgetall = original_hgetall

    assert match is not None
    assert match.entry is not None
    assert match.entry.id == winner.id
    assert call_count == 1


@pytest.mark.asyncio
async def test_ainvalidate_namespace_removes_all_entries(redis_storage):
    a = make_entry([1.0, 0.0], namespace="ns")
    b = make_entry([0.9, 0.1], namespace="ns")
    await redis_storage.astore(a)
    await redis_storage.astore(b)

    await redis_storage.ainvalidate_namespace("ns")

    assert await redis_storage.asearch([1.0, 0.0], "ns", "test-model", "abc123", 0.0) is None


@pytest.mark.asyncio
async def test_ainvalidate_namespace_returns_correct_count(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0], namespace="ns"))
    await redis_storage.astore(make_entry([0.9, 0.1], namespace="ns"))

    deleted = await redis_storage.ainvalidate_namespace("ns")

    assert deleted == 2


@pytest.mark.asyncio
async def test_ainvalidate_nonexistent_namespace_returns_zero(redis_storage):
    deleted = await redis_storage.ainvalidate_namespace("missing")

    assert deleted == 0


@pytest.mark.asyncio
async def test_aclear_removes_all_entries(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0], namespace="a"))
    await redis_storage.astore(make_entry([1.0, 0.0], namespace="b"))
    await redis_storage.astore(make_entry([1.0, 0.0], namespace="c"))

    await redis_storage.aclear()

    assert await redis_storage.anamespace_size("a") == 0
    assert await redis_storage.anamespace_size("b") == 0
    assert await redis_storage.anamespace_size("c") == 0


@pytest.mark.asyncio
async def test_namespace_isolation(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0], namespace="ns_a"))

    match = await redis_storage.asearch([1.0, 0.0], "ns_b", "test-model", "abc123", 0.0)

    assert match is None


@pytest.mark.asyncio
async def test_lazy_tombstone_cleanup(redis_storage):
    entry = make_entry([1.0, 0.0], namespace="ns", ttl=0.1)
    await redis_storage.astore(entry)
    await asyncio.sleep(0.2)

    match = await redis_storage.asearch([1.0, 0.0], "ns", "test-model", "abc123", 0.0)

    assert match is None
    assert await redis_storage.anamespace_size("ns") == 0


@pytest.mark.asyncio
async def test_namespace_size_returns_count(redis_storage):
    await redis_storage.astore(make_entry([1.0, 0.0], namespace="ns"))
    await redis_storage.astore(make_entry([0.9, 0.1], namespace="ns"))

    assert await redis_storage.anamespace_size("ns") == 2


@pytest.mark.asyncio
async def test_astore_sets_ttl_on_redis_key(redis_storage):
    entry = make_entry([1.0, 0.0], namespace="ns", ttl=0.1)
    await redis_storage.astore(entry)
    await asyncio.sleep(0.2)

    data = await redis_storage._client.hgetall(f"llmsc:entry:{entry.id}")

    assert data == {}


def test_store_search_sync_with_sync_client():
    async_client = fakeredis.aioredis.FakeRedis()
    sync_client = fakeredis.FakeRedis()
    storage = RedisStorage(async_client, sync_client=sync_client)
    entry = make_entry([1.0, 0.0], namespace="ns")

    storage.store(entry)
    match = storage.search([1.0, 0.0], "ns", "test-model", "abc123", 0.9)

    assert match is not None
    assert match.entry is not None
    assert match.entry.id == entry.id


def test_sync_methods_raise_without_sync_client():
    storage = RedisStorage(fakeredis.aioredis.FakeRedis())

    with pytest.raises(RuntimeError, match="initialized without a sync_client"):
        storage.store(make_entry([1.0, 0.0]))


def test_clear_sync_removes_all_entries():
    async_client = fakeredis.aioredis.FakeRedis()
    sync_client = fakeredis.FakeRedis()
    storage = RedisStorage(async_client, sync_client=sync_client)

    storage.store(make_entry([1.0, 0.0], namespace="a"))
    storage.store(make_entry([1.0, 0.0], namespace="b"))
    storage.store(make_entry([1.0, 0.0], namespace="c"))

    storage.clear()

    assert storage.namespace_size("a") == 0
    assert storage.namespace_size("b") == 0
    assert storage.namespace_size("c") == 0
