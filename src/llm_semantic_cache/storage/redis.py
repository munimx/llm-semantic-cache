"""RedisStorage — persistent cache backend using redis.asyncio."""
from __future__ import annotations

import json
from typing import Any, cast

from llm_semantic_cache.similarity import cosine_similarity
from llm_semantic_cache.storage.base import CacheEntry, StorageBackend

_ENTRY_PREFIX = "llmsc:entry:"
_NS_INDEX_PREFIX = "llmsc:ns:"


def _entry_key(entry_id: str) -> str:
    return f"{_ENTRY_PREFIX}{entry_id}"


def _ns_index_key(namespace: str) -> str:
    return f"{_NS_INDEX_PREFIX}{namespace}"


class RedisStorage(StorageBackend):
    """Redis-backed storage for SemanticCache.

    Uses native redis.asyncio for async operations and optional redis.Redis for
    sync operations.

    Scaling note: This backend fetches all candidate vectors from Redis to
    Python for cosine similarity computation. Performance degrades when a
    namespace exceeds ~5,000 entries. Use namespace partitioning for larger
    workloads, or wait for a vector-native backend.
    """

    def __init__(self, client: Any, sync_client: Any = None) -> None:
        """Initialize with async (redis.asyncio) and optional sync (redis.Redis) clients.

        Args:
            client: An async redis.asyncio client for async operations.
            sync_client: A synchronous redis.Redis client for sync operations.
                If None, sync methods (store, search, invalidate_namespace, clear,
                namespace_size) will raise RuntimeError. Use the async a*() methods
                in async frameworks (FastAPI, Starlette) instead.
        """
        self._client = client
        self._sync_client = sync_client

    def _require_sync_client(self) -> Any:
        """Return sync client or raise RuntimeError."""
        if self._sync_client is None:
            raise RuntimeError(
                "RedisStorage was initialized without a sync_client. "
                "Pass sync_client=redis.Redis(...) to use sync methods, "
                "or use the async a*() methods in async frameworks."
            )
        return self._sync_client

    def store(self, entry: CacheEntry) -> None:
        sync_client = self._require_sync_client()
        key = _entry_key(entry.id)
        payload = {
            "id": entry.id,
            "embedding": json.dumps(entry.embedding),
            "prompt_text": entry.prompt_text,
            "context_hash": entry.context_hash,
            "namespace": entry.namespace,
            "embedding_model_id": entry.embedding_model_id,
            "response": json.dumps(entry.response),
            "created_at": str(entry.created_at),
            "ttl": str(entry.ttl) if entry.ttl is not None else "",
        }
        pipe = sync_client.pipeline()
        pipe.hset(key, mapping=payload)
        pipe.sadd(_ns_index_key(entry.namespace), entry.id)
        if entry.ttl is not None:
            pipe.pexpire(key, max(1, int(entry.ttl * 1000)))
        pipe.execute()

    def search(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> CacheEntry | None:
        sync_client = self._require_sync_client()
        ns_key = _ns_index_key(namespace)
        entry_ids = sync_client.smembers(ns_key)
        if not entry_ids:
            return None

        decoded_ids = [
            raw_id.decode() if isinstance(raw_id, bytes) else raw_id for raw_id in entry_ids
        ]
        pipe = sync_client.pipeline()
        for entry_id in decoded_ids:
            pipe.hgetall(_entry_key(entry_id))
        results = pipe.execute()

        best_entry: CacheEntry | None = None
        best_score = -1.0
        dead_ids: list[str] = []

        for entry_id, data in zip(decoded_ids, results):
            if not data:
                dead_ids.append(entry_id)
                continue

            entry = _deserialize_entry(data)
            if entry.embedding_model_id != embedding_model_id:
                continue
            if entry.context_hash != context_hash:
                continue
            if entry.is_expired():
                dead_ids.append(entry_id)
                continue

            score = cosine_similarity(embedding, entry.embedding)
            if score >= threshold and score > best_score:
                best_score = score
                best_entry = entry

        if dead_ids:
            sync_client.srem(ns_key, *dead_ids)

        return best_entry

    def invalidate_namespace(self, namespace: str) -> int:
        sync_client = self._require_sync_client()
        ns_key = _ns_index_key(namespace)
        entry_ids = sync_client.smembers(ns_key)
        if not entry_ids:
            return 0

        entry_keys = [
            _entry_key(eid.decode() if isinstance(eid, bytes) else eid) for eid in entry_ids
        ]
        pipe = sync_client.pipeline()
        for key in entry_keys:
            pipe.delete(key)
        pipe.delete(ns_key)
        pipe.execute()
        return len(entry_ids)

    def clear(self) -> None:
        sync_client = self._require_sync_client()
        keys = sync_client.keys("llmsc:*")
        if keys:
            sync_client.delete(*keys)

    def namespace_size(self, namespace: str) -> int:
        sync_client = self._require_sync_client()
        return cast(int, sync_client.scard(_ns_index_key(namespace)))

    async def astore(self, entry: CacheEntry) -> None:
        """Store a cache entry in Redis."""
        key = _entry_key(entry.id)
        payload = {
            "id": entry.id,
            "embedding": json.dumps(entry.embedding),
            "prompt_text": entry.prompt_text,
            "context_hash": entry.context_hash,
            "namespace": entry.namespace,
            "embedding_model_id": entry.embedding_model_id,
            "response": json.dumps(entry.response),
            "created_at": str(entry.created_at),
            "ttl": str(entry.ttl) if entry.ttl is not None else "",
        }
        pipe = self._client.pipeline()
        pipe.hset(key, mapping=payload)
        pipe.sadd(_ns_index_key(entry.namespace), entry.id)
        if entry.ttl is not None:
            pipe.pexpire(key, max(1, int(entry.ttl * 1000)))
        await pipe.execute()

    async def asearch(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> CacheEntry | None:
        """Search for best matching entry in Redis with lazy tombstone cleanup."""
        ns_key = _ns_index_key(namespace)
        entry_ids = await self._client.smembers(ns_key)
        if not entry_ids:
            return None

        decoded_ids = [
            raw_id.decode() if isinstance(raw_id, bytes) else raw_id for raw_id in entry_ids
        ]
        pipe = self._client.pipeline()
        for entry_id in decoded_ids:
            pipe.hgetall(_entry_key(entry_id))
        results = await pipe.execute()

        best_entry: CacheEntry | None = None
        best_score = -1.0
        dead_ids: list[str] = []

        for entry_id, data in zip(decoded_ids, results):
            if not data:
                dead_ids.append(entry_id)
                continue

            entry = _deserialize_entry(data)
            if entry.embedding_model_id != embedding_model_id:
                continue
            if entry.context_hash != context_hash:
                continue
            if entry.is_expired():
                dead_ids.append(entry_id)
                continue

            score = cosine_similarity(embedding, entry.embedding)
            if score >= threshold and score > best_score:
                best_score = score
                best_entry = entry

        if dead_ids:
            await self._client.srem(ns_key, *dead_ids)

        return best_entry

    async def ainvalidate_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace atomically via pipeline."""
        ns_key = _ns_index_key(namespace)
        entry_ids = await self._client.smembers(ns_key)
        if not entry_ids:
            return 0

        entry_keys = [
            _entry_key(eid.decode() if isinstance(eid, bytes) else eid)
            for eid in entry_ids
        ]

        pipe = self._client.pipeline()
        for key in entry_keys:
            pipe.delete(key)
        pipe.delete(ns_key)
        await pipe.execute()

        return len(entry_ids)

    async def aclear(self) -> None:
        """Delete all llmsc:* keys. Use with caution in production."""
        keys = await self._client.keys("llmsc:*")
        if keys:
            await self._client.delete(*keys)

    async def anamespace_size(self, namespace: str) -> int:
        """Return the number of (potentially live) entries in a namespace."""
        return cast(int, await self._client.scard(_ns_index_key(namespace)))


def _deserialize_entry(data: dict[bytes | str, bytes | str]) -> CacheEntry:
    """Deserialize a Redis hash into a CacheEntry."""

    def d(key: str) -> str:
        value = data.get(key) or data.get(key.encode(), b"")
        return value.decode() if isinstance(value, bytes) else value

    ttl_str = d("ttl")
    ttl = float(ttl_str) if ttl_str else None

    return CacheEntry(
        id=d("id"),
        embedding=json.loads(d("embedding")),
        prompt_text=d("prompt_text"),
        context_hash=d("context_hash"),
        namespace=d("namespace"),
        embedding_model_id=d("embedding_model_id"),
        response=json.loads(d("response")),
        created_at=float(d("created_at")),
        ttl=ttl,
    )
