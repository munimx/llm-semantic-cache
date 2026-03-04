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

    Uses native redis.asyncio for all async operations. Sync methods are
    implemented via asyncio.run() — for async applications, use the a*()
    methods directly.

    Scaling note: This backend fetches all candidate vectors from Redis to
    Python for cosine similarity computation. Performance degrades when a
    namespace exceeds ~5,000 entries. Use namespace partitioning for larger
    workloads, or wait for a vector-native backend.
    """

    def __init__(self, client: Any) -> None:
        """Initialize with redis.asyncio client."""
        self._client = client

    def store(self, entry: CacheEntry) -> None:
        import asyncio

        asyncio.run(self.astore(entry))

    def search(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> CacheEntry | None:
        import asyncio

        return asyncio.run(
            self.asearch(embedding, namespace, embedding_model_id, context_hash, threshold)
        )

    def invalidate_namespace(self, namespace: str) -> int:
        import asyncio

        return asyncio.run(self.ainvalidate_namespace(namespace))

    def clear(self) -> None:
        import asyncio

        asyncio.run(self.aclear())

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

        best_entry: CacheEntry | None = None
        best_score = -1.0
        dead_ids: list[str] = []

        for raw_id in entry_ids:
            entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            data = await self._client.hgetall(_entry_key(entry_id))
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

    async def namespace_size(self, namespace: str) -> int:
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
