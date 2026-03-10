# Storage backends

## `InMemoryStorage`

Use `InMemoryStorage` for development, tests, and single-process single-thread deployments where you want zero dependencies and very low overhead. Do not use it for multi-threaded apps, multi-process workers, or any workload that needs persistence across restarts. It runs numpy-vectorized brute-force cosine search in memory and does no network I/O.

## `ThreadSafeInMemoryStorage`

`ThreadSafeInMemoryStorage` is an RLock-protected drop-in replacement for `InMemoryStorage`. Use it when multiple threads or concurrent async tasks share the same cache instance — for example in FastAPI, Starlette, or any other framework that dispatches requests on a thread pool or an event loop with concurrent coroutines.

```python
from recallm import CacheConfig, SemanticCache, ThreadSafeInMemoryStorage

cache = SemanticCache(
    storage=ThreadSafeInMemoryStorage(),
    config=CacheConfig(threshold="balanced"),
)
```

Use `InMemoryStorage` for single-threaded scripts and tests where you want minimal overhead. Use `ThreadSafeInMemoryStorage` everywhere else that stays in-process.

## `RedisStorage`

For async-first apps, initialize `RedisStorage` with only an async Redis client and use wrapped async callables:

```python
import redis.asyncio as redis
from recallm import RedisStorage

async_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
storage = RedisStorage(client=async_client)
```

If you need both sync and async call paths, provide `sync_client` too:

```python
import redis
import redis.asyncio as redis_async
from recallm import RedisStorage

async_client = redis_async.Redis(host="localhost", port=6379, decode_responses=False)
sync_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
storage = RedisStorage(client=async_client, sync_client=sync_client)
```

RedisStorage performs client-side cosine similarity: it fetches candidate vectors from Redis and scores them in Python, so lookup remains correct but gets slower as namespace size grows. At more than about 5,000 entries in one namespace, latency increases significantly; partition by namespace (tenant, corpus, time window, session) to keep each candidate set smaller. Tombstone cleanup is lazy: missing or expired entry IDs are pruned during searches, so `namespace_size()` can temporarily overcount. Cache hits also cost two round trips: one for candidate metadata scanning and one to fetch the winning entry payload.

## Implementing a custom backend

Implement the `StorageBackend` abstract methods:

- `store(entry: CacheEntry) -> None`
- `search(embedding, namespace, embedding_model_id, context_hash, threshold) -> SearchResult | None`
- `invalidate_namespace(namespace: str) -> int`
- `clear() -> None`
- `namespace_size(namespace: str) -> int`

Minimal SQLite-style stub:

```python
from llm_semantic_cache.storage.base import CacheEntry, SearchResult, StorageBackend

class SQLiteStorage(StorageBackend):
    def store(self, entry: CacheEntry) -> None:
        ...

    def search(self, embedding, namespace, embedding_model_id, context_hash, threshold) -> SearchResult | None:
        ...

    def invalidate_namespace(self, namespace: str) -> int:
        ...

    def clear(self) -> None:
        ...

    def namespace_size(self, namespace: str) -> int:
        ...
```

By default, async methods delegate to `asyncio.to_thread()` wrappers around sync methods. Override async methods when your backend has native async I/O so you avoid thread-pool overhead and keep behavior predictable under load.
