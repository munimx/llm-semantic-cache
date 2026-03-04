# LLM Semantic Cache

A Python library that adds **semantic caching** in front of any OpenAI-compatible LLM API.

When your application sends a prompt, instead of always forwarding it to the model, the library first checks whether a semantically similar prompt has been asked before. If yes, the cached response is returned immediately. If no, the request is forwarded, the response is stored, and future similar prompts benefit.

---

## The Problem

Exact-match caching (hashing the prompt string) is nearly useless for LLM workloads — prompts are dynamic by nature. Semantic caching, using embeddings to find near-matches, is the approach that actually works. LLM inference costs are typically the first thing teams optimize after shipping. A semantic cache directly cuts API bills with a concrete, measurable result.

---

## Architecture

```
Your Application
      │
      ▼
┌─────────────────────┐
│   SemanticCache      │  wrap() intercepts the call
│                     │
│  1. Embed prompt    │  local sentence-transformers model, runs in-process
│  2. Search index    │  cosine similarity against stored embeddings
│  3. Threshold check │  configurable per profile or raw float
│     hit → return    │  cached response returned immediately, no LLM call
│     miss → forward  │  original API call proceeds normally
│  4. Store result    │  embedding + response stored for future hits
└────────┬────────────┘
         │ on miss
         ▼
  Your LLM Provider
  (OpenAI / Anthropic / vLLM / Ollama / anything OpenAI-compatible)
```

---

## Design Decisions

### Local embeddings — no external dependency

The library uses `all-MiniLM-L6-v2` via `sentence-transformers`, running in-process. No embedding API calls. Small (80MB), fast (sub-10ms on CPU), good enough for semantic similarity on short to medium length text.

### Named threshold profiles

| Profile | Threshold | Intended for |
|---------|-----------|--------------|
| `strict` | 0.97 | Code generation, factual Q&A |
| `balanced` | 0.92 | General assistants, summarization |
| `loose` | 0.85 | Customer support, FAQ bots |

Override with a raw float for full control.

### Namespace-based cache invalidation

TTL alone is not sufficient. Cache entries are tagged at write time; entire namespaces can be invalidated in one operation without scanning or expiring unrelated entries.

### Pluggable storage

- **In-memory** — zero dependencies, good for development, lost on restart
- **Redis** — persistent, shared across workers and replicas

---

## What This Is Not

- Not a proxy, load balancer, or rate limiter
- Not a replacement for LiteLLM
- Not vLLM-specific

---

## Status

Early development. See [PROJECT_DIRECTION2.md](https://github.com/munimx/LLM-Inference-Optimization-Engine/blob/main/PROJECT_DIRECTION2.md) in the previous repo for the full project rationale and scope.

---

## Plan Addendum — Review Resolutions

The following addresses five issues raised during plan review. Each resolution is a binding design decision for implementation.

---

### Issue 1: Redis Scalability — ACCEPTED WITH MITIGATION

**Problem:** The Redis backend fetches all vectors for a namespace to Python for client-side cosine similarity. At scale (10k+ entries), this transfers ~15MB per cache miss and is slower than the LLM call itself.

**Resolution:** Two changes.

**1a. Document the scaling boundary.** The Redis backend README and docstring will state:

> Redis storage is designed for small-to-medium namespaces (under ~5,000 entries). For larger workloads, a vector store backend (pgvector, Chroma) is on the roadmap. If your namespace exceeds this range, Redis will degrade — use namespace partitioning or wait for a vector-native backend.

**1b. Add candidate pre-filtering via quantized bucketing.** Before the full cosine similarity scan, reduce the candidate set using a coarse locality filter. Each embedding is assigned a bucket key at write time derived from quantized top-k dimensions. At query time, only entries sharing the same bucket (plus adjacent buckets) are fetched.

This lives in `storage/redis.py` as an internal optimization — the `StorageBackend` ABC does not change.

```python
# storage/redis.py — internal pre-filtering

def _compute_bucket_key(self, embedding: list[float], num_dims: int = 4, num_bins: int = 8) -> str:
    """Quantize the top-k embedding dimensions into a coarse bucket key."""
    top_dims = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:num_dims]
    bins = [str(int(embedding[d] * num_bins) % num_bins) for d in top_dims]
    return ":".join(bins)

async def search(self, namespace: str, embedding: list[float], ...) -> list[CacheEntry]:
    bucket = self._compute_bucket_key(embedding)
    # Fetch candidates from this bucket + adjacent buckets only
    candidate_keys = await self._get_bucket_candidates(namespace, bucket)
    # Fall back to full scan if bucket is empty (cold start)
    if not candidate_keys:
        candidate_keys = await self._get_all_keys(namespace)
    # Cosine similarity on reduced candidate set
    ...
```

**New test cases:**
- `test_redis_bucket_reduces_candidate_set` — verify fewer entries fetched than total namespace size.
- `test_redis_bucket_cold_start_falls_back` — verify full scan when bucket is empty.
- `test_redis_search_correctness_with_bucketing` — verify top result is identical with and without pre-filtering for a known dataset.

---

### Issue 2: Streaming (`stream=True`) — ACCEPTED, BYPASS IN V1

**Problem:** `wrap()` has no strategy for `stream=True` requests. A naive implementation will crash or silently drop the stream.

**Resolution:** In v1, **streaming requests bypass the cache entirely**. This is the simplest correct behavior and avoids the complexity of stream teeing, which introduces backpressure, error-handling, and memory concerns that are disproportionate for an initial release.

Behavior in `cache.py`:

```python
def _is_streaming_request(self, kwargs: dict) -> bool:
    return kwargs.get("stream", False) is True

# Inside wrap():
if self._is_streaming_request(kwargs):
    log.info("cache.stream_bypass", reason="stream=True not cached in v1")
    STREAM_BYPASS_COUNTER.inc()
    return await original_fn(*args, **kwargs)
```

**Guarantees:**
- `stream=True` calls are forwarded to the provider unchanged — no interception, no accumulation, no modification.
- A structlog event is emitted on every bypass so users can measure how much traffic skips the cache.
- A Prometheus counter `semantic_cache_stream_bypass_total` tracks bypass volume.

**Documentation will state:**
> Streaming responses (`stream=True`) bypass the cache in the current version. The request is forwarded directly to the provider. This is a known limitation — stream caching is planned for a future release.

**New test cases:**
- `test_wrap_stream_true_bypasses_cache` — verify no cache lookup or store occurs.
- `test_wrap_stream_true_returns_original_response` — verify the provider's streaming response is returned unmodified.
- `test_wrap_stream_bypass_emits_log_and_metric` — verify structlog event and counter increment.

---

### Issue 3: Fail-Open Resilience — ACCEPTED

**Problem:** If cache infrastructure (Redis connection, embedding model, serialization) throws an exception, `wrap()` propagates it to the caller — crashing the application even though the LLM call itself would have succeeded.

**Resolution:** All cache operations inside `wrap()` are wrapped in a fail-open guard. If any cache operation fails, the original function is called as if the cache does not exist.

```python
# cache.py — fail-open guard

async def _cached_call(self, original_fn, *args, **kwargs):
    # Attempt cache lookup
    try:
        hit = await self._lookup(kwargs)
        if hit is not None:
            return hit
    except Exception as exc:
        log.error("cache.lookup_failed", error=str(exc), exc_info=True)
        CACHE_ERROR_COUNTER.labels(operation="lookup").inc()
        # Fall through — call the provider

    # Call the original function (always happens on miss or error)
    response = await original_fn(*args, **kwargs)

    # Attempt cache store — failure must not affect the response
    try:
        await self._store(kwargs, response)
    except Exception as exc:
        log.error("cache.store_failed", error=str(exc), exc_info=True)
        CACHE_ERROR_COUNTER.labels(operation="store").inc()

    return response
```

**Invariant:** `wrap()` never raises an exception that the unwrapped function would not have raised. The cache is purely additive — its failure is invisible to the caller except via logs and metrics.

**New Prometheus metric:** `semantic_cache_errors_total` with label `operation` (`lookup` | `store` | `embed`).

**New test cases:**
- `test_wrap_returns_response_when_lookup_raises` — Redis connection error during lookup, provider is still called.
- `test_wrap_returns_response_when_store_raises` — provider succeeds, store fails, response is still returned.
- `test_wrap_returns_response_when_embedding_raises` — model load failure, falls through to provider.
- `test_fail_open_logs_error_with_structlog` — verify structured log fields on failure.

---

### Issue 4: Context Serialization Robustness — ACCEPTED

**Problem:** `json.dumps` in `context.py` will fail with `TypeError` on common Python types: `set`, `datetime`, `bytes`, `UUID`, Pydantic models. Users will pass these as context values and get cryptic errors.

**Resolution:** `context.py` will include a canonical serializer that handles all common non-JSON-serializable types deterministically. The serializer is used exclusively for context fingerprinting — it produces a stable string, not a round-trippable format.

```python
# context.py — canonical serializer

import datetime
import uuid
from collections.abc import Set

def _canonical_default(obj: object) -> object:
    """json.dumps default hook for deterministic context hashing."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, Set):
        return sorted(str(item) for item in obj)  # deterministic order
    if isinstance(obj, frozenset):
        return sorted(str(item) for item in obj)
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    raise TypeError(f"Context value of type {type(obj).__name__} is not serializable. "
                    f"Convert it to a JSON-compatible type before passing it as context.")

def compute_context_hash(context: dict) -> str:
    """Deterministic hash of context dict for cache key construction."""
    serialized = json.dumps(context, sort_keys=True, default=_canonical_default, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()
```

**Design notes:**
- `sort_keys=True` ensures dict ordering does not affect the hash.
- Sets are sorted-then-stringified for deterministic ordering.
- The `TypeError` at the end of `_canonical_default` is intentional — unknown types should fail loudly with a clear message, not silently produce wrong hashes.
- `ensure_ascii=True` prevents encoding-dependent hash differences.

**New test cases:**
- `test_context_hash_with_datetime` — same datetime produces same hash.
- `test_context_hash_with_set` — `{1, 2, 3}` and `{3, 1, 2}` produce the same hash.
- `test_context_hash_with_uuid` — UUID context value serializes correctly.
- `test_context_hash_with_pydantic_model` — Pydantic v2 model in context values.
- `test_context_hash_with_bytes` — bytes value produces deterministic hash.
- `test_context_hash_unknown_type_raises_clear_error` — custom class raises `TypeError` with actionable message.
- `test_context_hash_deterministic_across_calls` — same input always produces same output.

---

### Issue 5: Sync/Async Wrapper Separation — ACCEPTED

**Problem:** `inspect.iscoroutinefunction()` is unreliable on decorated functions (e.g., functions wrapped by `@functools.wraps`, `@retry`, or framework decorators strip the coroutine flag). A single wrapper body that tries to handle both sync and async in one code path will produce subtle runtime failures.

**Resolution:** `wrap()` becomes a factory that inspects the target at decoration time and returns one of two distinct wrapper classes. The user can also force the mode explicitly to bypass detection entirely.

```python
# cache.py — wrapper factory

from typing import Literal

def wrap(
    self,
    fn: Callable,
    *,
    mode: Literal["auto", "sync", "async"] = "auto",
    **cache_opts,
) -> Callable:
    """Wrap a callable with semantic caching.

    Args:
        fn: The function to wrap.
        mode: "auto" detects sync/async (default). "sync" or "async"
              forces the wrapper type — use when auto-detection fails
              on heavily decorated functions.
    """
    if mode == "async" or (mode == "auto" and self._is_async(fn)):
        return self._make_async_wrapper(fn, **cache_opts)
    return self._make_sync_wrapper(fn, **cache_opts)

def _is_async(self, fn: Callable) -> bool:
    """Best-effort async detection, checking through common decorator layers."""
    unwrapped = inspect.unwrap(fn)  # follows __wrapped__ chain
    return inspect.iscoroutinefunction(unwrapped)

def _make_async_wrapper(self, fn: Callable, **cache_opts) -> Callable:
    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        return await self._cached_call(fn, *args, **kwargs)
    async_wrapper.__wrapped__ = fn
    return async_wrapper

def _make_sync_wrapper(self, fn: Callable, **cache_opts) -> Callable:
    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        return self._cached_call_sync(fn, *args, **kwargs)
    sync_wrapper.__wrapped__ = fn
    return sync_wrapper
```

**Key decisions:**
- `inspect.unwrap()` follows the `__wrapped__` chain before checking, which handles `@functools.wraps`-based decorators correctly.
- The `mode` parameter provides an explicit escape hatch — if auto-detection fails, the user sets `mode="async"` and it works. No guessing.
- Sync wrapper calls `_cached_call_sync`, which uses synchronous Redis/storage operations. Async wrapper calls `_cached_call`, which uses `await`. These are **separate code paths** — no `asyncio.run()` or `loop.run_until_complete()` bridging.
- Both wrappers set `__wrapped__` so downstream decorators can continue unwrapping.

**New test cases:**
- `test_wrap_auto_detects_async_function` — plain `async def` is wrapped as async.
- `test_wrap_auto_detects_sync_function` — plain `def` is wrapped as sync.
- `test_wrap_auto_detects_decorated_async` — `@functools.wraps`-decorated async function is correctly detected.
- `test_wrap_mode_override_forces_async` — `mode="async"` on a sync-looking function produces async wrapper.
- `test_wrap_mode_override_forces_sync` — `mode="sync"` on an async function produces sync wrapper.
- `test_sync_wrapper_does_not_use_event_loop` — sync wrapper never calls `asyncio.run` or touches an event loop.
- `test_async_wrapper_is_awaitable` — return value of async wrapper is a coroutine.

---

## Plan Addendum 2 — Round 2 Review Resolutions

The following addresses five issues raised during the second round of plan review. Each resolution is a binding design decision. Where a Round 1 decision is superseded, that is stated explicitly.

---

### Issue R2-1 (Critical): Sync/Async Storage Interface Mismatch — RESOLVED

**Problem:** The `StorageBackend` ABC defines only async methods (`async def store`, `async def search`, etc.). The sync wrapper from Round 1 calls `_cached_call_sync`, which cannot call async storage methods without `asyncio.run()` bridging — which was explicitly ruled out. This means the sync wrapper is broken for Flask, Django, and any non-async application.

**Resolution: Sync-primary ABC with async overrides.**

The `StorageBackend` ABC defines **sync abstract methods** as the primary interface. Async variants (prefixed with `a`) have default implementations that delegate to sync via `asyncio.get_running_loop().run_in_executor()`. Backends that have native async I/O (Redis) override the async methods.

```python
# storage/base.py

class StorageBackend(ABC):
    """Storage interface. Sync methods are abstract. Async methods have
    thread-pool defaults — override for native async I/O."""

    # --- Sync interface (abstract — every backend implements these) ---

    @abstractmethod
    def store(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    def search(
        self, namespace: str, embedding: list[float],
        embedding_model_id: str, threshold: float,
    ) -> list[CacheEntry]: ...

    @abstractmethod
    def invalidate_namespace(self, namespace: str) -> int: ...

    # --- Async interface (defaults run sync in thread pool) ---

    async def astore(self, entry: CacheEntry) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.store, entry)

    async def asearch(
        self, namespace: str, embedding: list[float],
        embedding_model_id: str, threshold: float,
    ) -> list[CacheEntry]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.search, namespace, embedding,
            embedding_model_id, threshold,
        )

    async def ainvalidate_namespace(self, namespace: str) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.invalidate_namespace, namespace,
        )
```

**How each backend implements this:**

| Backend | Sync methods | Async methods |
|---------|-------------|---------------|
| `MemoryStorage` | Implements all 3 (dict operations) | Overrides to call sync directly — no thread pool needed, no I/O | 
| `RedisStorage` | Implements all 3 via `redis.Redis` (blocking client) | Overrides all 3 via `redis.asyncio.Redis` (native async client) |

```python
# storage/memory.py — async overrides skip the thread pool (no I/O)
class MemoryStorage(StorageBackend):
    def store(self, entry: CacheEntry) -> None:
        self._entries[entry.key] = entry
        self._namespace_index[entry.namespace].add(entry.key)

    async def astore(self, entry: CacheEntry) -> None:
        self.store(entry)  # dict write, no I/O, thread pool is overhead

    # search and invalidate_namespace follow the same pattern
```

```python
# storage/redis.py — async overrides use native async Redis client
class RedisStorage(StorageBackend):
    def __init__(self, sync_client: redis.Redis, async_client: redis.asyncio.Redis):
        self._sync = sync_client
        self._async = async_client

    def store(self, entry: CacheEntry) -> None:
        self._sync.hset(...)  # blocking call

    async def astore(self, entry: CacheEntry) -> None:
        await self._async.hset(...)  # native async call

    # search and invalidate_namespace follow the same pattern
```

**How `wrap()` connects to this:**

- `_make_sync_wrapper` → calls `self.storage.store()`, `self.storage.search()` (sync methods directly)
- `_make_async_wrapper` → calls `await self.storage.astore()`, `await self.storage.asearch()` (async methods)

No `asyncio.run()`. No `loop.run_until_complete()`. No bridging. Each wrapper calls the method set that matches its execution model.

**Validation at `wrap()` time:** If a user passes `mode="async"` but their `StorageBackend` subclass hasn't overridden the async methods and its sync methods perform blocking I/O, the thread-pool default is technically correct but suboptimal. Documentation will note: *"For Redis, always pass both sync and async clients. The thread-pool fallback works but wastes a thread per cache operation."*

**Round 1 revision:** The Round 1 statement "Sync and async wrappers use completely separate code paths — no `asyncio.run()` bridging" remains true. This resolution fills in the missing detail: *how* the sync code path accesses storage.

**New test cases:**
- `test_sync_wrapper_calls_sync_storage_methods` — verify `store()` and `search()` are called, not `astore()`/`asearch()`.
- `test_async_wrapper_calls_async_storage_methods` — verify `astore()` and `asearch()` are called.
- `test_memory_async_does_not_use_executor` — verify `MemoryStorage.astore()` does not invoke `run_in_executor`.
- `test_redis_sync_uses_blocking_client` — verify `RedisStorage.store()` uses `redis.Redis`.
- `test_redis_async_uses_async_client` — verify `RedisStorage.astore()` uses `redis.asyncio.Redis`.
- `test_thread_pool_fallback_works_for_custom_backend` — a custom backend implementing only sync methods works correctly from an async wrapper via the default `run_in_executor` path.
- `test_sync_wrapper_in_flask_context` — sync wrapper functions correctly without an event loop present.

---

### Issue R2-2 (Correctness): Flawed Bucket Pre-filtering — DROPPED

**Problem:** The Round 1 "top-4 dims × 8 bins" LSH-style bucket key is mathematically broken for dense, normalized embeddings like those produced by `all-MiniLM-L6-v2` (384-dimensional, L2-normalized). High-cosine-similarity vectors frequently have different top-4 dimensions by absolute magnitude, because the information is distributed across all dimensions. The bucket key creates a coarse partition that splits genuinely similar vectors into different buckets, causing silent false negatives — the cache has the answer but fails to find it.

This is not a tuning problem. The approach is fundamentally unsound for this class of embeddings.

**Resolution: Drop bucket pre-filtering entirely.** Remove the following from the Round 1 plan:

- ~~`_compute_bucket_key()` method~~
- ~~Bucket key assignment at write time~~
- ~~Bucket-based candidate set reduction at query time~~
- ~~`test_redis_bucket_reduces_candidate_set`~~
- ~~`test_redis_bucket_cold_start_falls_back`~~
- ~~`test_redis_search_correctness_with_bucketing`~~

**Supersedes:** Round 1, Issue 1, section 1b ("Add candidate pre-filtering via quantized bucketing"). Section 1a (documenting the scaling boundary) remains in effect.

Both backends now use **brute-force cosine similarity** over all entries in the namespace filtered by `embedding_model_id`. For in-memory storage, this is a numpy vectorized operation — sub-millisecond for ≤5,000 entries. For Redis, see Issue R2-3 below.

**New test cases (replacing the removed ones):**
- `test_search_returns_correct_top_match_brute_force` — verify the highest-similarity entry is returned, not a bucket-adjacent one.
- `test_search_no_false_negatives_within_threshold` — for a known dataset, every entry above threshold is found.

---

### Issue R2-3 (Performance): Redis Data Transfer — ACCEPTED AS DOCUMENTED LIMITATION

**Problem:** With bucket pre-filtering dropped (Issue R2-2), the Redis backend must fetch all embeddings per namespace to Python for client-side cosine similarity. For a namespace with 5,000 entries of 384-dim float32 embeddings, this is ~7.5MB transferred per cache miss. At 10,000 entries, ~15MB. This is a performance cliff that can be slower than the LLM call it's trying to avoid.

**Resolution: Document honestly. No server-side workaround in v1.**

A Redis Lua script for server-side cosine similarity was considered and rejected for v1:

1. **Lua numeric precision.** Redis Lua uses double-precision floats, which is adequate, but the script must deserialize binary-packed vectors from hash fields — fragile and hard to test without a real Redis instance (violates the `fakeredis` testing constraint).
2. **Lua script size.** A cosine similarity kernel over variable-length vectors with threshold filtering is ~50 lines of Lua. This is a second language embedded in the codebase with its own testing and debugging story — disproportionate complexity for v1.
3. **Redis module alternative.** `RediSearch` with vector similarity (`FT.SEARCH ... KNN`) would solve this properly, but it requires a Redis module that is not available in all deployments and adds an infrastructure dependency. This belongs on the roadmap, not in v1.

**What ships in v1:**

1. **Client-side brute force with pipeline fetch.** Embeddings are fetched via a single `MGET` pipeline call (not N individual `HGETALL` calls). This minimizes round trips.

2. **Hard documentation of the boundary:**

   > **Redis storage scaling limit:** The Redis backend transfers all embeddings in a namespace to the client for similarity search. This is practical for namespaces up to ~5,000 entries. Beyond that, latency increases linearly with namespace size. For large namespaces, either partition into smaller namespaces or use a vector-native backend when available.

3. **A `namespace_size()` method** on `StorageBackend` that returns the entry count. This lets users monitor growth and set up alerts before hitting the cliff.

4. **A structlog warning** emitted when a namespace exceeds a configurable threshold (default: 5,000):

   ```python
   if namespace_count > self._size_warning_threshold:
       log.warning(
           "cache.namespace_large",
           namespace=namespace,
           entry_count=namespace_count,
           threshold=self._size_warning_threshold,
           msg="Namespace exceeds recommended size. Search latency may degrade.",
       )
   ```

**Roadmap note (not implemented now):** The proper fix is a vector-native backend (pgvector or RediSearch). When implemented, it will conform to the same `StorageBackend` ABC and be a drop-in replacement. The brute-force Redis backend will remain available for small deployments.

**New test cases:**
- `test_redis_search_uses_pipeline_fetch` — verify embeddings are fetched in a single pipeline, not individual calls.
- `test_namespace_size_returns_correct_count` — verify `namespace_size()` on both backends.
- `test_large_namespace_emits_warning` — verify structlog warning when namespace exceeds threshold.

---

### Issue R2-4 (Maintenance): Redis Namespace SET Leaks Dead Keys — RESOLVED

**Problem:** Each cache entry is stored as a Redis hash with a TTL. The namespace index is a Redis SET containing entry IDs. When a hash expires via TTL, Redis deletes the hash but does **not** remove its ID from the namespace SET. Over time, the SET accumulates IDs pointing to expired (nonexistent) hashes — tombstones. This causes:

1. `invalidate_namespace()` issues `DEL` commands for keys that no longer exist — wasted round trips.
2. `search()` fetches IDs from the SET, then issues `HGETALL` for each — expired ones return empty, wasting bandwidth.
3. The SET grows without bound if the namespace is long-lived.

**Resolution: Lazy cleanup during read operations.**

When `search()` or `invalidate_namespace()` encounters an ID in the namespace SET whose corresponding hash no longer exists, it removes that ID from the SET in the same pipeline. No separate cleanup job, no scan, no background task.

```python
# storage/redis.py — lazy cleanup during search

def search(self, namespace: str, embedding: list[float], ...) -> list[CacheEntry]:
    entry_ids = self._sync.smembers(self._namespace_key(namespace))
    if not entry_ids:
        return []

    pipeline = self._sync.pipeline()
    for entry_id in entry_ids:
        pipeline.hgetall(self._entry_key(entry_id))
    results = pipeline.execute()

    live_entries: list[CacheEntry] = []
    dead_ids: list[str] = []

    for entry_id, data in zip(entry_ids, results):
        if not data:
            dead_ids.append(entry_id)
            continue
        live_entries.append(self._deserialize(data))

    # Remove tombstones from the namespace SET
    if dead_ids:
        self._sync.srem(self._namespace_key(namespace), *dead_ids)
        log.debug("cache.tombstones_cleaned", namespace=namespace, count=len(dead_ids))

    # Cosine similarity on live entries only
    return self._rank_by_similarity(live_entries, embedding, threshold)
```

**Design notes:**

- Cleanup is O(dead keys found) per search — no full namespace scan.
- `SREM` is O(N) where N is the number of IDs to remove, executed in a single call.
- The cleanup is opportunistic. If a namespace is never searched or invalidated again, its SET retains tombstones. This is acceptable — orphaned SETs with no live hashes waste negligible memory (just the SET of string IDs) and will be caught on the next access.
- A separate `cleanup_namespace()` utility method is **not** added. The lazy approach is sufficient and avoids a maintenance API that users must remember to call. If this proves inadequate, a cleanup method can be added later without changing the storage interface.

**New test cases:**
- `test_redis_search_removes_expired_ids_from_set` — insert entries, let TTL expire (use `fakeredis` time manipulation), search, verify SET no longer contains expired IDs.
- `test_redis_invalidate_handles_already_expired_entries` — invalidate a namespace where some entries have already expired, verify no errors and SET is cleaned.
- `test_redis_search_ignores_tombstones_in_similarity` — expired entries do not participate in cosine similarity ranking.
- `test_redis_tombstone_cleanup_logs_count` — verify structlog debug event with correct count.

---

### Issue R2-5 (Debuggability): CacheEntry Missing prompt_text — RESOLVED

**Problem:** `CacheEntry` stores the embedding vector and the cached response but not the original prompt text. This makes three things impossible:

1. **Debugging:** When a user gets a wrong cached response, they cannot inspect which original prompt it matched against. They see an embedding vector (384 floats) and a response, but not the question that produced them.
2. **Re-embedding:** When the embedding model is swapped (e.g., from `all-MiniLM-L6-v2` to a newer model), existing cache entries cannot be re-embedded because the source text is gone. The only option is to discard the entire cache.
3. **Cache inspection tooling:** Any future CLI or admin UI for browsing cache contents would show opaque vectors instead of human-readable prompts.

**Resolution: Add `prompt_text` to `CacheEntry`.**

```python
@dataclass
class CacheEntry:
    key: str                    # deterministic hash of prompt + context
    namespace: str
    embedding: list[float]
    embedding_model_id: str
    prompt_text: str            # NEW — original prompt that produced this embedding
    response: dict              # serialized OpenAI-compatible response
    context_hash: str           # hash of context dict
    created_at: float           # unix timestamp
    ttl: int | None             # seconds, or None for no expiry
```

**Storage impact:**

- **In-memory:** No change — `CacheEntry` is stored as-is.
- **Redis:** `prompt_text` is stored as a field in the entry hash. For typical prompts (100–500 tokens, ~400–2,000 chars), this adds ~2KB per entry. At 5,000 entries, ~10MB additional — acceptable for the Redis scaling boundary already documented.

**What `prompt_text` is NOT:**

- It is **not** used in similarity search. Search is performed on the embedding vector. `prompt_text` is metadata for human inspection and tooling.
- It is **not** used in cache key construction. The cache key is derived from the embedding + context hash, as specified in the architecture.
- It is **not** optional. Every cache entry has a prompt — there is no scenario where the text is unavailable at write time. Making it optional would create a second class of entries that breaks tooling assumptions.

**Privacy consideration:** Storing prompt text means cache entries contain user input in plaintext. The documentation will note:

> Cache entries include the original prompt text for debugging and re-embedding. If your prompts contain sensitive data, ensure your storage backend has appropriate access controls. The in-memory backend is process-local. For Redis, use Redis ACLs and TLS as appropriate for your deployment.

**New test cases:**
- `test_cache_entry_stores_prompt_text` — verify `prompt_text` is preserved through store → search round-trip.
- `test_cache_entry_prompt_text_in_redis_hash` — verify `prompt_text` is stored as a field in the Redis hash.
- `test_cache_entry_prompt_text_not_used_in_similarity` — verify that two entries with different `prompt_text` but identical embeddings produce the same similarity score.
- `test_cache_entry_requires_prompt_text` — verify `CacheEntry` construction fails if `prompt_text` is omitted.

---

### Summary of Round 1 Decisions Modified by Round 2

| Round 1 Decision | Status After Round 2 |
|-----------------|---------------------|
| 1a. Document Redis scaling boundary | **Unchanged.** Still in effect. |
| 1b. Bucket pre-filtering in Redis | **Dropped.** Mathematically unsound (R2-2). Replaced by brute-force + documented limitation (R2-3). |
| 2. Streaming bypass | **Unchanged.** |
| 3. Fail-open resilience | **Unchanged.** |
| 4. Context serialization | **Unchanged.** |
| 5. Sync/async wrapper separation | **Extended.** R2-1 fills in the storage layer design that was missing. The wrapper factory design is unchanged; the storage ABC now has sync-primary methods with async overrides. |
