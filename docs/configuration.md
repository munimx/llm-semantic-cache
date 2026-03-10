# Configuration

## `CacheConfig` reference

| Field | Type | Default | Description |
|---|---|---|---|
| `threshold` | `str \| float` | `"balanced"` | Named profile (`"strict"`, `"balanced"`, `"loose"`) or raw cosine similarity in [0.0, 1.0] |
| `default_namespace` | `str` | `"default"` | Namespace used when `cache_namespace` is not passed |
| `default_ttl` | `float \| None` | `None` | Time-to-live in seconds. None = no expiration |
| `embedding_model` | `str` | `"BAAI/bge-small-en-v1.5"` | Embedding model identifier used for tagging cache entries. Must match the configured embedder's `model_id` |
| `cache_timeout_seconds` | `float` | `0.05` | Timeout for cache operations (async path only) |

## Threshold profiles

- `strict` (`0.97`): Use for code generation and factual answers where a false positive looks like confidently returning the wrong snippet or fact; a false negative looks like an extra provider call for a near-identical prompt.
- `balanced` (`0.92`): Use for general assistants and summarization where you want practical savings without aggressive matching; a false positive looks like subtle context drift, and a false negative looks like lower hit rate on paraphrases.
- `loose` (`0.85`): Use for repetitive support/FAQ bots where wording varies a lot; a false positive looks like a slightly off canned answer, and a false negative looks like avoidable misses in common support phrasing.

## `cache_timeout_seconds`

In async mode, cache lookup/store operations run with a hard timeout and fail open if they exceed `cache_timeout_seconds`, which means your wrapped function is still called and your app keeps serving responses. In v0.1.0 this timeout protection only applies to async code paths; sync callers (including sync `RedisStorage` usage) have no timeout guard and rely on the backend client's own behavior.

## `default_ttl`

Use `default_ttl` when cached answers age out naturally, such as short-lived support responses or rapidly changing operational data. Avoid relying on TTL alone for correctness when upstream content changes in bulk (for example model swaps, document re-indexes, API version changes). For content-change scenarios, use namespace invalidation so you can remove stale entries immediately instead of waiting for expiry.

## Per-call overrides

Pass `cache_namespace` on each wrapped call when you want request-level scoping, such as per-tenant or per-session isolation:

```python
response = cached_create(
    model="gpt-4o-mini",
    messages=messages,
    cache_context={"user_id": "u-42"},
    cache_namespace="session:s-123",
)
```
