# How it works

## The decision path

1. Extract the last user message from `messages` — if none found, bypass cache and call the LLM.
2. Compute SHA-256 of `cache_context` dict to produce `context_hash`.
3. Embed the prompt text using the configured embedding model.
4. Search the storage backend for entries matching `namespace`, `embedding_model_id`, and `context_hash` with cosine similarity ≥ threshold.
5. On hit: return the cached response immediately.
6. On miss: call the original function, store the result, return the response.

## The four conditions for a cache hit

| Condition | Value must match |
|---|---|
| Namespace | `cache_namespace` kwarg (default: `"default"`) |
| Embedding model | `embedding_model_id` from the configured embedder |
| Context hash | SHA-256 of `cache_context` dict |
| Cosine similarity | ≥ `threshold` (default: 0.92) |

## Conditions that prevent a cache hit

| Scenario | Outcome |
|---|---|
| `cache_context` missing | `ValueError` and original function is not called |
| `cache_context` is not a dict | `TypeError` and original function is not called |
| `cache_namespace` is not a string | `TypeError` and original function is not called |
| `stream=True` | Cache bypass, original function is called |
| No user message in `messages` | Cache bypass, original function is called |
| Different namespace | Miss |
| Different context hash | Miss |
| Different embedding model | Entry is invisible, miss |
| Best cosine similarity below threshold | Miss |
| Entry expired by TTL | Entry is ignored/evicted, miss |
| Namespace invalidated | Entry deleted, miss |
| Cache operation error | Fail open, original function is called |

## Embedding model isolation

Each cache entry stores `embedding_model_id`, and lookups only consider entries with the same model identifier as the currently configured embedder. Without this isolation, changing models would compare vectors from incompatible distributions and silently return wrong matches that look valid but are semantically incorrect.

## Fail-open guarantees

If embedding, lookup, or store operations fail, SemanticCache logs and counts the error, then calls the original function as if caching were disabled for that request. This keeps your application behavior available even when the cache path is unhealthy, while preserving observability for incident response.
