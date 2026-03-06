# Cache Decision Logic

This document is the complete specification of when the cache will and will not return a hit.

---

## 1. The Complete Decision Path

Every call to a `wrap()`-wrapped function follows this path in order:

1. **`cache_context` check** — Is `cache_context` present in kwargs?
   - **No** → `ValueError` raised. The wrapped function is **not** called.
   - **Yes, but not a dict** → `TypeError` raised. This is a programmer error and is **not** caught by fail-open.
   - **Yes, valid dict** → continue.
   - If `cache_namespace` is provided but is not a `str` → `TypeError` raised. This is a programmer error and is **not** caught by fail-open.

2. **`stream` check** — Is `stream=True`?
   - **Yes** → bypass cache entirely. Log `cache.stream_bypass`. Increment `semantic_cache_stream_bypass_total`. Call original function. Return response.
   - **No** → continue.

3. **Prompt extraction** — Extract last user message from `messages` list (reverse iteration).
   - **No user message found** → bypass cache. Log `cache.no_user_message_bypass`. Call original function. Return response.
   - **User message found** → continue.

4. **Context hashing** — Compute SHA-256 of `cache_context` dict with deterministic serialization.
   - **Hash fails** (unserializable value) → fail-open. Log `cache.lookup_params_failed`. Increment `semantic_cache_errors_total{operation="lookup"}`. Call original function. Return response.
   - **Hash succeeds** → continue.

5. **Embedding** — Compute embedding vector of prompt text using configured embedding model.
   - **Embedding fails** → fail-open. Log `cache.embed_failed`. Increment `semantic_cache_errors_total{operation="embed"}`. Call original function. Return response.
   - **Embedding succeeds** → measure latency in `semantic_cache_embedding_latency_seconds`. Continue.

6. **Storage search** — Query storage backend for best match. Candidates are filtered by:
   - `namespace` (exact match)
   - `embedding_model_id` (exact match — entries from other models are invisible)
   - `context_hash` (exact match)
   - Non-expired (TTL check; entries past their TTL are evicted)
   - Then: cosine similarity computed against all remaining candidates.
   - **Search fails or times out** → fail-open. Log `cache.lookup_failed`. Increment `semantic_cache_errors_total{operation="lookup"}`. Call original function.

7. **Threshold check** — Is `best_score >= threshold`?
   - **Yes → CACHE HIT.** Log `cache.hit` (with `similarity_score`, `threshold`, `embedding_model`, `namespace`). Increment `semantic_cache_hits_total`. Record score in `semantic_cache_similarity_score`. Return cached response. Original function is **not** called.
   - **No → CACHE MISS.** Call original function. Store embedding + response. Log `cache.miss` (with `best_score`, `threshold`, `embedding_model`, `namespace`). Increment `semantic_cache_misses_total`. Record score in `semantic_cache_similarity_score`. Return fresh response.

---

## 2. The Four Conditions for a Cache Hit

A hit occurs if and only if **all** of the following are true simultaneously:

1. **Semantic similarity** — The prompt embedding's cosine similarity with a stored entry's embedding is ≥ the configured threshold (profile name or raw float).
2. **Context match** — The SHA-256 hash of `cache_context` matches the stored entry's `context_hash` — meaning the exact same contextual inputs were present at write time.
3. **Embedding model match** — The stored entry was produced by the same embedding model (identified by `embedding_model_id`) as the current query. Entries from other models are invisible.
4. **Namespace + validity** — The entry is in the same namespace, has not exceeded its TTL, and has not been explicitly invalidated via `invalidate_namespace()`.

---

## 3. Conditions That Prevent a Cache Hit

| Condition | Outcome |
|---|---|
| `cache_context` not passed | `ValueError` raised — this is an error, not a bypass |
| `cache_context` wrong type (non-dict) | `TypeError` raised — programmer error, not fail-open |
| `cache_namespace` wrong type (non-str) | `TypeError` raised — programmer error, not fail-open |
| `stream=True` | Always bypass — streaming responses are not cacheable in v1 |
| No `role=user` message in `messages` | Always bypass — nothing to embed |
| Different `cache_context` values | Different SHA-256 hash → miss |
| Different `cache_namespace` | Entries are fully isolated per namespace |
| Different embedding model | Entries produced by other models are invisible |
| Similarity score below threshold | Not similar enough → miss |
| Entry TTL exceeded | Evicted during search, invisible |
| Namespace explicitly invalidated | All entries deleted by `invalidate_namespace()` |
| Any cache operation exception | Fail-open — original function called as if no cache exists |
| Async search timeout | Fail-open — configurable via `cache_timeout_seconds` |

---

## 4. Fail-Open Guarantees

These invariants hold unconditionally:

- `wrap()` **never raises an exception** that the unwrapped function would not have raised — with deliberate exceptions for invalid lookup kwargs: `ValueError` for missing `cache_context`, `TypeError` for non-dict `cache_context`, and `TypeError` for non-str `cache_namespace`. These are intentional and prevent silent context/key misuse that would produce wrong cache behavior.
- All cache operations (embed, search, store) are wrapped in `try/except`.
- Errors are logged via structlog and counted via `semantic_cache_errors_total`.
- On any failure, the original function is called as if the cache does not exist. The caller receives a valid response regardless of cache health.

---

## 5. Observability Signals Reference

| Decision Point | Structlog Event | Prometheus Metric | Fires When |
|---|---|---|---|
| Cache hit | `cache.hit` | `semantic_cache_hits_total` | Stored response returned |
| Cache miss | `cache.miss` | `semantic_cache_misses_total` | No match, forwarded to provider |
| Stream bypass | `cache.stream_bypass` | `semantic_cache_stream_bypass_total` | `stream=True` in request |
| No user message | `cache.no_user_message_bypass` | — | No `role=user` message found |
| Lookup failure | `cache.lookup_failed` | `semantic_cache_errors_total{operation="lookup"}` | Storage search raised exception |
| Store failure | `cache.store_failed` | `semantic_cache_errors_total{operation="store"}` | Storage write raised exception |
| Embed failure | `cache.embed_failed` | `semantic_cache_errors_total{operation="embed"}` | Embedding computation failed |
| Namespace too large | `cache.namespace_too_large` | — | Namespace exceeds 5,000 entries |
| Similarity score | — | `semantic_cache_similarity_score` | Every lookup with candidates |
| Embedding latency | — | `semantic_cache_embedding_latency_seconds` | Every embedding computation |
