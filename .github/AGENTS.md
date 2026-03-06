# Agent Instructions — LLM Semantic Cache

This document is the authoritative briefing for any AI agent working in this repository.
Read it fully before writing any code, creating any file, or making any decision.

---

## What This Project Is

A **Python library** that adds semantic caching in front of any OpenAI-compatible LLM API.

When an application sends a prompt, the library first checks whether a semantically similar prompt has been asked before. If yes, the cached response is returned immediately — no LLM call is made. If no, the request is forwarded to the provider, the response is stored, and future similar prompts benefit from the cache.

This is a single-purpose library. It does one thing and does it well.

---

## What This Project Is Not

Do not implement, suggest, or scaffold any of the following — they are explicitly out of scope and will be rejected:

- A proxy, reverse proxy, or API gateway
- Load balancing across multiple LLM instances
- Admission control or rate limiting
- Request coalescing or deduplication
- Model routing or fallback chains
- Circuit breakers
- A replacement for LiteLLM or any general-purpose gateway
- vLLM-specific integrations or GPU metric polling
- A managed cloud service or hosted version
- Support for non-OpenAI-compatible APIs requiring bespoke client code

These are permanent non-goals, not deferred work.

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

The library is a **thin wrapper**. Every component should be understandable by reading it directly.

---

## Core Technical Components

### 1. Local Embeddings (No External Dependency)

- Use `all-MiniLM-L6-v2` via `sentence-transformers`, running **in-process**
- No calls to any embedding API — that would defeat the purpose
- Target: sub-10ms on CPU for typical prompt lengths, ~80MB model size
- The embedding model must be swappable — users can override the default
- Every cache entry must be tagged with the `embedding_model_id` at write time
- Similarity search only runs against entries sharing the **same model identifier** as the current config — this prevents silent corruption when the model changes

### 2. Similarity Threshold Profiles

Three named profiles ship with the library:

| Profile    | Threshold | Intended For                          |
|------------|-----------|---------------------------------------|
| `strict`   | 0.97      | Code generation, factual Q&A          |
| `balanced` | 0.92      | General assistants, summarization     |
| `loose`    | 0.85      | Customer support, FAQ bots            |

- Users may override with a raw float
- Profiles are a starting point, not a constraint
- The threshold must be configurable per cache instance, not global

### 3. Cache Key Construction

The cache key is **never derived from the prompt embedding alone**. Hidden context — attached documents, conversation history, user state, session variables — affects the expected response. The library must make it easy to:

- Hash contextual inputs alongside the embedding as a deterministic fingerprint
- Scope the cache by namespace (e.g., per session, per document set)

Skipping context in the key must be hard to do accidentally, not the default.

### 4. Namespace-Based Cache Invalidation

- TTL alone is insufficient (a cached answer about a library's API surface becomes wrong after a breaking release)
- Cache entries are tagged with a namespace at write time
- Entire namespaces can be invalidated in a single operation, without scanning unrelated entries

### 5. Pluggable Storage Backends

Two backends ship with the library:

- **In-memory** — zero dependencies, suitable for development and testing, lost on restart
- **Redis** — persistent across restarts, shared across workers and replicas

The storage interface is an abstract base class. Implementations must be swappable without changing the cache logic.

A vector store backend (pgvector, Chroma) is on the roadmap but **not in scope for the initial version**. Do not implement it now.

---

## API Surface

The public interface is intentionally minimal. The primary entry point is `wrap()`, which intercepts calls to any OpenAI-compatible endpoint and applies the cache logic transparently.

The library accepts and returns OpenAI-shaped request and response types (Pydantic models matching the `/v1/chat/completions` contract). It must work with:

- OpenAI
- Anthropic (via adapter)
- vLLM
- Ollama
- Azure OpenAI
- Any service that speaks `/v1/chat/completions`

---

## Instrumentation

The library ships with structured logging and Prometheus metrics. Keep these:

- **Structlog** for structured logging
- **Prometheus** for metrics: cache hit/miss counters, embedding latency histograms, similarity score distributions

Logging and metrics must remain backend-agnostic.

---

## Code Philosophy — Non-Negotiable Rules

Every line of code in this repository must follow these rules. Violations will be caught in review.

### No abstraction without necessity
If a class wraps another class with no added behavior, remove it. If an interface has one implementation and no realistic prospect of a second, the interface is premature. Abstractions are earned, not speculative.

### Functions do one thing
Any function longer than 40 lines is a candidate for decomposition. Any block of code that needs an inline comment to explain *what* it does (not *why*) should be extracted into a named function instead.

### Names explain intent
- `cache.py` contains the cache. ✓
- `utils.py` that contains miscellaneous things. ✗
- If you must open a file to understand what it contains, the name is wrong.

### No clever code
No nested list comprehensions doing three things at once. No chained method calls that read inside-out. No one-liners that sacrifice readability for brevity. A new contributor must be able to read the code without pausing to mentally parse it.

### Tests are documentation
- Every non-trivial behavior has a test that demonstrates it
- New features require tests written first (TDD)
- Bug fixes require a test that reproduces the bug
- The test suite is the most reliable description of what the library does

### Config in one place
No magic numbers in source code. No environment variable checks scattered across modules. All tuneable values live in one configuration layer, documented there.

### Complexity must justify itself
Complexity is only acceptable when it is doing real work that cannot be done more simply. The bar is high and must be explicitly justified in code review.

---

## Testing Standards

The test suite is the most valuable asset in this codebase. It does not shrink.

- **Unit tests** cover all non-trivial behavior
- **Type checking**: strict `mypy`
- **Linting**: `ruff`
- **Cache tests**: use `fakeredis` — no real Redis instance required
- **Async tests**: `pytest-asyncio`
- Test file structure mirrors source structure

Target: a contributor should be able to clone the repo, read a single module, understand it completely, and submit a meaningful change — without needing to trace abstraction layers or ask maintainers for context.

---

## Honest Benchmarking

The benchmark suite measures hit rates across **realistic prompt distributions**, not synthetic ones engineered to show good numbers. Expected hit rates by use case:

| Use Case                        | Expected Hit Rate | Why                                            |
|---------------------------------|-------------------|------------------------------------------------|
| FAQ / support bot               | 40–70%            | High prompt repetition, forgiving similarity   |
| Document summarization pipeline | 20–50%            | Same docs re-processed, template prompts       |
| General chat assistant          | 5–15%             | High prompt diversity, dynamic context         |
| Code generation                 | 3–10%             | Exact problem statements vary, strict threshold|

If semantic caching does not help a given workload, documentation must say so plainly. The library does not oversell itself.

---

## Addendum — Binding Design Decisions (Review Resolutions)

The following decisions were made during plan review and are binding for implementation.

### Redis Scalability Constraint

The Redis backend performs client-side cosine similarity, which scales poorly beyond ~5,000 entries per namespace. Two mitigations:

1. **Document the limit** in the Redis backend docstring and README.
2. **Implement coarse bucket pre-filtering** inside `storage/redis.py`. Embeddings are assigned a bucket key at write time by quantizing the top-k dimensions. At query time, only entries in matching/adjacent buckets are fetched. Falls back to full scan on cold start. This is an internal optimization — the `StorageBackend` ABC does not change.

### Streaming Bypass (v1)

`stream=True` requests bypass the cache entirely in v1. The request is forwarded to the provider unchanged. A structlog event and Prometheus counter (`semantic_cache_stream_bypass_total`) are emitted on every bypass. Document this as a known limitation.

### Fail-Open Resilience

All cache operations inside `wrap()` are guarded with try/except. If any cache operation (lookup, embed, store) raises an exception, the original function is called as if the cache does not exist. Errors are logged via structlog and counted via `semantic_cache_errors_total` (labels: `lookup`, `store`, `embed`). **Invariant:** `wrap()` never raises an exception that the unwrapped function would not have raised.

### Context Serialization

`context.py` includes a canonical serializer (`_canonical_default`) that handles `datetime`, `date`, `UUID`, `bytes`, `set`, `frozenset`, and Pydantic models (v1 and v2). Sets are sorted for deterministic ordering. Unknown types raise `TypeError` with an actionable message. The serializer uses `sort_keys=True` and `ensure_ascii=True` for hash stability.

### Sync/Async Wrapper Separation

`wrap()` is a factory that returns either `_make_async_wrapper` or `_make_sync_wrapper` at decoration time — never a combined wrapper. Detection uses `inspect.unwrap()` before `inspect.iscoroutinefunction()` to handle decorated functions. An explicit `mode` parameter (`"auto"`, `"sync"`, `"async"`) overrides detection when needed. Sync and async wrappers use completely separate code paths — no `asyncio.run()` bridging.

---

## Addendum 2 — Round 2 Review Resolutions

The following decisions were made during the second round of plan review and are binding for implementation. Where they modify Round 1 decisions, that is noted.

### Sync/Async Storage Interface (R2-1)

The `StorageBackend` ABC defines **sync abstract methods** (`store`, `search`, `invalidate_namespace`) as the primary interface. Async variants (`astore`, `asearch`, `ainvalidate_namespace`) have default implementations via `run_in_executor`. Backends override async methods for native async I/O. Sync wrapper calls sync methods directly; async wrapper calls async methods directly. No `asyncio.run()` bridging. `MemoryStorage` overrides async methods to call sync directly (no thread pool — no I/O). `RedisStorage` accepts both `redis.Redis` and `redis.asyncio.Redis` clients and overrides both method sets.

### Bucket Pre-filtering Dropped (R2-2)

**Supersedes Round 1, Issue 1, section 1b.** The "top-4 dims × 8 bins" LSH-style bucket key is mathematically unsound for dense normalized embeddings. It causes silent false negatives. Both backends use brute-force cosine similarity over all entries in the namespace filtered by `embedding_model_id`. For in-memory, this is numpy vectorized. For Redis, this means client-side computation (see next section).

### Redis Data Transfer — Documented Limitation (R2-3)

**Extends Round 1, Issue 1, section 1a.** Redis backend fetches all embeddings per namespace for client-side cosine similarity. This is practical for ≤5,000 entries per namespace. Server-side computation (Lua scripts, RediSearch) is deferred to a future version. A `namespace_size()` method and a structlog warning at a configurable threshold (default: 5,000) are added. Embeddings are fetched via pipeline (`MGET`), not individual calls.

### Redis Namespace SET Tombstone Cleanup (R2-4)

Lazy cleanup during `search()` and `invalidate_namespace()`. When an ID in the namespace SET points to an expired hash, it is removed from the SET in the same pipeline. No separate cleanup job or background task. Cleanup is logged via structlog at debug level.

### CacheEntry Includes prompt_text (R2-5)

`CacheEntry` gains a required `prompt_text: str` field storing the original prompt. Used for debugging and future re-embedding when switching models. Not used in similarity search or cache key construction. Documentation notes the privacy implication of storing plaintext prompts.

---

## Known Risks — Handle These Correctly

### Context sensitivity
Semantically similar prompts may depend on different hidden context. A cache hit on the prompt text alone can return a correct-looking but contextually wrong response. The cache key construction rules above exist to prevent this. Do not simplify or bypass them.

### Embedding drift
Upgrading or swapping the embedding model changes the vector distribution. Old embeddings and new embeddings are not comparable — running both against the same cache produces wrong similarity scores silently (no error, just wrong answers). The `embedding_model_id` tagging on every cache entry is the mitigation. Do not remove or weaken it.

### Unrealistic user expectations
Users may deploy the library on workloads with low natural repetition and conclude it is broken. The documentation and benchmark suite exist to set correct expectations. Do not hide low hit rates in benchmarks.

---

## What to Do When Uncertain

1. **Simpler is correct.** If two approaches work, choose the one that is easier to read and explain.
2. **Scope is fixed.** If a feature is not listed in this document, it is either out of scope or needs explicit discussion before implementation.
3. **Tests first.** If you are unsure whether behavior is correct, write the test that defines correct behavior before implementing it.
4. **Names matter.** If you cannot name a function or module clearly, the design needs rethinking — not a better name.

---

## Project Status

Early development. This is the first clean iteration of this library with a correct, focused scope.

The project is framed as a learning artifact as well as a useful tool. Every component should be understandable end-to-end by a competent engineer reading it cold.
