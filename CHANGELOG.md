# Changelog

All notable changes to Recallm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-03-10

### Added
- `SemanticCache.stats()` returns a `CacheStats` dataclass with `hits`, `misses`, `hit_rate`, `avg_similarity`, and `namespace_sizes` for development-time visibility into cache behaviour
- `ThreadSafeInMemoryStorage` — RLock-protected drop-in replacement for `InMemoryStorage`, safe for multi-threaded and async-framework use
- `recallm` top-level package: `pip install recallm` now maps directly to `from recallm import ...`, eliminating the install/import name mismatch

### Changed
- Default embedding model changed from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` for improved first-run reliability
- Default `cache_timeout_seconds` raised from `0.05` to `0.2` (200 ms) to reduce silent cold-start bypasses
- Cache timeout now emits a structlog `warning` (`cache.timeout_exceeded`) with `elapsed_ms`, `timeout_ms`, and `action=bypass` instead of a silent error

## [0.1.0] — 2026-03-06

### Added
- `SemanticCache.wrap()` with sync and async support
- `InMemoryStorage` — zero-dependency in-process backend
- `RedisStorage` — persistent backend with lazy tombstone cleanup
- `FastEmbedEmbedder` — ONNX-based, ~20MB default embedder
- `SentenceTransformerEmbedder` — optional torch-based embedder
- Three similarity threshold profiles: `strict` (0.97), `balanced` (0.92), `loose` (0.85)
- Namespace-based cache invalidation
- TTL support on cache entries
- Prometheus metrics: hits, misses, errors, embedding latency, similarity scores, stream bypass
- Structlog structured logging on all cache events with rich operational fields
- `stream=True` bypass with per-namespace counter
- Fail-open on all cache operation failures
- `CacheContext` type alias for type-checking convenience
- `SemanticCache.async_warmup()` for non-blocking model load in async frameworks
- Grafana dashboard (8 panels, `$namespace` variable)
- Benchmark suite with four realistic prompt distributions
