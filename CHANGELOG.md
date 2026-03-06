# Changelog

All notable changes to Recallm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — YYYY-MM-DD

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
