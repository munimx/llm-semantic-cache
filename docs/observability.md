# Observability

## Prometheus metrics

| Metric | Type | Labels | Measures |
|---|---|---|---|
| `semantic_cache_hits_total` | Counter | `namespace` | Cache hits |
| `semantic_cache_misses_total` | Counter | `namespace` | Cache misses |
| `semantic_cache_errors_total` | Counter | `operation` | Failed cache ops |
| `semantic_cache_embedding_duration_seconds` | Histogram | — | Embedding latency |
| `semantic_cache_similarity_score` | Histogram | — | Similarity score distribution |
| `semantic_cache_stream_bypass_total` | Counter | `namespace` | `stream=True` bypasses |

## Structlog events

| Event name | Level | Key fields |
|---|---|---|
| `cache.hit` | `info` | `namespace`, `best_score`, `threshold`, `embedding_model` |
| `cache.miss` | `info` | `namespace`, `best_score`, `threshold`, `embedding_model` |
| `cache.stream_bypass` | `info` | `namespace`, `embedding_model` |
| `cache.no_user_message_bypass` | `info` | `namespace`, `embedding_model` |
| `cache.embed_failed` | `error` | `namespace`, `error` |
| `cache.lookup_failed` | `error` | `namespace`, `error` |
| `cache.lookup_params_failed` | `error` | `namespace`, `error` |
| `cache.store_failed` | `error` | `error` |
| `cache.namespace_too_large` | `warning` | `namespace`, `size`, `threshold` |

## Grafana dashboard

Import path: Grafana → Dashboards → Import → upload `dashboards/semantic-cache.json`.

The dashboard includes eight panels:

1. **Cache Hit Rate** — 5-minute hit-rate ratio for selected namespaces.
2. **Cache Hits/Misses** — hit and miss request rates over time.
3. **Embedding Latency** — p95 embedding latency from histogram buckets.
4. **Stream Bypass Rate** — rate of requests with `stream=True`.
5. **Cache Errors** — error rate grouped by `operation`.
6. **Similarity Score Distribution** — bucketed similarity scores over time window.
7. **Namespace Entry Counts** — per-namespace current entry count.
8. **Total Cache Entries** — sum of entries across namespaces.

Use `$namespace` to filter all panels to one or more namespaces or view aggregate behavior with “All”.

## Example log output

Console (development):

```text
2026-01-01T12:00:00Z [info     ] cache.hit namespace=default best_score=0.96 threshold=0.92 embedding_model=all-MiniLM-L6-v2
2026-01-01T12:00:02Z [info     ] cache.miss namespace=default best_score=0.81 threshold=0.92 embedding_model=all-MiniLM-L6-v2
2026-01-01T12:00:03Z [info     ] cache.stream_bypass namespace=default embedding_model=all-MiniLM-L6-v2
2026-01-01T12:00:04Z [error    ] cache.lookup_failed namespace=default error='redis timeout'
```

JSON (production):

```json
{"event":"cache.hit","level":"info","namespace":"default","best_score":0.96,"threshold":0.92,"embedding_model":"all-MiniLM-L6-v2","timestamp":"2026-01-01T12:00:00Z"}
{"event":"cache.miss","level":"info","namespace":"default","best_score":0.81,"threshold":0.92,"embedding_model":"all-MiniLM-L6-v2","timestamp":"2026-01-01T12:00:02Z"}
{"event":"cache.stream_bypass","level":"info","namespace":"default","embedding_model":"all-MiniLM-L6-v2","timestamp":"2026-01-01T12:00:03Z"}
{"event":"cache.lookup_failed","level":"error","namespace":"default","error":"redis timeout","timestamp":"2026-01-01T12:00:04Z"}
```
