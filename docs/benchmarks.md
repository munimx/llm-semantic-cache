# Benchmarks

## Expected hit rates

| Use case | Expected hit rate | Why |
|---|---|---|
| FAQ / support bot | 40–70% | Users repeat intent with small wording changes, so semantic similarity captures many near-duplicates. |
| Document summarization | 20–50% | Teams often reprocess the same corpus with templated prompts, creating moderate repetition. |
| General chat assistant | 5–15% | Open-ended chats vary heavily in user intent, so repeated semantic intent is limited. |
| Code generation | 3–10% | Requests are precise and threshold is strict, so near-matches are less likely to qualify as safe hits. |

These ranges are realistic workload expectations, not synthetic “best-case” numbers. Prompt distributions in this project intentionally include variation and noise so the benchmark reflects production-like repetition, not perfect duplicates.

## How to run

```bash
PYTHONPATH=src python -m benchmarks.run
```

## How to read results

Output is a markdown table with:

- **Use Case**: workload profile being simulated.
- **Expected Hit Rate**: target range based on typical production behavior.
- **Actual Hit Rate**: measured hits/total calls for this run.
- **Hits / Misses / Total**: raw counts behind the percentage.

“Good” results are when actual hit rate lands inside or near the expected range for that workload. If a run is far below range, tune threshold and `cache_context`; if far above range, validate answer quality to ensure you did not over-loosen matching.

## Benchmark limitations

CI benchmarks use a deterministic fake embedder, so hit rates differ from real embedding models. Always run with a real embedder before drawing production conclusions about quality, latency, or expected savings.

## Running with a real embedder

Swap in a real embedder when constructing `SemanticCache` in benchmark code:

```python
from llm_semantic_cache import CacheConfig, SemanticCache
from llm_semantic_cache.embeddings import FastEmbedEmbedder, SentenceTransformerEmbedder
from llm_semantic_cache.storage.memory import InMemoryStorage

cache_fastembed = SemanticCache(
    storage=InMemoryStorage(),
    config=CacheConfig(threshold="balanced"),
    embedder=FastEmbedEmbedder("all-MiniLM-L6-v2"),
)

cache_torch = SemanticCache(
    storage=InMemoryStorage(),
    config=CacheConfig(threshold="balanced"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
```
