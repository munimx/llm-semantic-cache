# Recallm
Semantic caching for LLMs. Ask once, recall forever.

![PyPI](https://img.shields.io/pypi/v/recallm) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![MIT License](https://img.shields.io/badge/license-MIT-green) ![CI](https://img.shields.io/github/actions/workflow/status/munimx/recallm/ci.yml?label=CI)

Exact-match caching is useless for LLMs — two users asking the same question in slightly different words both pay the full API cost. Recallm uses sentence embeddings to find near-matches and return cached responses instantly. The result: lower API costs, reduced latency, and no changes to your existing LLM client code.

## Install

```bash
pip install recallm
pip install "recallm[redis]"   # persistent cache, shared across workers
pip install "recallm[torch]"   # sentence-transformers embedder (700MB, PyTorch)
```

## Quickstart

```python
from llm_semantic_cache import CacheConfig, InMemoryStorage, SemanticCache

storage = InMemoryStorage()
cache = SemanticCache(storage=storage, config=CacheConfig(threshold="balanced"))

def fake_llm(**kwargs):
    return {"id": "resp-1", "choices": [{"message": {"content": "Paris"}}]}

cached = cache.wrap(fake_llm, mode="sync")

request = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "cache_context": {"user_id": "u-1", "document_id": "geo-v1"},
}

first = cached(**request)   # miss: calls fake_llm and stores response
second = cached(**request)  # hit: returns cached response
print(first["choices"][0]["message"]["content"], second["choices"][0]["message"]["content"])
```

> **Deployment note:** `SemanticCache(...)` loads the embedding model synchronously.
> In async frameworks (FastAPI, etc.), use `await cache.async_warmup()` during startup
> instead of relying on the constructor — see [getting started](docs/getting-started.md).

| Use case | Expected hit rate | Why |
|---|---|---|
| FAQ / support bot | 40–70% | High repetition, forgiving similarity |
| Document summarization | 20–50% | Same docs re-processed, template prompts |
| General chat assistant | 5–15% | High diversity, dynamic context |
| Code generation | 3–10% | Exact problem statements vary, strict threshold |

## Known limitations

- `stream=True` bypasses the cache entirely
- Redis backend is not suitable for namespaces > 5,000 entries without partitioning
- Sync callers using `RedisStorage` have no timeout protection (v0.1.0)

[Full docs](https://recallm.dev) · [Contributing](CONTRIBUTING.md) · [MIT License](LICENSE)
