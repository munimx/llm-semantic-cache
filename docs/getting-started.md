# Getting started

## Installation

Install Recallm with `pip install recallm` for the core package and the default `FastEmbedEmbedder`, which uses ONNX and keeps footprint small for local development. Add `pip install "recallm[torch]"` when you want `SentenceTransformerEmbedder`, which gives you a PyTorch-backed option with broader model ecosystem support. Add `pip install "recallm[redis]"` when you need a persistent cache shared across workers or replicas.

## Your first cache

```python
from llm_semantic_cache import CacheConfig, InMemoryStorage, SemanticCache

cache = SemanticCache(
    storage=InMemoryStorage(),
    config=CacheConfig(threshold="balanced", default_namespace="default"),
)

def fake_llm(**kwargs):
    return {"choices": [{"message": {"content": "Cached response"}}]}

cached = cache.wrap(fake_llm, mode="sync")

cached(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this changelog"}],
    cache_context={"document_id": "release-notes-v1"},
)
```

`storage` chooses where entries live, `threshold` controls similarity strictness, and `default_namespace` scopes entries when you do not pass `cache_namespace` per call. On a miss, Recallm embeds the prompt, searches candidates, misses, calls your LLM function, then stores the response and embedding metadata. On a hit, Recallm embeds the prompt, searches candidates, finds a match above threshold, and returns the cached response immediately.

## Understanding `cache_context`

`cache_context` is required because semantically similar prompts can still need different answers based on hidden state like user, document, or session; if you omit it, you risk incorrect hits. Treat it as part of correctness, not an optional optimization knob.

```python
# Good: context that affects the answer
cache_context={"user_id": user.id, "document_id": doc.id}

# Bad: context that doesn't affect the answer
cache_context={"request_id": str(uuid4())}  # unique every call = 0% hit rate

# Wrong: skipping context that does affect the answer
cache_context={}  # answer depends on user_id → wrong cache hits
```

## Choosing a threshold

Use `strict` (`0.97`) for code generation or factual Q&A where false positives are costly, `balanced` (`0.92`) for general assistants and summarization workloads, and `loose` (`0.85`) for repetitive support-style prompts where recall matters most. Start with `balanced`, measure hit rate and quality on your own traffic, then increase threshold to reduce wrong hits or decrease it to improve hit rate.

## Using Redis

```python
# Async-only (most common in FastAPI/Starlette)
import redis.asyncio as redis
from llm_semantic_cache import RedisStorage, SemanticCache

async_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
cache = SemanticCache(storage=RedisStorage(client=async_client))
```

```python
# Sync + async clients
import redis
import redis.asyncio as redis_async
from llm_semantic_cache import RedisStorage, SemanticCache

async_client = redis_async.Redis(host="localhost", port=6379, decode_responses=False)
sync_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
cache = SemanticCache(storage=RedisStorage(client=async_client, sync_client=sync_client))
```

RedisStorage fetches candidate embeddings into Python for cosine similarity, so performance is best below about 5,000 entries per namespace; above that it still works but lookup latency climbs. Partition large workloads into multiple namespaces (for example by tenant, corpus, or time window) to keep each namespace small.

## Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Missing `cache_context` | `ValueError` on every call | Pass `cache_context={}` |
| Unique value in `cache_context` | 0% hit rate | Only include context that affects the answer |
| Switching embedding models without clearing | Wrong hits (silent) | Call `invalidate_namespace()` after model change |
| Using `stream=True` | Cache always bypassed | Use non-streaming for cacheable requests |
