# LLM Semantic Cache

A Python library that adds **semantic caching** in front of any OpenAI-compatible LLM API.

When your application sends a prompt, instead of always forwarding it to the model, the library first checks whether a semantically similar prompt has been asked before. If yes, the cached response is returned immediately. If no, the request is forwarded, the response is stored, and future similar prompts benefit.

---

## The Problem

Exact-match caching (hashing the prompt string) is nearly useless for LLM workloads — prompts are dynamic by nature. Semantic caching, using embeddings to find near-matches, is the approach that actually works. LLM inference costs are typically the first thing teams optimize after shipping. A semantic cache directly cuts API bills with a concrete, measurable result.

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

---

## Design Decisions

### Local embeddings — no external dependency

The library uses `all-MiniLM-L6-v2` via `sentence-transformers`, running in-process. No embedding API calls. Small (80MB), fast (sub-10ms on CPU), good enough for semantic similarity on short to medium length text.

### Named threshold profiles

| Profile | Threshold | Intended for |
|---------|-----------|--------------|
| `strict` | 0.97 | Code generation, factual Q&A |
| `balanced` | 0.92 | General assistants, summarization |
| `loose` | 0.85 | Customer support, FAQ bots |

Override with a raw float for full control.

### Namespace-based cache invalidation

TTL alone is not sufficient. Cache entries are tagged at write time; entire namespaces can be invalidated in one operation without scanning or expiring unrelated entries.

### Pluggable storage

- **In-memory** — zero dependencies, good for development, lost on restart
- **Redis** — persistent, shared across workers and replicas

---

## What This Is Not

- Not a proxy, load balancer, or rate limiter
- Not a replacement for LiteLLM
- Not vLLM-specific

---

## Status

Early development. See [PROJECT_DIRECTION2.md](https://github.com/munimx/LLM-Inference-Optimization-Engine/blob/main/PROJECT_DIRECTION2.md) in the previous repo for the full project rationale and scope.
