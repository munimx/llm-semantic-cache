---
hide:
  - navigation
  - toc
---

<div class="hero-grid">
<div class="hero-content">

<span class="hero-eyebrow">Python Library · MIT License · v0.1.0</span>

<h1 class="hero-title">Ask once,<br>recall forever.</h1>

<p class="hero-subtitle">Stop paying for identical prompts. Recallm wraps your OpenAI-compatible client to return instant cached responses for semantically similar queries — no proxy, no infrastructure changes.</p>

<div class="hero-actions">
<button class="hero-install" onclick="navigator.clipboard.writeText('pip install recallm').then(()=>{var b=this.querySelector('.hero-copy-btn');b.textContent='✓';setTimeout(()=>{b.textContent='⧉'},1500)})"><span class="hero-install-icon">$</span> pip install recallm<span class="hero-copy-btn" title="Copy">⧉</span></button>
<a href="getting-started/" class="md-button md-button--primary">Get started →</a>
</div>

</div>
<div class="hero-code-panel">
<div class="hero-code-titlebar">
<span class="hero-code-dot hero-code-dot--red"></span>
<span class="hero-code-dot hero-code-dot--yellow"></span>
<span class="hero-code-dot hero-code-dot--green"></span>
<span class="hero-code-filename">app.py</span>
</div>

```python
from openai import OpenAI
from llm_semantic_cache import (
    SemanticCache, CacheConfig, InMemoryStorage
)

client = OpenAI()
cache = SemanticCache(
    storage=InMemoryStorage(),
    config=CacheConfig(threshold=0.92),
)
create = cache.wrap(client.chat.completions.create)

# First call — LLM is invoked, result cached
response = create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Capital of France?"}],
    cache_context={"user_id": "u123"},
)

# Semantically equivalent — cache hit, 0ms LLM latency
response = create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    cache_context={"user_id": "u123"},
)
```

</div>
</div>

---

<span class="section-label">How it works</span>

<div class="steps-grid">
<div class="step-card">
<div class="step-number">Step 01</div>
<h3>Intercept</h3>
<p>You wrap your client's <code>create</code> method once. Recallm intercepts every request before it leaves your application — no proxy servers, no infrastructure changes, no new services to run.</p>
</div>
<div class="step-card">
<div class="step-number">Step 02</div>
<h3>Embed &amp; search</h3>
<p>The prompt is converted to a vector locally using a fast ONNX model (~20MB, sub-10ms on CPU). We search your storage backend for entries with cosine similarity above your threshold.</p>
</div>
<div class="step-card">
<div class="step-number">Step 03</div>
<h3>Recall or forward</h3>
<p>Hit: the cached response is returned instantly. Miss: the original call proceeds, the response is stored, and future similar prompts benefit from the cache. You always get a response.</p>
</div>
</div>

---

<span class="section-label">By the numbers</span>

<div class="stats-strip">
<div class="stat-item">
<span class="stat-number">&lt;10ms</span>
<span class="stat-label">added latency per lookup<br>on CPU</span>
</div>
<div class="stat-item">
<span class="stat-number">0</span>
<span class="stat-label">external services required<br>runs entirely in-process</span>
</div>
<div class="stat-item">
<span class="stat-number">40–70%</span>
<span class="stat-label">cost reduction for<br>support &amp; FAQ workloads</span>
</div>
</div>

---

<span class="section-label">Honest hit rates</span>

Semantic caching is not magic. Hit rates depend entirely on how repetitive your workload is.

| Use case | Expected hit rate | Why |
|---|---|---|
| FAQ / support bot | 40–70% | High prompt repetition, forgiving similarity |
| Document summarization | 20–50% | Same docs re-processed, template prompts |
| General chat assistant | 5–15% | High diversity, dynamic context |
| Code generation | 3–10% | Exact problem statements vary, strict threshold |

If Recallm doesn't help your workload, the [benchmarks page](benchmarks.md) will tell you why before you ship it.

---

<span class="section-label">Known limitations</span>

- `stream=True` bypasses the cache entirely — streaming responses are not cacheable in v0.1.0
- Redis backend is not suitable for namespaces > 5,000 entries without partitioning
- Sync callers using `RedisStorage` have no timeout protection in v0.1.0

---

**[Getting started](getting-started.md)** · **[Configuration](configuration.md)** · **[Storage backends](storage-backends.md)** · **[GitHub](https://github.com/munimx/recallm)**

