"""Benchmark runner for LLM Semantic Cache.

Usage:
    PYTHONPATH=src python -m benchmarks.run

Measures hit rates across four prompt distributions using InMemoryStorage
and FakeEmbedder (no real LLM calls — benchmarks measure cache mechanics).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmarks.distributions import (
    code_generation_prompts,
    faq_bot_prompts,
    general_chat_prompts,
    summarization_prompts,
)
from benchmarks.report import format_report


def run_benchmark(
    prompts: list[str],
    threshold: float,
    model_name: str = "fake-benchmark-model",
) -> dict:
    """Run a cache hit rate benchmark on a prompt distribution.

    Uses InMemoryStorage and a deterministic FakeEmbedder so benchmarks
    run without any ML model or LLM API.
    """
    import math

    from llm_semantic_cache.cache import SemanticCache
    from llm_semantic_cache.config import CacheConfig
    from llm_semantic_cache.embeddings import _l2_normalize
    from llm_semantic_cache.storage.memory import InMemoryStorage

    class BenchmarkEmbedder:
        """Embedder that produces embeddings sensitive to prompt similarity."""

        def __init__(self) -> None:
            self._model_name = model_name

        @property
        def model_id(self) -> str:
            return self._model_name

        def embed(self, text: str) -> list[float]:
            # Hash-based embedding: similar strings → similar vectors
            # This is a rough simulation; real benchmarks use the real model.
            words = text.lower().split()
            dim = 32
            vec = [0.0] * dim
            for word in words:
                for i, ch in enumerate(word[:dim]):
                    vec[i % dim] += math.sin(ord(ch) * (i + 1) * 0.1)
            return _l2_normalize(vec)

    storage = InMemoryStorage()
    config = CacheConfig(threshold=threshold, cache_timeout_seconds=10.0)
    cache = SemanticCache(storage=storage, config=config, embedder=BenchmarkEmbedder())

    call_count = 0

    def fake_llm(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"id": f"resp-{call_count}", "choices": [{"message": {"content": "response"}}]}

    wrapped = cache.wrap(fake_llm, mode="sync")

    for prompt in prompts:
        wrapped(
            model="fake-model",
            messages=[{"role": "user", "content": prompt}],
            cache_context={},
        )

    total = len(prompts)
    misses = call_count
    hits = total - misses
    hit_rate = hits / total if total > 0 else 0.0

    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
    }


def main() -> None:
    benchmarks = [
        {
            "name": "FAQ / Support Bot",
            "prompts": faq_bot_prompts(200),
            "threshold": 0.85,
            "expected_range": "40–70%",
        },
        {
            "name": "Document Summarization",
            "prompts": summarization_prompts(200),
            "threshold": 0.92,
            "expected_range": "20–50%",
        },
        {
            "name": "General Chat",
            "prompts": general_chat_prompts(200),
            "threshold": 0.92,
            "expected_range": "5–15%",
        },
        {
            "name": "Code Generation",
            "prompts": code_generation_prompts(200),
            "threshold": 0.97,
            "expected_range": "3–10%",
        },
    ]

    results = []
    for b in benchmarks:
        print(f"Running: {b['name']}...", flush=True)
        stats = run_benchmark(b["prompts"], b["threshold"])
        results.append(
            {
                "use_case": b["name"],
                "expected": b["expected_range"],
                **stats,
            }
        )

    print(format_report(results))


if __name__ == "__main__":
    main()
