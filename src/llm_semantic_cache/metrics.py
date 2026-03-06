"""Prometheus metrics and structlog configuration for SemanticCache."""
from __future__ import annotations

import contextlib
import time as _time
from collections.abc import Generator
from typing import Any

import structlog

try:
    from prometheus_client import Counter, Histogram

    _CACHE_HITS = Counter(
        "semantic_cache_hits_total",
        "Number of cache hits",
        ["namespace"],
    )
    _CACHE_MISSES = Counter(
        "semantic_cache_misses_total",
        "Number of cache misses",
        ["namespace"],
    )
    _STREAM_BYPASSES = Counter(
        "semantic_cache_stream_bypass_total",
        "Number of stream=True requests that bypassed the cache",
        ["namespace"],
    )
    _CACHE_ERRORS = Counter(
        "semantic_cache_errors_total",
        "Number of cache operation failures",
        ["operation"],
    )
    _EMBEDDING_LATENCY = Histogram(
        "semantic_cache_embedding_latency_seconds",
        "Time spent computing embeddings",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    _SIMILARITY_SCORES = Histogram(
        "semantic_cache_similarity_score",
        "Distribution of cosine similarity scores for cache lookups",
        buckets=[0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure structlog for SemanticCache."""
    import logging

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def record_hit(namespace: str) -> None:
    """Increment cache hit counter."""
    if _PROMETHEUS_AVAILABLE:
        _CACHE_HITS.labels(namespace=namespace).inc()


def record_miss(namespace: str) -> None:
    """Increment cache miss counter."""
    if _PROMETHEUS_AVAILABLE:
        _CACHE_MISSES.labels(namespace=namespace).inc()


def record_stream_bypass(namespace: str) -> None:
    """Increment stream bypass counter for a namespace."""
    if _PROMETHEUS_AVAILABLE:
        _STREAM_BYPASSES.labels(namespace=namespace).inc()


def record_cache_error(operation: str) -> None:
    """Increment cache error counter for a given operation."""
    if _PROMETHEUS_AVAILABLE:
        _CACHE_ERRORS.labels(operation=operation).inc()


def record_similarity_score(score: float) -> None:
    """Observe a cosine similarity score in the histogram."""
    if _PROMETHEUS_AVAILABLE:
        _SIMILARITY_SCORES.observe(score)


@contextlib.contextmanager
def measure_embedding_latency() -> Generator[None, None, None]:
    """Context manager to measure and record embedding latency."""
    start = _time.perf_counter()
    try:
        yield
    finally:
        elapsed = _time.perf_counter() - start
        if _PROMETHEUS_AVAILABLE:
            _EMBEDDING_LATENCY.observe(elapsed)
