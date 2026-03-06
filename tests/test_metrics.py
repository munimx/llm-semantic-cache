"""Tests for metrics helpers and logging configuration."""
from __future__ import annotations

import time

import pytest
import structlog

from llm_semantic_cache import metrics


@pytest.fixture(autouse=True)
def reset_structlog_after_test() -> None:
    yield
    structlog.reset_defaults()


def test_configure_logging_does_not_raise() -> None:
    metrics.configure_logging()


def test_configure_logging_json_format_does_not_raise() -> None:
    metrics.configure_logging(json_format=True)


def test_record_hit_does_not_raise_without_prometheus() -> None:
    if metrics._PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client is available")
    metrics.record_hit("test")


def test_record_miss_does_not_raise() -> None:
    metrics.record_miss("test")


def test_record_stream_bypass_does_not_raise() -> None:
    metrics.record_stream_bypass("test")


def test_record_cache_error_does_not_raise() -> None:
    metrics.record_cache_error("lookup")


def test_record_similarity_score_does_not_raise() -> None:
    metrics.record_similarity_score(0.95)


def test_measure_embedding_latency_context_manager_does_not_raise() -> None:
    with metrics.measure_embedding_latency():
        time.sleep(0.001)


def test_measure_embedding_latency_does_not_suppress_exceptions() -> None:
    with pytest.raises(RuntimeError, match="boom"), metrics.measure_embedding_latency():
        raise RuntimeError("boom")


def test_record_hit_increments_counter_when_prometheus_available() -> None:
    pytest.importorskip("prometheus_client")
    if not metrics._PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus metrics are unavailable")
    counter = metrics._CACHE_HITS.labels(namespace="metrics-test")
    before = counter._value.get()
    metrics.record_hit("metrics-test")
    after = counter._value.get()
    assert after == before + 1


def test_record_miss_increments_counter_when_prometheus_available() -> None:
    pytest.importorskip("prometheus_client")
    if not metrics._PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus metrics are unavailable")
    counter = metrics._CACHE_MISSES.labels(namespace="metrics-test")
    before = counter._value.get()
    metrics.record_miss("metrics-test")
    after = counter._value.get()
    assert after == before + 1


def test_record_stream_bypass_increments_counter_with_namespace() -> None:
    pytest.importorskip("prometheus_client")
    if not metrics._PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus metrics are unavailable")
    counter = metrics._STREAM_BYPASSES.labels(namespace="bypass-test")
    before = counter._value.get()
    metrics.record_stream_bypass("bypass-test")
    after = counter._value.get()
    assert after == before + 1
