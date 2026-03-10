from dataclasses import FrozenInstanceError

import pytest

from llm_semantic_cache.config import CacheConfig, resolve_threshold


def test_resolve_threshold_from_named_profile_strict() -> None:
    assert resolve_threshold("strict") == 0.97


def test_resolve_threshold_from_named_profile_balanced() -> None:
    assert resolve_threshold("balanced") == 0.92


def test_resolve_threshold_from_named_profile_loose() -> None:
    assert resolve_threshold("loose") == 0.85


def test_resolve_threshold_from_raw_float() -> None:
    assert resolve_threshold(0.5) == 0.5


def test_resolve_threshold_rejects_out_of_range_float() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        resolve_threshold(1.2)


def test_resolve_threshold_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown threshold profile"):
        resolve_threshold("unknown")


def test_cache_config_defaults() -> None:
    config = CacheConfig()
    assert config.threshold == "balanced"
    assert config.default_namespace == "default"
    assert config.default_ttl is None
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.cache_timeout_seconds == 0.2


def test_cache_config_is_frozen() -> None:
    config = CacheConfig()
    with pytest.raises(FrozenInstanceError):
        config.threshold = "strict"  # type: ignore[misc]


def test_cache_config_resolved_threshold_uses_profile() -> None:
    config = CacheConfig(threshold="strict")
    assert config.resolved_threshold() == 0.97
