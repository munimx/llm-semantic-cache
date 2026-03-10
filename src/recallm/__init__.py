"""recallm — public interface for the llm-semantic-cache library.

Install:  pip install recallm
Import:   from recallm import SemanticCache, CacheConfig, InMemoryStorage

This module re-exports the full public API from the internal
``llm_semantic_cache`` package. All imports, type hints, and usage
should reference ``recallm`` directly.
"""

from llm_semantic_cache import (
    CacheConfig,
    CacheContext,
    CacheEntry,
    CacheStats,
    InMemoryStorage,
    SearchResult,
    SemanticCache,
    StorageBackend,
    ThreadSafeInMemoryStorage,
)

__all__ = [
    "SemanticCache",
    "CacheStats",
    "CacheConfig",
    "CacheEntry",
    "SearchResult",
    "StorageBackend",
    "InMemoryStorage",
    "ThreadSafeInMemoryStorage",
    "CacheContext",
]

try:
    from llm_semantic_cache.storage.redis import RedisStorage  # noqa: F401
    __all__.append("RedisStorage")
except ImportError:
    pass
