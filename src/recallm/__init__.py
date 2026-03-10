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
    InMemoryStorage,
    SearchResult,
    SemanticCache,
    StorageBackend,
)

__all__ = [
    "SemanticCache",
    "CacheConfig",
    "CacheEntry",
    "SearchResult",
    "StorageBackend",
    "InMemoryStorage",
    "CacheContext",
]

try:
    from llm_semantic_cache.storage.redis import RedisStorage  # noqa: F401
    __all__.append("RedisStorage")
except ImportError:
    pass
