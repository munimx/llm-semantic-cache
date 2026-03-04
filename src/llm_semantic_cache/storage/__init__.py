"""Storage backends for SemanticCache."""
from llm_semantic_cache.storage.base import CacheEntry, StorageBackend
from llm_semantic_cache.storage.memory import InMemoryStorage

__all__ = ["CacheEntry", "StorageBackend", "InMemoryStorage"]
