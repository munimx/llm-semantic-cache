"""Storage backends for SemanticCache."""
from llm_semantic_cache.storage.base import CacheEntry, StorageBackend
from llm_semantic_cache.storage.memory import InMemoryStorage
from llm_semantic_cache.storage.redis import RedisStorage

__all__ = ["CacheEntry", "StorageBackend", "InMemoryStorage", "RedisStorage"]
