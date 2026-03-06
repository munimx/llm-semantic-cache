"""llm-semantic-cache — semantic caching for OpenAI-compatible LLM APIs."""
from typing import Any

from llm_semantic_cache.cache import SemanticCache
from llm_semantic_cache.config import CacheConfig
from llm_semantic_cache.storage.base import CacheEntry, SearchResult, StorageBackend
from llm_semantic_cache.storage.memory import InMemoryStorage

CacheContext = dict[str, Any]
"""Type alias for cache context dicts passed to wrapped functions.

Use this type in your application code to document that a dict is
intended as a cache context:

    from llm_semantic_cache import wrap, CacheContext

    ctx: CacheContext = {"user_id": "u123", "document_id": "d456"}
    response = client.chat.completions.create(messages=..., cache_context=ctx)
"""

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
    from llm_semantic_cache.storage.redis import RedisStorage
    __all__.append("RedisStorage")
except ImportError:
    pass
