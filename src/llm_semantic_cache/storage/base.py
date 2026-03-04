"""StorageBackend ABC and CacheEntry — the storage contract."""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A single cached response with its embedding and metadata."""

    embedding: list[float]
    """The embedding vector of the cached prompt."""

    prompt_text: str
    """The original prompt text (last user message). Stored for debugging and re-embedding."""

    context_hash: str
    """SHA-256 hash of the cache_context dict. Part of the cache key."""

    namespace: str
    """Namespace this entry belongs to. Used for scoped invalidation."""

    embedding_model_id: str
    """Identifier of the embedding model used. Entries are only matched against
    entries sharing the same model_id."""

    response: dict[str, Any]
    """The cached response (ChatCompletionResponse serialized to dict)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this entry. Auto-generated."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when this entry was created."""

    ttl: float | None = None
    """Time-to-live in seconds. None means no expiration."""

    def is_expired(self) -> bool:
        """Return True if this entry has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class StorageBackend(ABC):
    """Abstract base class for SemanticCache storage backends.

    All cache operations are available in both sync and async forms.
    Subclasses must implement the sync abstract methods. The default async
    implementations delegate to sync via asyncio.to_thread(). Production
    backends (e.g., RedisStorage) override the async methods with native
    async I/O.
    """

    @abstractmethod
    def store(self, entry: CacheEntry) -> None:
        """Store a cache entry."""

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> CacheEntry | None:
        """Find the best matching cache entry above the similarity threshold.

        Only considers entries that match:
        - namespace (exact)
        - embedding_model_id (exact)
        - context_hash (exact)
        - cosine similarity >= threshold

        Returns the highest-similarity match, or None if no match is found.
        """

    @abstractmethod
    def invalidate_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace. Returns the count deleted."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all entries in all namespaces."""

    async def astore(self, entry: CacheEntry) -> None:
        """Async version of store()."""
        import asyncio

        await asyncio.to_thread(self.store, entry)

    async def asearch(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> CacheEntry | None:
        """Async version of search()."""
        import asyncio

        return await asyncio.to_thread(
            self.search, embedding, namespace, embedding_model_id, context_hash, threshold
        )

    async def ainvalidate_namespace(self, namespace: str) -> int:
        """Async version of invalidate_namespace()."""
        import asyncio

        return await asyncio.to_thread(self.invalidate_namespace, namespace)

    async def aclear(self) -> None:
        """Async version of clear()."""
        import asyncio

        await asyncio.to_thread(self.clear)
