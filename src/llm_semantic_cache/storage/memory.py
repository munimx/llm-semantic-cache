"""InMemoryStorage — dict-based storage with brute-force cosine similarity search."""
from __future__ import annotations

from llm_semantic_cache.similarity import cosine_similarity
from llm_semantic_cache.storage.base import CacheEntry, StorageBackend


class InMemoryStorage(StorageBackend):
    """Zero-dependency in-memory storage for SemanticCache.

    Suitable for development, testing, and single-process deployments.
    Data is lost when the process exits — this is by design.

    Thread safety: basic dict operations in CPython are GIL-protected, but
    concurrent writes from multiple threads are not guaranteed to be atomic.
    For multi-threaded production use, use RedisStorage.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[CacheEntry]] = {}

    def store(self, entry: CacheEntry) -> None:
        """Store a cache entry, creating the namespace bucket if needed."""
        if entry.namespace not in self._store:
            self._store[entry.namespace] = []
        self._store[entry.namespace].append(entry)

    def search(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> CacheEntry | None:
        """Find the best matching entry using brute-force cosine similarity.

        Filters candidates by namespace, embedding_model_id, context_hash,
        and non-expiry before computing similarity.
        """
        candidates = self._store.get(namespace, [])
        best_entry: CacheEntry | None = None
        best_score = -1.0

        for entry in candidates:
            if entry.embedding_model_id != embedding_model_id:
                continue
            if entry.context_hash != context_hash:
                continue
            if entry.is_expired():
                continue
            score = cosine_similarity(embedding, entry.embedding)
            if score >= threshold and score > best_score:
                best_score = score
                best_entry = entry

        return best_entry

    def invalidate_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace. Returns count of deleted entries."""
        entries = self._store.pop(namespace, [])
        return len(entries)

    def clear(self) -> None:
        """Delete all entries in all namespaces."""
        self._store.clear()

    def namespace_size(self, namespace: str) -> int:
        """Return the number of entries in a namespace (for monitoring)."""
        return len(self._store.get(namespace, []))
