"""InMemoryStorage — dict-based storage with numpy-vectorized similarity search."""
from __future__ import annotations

import numpy as np

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
        # namespace -> list of CacheEntry (live entries only)
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
        """Find the best matching entry using numpy-vectorized cosine similarity.

        Filters candidates by namespace, embedding_model_id, context_hash,
        and non-expiry. Expired entries are evicted from the store during search.
        Uses a single matrix dot product for all similarity computations.
        Assumes embeddings are L2-normalized unit vectors.
        """
        all_entries = self._store.get(namespace, [])
        if not all_entries:
            return None

        # Separate live from expired; write back only live entries (eviction).
        live: list[CacheEntry] = []
        expired: list[CacheEntry] = []
        for entry in all_entries:
            if entry.is_expired():
                expired.append(entry)
            else:
                live.append(entry)

        if expired:
            self._store[namespace] = live

        # Filter by model_id and context_hash.
        candidates = [
            e
            for e in live
            if e.embedding_model_id == embedding_model_id and e.context_hash == context_hash
        ]
        if not candidates:
            return None

        # Vectorized cosine similarity: all embeddings are already L2-normalized,
        # so dot product equals cosine similarity.
        query = np.array(embedding, dtype=np.float64)
        matrix = np.array([e.embedding for e in candidates], dtype=np.float64)
        scores = matrix @ query  # shape: (n_candidates,)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return candidates[best_idx]
        return None

    def invalidate_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace. Returns count of deleted entries."""
        entries = self._store.pop(namespace, [])
        return len(entries)

    def clear(self) -> None:
        """Delete all entries in all namespaces."""
        self._store.clear()

    def namespace_size(self, namespace: str) -> int:
        """Return the count of live (non-expired) entries in a namespace."""
        return sum(1 for e in self._store.get(namespace, []) if not e.is_expired())
