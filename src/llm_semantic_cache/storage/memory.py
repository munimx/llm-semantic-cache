"""InMemoryStorage — dict-based storage with numpy-vectorized similarity search."""
from __future__ import annotations

import numpy as np

from llm_semantic_cache.storage.base import CacheEntry, SearchResult, StorageBackend


class InMemoryStorage(StorageBackend):
    """Zero-dependency in-memory storage for SemanticCache.

    IMPORTANT: Not safe for concurrent use from multiple threads.
    Concurrent writes from different threads can silently lose entries
    due to a read-modify-write race in search(): thread A reads the
    namespace list, thread B appends to it, then thread A overwrites
    with its live-only list, losing thread B's entry.

    This is by design — InMemoryStorage is the development and testing
    backend. No locking is added because the overhead is inappropriate
    for a zero-dependency dev backend.

    For multi-threaded or multi-process production deployments, use
    RedisStorage. For single-threaded async frameworks (FastAPI, etc.),
    InMemoryStorage is safe — async tasks are cooperative and do not
    race at the Python level.
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
    ) -> SearchResult | None:
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
            return SearchResult(entry=candidates[best_idx], best_score=best_score)
        return SearchResult(entry=None, best_score=best_score)

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

    # Async overrides — call sync methods directly without thread pool.
    # InMemoryStorage operations are instant dict/list operations;
    # running them through asyncio.to_thread() adds unnecessary overhead.

    async def astore(self, entry: CacheEntry) -> None:
        self.store(entry)

    async def asearch(
        self,
        embedding: list[float],
        namespace: str,
        embedding_model_id: str,
        context_hash: str,
        threshold: float,
    ) -> SearchResult | None:
        return self.search(embedding, namespace, embedding_model_id, context_hash, threshold)

    async def ainvalidate_namespace(self, namespace: str) -> int:
        return self.invalidate_namespace(namespace)

    async def aclear(self) -> None:
        self.clear()

    async def anamespace_size(self, namespace: str) -> int:
        return self.namespace_size(namespace)
