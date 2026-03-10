"""SemanticCache — the main entry point for semantic caching."""
from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from llm_semantic_cache.config import LARGE_NAMESPACE_THRESHOLD, CacheConfig
from llm_semantic_cache.context import hash_context
from llm_semantic_cache.embeddings import Embedder, FastEmbedEmbedder
from llm_semantic_cache.metrics import (
    measure_embedding_latency,
    record_cache_error,
    record_hit,
    record_miss,
    record_similarity_score,
    record_stream_bypass,
)
from llm_semantic_cache.prompt import extract_prompt_text
from llm_semantic_cache.storage.base import CacheEntry, StorageBackend

log = structlog.get_logger(__name__)


@dataclass
class CacheStats:
    """Snapshot of cache performance since the SemanticCache was instantiated.

    Intended for development and debugging — inspect hit rates and similarity
    distributions without wiring up Prometheus. For production observability,
    use the Prometheus metrics emitted alongside these counters.
    """

    hits: int
    """Total cache hits since instantiation."""

    misses: int
    """Total cache misses since instantiation."""

    hit_rate: float
    """hits / (hits + misses). Returns 0.0 when no requests have been made yet."""

    avg_similarity: float
    """Rolling mean of cosine similarity scores on cache hits. Returns 0.0 when no hits yet."""

    namespace_sizes: dict[str, int] = field(default_factory=dict)
    """Live entry count per namespace, reflecting current storage state."""


class SemanticCache:
    """Adds semantic caching to any OpenAI-compatible callable."""

    def __init__(
        self,
        storage: StorageBackend,
        config: CacheConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or CacheConfig()
        self._embedder = embedder or FastEmbedEmbedder(self._config.embedding_model)
        self._threshold = self._config.resolved_threshold()
        self._last_size_check: dict[str, float] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._similarity_sum: float = 0.0
        self._similarity_hit_count: int = 0
        self._touched_namespaces: set[str] = set()
        # Warm up the embedding model eagerly to prevent cold-start cache bypass.
        # The lazy load on first use can take seconds; we absorb this cost at init time.
        self._embedder.embed("warmup")
        # Note: This blocks the calling thread while the ONNX model loads (1–10s on
        # first run). In async frameworks, use async_warmup() instead — construct
        # SemanticCache before the event loop starts, or call
        # await asyncio.to_thread(lambda: SemanticCache(...)) from async code.

    def _validate_call_kwargs(
        self, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        """Extract and validate cache_context and cache_namespace from kwargs.

        Raises ValueError if cache_context is missing.
        Raises TypeError if cache_context is not a dict or cache_namespace
        is not a str — these are programmer errors, not runtime failures
        and are NOT caught by the fail-open handler.
        """
        if "cache_context" not in kwargs:
            raise ValueError(
                "cache_context is required when calling a cached function. "
                "Pass cache_context={} explicitly if this request has no additional context."
            )
        cache_context = kwargs.pop("cache_context")
        namespace = kwargs.pop("cache_namespace", self._config.default_namespace)

        if not isinstance(cache_context, dict):
            raise TypeError(
                f"cache_context must be a dict, got {type(cache_context).__name__!r}. "
                "Pass a dict with string keys and JSON-serializable values."
            )
        if not isinstance(namespace, str):
            raise TypeError(
                f"cache_namespace must be a str, got {type(namespace).__name__!r}."
            )

        return cache_context, namespace

    def wrap(
        self,
        fn: Callable[..., Any],
        *,
        mode: Literal["auto", "sync", "async"] = "auto",
    ) -> Callable[..., Any]:
        """Return a wrapped callable that applies semantic caching."""
        if mode == "async" or (mode == "auto" and self._is_async(fn)):
            return self._make_async_wrapper(fn)
        return self._make_sync_wrapper(fn)

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all cache entries in a namespace. Returns count deleted."""
        return self._storage.invalidate_namespace(namespace)

    async def ainvalidate_namespace(self, namespace: str) -> int:
        """Async version of invalidate_namespace()."""
        return await self._storage.ainvalidate_namespace(namespace)

    def stats(self) -> CacheStats:
        """Return a snapshot of cache performance since instantiation.

        Counters are instance-level and reset when the SemanticCache is
        recreated. They are not persisted to storage. Use Prometheus metrics
        for production monitoring; this method is for development and debugging.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        avg_similarity = (
            self._similarity_sum / self._similarity_hit_count
            if self._similarity_hit_count > 0
            else 0.0
        )
        namespace_sizes = {
            ns: self._storage.namespace_size(ns) for ns in self._touched_namespaces
        }
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate,
            avg_similarity=avg_similarity,
            namespace_sizes=namespace_sizes,
        )

    async def async_warmup(self) -> None:
        """Warm up the embedding model without blocking the event loop.

        Use this in async frameworks (FastAPI lifespan, etc.) instead of
        relying on the blocking warmup in __init__:

            @asynccontextmanager
            async def lifespan(app):
                await cache.async_warmup()
                yield
        """
        await asyncio.to_thread(self._embedder.embed, "warmup")

    def _make_async_wrapper(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self._async_cached_call(fn, *args, **kwargs)

        async_wrapper.__wrapped__ = fn
        return async_wrapper

    def _make_sync_wrapper(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._sync_cached_call(fn, *args, **kwargs)

        sync_wrapper.__wrapped__ = fn
        return sync_wrapper

    async def _async_cached_call(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Async cache execution path with fail-open and timeout."""
        cache_context, namespace = self._validate_call_kwargs(kwargs)

        if kwargs.get("stream", False):
            record_stream_bypass(namespace)
            log.info(
                "cache.stream_bypass",
                namespace=namespace,
                embedding_model=self._embedder.model_id,
            )
            return await fn(*args, **kwargs)

        try:
            prompt_text = self._extract_prompt(kwargs.get("messages", []))
            if prompt_text is None:
                log.info(
                    "cache.no_user_message_bypass",
                    namespace=namespace,
                    embedding_model=self._embedder.model_id,
                )
                return await fn(*args, **kwargs)
            context_hash = hash_context(cache_context)
        except Exception as exc:
            record_cache_error("lookup")
            log.error("cache.lookup_params_failed", error=str(exc), namespace=namespace)
            return await fn(*args, **kwargs)

        embedding, cached, best_score = await self._async_lookup(
            prompt_text, namespace, context_hash
        )
        self._touched_namespaces.add(namespace)
        if best_score is not None:
            record_similarity_score(best_score)
        if cached is not None:
            assert best_score is not None  # cached is only set when search returns a score
            self._hits += 1
            self._similarity_sum += best_score
            self._similarity_hit_count += 1
            record_hit(namespace)
            log.info(
                "cache.hit",
                namespace=namespace,
                best_score=best_score,
                threshold=self._threshold,
                embedding_model=self._embedder.model_id,
            )
            return cached

        response = await fn(*args, **kwargs)
        if embedding:
            await self._async_store(
                prompt_text,
                embedding,
                context_hash,
                namespace,
                response,
            )

        self._misses += 1
        record_miss(namespace)
        log.info(
            "cache.miss",
            namespace=namespace,
            best_score=best_score,
            threshold=self._threshold,
            embedding_model=self._embedder.model_id,
        )
        return response

    def _sync_cached_call(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Sync cache execution path with fail-open."""
        cache_context, namespace = self._validate_call_kwargs(kwargs)

        if kwargs.get("stream", False):
            record_stream_bypass(namespace)
            log.info(
                "cache.stream_bypass",
                namespace=namespace,
                embedding_model=self._embedder.model_id,
            )
            return fn(*args, **kwargs)

        try:
            prompt_text = self._extract_prompt(kwargs.get("messages", []))
            if prompt_text is None:
                log.info(
                    "cache.no_user_message_bypass",
                    namespace=namespace,
                    embedding_model=self._embedder.model_id,
                )
                return fn(*args, **kwargs)
            context_hash = hash_context(cache_context)
        except Exception as exc:
            record_cache_error("lookup")
            log.error("cache.lookup_params_failed", error=str(exc), namespace=namespace)
            return fn(*args, **kwargs)

        embedding, cached, best_score = self._sync_lookup(prompt_text, namespace, context_hash)
        self._touched_namespaces.add(namespace)
        if best_score is not None:
            record_similarity_score(best_score)
        if cached is not None:
            assert best_score is not None  # cached is only set when search returns a score
            self._hits += 1
            self._similarity_sum += best_score
            self._similarity_hit_count += 1
            record_hit(namespace)
            log.info(
                "cache.hit",
                namespace=namespace,
                best_score=best_score,
                threshold=self._threshold,
                embedding_model=self._embedder.model_id,
            )
            return cached

        response = fn(*args, **kwargs)
        if embedding:
            self._sync_store(
                prompt_text,
                embedding,
                context_hash,
                namespace,
                response,
            )

        self._misses += 1
        record_miss(namespace)
        log.info(
            "cache.miss",
            namespace=namespace,
            best_score=best_score,
            threshold=self._threshold,
            embedding_model=self._embedder.model_id,
        )
        return response

    async def _async_lookup(
        self, prompt_text: str, namespace: str, context_hash: str
    ) -> tuple[list[float], Any | None, float | None]:
        """Attempt cache lookup with timeout and fail-open on error."""
        try:
            with measure_embedding_latency():
                embedding = self._embedder.embed(prompt_text)
        except Exception as exc:
            record_cache_error("embed")
            log.error("cache.embed_failed", error=str(exc), namespace=namespace)
            return [], None, None
        try:
            op_start = time.perf_counter()
            result = await asyncio.wait_for(
                self._storage.asearch(
                    embedding=embedding,
                    namespace=namespace,
                    embedding_model_id=self._embedder.model_id,
                    context_hash=context_hash,
                    threshold=self._threshold,
                ),
                timeout=self._config.cache_timeout_seconds,
            )
            if result is None:
                return embedding, None, None
            response = result.entry.response if result.entry is not None else None
            return embedding, response, result.best_score
        except TimeoutError:
            elapsed_ms = int((time.perf_counter() - op_start) * 1000)
            timeout_ms = int(self._config.cache_timeout_seconds * 1000)
            record_cache_error("lookup")
            log.warning(
                "cache.timeout_exceeded",
                elapsed_ms=elapsed_ms,
                timeout_ms=timeout_ms,
                action="bypass",
                namespace=namespace,
            )
            return embedding, None, None
        except Exception as exc:
            record_cache_error("lookup")
            log.error("cache.lookup_failed", error=str(exc), namespace=namespace)
            return embedding, None, None

    def _sync_lookup(
        self, prompt_text: str, namespace: str, context_hash: str
    ) -> tuple[list[float], Any | None, float | None]:
        """Sync cache lookup with fail-open on error."""
        try:
            with measure_embedding_latency():
                embedding = self._embedder.embed(prompt_text)
        except Exception as exc:
            record_cache_error("embed")
            log.error("cache.embed_failed", error=str(exc), namespace=namespace)
            return [], None, None
        try:
            result = self._storage.search(
                embedding=embedding,
                namespace=namespace,
                embedding_model_id=self._embedder.model_id,
                context_hash=context_hash,
                threshold=self._threshold,
            )
            if result is None:
                return embedding, None, None
            response = result.entry.response if result.entry is not None else None
            return embedding, response, result.best_score
        except Exception as exc:
            record_cache_error("lookup")
            log.error("cache.lookup_failed", error=str(exc), namespace=namespace)
            return embedding, None, None

    async def _async_store(
        self,
        prompt_text: str,
        embedding: list[float],
        context_hash: str,
        namespace: str,
        response: Any,
    ) -> None:
        """Attempt to store a response in the cache. Fail-open on error."""
        try:
            entry = self._build_entry(
                prompt_text,
                embedding,
                context_hash,
                namespace,
                response,
            )
            await asyncio.wait_for(
                self._storage.astore(entry),
                timeout=self._config.cache_timeout_seconds,
            )
            now = time.monotonic()
            last = self._last_size_check.get(namespace, 0.0)
            if now - last > 60.0:
                self._last_size_check[namespace] = now
                size = await self._storage.anamespace_size(namespace)
                if size > LARGE_NAMESPACE_THRESHOLD:
                    log.warning(
                        "cache.namespace_large",
                        namespace=namespace,
                        size=size,
                        threshold=LARGE_NAMESPACE_THRESHOLD,
                    )
        except Exception as exc:
            record_cache_error("store")
            log.error("cache.store_failed", error=str(exc))

    def _sync_store(
        self,
        prompt_text: str,
        embedding: list[float],
        context_hash: str,
        namespace: str,
        response: Any,
    ) -> None:
        """Sync store with fail-open on error."""
        try:
            entry = self._build_entry(
                prompt_text,
                embedding,
                context_hash,
                namespace,
                response,
            )
            self._storage.store(entry)
            now = time.monotonic()
            last = self._last_size_check.get(namespace, 0.0)
            if now - last > 60.0:
                self._last_size_check[namespace] = now
                size = self._storage.namespace_size(namespace)
                if size > LARGE_NAMESPACE_THRESHOLD:
                    log.warning(
                        "cache.namespace_large",
                        namespace=namespace,
                        size=size,
                        threshold=LARGE_NAMESPACE_THRESHOLD,
                    )
        except Exception as exc:
            record_cache_error("store")
            log.error("cache.store_failed", error=str(exc))

    def _extract_prompt(self, messages: Any) -> str | None:
        """Extract prompt text from OpenAI-style messages (Pydantic or dict)."""
        if not isinstance(messages, list):
            return None
        return extract_prompt_text(messages)

    def _build_entry(
        self,
        prompt_text: str,
        embedding: list[float],
        context_hash: str,
        namespace: str,
        response: Any,
    ) -> CacheEntry:
        """Construct a CacheEntry from a response."""
        if isinstance(response, dict):
            response_dict = response
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump(mode="json")
        else:
            raise TypeError(
                f"Cannot serialize response of type {type(response).__name__!r}. "
                "SemanticCache expects an OpenAI-compatible response (dict or Pydantic model). "
                "For non-standard responses, convert to a dict before passing to the cached function."  # noqa: E501
            )

        return CacheEntry(
            id=str(uuid.uuid4()),
            embedding=embedding,
            prompt_text=prompt_text,
            context_hash=context_hash,
            namespace=namespace,
            embedding_model_id=self._embedder.model_id,
            response=response_dict,
            ttl=self._config.default_ttl,
        )

    @staticmethod
    def _is_async(fn: Callable[..., Any]) -> bool:
        """Detect if fn is async, following __wrapped__ chain."""
        unwrapped = inspect.unwrap(fn)
        return inspect.iscoroutinefunction(unwrapped)
