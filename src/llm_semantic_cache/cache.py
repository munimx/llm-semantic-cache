"""SemanticCache — the main entry point for semantic caching."""
from __future__ import annotations

import asyncio
import functools
import inspect
import random
import uuid
from collections.abc import Callable
from typing import Any, Literal

import structlog

from llm_semantic_cache.config import CacheConfig, LARGE_NAMESPACE_THRESHOLD
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
        # Warm up the embedding model eagerly to prevent cold-start cache bypass.
        # The lazy load on first use can take seconds; we absorb this cost at init time.
        self._embedder.embed("warmup")
        # Note: This blocks the calling thread while the ONNX model loads (1–10s on
        # first run). In async frameworks, use async_warmup() instead — construct
        # SemanticCache before the event loop starts, or call
        # await asyncio.to_thread(lambda: SemanticCache(...)) from async code.

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
        if "cache_context" not in kwargs:
            raise ValueError(
                "cache_context is required when calling a cached function. "
                "Pass cache_context={} explicitly if this request has no additional context."
            )
        cache_context = kwargs.pop("cache_context")
        namespace = kwargs.pop("cache_namespace", self._config.default_namespace)

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
        if best_score is not None:
            record_similarity_score(best_score)
        if cached is not None:
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
        if "cache_context" not in kwargs:
            raise ValueError(
                "cache_context is required when calling a cached function. "
                "Pass cache_context={} explicitly if this request has no additional context."
            )
        cache_context = kwargs.pop("cache_context")
        namespace = kwargs.pop("cache_namespace", self._config.default_namespace)

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
        if best_score is not None:
            record_similarity_score(best_score)
        if cached is not None:
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
            if random.random() < 0.01:
                size = await self._storage.anamespace_size(namespace)
                if size > LARGE_NAMESPACE_THRESHOLD:
                    log.warning(
                        "cache.namespace_too_large",
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
            if random.random() < 0.01:
                size = self._storage.namespace_size(namespace)
                if size > LARGE_NAMESPACE_THRESHOLD:
                    log.warning(
                        "cache.namespace_too_large",
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
                "For non-standard responses, convert to a dict before passing to the cached function."
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
