"""SemanticCache — the main entry point for semantic caching."""
from __future__ import annotations

import asyncio
import functools
import inspect
import uuid
from collections.abc import Callable
from typing import Any, Literal

import structlog

from llm_semantic_cache.config import CacheConfig
from llm_semantic_cache.context import hash_context
from llm_semantic_cache.embeddings import Embedder, FastEmbedEmbedder
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
        namespace, context_hash, prompt_text = self._extract_cache_params(kwargs)

        if kwargs.get("stream", False):
            log.info("cache.stream_bypass", namespace=namespace)
            return await fn(*args, **kwargs)

        if prompt_text is None:
            log.info("cache.no_user_message_bypass", namespace=namespace)
            return await fn(*args, **kwargs)

        cached = await self._async_lookup(prompt_text, namespace, context_hash)
        if cached is not None:
            log.info("cache.hit", namespace=namespace)
            return cached

        response = await fn(*args, **kwargs)
        await self._async_store(prompt_text, context_hash, namespace, response)

        log.info("cache.miss", namespace=namespace)
        return response

    def _sync_cached_call(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Sync cache execution path with fail-open."""
        namespace, context_hash, prompt_text = self._extract_cache_params(kwargs)

        if kwargs.get("stream", False):
            log.info("cache.stream_bypass", namespace=namespace)
            return fn(*args, **kwargs)

        if prompt_text is None:
            log.info("cache.no_user_message_bypass", namespace=namespace)
            return fn(*args, **kwargs)

        cached = self._sync_lookup(prompt_text, namespace, context_hash)
        if cached is not None:
            log.info("cache.hit", namespace=namespace)
            return cached

        response = fn(*args, **kwargs)
        self._sync_store(prompt_text, context_hash, namespace, response)

        log.info("cache.miss", namespace=namespace)
        return response

    async def _async_lookup(
        self, prompt_text: str, namespace: str, context_hash: str
    ) -> Any | None:
        """Attempt cache lookup with timeout and fail-open on error."""
        try:
            embedding = self._embedder.embed(prompt_text)
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
            return result.response if result is not None else None
        except Exception as exc:
            log.error("cache.lookup_failed", error=str(exc))
            return None

    def _sync_lookup(
        self, prompt_text: str, namespace: str, context_hash: str
    ) -> Any | None:
        """Sync cache lookup with fail-open on error."""
        try:
            embedding = self._embedder.embed(prompt_text)
            result = self._storage.search(
                embedding=embedding,
                namespace=namespace,
                embedding_model_id=self._embedder.model_id,
                context_hash=context_hash,
                threshold=self._threshold,
            )
            return result.response if result is not None else None
        except Exception as exc:
            log.error("cache.lookup_failed", error=str(exc))
            return None

    async def _async_store(
        self,
        prompt_text: str,
        context_hash: str,
        namespace: str,
        response: Any,
    ) -> None:
        """Attempt to store a response in the cache. Fail-open on error."""
        try:
            embedding = self._embedder.embed(prompt_text)
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
        except Exception as exc:
            log.error("cache.store_failed", error=str(exc))

    def _sync_store(
        self,
        prompt_text: str,
        context_hash: str,
        namespace: str,
        response: Any,
    ) -> None:
        """Sync store with fail-open on error."""
        try:
            embedding = self._embedder.embed(prompt_text)
            entry = self._build_entry(
                prompt_text,
                embedding,
                context_hash,
                namespace,
                response,
            )
            self._storage.store(entry)
        except Exception as exc:
            log.error("cache.store_failed", error=str(exc))

    def _extract_cache_params(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, str, str | None]:
        """Extract and validate cache kwargs, returning namespace/hash/prompt."""
        if "cache_context" not in kwargs:
            raise ValueError(
                "cache_context is required when calling a cached function. "
                "Pass cache_context={} explicitly if this request has no additional context."
            )

        cache_context = kwargs.pop("cache_context")
        namespace = kwargs.pop("cache_namespace", self._config.default_namespace)
        context_hash = hash_context(cache_context)
        prompt_text = self._extract_prompt(kwargs.get("messages", []))
        return namespace, context_hash, prompt_text

    def _extract_prompt(self, messages: Any) -> str | None:
        """Extract prompt text from model or dict messages."""
        try:
            return extract_prompt_text(messages)
        except Exception:
            pass
        if not isinstance(messages, list):
            return None
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            return None
        return None

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
        elif hasattr(response, "__dict__"):
            response_dict = vars(response)
        else:
            response_dict = {"value": response}

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
