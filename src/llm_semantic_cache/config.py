"""CacheConfig — all tuneable values for SemanticCache in one place."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

THRESHOLD_PROFILES: Final[dict[str, float]] = {
    "strict": 0.97,
    "balanced": 0.92,
    "loose": 0.85,
}
LARGE_NAMESPACE_THRESHOLD: Final[int] = 5_000
"""Warn when a namespace exceeds this many entries. See AGENTS.md R2-3."""


def resolve_threshold(profile_or_float: str | float) -> float:
    """Resolve a named profile or raw float to a threshold value."""
    if isinstance(profile_or_float, float):
        if not 0.0 <= profile_or_float <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {profile_or_float}"
            )
        return profile_or_float
    if profile_or_float not in THRESHOLD_PROFILES:
        valid = ", ".join(f'"{k}"' for k in THRESHOLD_PROFILES)
        raise ValueError(
            f"Unknown threshold profile {profile_or_float!r}. Valid profiles: {valid}"
        )
    return THRESHOLD_PROFILES[profile_or_float]


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for a SemanticCache instance.

    All tuneable values live here. Pass a CacheConfig to SemanticCache.__init__.
    """

    threshold: str | float = "balanced"
    """Similarity threshold. Named profile or raw float in [0.0, 1.0]."""

    default_namespace: str = "default"
    """Default namespace for cache entries when no namespace is specified."""

    default_ttl: float | None = None
    """Default TTL in seconds for cache entries. None means no expiration."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """Embedding model identifier. Used for model_id tagging on cache entries."""

    cache_timeout_seconds: float = 0.05
    """Timeout in seconds for cache operations (embed + storage).

    IMPORTANT: This timeout is enforced only in the async path (asyncio.wait_for).
    Sync callers using RedisStorage have no timeout protection — if Redis is slow
    or unresponsive, sync calls will block indefinitely. For production use with
    RedisStorage, use the async path (async def + await) which enforces this timeout.
    InMemoryStorage is always fast (no I/O) and the timeout is not meaningful there.
    """

    def resolved_threshold(self) -> float:
        """Return the numeric threshold, resolving named profiles."""
        return resolve_threshold(self.threshold)
