"""hash_context — deterministic SHA-256 fingerprint of a context dict."""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any


def _canonical_default(obj: object) -> object:
    """JSON serialization hook for deterministic context hashing.

    Handles common non-JSON-serializable Python types. Unknown types raise
    TypeError with an actionable message so callers know what to convert.
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, frozenset):
        return sorted(str(item) for item in obj)
    if isinstance(obj, set):
        return sorted(str(item) for item in obj)
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    raise TypeError(
        f"Context value of type {type(obj).__name__!r} is not serializable. "
        f"Convert it to a JSON-compatible type (str, int, float, bool, list, dict) "
        f"before passing it as context."
    )


def hash_context(context: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a context dict.

    The hash is stable across Python restarts. Key ordering does not affect
    the result. Common non-JSON types (datetime, UUID, bytes, sets, Pydantic
    models) are handled. Unknown types raise TypeError.
    """
    serialized = json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_default,
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
