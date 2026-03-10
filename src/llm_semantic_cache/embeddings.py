"""Embedder Protocol and default FastEmbedEmbedder implementation."""
from __future__ import annotations

import math
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding models.

    Any object with embed() and model_id satisfies this protocol.
    The library ships FastEmbedEmbedder as the default. Users can supply
    their own implementation by satisfying this protocol.
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for this embedding model.

        Used to tag cache entries. Entries written by a different model
        are invisible during search — this prevents cross-model hits.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a text string into a fixed-dimension float vector.

        The returned vector must be L2-normalized (unit length).
        """
        ...


def _l2_normalize(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm = 1.0)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class FastEmbedEmbedder:
    """Default embedder using fastembed (ONNX-based, ~20MB install).

    Uses BAAI/bge-small-en-v1.5 by default. The model is loaded lazily on the
    first call to embed() — not at initialization time.

    fastembed is ~20MB vs ~700MB for sentence-transformers/torch. It is the
    recommended embedder for production deployments.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_name

    def _load_model(self) -> object:
        """Lazy model loading — imports and loads on first use."""
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the default embedder. "
                "Install it with: pip install fastembed\n"
                "Or use the [torch] extra for sentence-transformers: "
                "pip install 'llm-semantic-cache[torch]'"
            ) from exc
        return TextEmbedding(model_name=self._model_name)

    def embed(self, text: str) -> list[float]:
        """Embed text and return an L2-normalized vector."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
        # fastembed returns a generator of numpy arrays
        result = list(self._model.embed([text]))[0]  # type: ignore[attr-defined]
        return _l2_normalize(result.tolist())


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers (optional [torch] extra).

    Requires: pip install 'llm-semantic-cache[torch]'

    Uses BAAI/bge-small-en-v1.5 by default. Model loads lazily on first embed().
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_name

    def _load_model(self) -> object:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install 'llm-semantic-cache[torch]'"
            ) from exc
        return SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        """Embed text and return an L2-normalized vector."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
        result = self._model.encode([text], normalize_embeddings=True)  # type: ignore[attr-defined]
        return list(result[0].tolist())
