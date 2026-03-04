"""Tests for Embedder protocol and FakeEmbedder."""
from __future__ import annotations

import builtins
import math

import pytest

from llm_semantic_cache.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    _l2_normalize,
)


class FakeEmbedder:
    """Deterministic embedder for testing — no ML dependencies."""

    def __init__(self, model_name: str = "fake-model") -> None:
        self._model_name = model_name

    @property
    def model_id(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        # Simple deterministic embedding based on text length
        dim = 4
        raw = [float(len(text) % (i + 2)) for i in range(dim)]
        return _l2_normalize(raw)


def test_fake_embedder_satisfies_protocol() -> None:
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_embedder_model_id_returns_string() -> None:
    assert isinstance(FakeEmbedder().model_id, str)


def test_fake_embedder_embed_returns_list_of_floats() -> None:
    embedding = FakeEmbedder().embed("hello")
    assert isinstance(embedding, list)
    assert all(isinstance(value, float) for value in embedding)


def test_fake_embedder_embed_returns_unit_vector() -> None:
    embedding = FakeEmbedder().embed("hello world")
    norm = math.sqrt(sum(value * value for value in embedding))
    assert norm == pytest.approx(1.0)


def test_l2_normalize_unit_vector_unchanged() -> None:
    vector = [1.0, 0.0, 0.0]
    assert _l2_normalize(vector) == vector


def test_l2_normalize_zero_vector_returns_original() -> None:
    vector = [0.0, 0.0, 0.0]
    assert _l2_normalize(vector) == vector


def test_l2_normalize_normalizes_to_unit_length() -> None:
    normalized = _l2_normalize([3.0, 4.0])
    norm = math.sqrt(sum(value * value for value in normalized))
    assert norm == pytest.approx(1.0)


def test_fake_embedder_deterministic() -> None:
    embedder = FakeEmbedder()
    assert embedder.embed("same") == embedder.embed("same")


def test_embedder_model_id_changes_with_different_model_name() -> None:
    assert FakeEmbedder("model-a").model_id != FakeEmbedder("model-b").model_id


def test_fastembed_embedder_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastembed":
            raise ImportError("No module named 'fastembed'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    embedder = FastEmbedEmbedder()
    with pytest.raises(ImportError) as exc_info:
        embedder._load_model()

    assert "fastembed is required for the default embedder" in str(exc_info.value)
    assert "pip install fastembed" in str(exc_info.value)
