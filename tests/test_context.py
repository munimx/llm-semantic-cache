import datetime
import uuid

import pytest

from llm_semantic_cache.context import hash_context


class Unserializable:
    pass


def test_empty_dict_produces_stable_hash() -> None:
    assert hash_context({}) == hash_context({})


def test_key_order_does_not_affect_hash() -> None:
    assert hash_context({"a": 1, "b": 2}) == hash_context({"b": 2, "a": 1})


def test_same_input_same_output() -> None:
    context = {"nested": {"x": 1}, "items": [1, 2, 3]}
    assert hash_context(context) == hash_context(context)


def test_different_inputs_different_hashes() -> None:
    assert hash_context({"a": 1}) != hash_context({"a": 2})


def test_datetime_is_handled() -> None:
    dt = datetime.datetime(2024, 1, 1, 12, 0, 0)
    assert hash_context({"dt": dt})


def test_uuid_is_handled() -> None:
    value = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert hash_context({"id": value})


def test_bytes_is_handled() -> None:
    assert hash_context({"blob": b"abc"})


def test_set_is_handled_deterministically() -> None:
    assert hash_context({"set": {"a", "b"}}) == hash_context({"set": {"b", "a"}})


def test_frozenset_is_handled() -> None:
    assert hash_context({"f": frozenset({"x", "y"})})


def test_unknown_type_raises_clear_type_error() -> None:
    with pytest.raises(TypeError, match="Unserializable"):
        hash_context({"x": Unserializable()})


def test_nested_dict_is_handled() -> None:
    assert hash_context({"outer": {"inner": {"k": "v"}}})


def test_unicode_values_work() -> None:
    assert hash_context({"emoji": "✅", "text": "café"}) == hash_context(
        {"text": "café", "emoji": "✅"}
    )
