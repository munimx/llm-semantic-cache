from llm_semantic_cache.models import ChatMessage
from llm_semantic_cache.prompt import extract_prompt_text


def test_returns_last_user_message() -> None:
    messages = [
        ChatMessage(role="system", content="You are helpful"),
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="second"),
    ]
    assert extract_prompt_text(messages) == "second"


def test_returns_none_when_no_user_message() -> None:
    messages = [ChatMessage(role="system", content="rules")]
    assert extract_prompt_text(messages) is None


def test_returns_none_when_user_content_empty() -> None:
    messages = [ChatMessage(role="user", content="")]
    assert extract_prompt_text(messages) is None


def test_returns_none_when_user_content_whitespace_only() -> None:
    messages = [ChatMessage(role="user", content="   ")]
    assert extract_prompt_text(messages) is None


def test_multiple_user_messages_returns_last() -> None:
    messages = [
        ChatMessage(role="user", content="one"),
        ChatMessage(role="user", content="two"),
    ]
    assert extract_prompt_text(messages) == "two"


def test_strips_whitespace() -> None:
    messages = [ChatMessage(role="user", content="  hello  ")]
    assert extract_prompt_text(messages) == "hello"


def test_mixed_roles_returns_correct_user_message() -> None:
    messages = [
        ChatMessage(role="assistant", content="a"),
        ChatMessage(role="user", content="target"),
        ChatMessage(role="assistant", content="b"),
    ]
    assert extract_prompt_text(messages) == "target"
