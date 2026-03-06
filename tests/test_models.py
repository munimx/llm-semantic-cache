from llm_semantic_cache.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    UsageInfo,
)


def test_chat_message_basic() -> None:
    message = ChatMessage(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"


def test_chat_message_with_optional_fields() -> None:
    message = ChatMessage(
        role="assistant",
        content="Hi",
        name="bot",
        tool_calls=[{"id": "1", "type": "function"}],
    )
    assert message.name == "bot"
    assert message.tool_calls is not None


def test_chat_completion_request_defaults() -> None:
    request = ChatCompletionRequest(model="gpt-4", messages=[ChatMessage(role="user", content="Q")])
    assert request.stream is False
    assert request.extra == {}
    assert request.temperature is None
    assert request.max_tokens is None


def test_chat_completion_request_extra_fields_allowed() -> None:
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="Q")],
        custom_field="value",
    )
    assert request.custom_field == "value"


def test_chat_completion_response_round_trip_json() -> None:
    response = ChatCompletionResponse(
        id="cmpl-1",
        created=123,
        model="gpt-4",
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content="A"))],
    )
    parsed = ChatCompletionResponse.model_validate_json(response.model_dump_json())
    assert parsed == response


def test_usage_info_defaults_to_zero() -> None:
    usage = UsageInfo()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_choice_with_finish_reason() -> None:
    choice = Choice(
        index=0,
        message=ChatMessage(role="assistant", content="Done"),
        finish_reason="stop",
    )
    assert choice.finish_reason == "stop"
