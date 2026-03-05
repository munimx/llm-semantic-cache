"""extract_prompt_text — canonical prompt extraction from chat messages."""
from __future__ import annotations

from llm_semantic_cache.models import ChatMessage


def extract_prompt_text(messages: list[ChatMessage] | list[dict]) -> str | None:
    """Extract the canonical prompt text from a list of chat messages.

    Accepts either a list of ChatMessage objects or a list of dicts with
    'role' and 'content' keys (raw OpenAI API format).

    Returns the content of the last message with role='user' that has
    non-empty string content. Returns None if no such message exists.

    System prompts and assistant messages are intentionally excluded from
    the embedding. They are part of the context (passed via cache_context),
    not the prompt.
    """
    if not messages:
        return None
    normalized: list[ChatMessage] = []
    for m in messages:
        if isinstance(m, dict):
            normalized.append(ChatMessage(role=m.get("role", ""), content=m.get("content") or ""))
        else:
            normalized.append(m)
    user_messages = [m for m in normalized if m.role == "user" and m.content and m.content.strip()]
    if not user_messages:
        return None
    return user_messages[-1].content.strip()
