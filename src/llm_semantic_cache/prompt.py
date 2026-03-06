"""extract_prompt_text — canonical prompt extraction from chat messages."""
from __future__ import annotations

from llm_semantic_cache.models import ChatMessage


def extract_prompt_text(messages: list[ChatMessage] | list[dict]) -> str | None:
    """Extract the canonical prompt text from a list of chat messages.

    Accepts either a list of ChatMessage objects or a list of dicts with
    'role' and 'content' keys (raw OpenAI API format).

    Returns the content of the last non-empty message with role='user'.
    Iterates in reverse so long conversation histories pay minimal cost.

    System prompts and assistant messages are intentionally excluded from
    the embedding. They are part of the context (passed via cache_context),
    not the prompt.
    """
    if not messages:
        return None
    for m in reversed(messages):
        if isinstance(m, ChatMessage):
            role = m.role
            content = m.content or ""
        elif isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content") or ""
        else:
            continue
        if role == "user" and content.strip():
            return content.strip()
    return None
