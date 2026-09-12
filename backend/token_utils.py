"""Lightweight token budget helpers.

Rough token estimate: 1 token ≈ 4 characters of English text.
Intentionally conservative — no external dependency needed.
"""


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: 1 token ≈ 4 characters of English text.

    This is intentionally conservative — no external dependency needed.
    """
    total_chars = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if content is None:
            # Support plain dict messages like {"content": "..."}.
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = ""
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        total_chars += len(str(content))
    return total_chars // 4


def truncate_messages_if_needed(messages: list, max_tokens: int = 28_000) -> list:
    """Drop oldest non-system messages if estimated tokens exceed max_tokens.

    Always preserves the first message (usually the system prompt) and the
    last message (current user turn).
    """
    if not messages:
        return messages
    if estimate_tokens(messages) <= max_tokens:
        return messages
    # Keep first (system) and last (current user turn) always.
    # Drop from oldest non-system messages until under budget.
    result = [messages[0]] + list(messages[1:])
    while estimate_tokens(result) > max_tokens and len(result) > 2:
        result.pop(1)  # remove second-oldest
    import logging
    logging.getLogger("agentflow.tokens").warning(
        "token_budget_truncated",
        extra={"remaining_messages": len(result), "original": len(messages)},
    )
    return result
