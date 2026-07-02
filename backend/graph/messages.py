"""Normalize LangChain message content to plain strings."""


def content_to_str(content) -> str:
    """Coerce AIMessage/HumanMessage content to a single string.

    Most providers return str; Anthropic-style models may return a list of
    content blocks. Used by agents, router, synthesizer, API serializers,
    and the SSE stream handler.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)
