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


def _msg_type(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("type", "")
    return getattr(msg, "type", "")

def is_tool_message(msg) -> bool:
    return _msg_type(msg) == "tool"

def is_ai_message(msg) -> bool:
    return _msg_type(msg) == "ai"

def is_human_message(msg) -> bool:
    return _msg_type(msg) == "human"

def get_msg_content(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")

import re

_URL_RE = re.compile(r"https?://[^\s\)\]\"',<>]+")

def extract_sources(messages) -> list[str]:
    """Pull web URLs and retrieved document labels out of ToolMessages.

    Robust to both string and list-content ToolMessage variants. Returns
    first-seen order; duplicates are not removed.
    """
    sources: list[str] = []
    for msg in messages:
        if is_tool_message(msg):
            content = get_msg_content(msg)
            if isinstance(content, str):
                sources.extend(_URL_RE.findall(content))
                name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None)
                if name == "retrieve_documents":
                    sources.extend(re.findall(r"<<UNTRUSTED EXCERPT — source: ([^>]+)>>", content))
            else:
                sources.extend(_URL_RE.findall(str(content)))
    return list(dict.fromkeys(sources))
