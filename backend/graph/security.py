"""Prompt-injection boundary helpers for untrusted tool/document text."""


def escape_untrusted(text: str) -> str:
    """Escape `<<` / `>>` so crafted payloads cannot break UNTRUSTED markers."""
    return text.replace("<<", "«").replace(">>", "»")
