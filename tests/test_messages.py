"""Tests for backend.graph.messages.content_to_str."""

from backend.graph.messages import content_to_str


def test_content_to_str_plain():
    assert content_to_str("hello") == "hello"


def test_content_to_str_list_blocks():
    blocks = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
    assert content_to_str(blocks) == "hello"


def test_content_to_str_none():
    assert content_to_str(None) == ""
