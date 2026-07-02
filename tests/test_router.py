"""Offline router label parsing tests (DESIGN_DOC §9)."""

import pytest

from backend.graph.router import _parse_label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("research", "research"),
        ("analysis", "analysis"),
        ("chat", "chat"),
        ("Research", "research"),
        ("analysis.", "analysis"),
        ("The answer is research", "research"),
        ("category: analysis", "analysis"),
        ("please route to chat", "chat"),
        ("", "chat"),
        ("unknown", "chat"),
        ("research\n", "research"),
        ("  analysis  ", "analysis"),
        ("I think this is research because...", "research"),
        ("compare and summarize → analysis", "analysis"),
        ("thanks, shorten that → chat", "chat"),
        ("latest AI papers", "chat"),
        ("research: web search needed", "research"),
        ("analysis: compare docs", "analysis"),
        ("chat: casual follow-up", "chat"),
        ("RESEARCH", "research"),
        ("foo bar baz", "chat"),
    ],
)
def test_parse_label_offline(raw, expected):
    assert _parse_label(raw) == expected
