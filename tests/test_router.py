"""Offline router label parsing tests (DESIGN_DOC §9).

Tier A: pure string parsing only — zero LLM / zero network calls.
``_parse_label`` is the deterministic post-processor for the router node's
raw LLM output, so every case here is exercisable without any API key.
"""

import pytest

from backend.graph.router import _parse_label

pytestmark = pytest.mark.tier_a


@pytest.mark.parametrize(
    "raw,expected",
    [
        # ----- original 20 cases -----
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
        ("compare and summarize \u2192 analysis", "analysis"),
        ("thanks, shorten that \u2192 chat", "chat"),
        ("latest AI papers", "chat"),
        ("research: web search needed", "research"),
        ("analysis: compare docs", "analysis"),
        ("chat: casual follow-up", "chat"),
        ("RESEARCH", "research"),
        ("foo bar baz", "chat"),
        # ----- blog label coverage (previously untested) -----
        ("blog", "blog"),
        ("BLOG", "blog"),
        ("route: blog", "blog"),
        ("write a blog post \u2192 blog", "blog"),
        ("The route is blog.", "blog"),
        # ----- whitespace / punctuation edge cases -----
        ("   ", "chat"),           # whitespace-only \u2192 default
        ("...", "chat"),            # punctuation-only \u2192 default
        ("!@#$%", "chat"),         # symbol-only \u2192 default
        # ----- first-token fallback path -----
        ("analysis because the user wants a summary", "analysis"),
        ("research needed; live web data", "research"),
        # ----- mixed-case in a longer sentence -----
        ("ANALYSIS: compare revenue figures", "analysis"),
        ("Please write a BLOG about AI", "blog"),
    ],
)
def test_parse_label_offline(raw, expected):
    assert _parse_label(raw) == expected
