"""Unit tests for token_utils.py (Session 5-A)."""

from langchain_core.messages import HumanMessage, SystemMessage

from backend.token_utils import estimate_tokens, truncate_messages_if_needed


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_estimate_tokens_str_and_list():
    msgs = [
        SystemMessage(content="Hello world"),  # 11 chars -> 2 tokens
        HumanMessage(content=[{"text": "Foo bar baz"}]),  # 11 chars -> 2 tokens
        {"content": "Plain dict message"},  # 18 chars -> 4 tokens
    ]
    total = estimate_tokens(msgs)
    assert total >= 8


def test_truncate_messages_if_needed_under_budget():
    msgs = [
        SystemMessage(content="System prompt"),
        HumanMessage(content="User query"),
    ]
    truncated = truncate_messages_if_needed(msgs, max_tokens=1000)
    assert len(truncated) == 2


def test_truncate_messages_if_needed_over_budget():
    sys_msg = SystemMessage(content="System prompt " * 10)
    msgs = [sys_msg]
    # Add 5 long turns
    for i in range(5):
        msgs.append(HumanMessage(content=f"Turn {i} " + "long text string " * 20))

    # Set budget low enough to force truncation
    truncated = truncate_messages_if_needed(msgs, max_tokens=30)
    # First (system) and last (current turn) must be preserved
    assert truncated[0] == msgs[0]
    assert truncated[-1] == msgs[-1]
    assert len(truncated) < len(msgs)
