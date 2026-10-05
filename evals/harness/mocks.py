"""
Test stubs for harness unit tests.

Provides zero-LLM fake objects for testing report.py and other harness
components without making real LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeLLMResponse:
    """Minimal AI message stub for harness unit tests.

    Mimics the interface of a LangChain ChatModel response enough to
    exercise token tracking and response parsing code paths.

    Usage:
        response = FakeLLMResponse(content='{"score": 0.8}', usage={"input": 100, "output": 50})
    """

    content: str
    usage: dict[str, int] = field(default_factory=dict)

    # response_metadata mimics ChatGroq's structure
    response_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sync response_metadata["usage"] with usage dict
        if self.usage and "usage" not in self.response_metadata:
            self.response_metadata["usage"] = self.usage

    @property
    def usage_metadata(self) -> dict[str, int] | None:
        """Provide usage_metadata attribute for token extraction."""
        if self.usage:
            return {
                "input_tokens": self.usage.get("input", 0),
                "output_tokens": self.usage.get("output", 0),
                "total_tokens": self.usage.get("total", self.usage.get("input", 0) + self.usage.get("output", 0)),
            }
        return None


def fake_graph_invoke(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a canned AgentState dict for testing report.py without real calls.

    Args:
        state: optional base state to extend. If None, returns a default
               fake state with sample data.

    Returns:
        A dict that looks enough like an AgentState for testing purposes.
    """
    if state is None:
        state = {}

    return {
        **state,
        "messages": [
            {
                "type": "human",
                "content": "What are the latest AI research papers?",
            },
        ],
        "route": "research",
        "agent_output": "Here are some recent AI research papers...",
        "sources": [
            {"title": "Paper 1", "url": "https://example.com/paper1"},
            {"title": "Paper 2", "url": "https://example.com/paper2"},
        ],
        "final_response": "Based on my research, here are the latest AI papers...",
        "router_fallback": False,
        "blog_output": None,
    }


def fake_router_rows() -> list[dict]:
    """Return a small set of fake router evaluation rows for testing."""
    return [
        {"input": "What are AI papers?", "expected": "research", "reason": "test"},
        {"input": "Summarize this", "expected": "analysis", "reason": "test"},
        {"input": "Write a blog", "expected": "blog", "reason": "test"},
        {"input": "Hello", "expected": "chat", "reason": "test"},
        {"input": "Hi there", "expected": "chat", "reason": "test"},
    ]


def fake_predictions() -> list[str]:
    """Return fake predictions (mostly correct, one wrong) for testing."""
    return ["research", "analysis", "blog", "chat", "analysis"]  # Last one wrong
