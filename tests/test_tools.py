"""Tool-level unit tests (no LLM)."""

from unittest.mock import patch

import pytest

from backend.graph.tools import tavily_search


def test_tavily_search_handles_api_failure():
    with patch("backend.graph.tools._get_tavily") as mock_get:
        mock_get.return_value.invoke.side_effect = RuntimeError("quota exceeded")
        result = tavily_search.invoke("test query")
    assert result.startswith("Error: web search unavailable")

def test_wikipedia_search_handles_api_failure():
    from backend.graph.tools import wikipedia_search
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = wikipedia_search.invoke("test query")
    assert result.startswith("Error: Wikipedia lookup failed")


@pytest.mark.parametrize("wrapped", [True, False])
def test_search_results_keep_sources_without_unbounded_payloads(monkeypatch, wrapped):
    from types import SimpleNamespace

    import backend.graph.tools as tools

    rows = [{"title": f"Result {i}", "url": f"https://example.com/{i}",
             "content": "x" * 10000, "raw_content": "UNUSED RAW PAGE"} for i in range(3)]
    response = {"results": rows} if wrapped else rows
    monkeypatch.setattr(tools, "_get_tavily", lambda: SimpleNamespace(invoke=lambda _query: response))
    result = tools.tavily_search.invoke("latest release")
    assert len(result) < 7000
    assert all(f"https://example.com/{i}" in result for i in range(3))
    assert "UNUSED RAW PAGE" not in result
