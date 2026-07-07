"""Tool-level unit tests (no LLM)."""

from unittest.mock import MagicMock, patch

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
