"""Tests for backend.graph.messages.content_to_str."""

from langchain_core.messages import ToolMessage

from backend.graph.messages import content_to_str, extract_sources


def test_content_to_str_plain():
    assert content_to_str("hello") == "hello"


def test_content_to_str_list_blocks():
    blocks = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
    assert content_to_str(blocks) == "hello"


def test_content_to_str_none():
    assert content_to_str(None) == ""


def test_retrieved_document_labels_are_counted_as_sources():
    message = ToolMessage(
        content="[1] <<UNTRUSTED EXCERPT — source: report.pdf, page 2>> revenue <<END EXCERPT>>",
        name="retrieve_documents",
        tool_call_id="retrieval-1",
    )
    assert extract_sources([message]) == ["report.pdf, page 2"]
