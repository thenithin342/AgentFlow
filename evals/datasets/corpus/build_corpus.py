"""Build the golden retrieval corpus PDFs for offline evals.

Deterministic: regenerates every PDF on each run from the source paragraphs
(same inputs -> same bytes), so edited paragraph sources always refresh the
corpus — stale content can never linger behind an existing file. Uses pypdf
directly (same writer pattern as tests/test_graph.py ``_ensure_sample_pdf``)
so no reportlab/fpdf dependency is needed.

One fact per page, one sentence per fact, so chunks stay anchorable:
``retriever.jsonl`` ``relevant_text`` anchors and ``rag_qa.jsonl``
``expected_facts`` are verbatim substrings of these pages.

Content is ASCII-only (the writer encodes latin-1) and avoids PDF
literal-string metacharacters; ``_escape`` still guards parens/backslash.

Slugs (used as ``source_doc`` in retriever.jsonl / rag_qa.jsonl):
    agentflow_facts    - AgentFlow company / product facts
    langgraph_reference - brief LangGraph multi-agent architecture reference
    northwind_2024     - fictional Northwind business dataset (FY2024)

Northwind numbers are internally consistent: product lines
(4.1 + 2.8 + 3.2 + 2.3 = 12.4) and regions (5.0 + 3.1 + 2.5 + 1.8 = 12.4)
both sum to total revenue; 12.4 - 7.9 expenses = 4.5 net profit.
"""

from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent

AGENTFLOW_FACTS = [
    "AgentFlow was founded in 2024 by a small team of engineers.",
    "AgentFlow is a multi-agent orchestration framework built on LangGraph.",
    "AgentFlow routes every user query through a router node that classifies intent as research, analysis, chat, or blog.",
    "The Research agent answers current-events questions using Tavily web search and Wikipedia.",
    "The Analysis agent performs comparisons, summaries, and calculations using a safe calculator and a Python code interpreter.",
    "The Chat agent handles casual follow-ups directly with no tool calls for low latency.",
    "The Blog Writer agent produces structured SEO-friendly posts with a title, meta description, tags, and sections.",
    "The synthesizer node rewrites raw agent output into the final user-facing response.",
    "Conversation state persists across restarts in SQLite, keyed by thread_id.",
    "When review mode is enabled, a human-review checkpoint pauses for approval or edits before responding.",
    "AgentFlow supports retrieval-augmented generation over user-uploaded PDFs indexed into FAISS.",
    "Long-term memory stores user facts across threads, and short-term memory compresses older turns into a summary.",
]

LANGGRAPH_REFERENCE = [
    "LangGraph is a library for building stateful multi-agent applications as graphs.",
    "A LangGraph graph is composed of nodes, which are Python functions, and edges, which define transitions.",
    "Conditional edges route execution dynamically based on the current state.",
    "The add_messages reducer appends new messages to conversation state instead of overwriting it.",
    "A checkpointer such as SqliteSaver persists graph state across runs, keyed by thread_id.",
    "The interrupt function pauses graph execution to wait for human input.",
    "A Command object with a resume value restarts a paused graph with the human decision.",
    "The create_react_agent helper wraps a language model in a reason-and-act loop bound to tools.",
    "Subgraphs let teams compose smaller graphs into a larger multi-agent workflow.",
    "LangGraph supports streaming so tokens reach the user interface as they are generated.",
]

NORTHWIND_2024 = [
    "Northwind Traders reported total revenue of 12.4 million dollars for fiscal year 2024.",
    "The Beverages product line led sales with 4.1 million dollars in revenue.",
    "The Condiments product line earned 2.8 million dollars in revenue.",
    "The Confections product line earned 3.2 million dollars in revenue.",
    "The Seafood product line earned 2.3 million dollars in revenue.",
    "The North region contributed 5.0 million dollars, the largest regional share.",
    "The South region contributed 3.1 million dollars in revenue.",
    "The East region contributed 2.5 million dollars in revenue.",
    "The West region contributed 1.8 million dollars in revenue.",
    "Chai was the top-selling product with 850 thousand dollars in sales.",
    "Operating expenses for 2024 totaled 7.9 million dollars.",
    "Net profit for 2024 was 4.5 million dollars, a margin of about 36 percent.",
]

DOCS = {
    "agentflow_facts": AGENTFLOW_FACTS,
    "langgraph_reference": LANGGRAPH_REFERENCE,
    "northwind_2024": NORTHWIND_2024,
}


def _escape(text: str) -> str:
    """Escape PDF literal-string metacharacters (backslash, parens)."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(path: Path, paras: list) -> None:
    """Write one paragraph per page (writer pattern from tests/test_graph.py)."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for para in paras:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 750 Td ({_escape(para)}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_obj = writer._add_object(font)
        resources = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_obj})}
        )
        page[NameObject("/Resources")] = resources
    with open(path, "wb") as handle:
        writer.write(handle)


def build_corpus() -> list:
    """Regenerate all corpus PDFs from the source paragraphs; return paths."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    built = []
    for slug, paras in DOCS.items():
        path = CORPUS_DIR / f"{slug}.pdf"
        _write_pdf(path, paras)
        print(f"wrote {path.name} ({len(paras)} pages)")
        built.append(path)
    return built


def main() -> None:
    build_corpus()


if __name__ == "__main__":
    main()
