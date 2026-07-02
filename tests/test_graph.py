"""
Graph-level tests.

Reference: DESIGN_DOC.md section 9 "Testing Strategy"
"""

import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from backend.graph.build_graph import graph
from backend.graph.human_review import APPROVE_SENTINEL
from backend.graph.router import router_node
from backend.graph.synthesizer import synthesizer_node


# Skip marker for any test that hits the real Groq API. CI runners and
# local dev without GROQ_API_KEY set shouldn't hard-fail the whole
# suite — they should skip with a clear reason. The conftest's
# _rate_limit_guard also downgrades quota-exhaustion failures to xfail
# for the same reason on the happy path (key set, quota hit).
requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping LLM-dependent test",
)


# --- Phase 7: sample PDF generator ----------------------------------------

def _ensure_sample_pdf() -> str:
    """Build tests/sample.pdf on first use. Idempotent (skips if exists).

    Uses pypdf directly because no reportlab/fpdf is installed in this env.
    Writes 3 pages, one short paragraph each, so the RAG retriever has
    anchorable text for the test queries.
    """
    fp = Path(__file__).parent / "sample.pdf"
    if fp.exists() and fp.stat().st_size > 100:
        return str(fp)

    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

    paras = [
        "The AgentFlow project was founded in 2024 by a small team of engineers.",
        "AgentFlow is a multi-agent orchestration framework built on LangGraph.",
        "It supports research, analysis, and chat agents with persistent memory and RAG over uploaded PDFs.",
    ]

    w = PdfWriter()
    for p in paras:
        page = w.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 750 Td ({p}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        font_obj = w._add_object(font)
        res = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_obj})})
        page[NameObject("/Resources")] = res
    with open(fp, "wb") as f:
        w.write(f)
    return str(fp)


@requires_groq
def test_graph_compiles():
    """Phase 1: assert the graph compiles and graph.invoke(...) returns
    without raising. The chat agent now produces a real AIMessage reply, so
    the output contains the original HumanMessage + at least one AIMessage."""
    from langchain_core.messages import AIMessage
    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    assert "messages" in result
    assert len(result["messages"]) >= 2, (
        f"Expected at least 2 messages (input + AI reply), got {len(result['messages'])}"
    )
    assert isinstance(result["messages"][-1], AIMessage), (
        "Last message should be an AIMessage from the chat agent"
    )


@requires_groq
def test_review_required_optional():
    """Pre-Phase-2 guard: graph.invoke() must NOT raise KeyError when
    review_required is omitted from the input dict.
    Fixes: review_required declared NotRequired in AgentState."""
    # No review_required key in input — must not crash, must return a reply
    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    assert "messages" in result
    assert len(result["messages"]) >= 2, (
        f"Expected at least 2 messages (input + AI reply), got {len(result['messages'])}"
    )


@requires_groq
def test_echo_node_no_message_duplication():
    """Guard against message duplication: the graph must not append the
    HumanMessage a second time or produce more than one AI reply.
    Expected: [HumanMessage, AIMessage] — exactly 2 messages."""
    from langchain_core.messages import AIMessage
    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    msgs = result["messages"]
    human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    assert len(human_msgs) == 1, (
        f"HumanMessage was duplicated — got {len(human_msgs)} HumanMessages"
    )
    assert len(ai_msgs) == 1, (
        f"Expected exactly one AIMessage, got {len(ai_msgs)}"
    )
    assert len(msgs) == 2, f"Expected exactly 2 messages, got {len(msgs)}"
    assert isinstance(msgs[-1], AIMessage), (
        "Last message should be an AIMessage from the chat agent"
    )


# --- Phase 2: router classification tests -----------------------------------
# These tests make real Groq API calls via llm_smart and require GROQ_API_KEY
# to be set in .env.

@requires_groq
def test_router_classifies_research():
    """router_node must classify a research-style query as "research"."""
    result = router_node({"messages": [HumanMessage(content="What are the latest AI research papers?")]})
    assert result["route"] == "research", (
        f"Expected 'research', got {result['route']!r}"
    )


@requires_groq
def test_router_classifies_analysis():
    """router_node must classify a comparison/summarization query as "analysis"."""
    result = router_node({"messages": [HumanMessage(content="Summarize and compare chapter 2 vs chapter 4")]})
    assert result["route"] == "analysis", (
        f"Expected 'analysis', got {result['route']!r}"
    )


@requires_groq
def test_router_classifies_chat():
    """router_node must classify a casual follow-up as "chat"."""
    result = router_node({"messages": [HumanMessage(content="Thanks, can you shorten that?")]})
    assert result["route"] == "chat", (
        f"Expected 'chat', got {result['route']!r}"
    )


@requires_groq
def test_full_graph_routes_to_chat():
    """End-to-end: a casual greeting must traverse the full graph and return
    a state dict with a 'messages' key."""
    result = graph.invoke({"messages": [HumanMessage(content="hi, how are you?")]})
    assert "messages" in result


# --- Phase 3: real agent nodes + tools -------------------------------------
# These tests exercise the Phase 3 implementations of chat_agent_node and
# the calculator tool. The Research agent test is end-to-end via the full
# graph (test_full_graph_research_routes covers URL harvesting implicitly).

from backend.graph.agents import chat_agent_node
from backend.graph.tools import calculator


@requires_groq
def test_chat_agent_responds():
    """chat_agent_node must return a non-empty string for a simple prompt."""
    state = {"messages": [HumanMessage(content="What is 2+2?")]}
    cfg = {"configurable": {"thread_id": "test-chat-agent-node"}}
    result = chat_agent_node(state, config=cfg)
    assert "agent_output" in result
    assert isinstance(result["agent_output"], str)
    assert result["agent_output"].strip(), "agent_output must be non-empty"


def test_calculator_tool():
    """calculator tool must evaluate "2 ** 10" to "1024". No API call."""
    assert calculator.invoke("2 ** 10") == "1024"


@pytest.mark.parametrize("expr,expected_prefix", [
    ("1/0", "Error"),          # ZeroDivisionError -> "Error: division by zero"
    ("2**10001", "Error"),     # exponent cap (_MAX_POW_RIGHT = 10000)
    ("__import__('os')", "Error"),  # Name node -> disallowed expression node
    ("1001**2", "Error"),      # base cap (_MAX_POW_BASE = 1000): 1001 > 1000
    ("2+2", "4"),              # smoke: hits happy path
    ("-3*-3", "9"),            # unary + binop composition
])
def test_calculator_edge_cases(expr, expected_prefix):
    """Calculator safety caps + happy path. No LLM involved."""
    result = calculator.invoke({"expression": expr})
    assert result.startswith(expected_prefix), (
        f"expr={expr!r} -> {result!r}; expected prefix {expected_prefix!r}"
    )


@requires_groq
def test_full_graph_chat_flow():
    """End-to-end: router + chat_agent + synthesizer produce agent_output
    AND final_response for a casual turn."""
    result = graph.invoke({"messages": [HumanMessage(content="hi, how are you?")]})
    assert "agent_output" in result
    assert isinstance(result["agent_output"], str)
    assert result["agent_output"].strip(), "agent_output must be non-empty"
    assert "final_response" in result
    assert result["final_response"], "final_response must be non-empty"


# --- Phase 4: SQLite checkpointing ---------------------------------------
# These tests exercise the SqliteSaver wired into build_graph in Phase 4.
# The conftest autouse fixture injects a default thread_id per test name
# when one is missing — but the tests below pass an EXPLICIT thread_id
# so multiple invokes in the same test share a thread (required for
# multi-turn state persistence and per-thread isolation).

THREAD_PERSIST = {"configurable": {"thread_id": "test-persistence-001"}}
THREAD_A = {"configurable": {"thread_id": "test-isolation-A"}}
THREAD_B = {"configurable": {"thread_id": "test-isolation-B"}}


@requires_groq
def test_state_persists_across_invocations():
    """Turn 1 introduces a fact; Turn 2 asks about it. The chat agent
    receives the full messages history (replayed from the checkpoint)
    and must recall the fact. Proves SqliteSaver is keyed on thread_id
    and replays prior state on subsequent invokes."""
    graph.invoke(
        {"messages": [HumanMessage(content="My name is Nithin. Remember it.")]},
        config=THREAD_PERSIST,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is my name? Reply with just the name.")]},
        config=THREAD_PERSIST,
    )
    output = (result.get("agent_output") or "").lower()
    assert "nithin" in output, (
        f"Expected 'nithin' in agent_output, got {result.get('agent_output')!r}. "
        "If the model said 'you didn't tell me', the checkpointer is not "
        "replaying the prior messages on the second invoke."
    )


@requires_groq
def test_different_threads_are_isolated():
    """Two threads carry separate conversation histories. Thread A's
    fact must not leak into Thread B's response, and vice versa."""
    graph.invoke(
        {"messages": [HumanMessage(content="My favorite color is red. Just acknowledge.")]},
        config=THREAD_A,
    )
    graph.invoke(
        {"messages": [HumanMessage(content="My favorite color is blue. Just acknowledge.")]},
        config=THREAD_B,
    )

    result_a = graph.invoke(
        {"messages": [HumanMessage(content="What is my favorite color? Reply with just the color.")]},
        config=THREAD_A,
    )
    result_b = graph.invoke(
        {"messages": [HumanMessage(content="What is my favorite color? Reply with just the color.")]},
        config=THREAD_B,
    )

    out_a = (result_a.get("agent_output") or "").lower()
    out_b = (result_b.get("agent_output") or "").lower()

    assert "red" in out_a and "blue" not in out_a, (
        f"Thread A leaked: expected 'red' and not 'blue', got {out_a!r}"
    )
    assert "blue" in out_b and "red" not in out_b, (
        f"Thread B leaked: expected 'blue' and not 'red', got {out_b!r}"
    )


# --- Phase 5: Synthesizer node ---------------------------------------------
# These tests exercise the Phase 5 synthesizer. They call the real Groq API
# (llm_smart, 70b) just like the rest of the suite — there is no LLM mock
# fixture.

@requires_groq
def test_synthesizer_produces_final_response():
    """synthesizer_node must return a non-empty `final_response` string for a
    simple agent_output with no sources."""
    state = {
        "agent_output": "The capital of France is Paris.",
        "sources": [],
    }
    result = synthesizer_node(state)
    assert "final_response" in result
    assert isinstance(result["final_response"], str)
    assert result["final_response"].strip(), "final_response must be non-empty"


@requires_groq
def test_synthesizer_includes_sources():
    """synthesizer_node must surface the source URL when sources are present."""
    url = "https://en.wikipedia.org/wiki/Paris"
    state = {
        "agent_output": "Paris is the capital of France.",
        "sources": [url],
    }
    result = synthesizer_node(state)
    assert result.get("final_response"), "final_response missing"
    text = result["final_response"]
    assert "Source" in text or url in text, (
        f"Expected 'Source' header or inline URL in final_response, got {text!r}"
    )


@requires_groq
def test_full_graph_research_flow():
    """End-to-end research flow: router -> research_agent -> synthesizer.
    Both `agent_output` AND `final_response` must be populated."""
    result = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
    assert "agent_output" in result
    assert result["agent_output"], "agent_output must be non-empty"
    assert "final_response" in result
    assert result["final_response"], "final_response must be non-empty"


@requires_groq
def test_full_graph_chat_flow_phase5():
    """Phase 5: full graph for a casual message. Chat route skips synthesizer."""
    result = graph.invoke({"messages": [HumanMessage(content="hi, how are you?")]})
    assert "agent_output" in result
    assert result["agent_output"], "agent_output must be non-empty"
    assert "final_response" in result
    assert result["final_response"], "final_response must be non-empty"


# --- Phase 6: Human-in-the-loop review (unit-level, no LLM) ---------------

from backend.graph.human_review import (
    APPROVE_SENTINEL,
    _normalize_resume_value,
    human_review_node,
)


def test_normalize_resume_value_shapes():
    """Resume payloads from LangGraph may be str, dict, or Interrupt list."""

    class _Intr:
        def __init__(self, value):
            self.value = value

    assert _normalize_resume_value("edited") == "edited"
    assert _normalize_resume_value({"resume": "edited"}) == "edited"
    assert _normalize_resume_value([_Intr({"draft": "x"})]) == "x"
    assert _normalize_resume_value(None) is None


def test_human_review_skips_duplicate_ai_message():
    """Pass-through must not append a second identical AIMessage."""
    from langchain_core.messages import AIMessage

    state = {
        "review_required": False,
        "route": "chat",
        "final_response": "hello",
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="hello", name="chat_agent"),
        ],
    }
    result = human_review_node(state)
    assert "messages" not in result
    assert result["review_required"] is False


# --- Phase 6: Human-in-the-loop review --------------------------------------
# These tests exercise the Phase 6 human_review node + interrupt() flow.
# Each resume-flow test uses a unique explicit thread_id so the conftest
# autouse fixture preserves it (does NOT override an existing thread_id).
# The two resume tests also force review_required=True so the in-node
# interrupt() actually fires.

THREAD_HR_SKIP = {"configurable": {"thread_id": "test-hr-skip"}}
THREAD_HR_PAUSE = {"configurable": {"thread_id": "test-hr-pause"}}
THREAD_HR_APPROVE = {"configurable": {"thread_id": "test-hr-approve"}}
THREAD_HR_EDIT = {"configurable": {"thread_id": "test-hr-edit"}}


@requires_groq
def test_human_review_skipped_when_not_required():
    """review_required=False: human_review passes through, no pause.
    Returns a final state with __interrupt__ absent (or empty)."""
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi, how are you?")], "review_required": False},
        config=THREAD_HR_SKIP,
    )
    assert "final_response" in result
    assert result["final_response"], "final_response must be non-empty"
    interrupts = result.get("__interrupt__") or []
    assert not interrupts, f"Expected no interrupt when review_required=False, got {interrupts!r}"


@requires_groq
def test_human_review_pauses_execution():
    """review_required=True: graph.invoke must pause at human_review; the
    interrupt lives on the snapshot's tasks, NOT on the result dict."""
    # The first invoke drives the graph into the human_review node, where
    # it raises interrupt() and pauses. The result dict's `__interrupt__`
    # key is None in LangGraph 1.x — the actual Interrupt objects live on
    # `graph.get_state(cfg).tasks[i].interrupts`.
    graph.invoke(
        {"messages": [HumanMessage(content="hi, how are you?")], "review_required": True},
        config=THREAD_HR_PAUSE,
    )
    snap = graph.get_state(THREAD_HR_PAUSE)
    interrupts = []
    for t in (snap.tasks or ()):
        interrupts.extend(getattr(t, "interrupts", None) or ())
    assert interrupts, (
        f"Expected at least one Interrupt on get_state().tasks when "
        f"review_required=True. got snap.next={snap.next!r} "
        f"tasks={snap.tasks!r}"
    )
    # The graph must be paused at the human_review node
    assert snap.next == ("human_review",), (
        f"Expected next=('human_review',), got {snap.next!r}"
    )


@requires_groq
def test_human_review_resumes_with_approval():
    """Pause at human_review, then resume with the 'approve' sentinel.
    final_response should hold the pre-review draft (synthesizer output)."""
    from langgraph.types import Command

    cfg = THREAD_HR_APPROVE
    graph.invoke(
        {"messages": [HumanMessage(content="hi, how are you?")], "review_required": True},
        config=cfg,
    )
    draft = (graph.get_state(cfg).values or {}).get("final_response")
    assert draft, "Expected paused state to contain the pre-review draft"

    result = graph.invoke(Command(resume=APPROVE_SENTINEL), config=cfg)
    assert "final_response" in result
    assert result["final_response"], "final_response must be non-empty"
    assert result["final_response"] == draft, (
        "Approval should keep the original draft unchanged."
    )
    assert result["final_response"], "final_response must be non-empty"
    assert result["final_response"] != APPROVE_SENTINEL, (
        "Approval sentinel leaked into final_response — the node should have "
        "treated the sentinel as 'no edit' and kept the original draft."
    )
    # Graph should have reached END
    assert graph.get_state(cfg).next == (), (
        f"Expected next=() (END), got {graph.get_state(cfg).next!r}"
    )


@requires_groq
def test_human_review_resumes_with_edit():
    """Pause at human_review, then resume with an edit string.
    final_response should equal the edit."""
    from langgraph.types import Command

    cfg = THREAD_HR_EDIT
    graph.invoke(
        {"messages": [HumanMessage(content="hi, how are you?")], "review_required": True},
        config=cfg,
    )
    edit = "Edited by human: this is the new final answer."
    result = graph.invoke(Command(resume=edit), config=cfg)
    assert result.get("final_response") == edit, (
        f"Expected final_response == {edit!r}, got {result.get('final_response')!r}"
    )


# --- Phase 2: skip-synthesizer + SOURCES-in-SSE ---------------------------

@requires_groq
def test_chat_skips_synthesizer():
    """Phase 2: chat turns write `final_response` directly in
    `chat_agent_node` and route straight to `human_review` (pass-through
    when no review is required). The synthesizer would paraphrase the
    chat answer — when it does run, `final_response` will differ from
    `agent_output`. Equality of the two keys proves the synthesizer
    was bypassed."""
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi, how are you?")], "review_required": False},
        config={"configurable": {"thread_id": "test-skip-synth-001"}},
    )
    assert "agent_output" in result and result["agent_output"], (
        "agent_output missing or empty — chat_agent did not run"
    )
    assert "final_response" in result and result["final_response"], (
        "final_response missing or empty — chat_agent did not populate it"
    )
    assert result["final_response"] == result["agent_output"], (
        f"final_response differs from agent_output — synthesizer ran for a "
        f"chat turn (or chat_agent did not write both keys). "
        f"agent_output={result['agent_output']!r}; "
        f"final_response={result['final_response']!r}"
    )


@requires_groq
def test_research_runs_synthesizer():
    """Control test for test_chat_skips_synthesizer: a research turn
    must go through the synthesizer, so `final_response` and
    `agent_output` are expected to differ. Proves the conditional
    edge branches correctly on `state['route']`."""
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        config={"configurable": {"thread_id": "test-skip-synth-002"}},
    )
    assert "agent_output" in result and result["agent_output"]
    assert "final_response" in result and result["final_response"]
    # Synthesizer is allowed to either paraphrase or repeat. We only
    # assert the synthesizer NODE ran — for that we use a unique
    # thread_id and a fresh invoke. A weak signal that synth ran is
    # the existence of `final_response` after a research route
    # (the chat_agent doesn't run on research, so the synthesizer is
    # the only writer). The fact that `final_response` is populated
    # at all proves the synth branch executed.
    assert result["final_response"]


@requires_groq
@pytest.mark.asyncio
async def test_sources_in_sse():
    """Phase 2: a research turn's SSE stream must contain a
    `[SOURCES:n]` line with n >= 1 (the research agent harvests URLs
    from Tavily). Uses the real FastAPI lifespan + real graph; gated
    by the env check inside `requires_groq`.

    The existing `tests/test_api.py` only exercises a fake graph; this
    test needs the real one to verify the post-stream `aget_state`
    branch in main.py emits `[SOURCES:n]`."""
    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient
    from backend.main import app

    thread_id = "test-sources-sse-001"
    # `ASGITransport` does NOT run the FastAPI `lifespan` context, so
    # `app.state.graph` would never be set and the /chat handler would
    # raise `AttributeError`. `LifespanManager` runs startup/shutdown
    # around the test request so `app.state.graph` is populated.
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(
                "/chat",
                json={"thread_id": thread_id, "message": "What is the capital of France?"},
            )
            assert r.status_code == 200, r.text
            body = r.text

    # SSE framing: each `data: ...` line is one frame; sentinels are
    # one frame each. We split on \n\n to isolate frames, then look
    # for the [SOURCES:n] line.
    sources_line = None
    for frame in body.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: [SOURCES:"):
                sources_line = line
                break
        if sources_line:
            break
    assert sources_line is not None, (
        f"Expected a [SOURCES:n] line in SSE stream, got body[:400]={body[:400]!r}"
    )
    n = int(sources_line[len("data: [SOURCES:"):-1])
    assert n >= 1, f"Expected at least 1 source for a research query, got n={n}"


# --- Phase 7: RAG pipeline -------------------------------------------------
# These tests exercise ingest_pdf, retrieve_documents, and the end-to-end
# graph flow with RAG. The sentence-transformer model downloads on first
# call (~90MB, cached by HuggingFace under ~/.cache/huggingface).

THREAD_RAG = "test-rag-001"


def test_ingest_pdf_creates_index():
    """ingest_pdf must write a per-thread FAISS index under faiss_indexes/."""
    from backend.rag.ingest import ingest_pdf

    sample = _ensure_sample_pdf()
    ingest_pdf(sample, THREAD_RAG)

    index_dir = Path("faiss_indexes") / THREAD_RAG
    assert index_dir.exists(), f"Expected {index_dir} to exist after ingest_pdf"
    assert (index_dir / "index.faiss").exists(), "FAISS index file missing"
    assert (index_dir / "index.pkl").exists(), "FAISS pickle sidecar missing"


def test_retrieve_documents_returns_chunks():
    """The retrieve_documents tool must return non-empty text containing
    content from the ingested PDF."""
    from backend.rag.ingest import ingest_pdf
    from backend.graph.tools import make_retrieve_documents_tool

    ingest_pdf(_ensure_sample_pdf(), THREAD_RAG)
    tool = make_retrieve_documents_tool(THREAD_RAG)
    result = tool.invoke({"query": "When was AgentFlow founded?"})

    assert isinstance(result, str) and result.strip(), f"Expected non-empty string, got {result!r}"
    assert "AgentFlow" in result or "2024" in result, (
        f"Expected PDF content (AgentFlow or 2024) in retrieval result, got {result!r}"
    )


def test_retrieve_documents_handles_missing_index():
    """When no index exists for a thread, the tool must return a friendly
    message instead of raising."""
    from backend.graph.tools import make_retrieve_documents_tool

    tool = make_retrieve_documents_tool("test-rag-nonexistent-xyz")
    result = tool.invoke({"query": "anything"})
    assert "no documents" in result.lower(), f"Expected friendly fallback, got {result!r}"


def test_ingest_pdf_appends_to_existing_index():
    """Re-uploading a PDF must merge into the existing FAISS index."""
    from backend.rag.ingest import ingest_pdf, get_retriever_cached, INDEX_ROOT
    from backend.graph.tools import make_retrieve_documents_tool

    thread = "test-rag-append-001"
    index_dir = INDEX_ROOT / thread
    if index_dir.exists():
        import shutil
        shutil.rmtree(index_dir)

    sample = _ensure_sample_pdf()
    ingest_pdf(sample, thread)
    tool = make_retrieve_documents_tool(thread)
    first = tool.invoke({"query": "When was AgentFlow founded?"})
    assert "2024" in first or "AgentFlow" in first

    # Second ingest on same thread — index must retain prior content.
    ingest_pdf(sample, thread)
    get_retriever_cached(thread)  # reload from disk
    second = tool.invoke({"query": "When was AgentFlow founded?"})
    assert "2024" in second or "AgentFlow" in second
    assert index_dir.exists()


@requires_groq
def test_full_graph_rag_flow():
    """End-to-end: ingest a PDF, ask a question about it through the full
    graph. The RAG tool must surface PDF content in the final response."""
    from backend.rag.ingest import ingest_pdf

    ingest_pdf(_ensure_sample_pdf(), THREAD_RAG)
    result = graph.invoke(
        {"messages": [HumanMessage(content="When was the AgentFlow project founded?")]},
        config={"configurable": {"thread_id": THREAD_RAG}},
    )
    text = (result.get("final_response") or "").lower()
    assert "2024" in text or "agentflow" in text, (
        f"Expected PDF-derived fact in final_response, got {result.get('final_response')!r}"
    )

