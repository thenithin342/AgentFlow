"""
Graph construction — wires every node and edge together into the compiled
AgentFlow StateGraph.

Phase 9 (current): Added memory nodes, blog writer, and new tools.
    - memory_reader_node fires at START (injects LTM context)
    - stm_compressor_node fires conditionally after human_review
    - memory_writer_node fires after STM (or directly after human_review)
    - blog_writer_node added as a 4th agent route

Phase 3: real router + 3 agent nodes (research, analysis, chat). Each
agent goes straight to END — the Synthesizer arrives in Phase 5 and
Human Review arrives in Phase 6.

Phase 5: Synthesizer node added. All three agents now route through
`synthesizer` before END. Human Review arrives in Phase 6.

Phase 4: SqliteSaver checkpointer wired in. The compiled graph holds a
live reference to the checkpointer (and therefore the underlying
sqlite3.Connection), so the DB file stays locked for the lifetime of
this process. See the comment block above `graph = builder.compile(...)`
for thread_id requirements.

Phase 8: graph topology and the checkpointer/DB side effect are now
separated. `builder` is the uncompiled StateGraph (topology only).
`build_compiled_graph(db_path=None)` is the factory that owns the
sqlite3 connection + SqliteSaver + compile call.

`get_default_graph()` returns the lazily-created default compiled
instance used by the test suite and CLI — it is NOT built at import
time so that `backend/main.py` importing `builder` does not open a
second sync SQLite connection before the async lifespan one does.

Reference: DESIGN_DOC.md section 1 "Architecture Overview",
section 4 "Routing Logic", section 5 "Persistence Design"
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.graph.agents import (
    analysis_agent_node,
    chat_agent_node,
    research_agent_node,
)
from backend.graph.blog_agent import blog_writer_node
from backend.graph.human_review import human_review_node
from backend.graph.memory_nodes import (
    memory_reader_node,
    memory_writer_node,
    should_run_stm,
    stm_compressor_node,
)
from backend.graph.router import route_query, router_node
from backend.graph.state import AgentState
from backend.graph.synthesizer import synthesizer_node

# --- Graph topology (pure — no DB, no side effects) ------------------------

builder = StateGraph(AgentState)

# Memory nodes
builder.add_node("memory_reader", memory_reader_node)
builder.add_node("memory_writer", memory_writer_node)
builder.add_node("stm_compressor", stm_compressor_node)

# Core pipeline nodes
builder.add_node("router", router_node)
builder.add_node("research_agent", research_agent_node)
builder.add_node("analysis_agent", analysis_agent_node)
builder.add_node("chat_agent", chat_agent_node)
builder.add_node("blog_writer", blog_writer_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("human_review", human_review_node)

# START → memory_reader → router
builder.add_edge(START, "memory_reader")
builder.add_edge("memory_reader", "router")

# Router → agents (conditional)
builder.add_conditional_edges(
    "router",
    route_query,
    {
        "research": "research_agent",
        "analysis": "analysis_agent",
        "chat": "chat_agent",
        "blog": "blog_writer",
    },
)

# Research + Analysis → synthesizer
builder.add_edge("research_agent", "synthesizer")
builder.add_edge("analysis_agent", "synthesizer")


# Chat and blog turns write `final_response` directly and skip the synthesizer.
# The conditional edge branches on state["route"].
def route_query_to_synth(state: AgentState) -> str:
    route = state.get("route")
    if route in {"research", "analysis"}:
        return "synthesizer"
    return "human_review"


builder.add_conditional_edges(
    "chat_agent",
    route_query_to_synth,
    {"synthesizer": "synthesizer", "human_review": "human_review"},
)

# Blog agent writes final_response + blog_output directly, skips synthesizer
builder.add_edge("blog_writer", "human_review")

builder.add_edge("synthesizer", "human_review")

# human_review → conditional: STM compression or straight to memory_writer
builder.add_conditional_edges(
    "human_review",
    should_run_stm,
    {"stm_compressor": "stm_compressor", "memory_writer": "memory_writer"},
)
builder.add_edge("stm_compressor", "memory_writer")
builder.add_edge("memory_writer", END)


# --- Checkpointer / DB side effect ---------------------------------------
#
# SqliteSaver (sync) is compiled into the graph here. The DB file is held
# open for the lifetime of the process (Windows refuses to delete a locked
# file), so test-suite cleanup uses TRUNCATE rather than unlink — see
# `tests/conftest.py::_clean_checkpoint_db`.
#
# THREAD_ID IS REQUIRED:
#   Every call to graph.invoke / graph.ainvoke / graph.astream_events
#   MUST pass config={"configurable": {"thread_id": "<id>"}}.
#   Without it, LangGraph 1.x raises ValueError("Missing thread_id").
#   Use a stable ID per browser session (or per "new conversation"
#   click) to scope memory — different thread_ids do NOT share state.
#
#   Example:
#     config = {"configurable": {"thread_id": "user-42-conv-7"}}
#     result = graph.invoke({"messages": [HumanMessage(...)]}, config)
#
# Override the default path with the CHECKPOINT_DB_PATH env var.
# PHASE 8 NOTE — async streaming upgrade:
#   The FastAPI server does NOT import `graph` from here for streaming.
#   It creates its own AsyncSqliteSaver inside the FastAPI async lifespan
#   context and re-compiles the graph there.
#
# This is the correct LangGraph pattern: sync graph for tests/CLI,
# async graph (re-compiled with AsyncSqliteSaver) for the API server.
_DEFAULT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "agentflow.db")


def build_compiled_graph(db_path: str | None = None) -> CompiledStateGraph:
    """Compile `builder` with a fresh sync SqliteSaver checkpointer.

    Args:
        db_path: Path to the SQLite file. `None` → env-defaulted
            `CHECKPOINT_DB_PATH` ("agentflow.db"). Pass `":memory:"` for a
            transient, per-call checkpointer (useful in unit tests that
            want full isolation without touching the on-disk DB).

    Returns:
        A compiled `CompiledStateGraph` ready for `.invoke(...)`. The
        underlying sqlite3 connection is held open for the lifetime of
        the returned graph.
    """
    path = db_path if db_path is not None else _DEFAULT_DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False)
    cp = SqliteSaver(conn)
    g = builder.compile(checkpointer=cp)
    g.name = "AgentFlow"
    return g


# ---------------------------------------------------------------------------
# Lazy default graph — used by the test suite and CLI tools.
# ---------------------------------------------------------------------------

import threading

_default_graph: CompiledStateGraph | None = None
_default_graph_lock = threading.Lock()
# Serializes all sync .invoke() / .stream() / .astream_events() calls into the
# default graph. The default graph's SqliteSaver holds a single sync
# sqlite3.Connection with `check_same_thread=False`; even with that flag, the
# Python sqlite3 module serialises the *Connection* via an internal lock but
# does NOT serialise LangGraph's per-call cursor + checkpoint write/read
# sequences — concurrent sync calls on the same connection can interleave
# cursor state and corrupt checkpoint rows. This lock wraps the entire call
# so the connection sees one request at a time.
# RLock (not Lock) so the proxy's `__getattr__` wrapper can re-acquire when a
# caller like `tests/conftest.py::_default_thread_id` wraps the locked entry
# point with a thread_id injector: that wrapper calls the original (locked)
# entry point, and the proxy's getter re-wraps the patched version. The
# second acquire is on the same thread → RLock succeeds where Lock would
# deadlock. Async callers (FastAPI astream_events/ainvoke in main.py) use a
# separate AsyncSqliteSaver and are NOT routed through this lock.
_invoke_lock = threading.RLock()

_SYNC_ENTRY_POINTS = frozenset({"invoke", "stream", "astream_events"})


def get_default_graph() -> CompiledStateGraph:
    """Return the lazily-created default sync compiled graph."""
    global _default_graph
    if _default_graph is None:
        with _default_graph_lock:
            if _default_graph is None:
                _default_graph = build_compiled_graph()
    return _default_graph


class _GraphProxy:
    """Thin proxy that forwards every attribute to the lazily-built default
    graph.  This preserves the ``from build_graph import graph`` import
    style used in tests/conftest.py and test_graph.py while avoiding
    the eager DB connection at import time.

    `invoke` / `stream` / `astream_events` are funnelled through
    `_invoke_lock` so concurrent sync calls on the underlying
    sqlite3.Connection cannot interleave checkpoint reads/writes. Async
    callers (FastAPI astream_events in main.py) use a different
    AsyncSqliteSaver and are NOT routed through this lock.
    """

    def __getattr__(self, name: str):
        target = getattr(get_default_graph(), name)
        if name in _SYNC_ENTRY_POINTS:
            def locked_call(*args, **kwargs):
                with _invoke_lock:
                    return target(*args, **kwargs)
            return locked_call
        return target

    def __setattr__(self, name: str, value):
        setattr(get_default_graph(), name, value)


graph = _GraphProxy()
