"""
Graph construction — wires every node and edge together into the compiled
AgentFlow StateGraph.

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
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from backend.graph.state import AgentState
from backend.graph.router import router_node, route_query
from backend.graph.agents import (
    research_agent_node,
    analysis_agent_node,
    chat_agent_node,
)
from backend.graph.synthesizer import synthesizer_node
from backend.graph.human_review import human_review_node


# --- Graph topology (pure — no DB, no side effects) ------------------------

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("research_agent", research_agent_node)
builder.add_node("analysis_agent", analysis_agent_node)
builder.add_node("chat_agent", chat_agent_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("human_review", human_review_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_query,
    {
        "research": "research_agent",
        "analysis": "analysis_agent",
        "chat": "chat_agent",
    },
)
builder.add_edge("research_agent", "synthesizer")
builder.add_edge("analysis_agent", "synthesizer")


# Phase 2: chat turns write `final_response` directly in chat_agent_node
# and skip the synthesizer. Research/analysis still pass through synth to
# polish the answer into `final_response`. The conditional edge branches
# on `state["route"]` set by the router — non-chat routes go to synth,
# chat goes straight to human_review (pass-through when no review needed).
def route_query_to_synth(state: AgentState) -> str:
    return "synthesizer" if state.get("route") in {"research", "analysis"} else "human_review"


builder.add_conditional_edges(
    "chat_agent",
    route_query_to_synth,
    {"synthesizer": "synthesizer", "human_review": "human_review"},
)
builder.add_edge("synthesizer", "human_review")
builder.add_edge("human_review", END)


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
#   context and re-compiles the graph there:
#
#     from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
#     import aiosqlite
#     async def lifespan(app):
#         async with aiosqlite.connect(db_path) as conn:
#             checkpointer = AsyncSqliteSaver(conn)
#             app.state.graph = builder.compile(checkpointer=checkpointer)
#             yield
#
# This is the correct LangGraph pattern: sync graph for tests/CLI,
# async graph (re-compiled with AsyncSqliteSaver) for the API server.
# TODO: implement retention policy before production (Phase 7+).
_DEFAULT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "agentflow.db")


def build_compiled_graph(db_path: Optional[str] = None) -> CompiledStateGraph:
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
#
# We do NOT call build_compiled_graph() at module import time because
# backend/main.py imports `builder` from this module. That import would
# trigger build_compiled_graph() and open a *second* sync SQLite
# connection on the same DB file before the FastAPI lifespan opens its
# async one — undermining the topology/side-effect separation and
# potentially locking the DB on Windows.
#
# Call get_default_graph() explicitly in tests and CLI scripts instead
# of relying on the module-level `graph` name.
# ---------------------------------------------------------------------------

_default_graph: CompiledStateGraph | None = None


def get_default_graph() -> CompiledStateGraph:
    """Return the lazily-created default sync compiled graph.

    The graph (and its underlying sqlite3 connection) is built on the
    first call and reused for all subsequent calls within the same
    process.  Tests and CLI tools should call this instead of importing
    the old module-level ``graph`` name.
    """
    global _default_graph
    if _default_graph is None:
        _default_graph = build_compiled_graph()
    return _default_graph


# Backwards-compatible alias so that
#   from backend.graph.build_graph import graph
# still works in existing test files without modification.
# The property resolves lazily on first attribute access.
class _GraphProxy:
    """Thin proxy that forwards every attribute to the lazily-built default
    graph.  This preserves the ``from build_graph import graph`` import
    style used in tests/conftest.py and test_graph.py while avoiding
    the eager DB connection at import time."""

    def __getattr__(self, name: str):
        return getattr(get_default_graph(), name)

    def __setattr__(self, name: str, value):
        setattr(get_default_graph(), name, value)


graph = _GraphProxy()
