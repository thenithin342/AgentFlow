# AgentFlow — Technical Design Document

**Version:** 1.0
**Companion to:** `PRD.md`, `TECH_STACK.md`

---

## 1. Architecture Overview

AgentFlow is implemented as a single LangGraph `StateGraph` with seven nodes: Router, Research Agent, Analysis Agent, Chat Agent, Synthesizer, Human Review, and an implicit terminal node (`END`). The graph is wrapped by a FastAPI backend that streams execution events to a React frontend. See `architecture.svg` for the visual diagram.

```
User Query → Router → Research Agent ─┐
              ├────── Analysis Agent ─┼─→ Synthesizer → Human Review → Final Output
              └────── Chat Agent ─────┘
```

The Router picks one of the three agents per turn (current implementation — see §4). A future extension is to fan out to multiple agents in parallel for compound queries; LangGraph supports this via fan-out edges re-joining at the Synthesizer, but it is not in scope today.

All nodes read and write a single shared state object. Execution is checkpointed after every node transition, so the graph can be paused, resumed, or replayed from any point in its history.

---

## 2. State Schema

The graph state is a `TypedDict` (or `Annotated` `BaseModel` if stricter validation is needed). This is the contract every node reads from and writes to.

```python
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]      # full conversation history
    route: Optional[Literal["research", "analysis", "chat"]]
    agent_output: Optional[str]                  # raw output from the dispatched agent
    sources: Optional[list[str]]                 # citations gathered during research
    documents: Optional[list[str]]                # IDs of uploaded docs relevant to this thread
    review_required: bool                          # whether human-in-the-loop is active
    final_response: Optional[str]
```

`messages` uses LangGraph's built-in `add_messages` reducer so that each node can simply append to the list rather than manually merging conversation history — this is what makes multi-turn memory work correctly under the checkpointer.

---

## 3. Node Specifications

**Router.** Pure LLM classification call, no tools. Takes `messages[-1]` (the latest user turn) plus a short system prompt with few-shot examples of each category, and returns one of `research | analysis | chat`. Implemented as a conditional edge function, not a regular node — `add_conditional_edges(START, route_query, {...})`. Uses `llm_smart` (70b) since misclassification cascades into the wrong agent entirely.

**Research Agent.** Built with `create_react_agent` or a manual `ToolNode` loop. Tools: `tavily_search` (web search) and `retrieve_documents` (FAISS similarity search over uploaded PDFs). Uses `llm_fast` (8b) since tool-calling doesn't require heavy reasoning, just reliable function-call formatting.

**Analysis Agent.** Same scaffolding as Research, with tools: `python_repl` or a sandboxed calculator tool, plus `retrieve_documents` for comparing document sections. Uses `llm_fast`.

**Chat Agent.** No tools bound. Directly calls the LLM with the full message history and returns a response. This is the fast path — most casual turns should resolve here without touching external APIs. Uses `llm_fast`.

**Synthesizer.** Takes whichever agent(s) ran and their `agent_output` plus `sources`, and produces a single coherent `final_response` with citations formatted. Uses `llm_smart` since this is the last quality gate before the user sees anything.

**Human Review.** Implemented with LangGraph's `interrupt()` function inside the node. When `review_required` is `True`, the node raises an interrupt that pauses graph execution and returns control to the calling application; the frontend displays the draft and the human approves or edits it; the decision is fed back via `graph.invoke(Command(resume=...))` (the LangGraph 1.x native resume primitive), where `resume="approve"` keeps the existing draft and any other string replaces `final_response` with the edit. Execution then resumes from the paused checkpoint and drains to `END`.

---

## 4. Routing Logic

The conditional edge function inspects the router's classification and returns the matching node name as a string, which LangGraph uses to select the next node:

```python
def route_query(state: AgentState) -> str:
    return state["route"]

graph.add_conditional_edges(
    "router",
    route_query,
    {"research": "research_agent", "analysis": "analysis_agent", "chat": "chat_agent"},
)
```

A future extension (stretch goal) is to allow `route` to be a list rather than a single value, so the router can dispatch to multiple agents in parallel for compound queries (e.g. "research X and also calculate Y") — LangGraph supports this via fan-out edges, with all branches re-joining at the Synthesizer.

---

## 5. Persistence Design

Checkpointing uses `SqliteSaver` (or `PostgresSaver` if Postgres is available, per Nithin's existing AgentFlow checkpointer work) keyed by `thread_id`. Every node transition writes a new checkpoint, so:

- Conversations resume exactly where they left off after a backend restart.
- The human-review interrupt relies on this same mechanism — pausing is just "stop after this checkpoint and wait."
- A `thread_id` per browser session (or per explicit "new conversation" action) scopes memory correctly so users don't see each other's history.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("agentflow.db")
graph = builder.compile(checkpointer=checkpointer, interrupt_after=["synthesizer"])
```

---

## 6. RAG Pipeline

Document upload triggers: PDF text extraction → recursive character text splitting (chunk size ~800, overlap ~150) → embedding via a local sentence-transformer or the Groq/Gemini embedding endpoint → upsert into a per-thread FAISS index. The Research agent's `retrieve_documents` tool performs a similarity search (top-k = 4) against this index and returns the matched chunks as tool output, which the agent then cites in its response.

---

## 7. API Design (FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Submit a message for a given `thread_id`; returns a streaming response |
| `/upload` | POST | Upload a PDF; triggers the RAG ingestion pipeline |
| `/threads/{thread_id}/state` | GET | Fetch current graph state (for debugging / resuming) |
| `/review/{thread_id}` | POST | Submit human approval or edits during an interrupt |

Streaming uses `graph.astream_events(..., version="v2")`, filtered to `on_chat_model_stream` events, piped into a FastAPI `StreamingResponse` so the frontend can render tokens as they arrive.

---

## 8. Frontend Design

A minimal React + Tailwind single-page chat UI: message list, input box, file upload control, and a small badge per assistant message showing which agent(s) handled it (pulled from `state["route"]`) — useful for demoing the routing logic live. When `review_required` is on, an approval panel renders the draft with "Approve" / "Edit & Resend" actions that call `/review/{thread_id}`.

---

## 9. Testing Strategy

Each node is unit-testable in isolation by constructing a minimal `AgentState` dict and invoking the node function directly, without running the full graph. The router is tested against a fixed set of ~20 labeled example queries to catch classification drift. The full graph is tested end-to-end with `graph.invoke()` against a few canonical scenarios (research question, document Q&A, casual chat, human-review flow) to confirm wiring is correct.

---

## 10. Open Questions / Future Extensions

Parallel multi-agent dispatch for compound queries (noted in §4). Swapping `SqliteSaver` for `PostgresSaver` for true concurrent-write durability.
