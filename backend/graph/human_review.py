"""
Human-in-the-loop review node.

Phase 6 implementation. Pauses the graph between the Synthesizer and END so the
caller can inspect and edit `final_response` before the user sees it.

Behavior:

- If `state["review_required"]` is False → pass-through, return `{}` (no state
  change, no pause).
- If True → call `interrupt({"draft": state["final_response"]})`. LangGraph
  raises a `GraphInterrupt`, snapshots the state under the active `thread_id`,
  and returns control to the caller with the snapshot exposed via
  `__interrupt__` on the invoke result and via `graph.get_state(config)`.
- On resume, the caller invokes `graph.invoke(Command(resume=<value>), config)`,
  which feeds the value into the pending `interrupt()` call. We treat the
  APPROVE_SENTINEL token and `None` as approval (keep current draft), and any
  other non-empty string as an edit (return it as the new `final_response`).
  `Command(resume=...)` is the LangGraph 1.x native resume primitive —
  `update_state` is the lower-level state-mutation API and is NOT what we use
  here.

Resume contract:

- `Command(resume=APPROVE_SENTINEL)` → keep draft.
- `Command(resume=<edit string>)` → replace draft with the edit.
- The approval sentinel is a fixed UUID that the API layer sends on the
  "approve" action. It is intentionally not human-typeable so a user who
  literally writes "approve" as their edit does NOT accidentally trigger the
  approval path. `None` is also accepted as "no input".

Reference: DESIGN_DOC.md section 3 "Human Review", section 5 "Persistence Design".
"""

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from backend.graph.messages import content_to_str
from backend.graph.state import AgentState


# Non-guessable sentinel — a user who literally types "approve" as an edit
# must NOT hit this branch. The API layer (main.py) sends this exact token;
# any other string (including the word "approve") is treated as an edit.
APPROVE_SENTINEL = "__AGENTFLOW_APPROVE_7f3a9c2e-1b4d-4e8f-9d0a-6c5b7e8f1a2b__"


def _agent_name_for_route(state: AgentState) -> str:
    route = state.get("route")
    if route == "chat":
        return "chat_agent"
    if route == "research":
        return "research_agent"
    if route == "analysis":
        return "analysis_agent"
    if route == "blog":
        return "blog_writer"
    return "synthesizer"


def _normalize_resume_value(value) -> str | None:
    """Coerce LangGraph interrupt resume payloads to a plain string.

    LangGraph versions may return the resume value directly, wrapped in a
    dict, or as a one-element list/tuple of Interrupt objects.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
        if hasattr(value, "value"):
            value = value.value
    if isinstance(value, dict):
        if "resume" in value:
            value = value["resume"]
        elif "draft" in value:
            value = value["draft"]
        else:
            return None
    if not isinstance(value, str):
        value = str(value)
    return value


def _last_ai_content(state: AgentState) -> str | None:
    messages = state.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    if getattr(last, "type", None) != "ai":
        return None
    content = getattr(last, "content", None)
    return content_to_str(content) if content is not None else None


def _should_append_assistant_message(state: AgentState, content: str) -> bool:
    """Skip duplicate AIMessage when an agent node already appended the reply."""
    last_ai = _last_ai_content(state)
    return last_ai != content


def human_review_node(state: AgentState) -> dict:
    """Gate `final_response` behind a human checkpoint.

    Returns:
        - `{"review_required": False}` if review not required (pass-through).
        - `{"review_required": False}` if human approved (keep draft).
        - `{"final_response": <edit>, "review_required": False}` on edit.

    Every return path clears `review_required` so the next turn in the same
    thread does NOT re-trigger the interrupt when the user did not request
    a review.
    """
    name = _agent_name_for_route(state)

    if not state.get("review_required", False):
        final = state.get("final_response") or ""
        out: dict = {"review_required": False}
        if final and _should_append_assistant_message(state, final):
            out["messages"] = [AIMessage(content=final, name=name)]
        return out

    human_input = _normalize_resume_value(
        interrupt({"draft": state.get("final_response")})
    )

    if human_input and human_input != APPROVE_SENTINEL:
        out = {"final_response": human_input, "review_required": False}
        if _should_append_assistant_message(state, human_input):
            out["messages"] = [AIMessage(content=human_input, name=name)]
        return out

    final = state.get("final_response") or ""
    out = {"review_required": False}
    if final and _should_append_assistant_message(state, final):
        out["messages"] = [AIMessage(content=final, name=name)]
    return out
