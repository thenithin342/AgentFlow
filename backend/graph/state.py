"""
AgentFlow graph state schema.

Reference: DESIGN_DOC.md section 2 "State Schema"

Design note on `review_required`:
  Declared with `NotRequired[bool]` — this is a *type-system* signal that the
  key may be absent from the dict. It is NOT a runtime default: TypedDict
  does not synthesize missing keys at runtime, so a node that reads
  `state["review_required"]` will still raise KeyError if the caller omits it.
  Nodes must read it via `state.get("review_required", False)` to get the
  effective False default; the `NotRequired` annotation just tells the type
  checker (and readers) that omission is permitted.
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between every node in the AgentFlow graph.

    `messages` uses LangGraph's built-in `add_messages` reducer so that each
    node can simply append to the list rather than manually merging
    conversation history.
    """

    messages: Annotated[list, add_messages]
    route: Optional[Literal["research", "analysis", "chat"]]
    agent_output: Optional[str]
    sources: Optional[list[str]]
    documents: Optional[list[str]]
    # NotRequired = "the key may be omitted"; the runtime default of False is
    # provided by `state.get("review_required", False)` at read sites.
    review_required: NotRequired[bool]
    final_response: Optional[str]
