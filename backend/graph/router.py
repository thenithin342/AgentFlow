"""
Router node and conditional edge function.

Classifies the latest user message into one of:
    "research" | "analysis" | "chat"

Architecture:
    router_node writes `state["route"]` by calling llm_smart with a few-shot
    system prompt. route_query is a conditional edge function — it reads
    `state["route"]` and returns the next node name for LangGraph's path-map
    lookup (see DESIGN_DOC.md §4).

Reference: DESIGN_DOC.md section 3 "Router" and section 4 "Routing Logic"
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.messages import content_to_str
from backend.llm import llm_fast
from backend.graph.state import AgentState


logger = logging.getLogger("agentflow.router")


# --- System prompt -----------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are a routing classifier for a multi-agent assistant. Your only job is to
classify the user's latest message into exactly one of four categories:

- research: the user wants fresh, external information retrieved from the web
  (news, latest papers, current events, lookups about real-world entities,
  "what is X", "find me", "latest", "search for").
- analysis: the user wants reasoning or computation over data they provided or
  over content already in the conversation (summarize, compare, calculate,
  contrast chapters, compute totals, analyze). ALSO use analysis when the user
  asks about an uploaded document or PDF — e.g. "what is this about?",
  "summarize the document", "what does the file say about X?".
- blog: the user wants a blog post, article, or long-form written content created
  on a topic. Triggers include: "write a blog post", "draft an article",
  "create a blog", "write about X for my blog", "article about", "write up on".
- chat: the user is making small talk, asking a follow-up clarification,
  rephrasing, thanking, asking to shorten or rewrite, or holding a normal
  conversation turn that needs no tool.

Examples:

User: "What are the latest AI research papers?"
Answer: research

User: "Find me recent news about quantum computing breakthroughs."
Answer: research

User: "Search for the current population of Tokyo."
Answer: research

User: "Look up the latest iPhone release date and specs."
Answer: research

User: "Summarize and compare chapter 2 vs chapter 4."
Answer: analysis

User: "Calculate the compound interest on $5000 at 4% for 10 years."
Answer: analysis

User: "Compare the revenue growth of Apple vs Microsoft over the last 5 years."
Answer: analysis

User: "Analyze the sentiment of this customer review."
Answer: analysis

User: "What is this document about?"
Answer: analysis

User: "I have uploaded the PDF, what is that about?"
Answer: analysis

User: "Summarize the file I uploaded."
Answer: analysis

User: "What does the document say about X?"
Answer: analysis

User: "Give me key points from the PDF."
Answer: analysis

User: "What is in the uploaded file?"
Answer: analysis

User: "Write a blog post about the future of AI."
Answer: blog

User: "Draft an article on climate change solutions."
Answer: blog

User: "Create a blog post explaining how LangGraph works."
Answer: blog

User: "Write up an article about Python for beginners."
Answer: blog

User: "I need a blog post about healthy eating habits."
Answer: blog

User: "Thanks, can you shorten that?"
Answer: chat

User: "Hi, how are you?"
Answer: chat

User: "Can you rephrase that in a more formal tone?"
Answer: chat

User: "Got it, please continue."
Answer: chat

Respond with ONLY the word: research, analysis, blog, or chat.
No explanation, no punctuation, no extra text.
"""


# --- Helpers ----------------------------------------------------------------

_VALID_LABELS = {"research", "analysis", "chat", "blog"}
_PUNCT = ",.!?:;\"'`"


def _parse_label(text: str) -> str:
    """Extract a valid route label from the LLM's raw response."""
    if not text:
        return "chat"
    lowered = text.strip().lower()
    for label in _VALID_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", lowered):
            return label
    token = lowered.split()[0] if lowered else ""
    token = token.strip(_PUNCT)
    if token in _VALID_LABELS:
        return token
    return "chat"


# --- Node + edge ------------------------------------------------------------

def _route_for_message(user_text: str) -> str:
    """Classify a single user message via the router LLM (no cross-request cache)."""
    response = llm_fast.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ],
        timeout=5.0,
        config={"tags": ["router"]},
    )
    return _parse_label(content_to_str(response.content))


def router_node(state: AgentState) -> dict:
    """Classify the latest user message and write `route` into state.

    Reads `state["messages"][-1]`, calls llm_fast with a few-shot system
    prompt, parses the single-word response, and returns
    `{"route": <label>}`.
    """
    last_message = state["messages"][-1]
    user_text = content_to_str(
        last_message.content if hasattr(last_message, "content") else last_message
    )

    try:
        label = _route_for_message(user_text)
    except Exception:
        # Groq 5xx, network blip, malformed response — any of these would
        # otherwise propagate and 500 the whole graph run. "chat" is the
        # safest default: the chat agent has no tools and won't fabricate
        # external facts. Logging at warning keeps the trail for ops to
        # see the real exception in the server log.
        logger.warning(
            "[AgentFlow] router LLM failed; falling back to 'chat'",
            exc_info=True,
        )
        label = "chat"

    return {
        "route": label,
        # Reset per-turn scratch fields so stale agent_output / sources /
        # final_response / blog_output from a prior turn cannot leak into the
        # new route.
        "agent_output": None,
        "sources": [],
        "final_response": None,
        "blog_output": None,
    }


def route_query(state: AgentState) -> str:
    """Conditional edge function — returns the name of the next node.

    Reads `state["route"]` (written by router_node) and returns it.
    LangGraph uses this string as a key into the path-map passed to
    `add_conditional_edges`.

    Falls back to "chat" if route is None (e.g. state was built without
    going through router_node) so the graph never dead-ends.
    """
    return state.get("route") or "chat"
