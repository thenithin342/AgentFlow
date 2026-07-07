"""
Short-Term Memory (STM) for AgentFlow.

STM compresses older messages into a summary paragraph every STM_WINDOW turns,
keeping the in-context history manageable without losing conversational thread.

Design:
    - Every time the turn counter reaches a multiple of STM_WINDOW the compressor
      fires and replaces the *oldest* (len(messages) - STM_KEEP_RECENT) messages
      with a single SystemMessage summarising them.
    - The summary is stored in `state["stm_summary"]` AND prepended as a
      SystemMessage at the start of the conversation so agents always see it.
    - `STM_KEEP_RECENT` recent messages are always kept verbatim — the LLM needs
      the immediate context uncompressed.

Reference: DESIGN_DOC.md §10 "Open Questions / Future Extensions" — memory
summarisation node.
"""

from __future__ import annotations
import json
import logging
from typing import Sequence, TYPE_CHECKING

from backend.graph.messages import _msg_type, get_msg_content

from langchain_core.messages import SystemMessage

if TYPE_CHECKING:
    pass

logger = logging.getLogger("agentflow.memory.stm")

# Compress once every N *human* turns.
STM_WINDOW = 10
# Always keep this many of the most-recent messages verbatim.
STM_KEEP_RECENT = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_compress(turn_count: int) -> bool:
    """Return True when the turn counter has just crossed a compression window."""
    return turn_count > 0 and turn_count % STM_WINDOW == 0


def compress_messages(messages: list, llm) -> str:
    """Ask the LLM to summarise `messages` into a short paragraph.

    The summary is written in third-person past tense so it reads naturally
    when prepended to the system context.  The LLM is given only the message
    *content* — no tool call metadata or IDs — to keep the prompt tight.

    Returns a plain-text summary string (never empty; falls back to a
    minimal placeholder on any LLM error).
    """
    if not messages:
        return ""

    # Build a compact transcript for the compressor prompt.
    lines: list[str] = []
    for m in messages:
        role = _msg_type(m) or "unknown"
        label = {"human": "User", "ai": "Assistant", "system": "System"}.get(role, role.title())
        c = get_msg_content(m)
        content = c if isinstance(c, str) else str(c)
        if content.strip():
            lines.append(f"{label}: {content.strip()}")

    transcript = "\n".join(lines)
    if not transcript:
        return ""

    prompt = (
        "You are summarising an earlier portion of a conversation for context.\n"
        "Write a concise paragraph (3-6 sentences) in the third person that captures:\n"
        "- what the user was trying to accomplish\n"
        "- the key facts, answers, or results produced\n"
        "- any important decisions or constraints mentioned\n\n"
        "Transcript to summarise:\n"
        "---\n"
        f"{transcript}\n"
        "---\n"
        "Summary:"
    )

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        text = text if isinstance(text, str) else str(text)
        return text.strip()
    except Exception:
        logger.warning("[STM] compression LLM call failed", exc_info=True)
        return f"[Earlier conversation summarised: {len(messages)} messages compressed.]"


def build_stm_prefix(summary: str) -> SystemMessage | None:
    """Wrap the STM summary in a SystemMessage suitable for injection into
    the agent's message list.  Returns None if summary is empty."""
    if not summary or not summary.strip():
        return None
    text = (
        "## Conversation context (earlier messages summarised)\n"
        f"{summary}\n\n"
        "---\n"
        "The above is a summary of earlier turns. The recent messages below are verbatim."
    )
    return SystemMessage(content=text)
