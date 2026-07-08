"""
Memory graph nodes for AgentFlow Phase 9.

Three nodes:
    memory_reader_node  — fires at START, injects LTM context into state
    memory_writer_node  — fires before END, extracts facts → writes to LTM
    stm_compressor_node — fires conditionally every STM_WINDOW turns,
                          compresses old messages into stm_summary

These nodes are designed to be lightweight and fault-tolerant: any failure
is logged as a warning and the graph continues without memory — degraded
but not broken.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from backend.graph.state import AgentState
from backend.graph.messages import content_to_str, is_human_message
from backend.memory.stm import (
    STM_KEEP_RECENT,
    should_compress,
    compress_messages,
)
from backend.memory.ltm import read_ltm, write_ltm, extract_facts

logger = logging.getLogger("agentflow.memory.nodes")

# Default user_id for single-user local setups.
_DEFAULT_USER_ID = "default"


def _user_id_from_config(config: RunnableConfig | None) -> str:
    """Extract user_id from the LangGraph RunnableConfig.

    Falls back to "default" for single-user local setups so existing
    tests and CLI tools that don't pass user_id continue working.
    """
    if not config:
        return _DEFAULT_USER_ID
    return (config.get("configurable") or {}).get("user_id", _DEFAULT_USER_ID)


def _thread_id_from_config(config: RunnableConfig | None) -> str:
    if not config:
        return ""
    return (config.get("configurable") or {}).get("thread_id", "")


# ---------------------------------------------------------------------------
# memory_reader_node
# ---------------------------------------------------------------------------

def memory_reader_node(state: AgentState, config: RunnableConfig) -> dict:
    """Query LTM and inject relevant context into the state.

    Reads the latest user message as the query, retrieves the top-k facts from
    the user's LTM store, and writes them into `state["ltm_context"]`.

    This runs synchronously (like all other nodes in this project) and is
    fast enough: LTM retrieval is a local FAISS similarity search, not a
    network call.
    """
    user_id = _user_id_from_config(config)

    # Extract the latest user message as the retrieval query.
    messages = state.get("messages") or []
    query = ""
    for m in reversed(messages):
        if is_human_message(m):
            query = content_to_str(m.content if hasattr(m, "content") else m)
            break

    if not query:
        return {"ltm_context": None}

    ltm_context = read_ltm(user_id, query)
    if ltm_context:
        logger.debug("[memory_reader] injected %d chars of LTM for user %s", len(ltm_context), user_id)
    turn_count = state.get("turn_count") or 0
    return {"ltm_context": ltm_context or None, "turn_count": turn_count + 1}


# ---------------------------------------------------------------------------
# memory_writer_node
# ---------------------------------------------------------------------------

def memory_writer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Extract facts from the completed turn and write them to LTM.

    Runs after human_review (or synthesizer for chat turns) so it has access
    to the final_response.  Extracts facts from the (user_query, final_response)
    pair to capture what was accomplished this turn.

    This is a best-effort node — any failure is swallowed so the turn always
    completes even if LTM write fails.
    """
    user_id = _user_id_from_config(config)
    thread_id = _thread_id_from_config(config)

    # Build a compact turn summary for fact extraction.
    messages = state.get("messages") or []
    final = state.get("final_response") or state.get("agent_output") or ""

    user_query = ""
    for m in reversed(messages):
        if is_human_message(m):
            user_query = content_to_str(m.content if hasattr(m, "content") else m)
            break

    if not user_query or not final:
        return {}

    turn_text = f"User: {user_query}\n\nAssistant: {final}"

    try:
        from backend.llm import llm_fast
        facts = extract_facts(turn_text, llm_fast)
        if facts:
            write_ltm(user_id, facts, source_thread_id=thread_id)
    except Exception:
        logger.warning("[memory_writer] failed to write LTM for user %s", user_id, exc_info=True)

    return {}


# ---------------------------------------------------------------------------
# stm_compressor_node
# ---------------------------------------------------------------------------

def stm_compressor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Compress old messages into an STM summary when the turn window is hit.

    Increments `turn_count` on every call. When `should_compress` returns True,
    replaces the oldest messages with a SystemMessage summary and stores the
    summary text in `state["stm_summary"]` for downstream injection.

    Returns:
        dict with updated `turn_count` and optionally `stm_summary` +
        `messages` (the pruned list with the summary SystemMessage prepended).
    """
    messages = list(state.get("messages") or [])
    turn_count = state.get("turn_count") or 0

    if not should_compress(turn_count):
        return {"turn_count": turn_count}

    # Split messages: keep recent verbatim, compress the rest.
    if len(messages) <= STM_KEEP_RECENT:
        return {"turn_count": turn_count}

    older = messages[: len(messages) - STM_KEEP_RECENT]
    recent = messages[len(messages) - STM_KEEP_RECENT :]

    try:
        from backend.llm import llm_fast
        from backend.memory.stm import build_stm_prefix
        summary = compress_messages(older, llm_fast)
        prefix = build_stm_prefix(summary)
        new_messages = ([prefix] if prefix else []) + recent
        logger.info(
            "[STM] compressed %d messages into summary (%d chars); turn=%d",
            len(older),
            len(summary),
            turn_count,
        )
        return {
            "turn_count": turn_count,
            "stm_summary": summary,
            "messages": new_messages,
        }
    except Exception:
        logger.warning("[STM] compression failed at turn %d", turn_count, exc_info=True)
        return {"turn_count": turn_count}


# ---------------------------------------------------------------------------
# Conditional edge helper for STM
# ---------------------------------------------------------------------------

def should_run_stm(state: AgentState) -> str:
    """Conditional edge: run stm_compressor or skip straight to memory_writer."""
    current = state.get("turn_count") or 0
    if should_compress(current) and len(state.get("messages") or []) > STM_KEEP_RECENT:
        return "stm_compressor"
    return "memory_writer"
