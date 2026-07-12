"""
Research, Analysis, and Chat agent nodes.

Each agent runs through `langgraph.prebuilt.create_react_agent`, which wraps an
LLM in a ReAct loop bound to a fixed tool list. We extract:
- the final AIMessage content (`state["agent_output"]`)
- URLs harvested from any ToolMessage the loop produced (`state["sources"]`)

The chat agent has no tools — it is a direct `llm_fast` call over the user's
message. It is the fast path for casual turns and follow-ups.

PHASE 7: The Research and Analysis agents now accept a `RunnableConfig` so
they can read the active `thread_id` from `config["configurable"]` and bind a
per-thread `retrieve_documents` tool to a freshly-built ReAct agent. The
factory is invoked on every node call so the tool's `thread_id` closure
matches whatever conversation is currently invoking the graph.

Reference: DESIGN_DOC.md section 3 "Node Specifications", section 6 "RAG
Pipeline".
"""

import hashlib
from collections import OrderedDict
from typing import Any

from langchain_core.runnables import RunnableConfig

# LangGraph 1.x deprecation note: the warning says to import from langchain.agents,
# but that symbol does not exist in langchain 1.3.x yet. The prebuilt import still
# works in LangGraph 1.x and will be updated when langchain.agents exports it.
# Track: https://github.com/langchain-ai/langgraph/issues — revisit at Phase 8 / upgrade.
from langgraph.prebuilt import create_react_agent

from backend.graph.messages import (
    content_to_str,
    extract_sources,
    get_msg_content,
    is_ai_message,
)
from backend.graph.state import AgentState
from backend.graph.tools import (
    calculator,
    code_interpreter,
    datetime_tool,
    make_retrieve_documents_tool,
    tavily_search,
    url_reader,
    wikipedia_search,
)

# --- Thread ID helper ------------------------------------------------------


def _thread_id_from_config(config: RunnableConfig) -> str:
    """Extract thread_id from a LangGraph RunnableConfig.

    LangGraph 1.x injects the per-invoke config (containing
    `configurable.thread_id`) into node functions that declare a second
    `config: RunnableConfig` parameter. We do NOT fall back to a literal
    "default" — that would silently merge two unrelated users' state into
    one shared bucket. If the caller (graph driver, API layer, test) does
    not provide a thread_id, fail closed with a clear error.

    For direct unit-test invocation of a node function, pass the config
    explicitly: `research_agent_node(state, config={"configurable": {"thread_id": "t"}})`.
    """
    if not config:
        raise ValueError(
            "thread_id is required: pass config={'configurable': {'thread_id': '...'}} "
            "to the node function. LangGraph always injects this in production; "
            "direct test calls must do the same."
        )
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if not thread_id:
        raise ValueError(
            "thread_id is required inside config['configurable']; got None or empty."
        )
    return thread_id


# --- Cached ReAct agent factory -------------------------------------------

# `create_react_agent` is not free — it builds a CompiledStateGraph,
# wires tool schemas into the LLM binding, etc. The agent's
# graph topology is a function of (tools, llm, thread_id). The
# `thread_id` is included in the cache key because each call builds a
# fresh `make_retrieve_documents_tool(thread_id)` closure whose name is
# identical across threads — without the thread scope a graph compiled
# for thread A would be reused for thread B, silently binding the wrong
# retriever. Caching per (tool names, llm, prompt, thread_id) still
# avoids redundant `create_react_agent` calls within the same thread.
import threading

_AGENT_CACHE: OrderedDict = OrderedDict()
_MAX_AGENT_CACHE = 128
_AGENT_CACHE_LOCK = threading.Lock()
_AGENT_CACHE_EVENTS: dict = {}


def _get_cached_agent(
    tools: list,
    llm,
    prompt: str | None = None,
    thread_id: str | None = None,
) -> Any:
    """Return a memoized `create_react_agent(llm, tools, prompt)` result.

    Key is ``(tool names sorted, llm model name, prompt hash, thread_id)``.
    Including ``thread_id`` prevents a graph compiled with one thread's
    ``retrieve_documents`` closure from being reused for a different thread
    and leaking documents across conversations.  The prompt hash ensures
    different agents (research vs analysis vs chat) each get their own
    compiled graph with the correct system message injected.

    `state_modifier` is the LangGraph API for prepending a system message
    to the agent's message list on every invocation — equivalent to setting
    a system prompt without rebuilding the graph per call.
    """
    prompt_key = hashlib.sha256(prompt.encode()).hexdigest() if prompt else None
    # Resolve a stable model identifier for the cache key. `RunnableWithFallbacks`
    # does forward `model_name` (verified on langchain-core 1.x), but its identity
    # includes the fallback chain -- different sets of keys collapse to the same
    # string. Use the FIRST runnable's model name (the primary key) as the
    # distinguishing identifier. If the key cannot be resolved, fall back to id()
    # so we never silently merge two distinct runnables into one cache entry.
    llm_model_key = getattr(llm, "model_name", None)
    if not llm_model_key:
        try:
            llm_model_key = type(llm.first).__name__
        except AttributeError:
            llm_model_key = f"id:{id(llm)}"
    key = (
        tuple(sorted(t.name for t in tools)),
        llm_model_key,
        prompt_key,
        thread_id,
    )
    with _AGENT_CACHE_LOCK:
        cached = _AGENT_CACHE.get(key)
        if cached is not None:
            _AGENT_CACHE.move_to_end(key)
            return cached
            
        event = _AGENT_CACHE_EVENTS.get(key)
        if event is not None:
            wait = True
        else:
            event = threading.Event()
            _AGENT_CACHE_EVENTS[key] = event
            wait = False

    if wait:
        event.wait()
        with _AGENT_CACHE_LOCK:
            cached = _AGENT_CACHE.get(key)
            if cached is not None:
                _AGENT_CACHE.move_to_end(key)
                return cached
            
    try:
        if prompt:
            agent = create_react_agent(llm, tools, prompt=prompt)
        else:
            agent = create_react_agent(llm, tools)
            
        with _AGENT_CACHE_LOCK:
            while len(_AGENT_CACHE) >= _MAX_AGENT_CACHE:
                _AGENT_CACHE.popitem(last=False)
            _AGENT_CACHE[key] = agent
            
        return agent
    finally:
        if not wait:
            event.set()
            with _AGENT_CACHE_LOCK:
                _AGENT_CACHE_EVENTS.pop(key, None)


# --- Agent system prompts ---------------------------------------------------

RESEARCH_AGENT_PROMPT = """\
You are AgentFlow's Research Agent — a sharp, thorough investigator with access \
to live web search, Wikipedia, URL reading, and the user's uploaded documents.

## Strategy
1. First check `retrieve_documents` to see if the user's uploaded files contain \
   relevant information. If they do, combine it with web results.
2. Use `tavily_search` for current events, real-world facts, recent papers, or \
   anything that needs up-to-date data.
3. Use `wikipedia_search` for encyclopaedic background, definitions, or historical \
   context — it's fast and reliable for well-known topics.
4. Use `url_reader` when you have a specific URL you want to read in full.
5. Use `datetime_tool` when the question involves the current date or time.
6. Run 1–3 targeted searches. Each query should be specific — avoid vague queries \
   like "tell me about X". Prefer "X key statistics 2024" or "X vs Y comparison".
7. Synthesize across sources. Do not dump raw search snippets — draw conclusions.

## Long-term memory
If a `## Long-term memory` section appears in the conversation context, use those \
facts to personalise your response (e.g. if the user's domain is mentioned, tailor \
examples accordingly).

## Output format
- Lead with the direct answer to the user's question.
- Follow with supporting evidence, key facts, or notable nuances.
- If information conflicts across sources, call that out explicitly.
- Cite sources inline using the URLs returned by Tavily.
- Keep the response focused — quality over quantity.

## Constraints
- Never fabricate statistics or quote sources you have not seen in tool results.
- If search returns nothing useful, say so and explain what you could/couldn't find.
"""

ANALYSIS_AGENT_PROMPT = """\
You are AgentFlow's Analysis Agent — a meticulous analyst with access to the \
user's uploaded documents, a safe calculator, and a Python code interpreter.

## Strategy
1. ALWAYS call `retrieve_documents` first. The user almost certainly uploaded a \
   document for you to analyse. Read the retrieved excerpts carefully.
2. Use `calculator` for quick arithmetic — single expressions.
3. Use `code_interpreter` for complex numeric computation, statistics, data \
   transformations, or multi-step calculations. Write clean Python snippets.
4. If the retrieved excerpts are insufficient, explicitly note what's missing \
   and answer based on what you do have.

## Output format
- Lead with the key finding or answer.
- Show your reasoning step-by-step for calculations or multi-part analysis.
- Use bullet points or a short table for comparisons.
- For document summaries: cover main topic, key arguments/findings, and any \
  notable conclusions or caveats from the document.
- Be specific — reference actual content from the documents, not generic statements.

## Constraints
- Ground every claim in retrieved document content or calculator/code output.
- Never say "I cannot access files" — you have the `retrieve_documents` tool.
- Do not pad. If the answer is short, be short.
"""

CHAT_AGENT_PROMPT = """\
You are AgentFlow — a helpful, direct AI assistant with access to the user's \
uploaded documents.

## Tool Usage
- You have exactly ONE tool: `retrieve_documents`. Use it ONLY when the user's \
  message clearly refers to something they've uploaded (e.g. "what does the file say", \
  "summarize the PDF", "in the document", etc.).
- For ALL other tasks — coding, math, writing, general knowledge, casual chat — \
  respond directly with text. Do NOT attempt to call any tool.
- NEVER invent or call tools that are not listed above (e.g. `import_requests`, \
  `run_code`, `execute`, `search`, etc.). If you try to call an unlisted tool, \
  the request will fail.

## How to respond
- Match the user's register: casual for casual, technical for technical.
- Be concise. One clear paragraph is better than three vague ones.
- No filler phrases: never start with "Certainly!", "Great question!", "Of course!"
- If you genuinely don't know something, say so briefly and suggest next steps.

## Tone
Friendly peer, not a corporate chatbot. Direct, helpful, human.
"""



def _output_from_messages(messages) -> str:
    last = messages[-1]
    if is_ai_message(last):
        return content_to_str(get_msg_content(last))
    return content_to_str(last)


def research_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Sync ReAct agent bound to tavily_search + retrieve_documents.

    SYNC on purpose: the pytest suite drives the graph via
    `graph.invoke(...)`. LangGraph 1.x refuses to dispatch into an async
    node from a sync driver (raises `TypeError: No synchronous function
    provided`). The FastAPI server uses `astream_events` (async), which
    runs sync nodes in a worker thread — the 5-30s ReAct loop never
    blocks the event loop. This matches the original Phase 7 design.

    The `retrieve_documents` tool closure is rebuilt per call so it
    carries the active thread's `thread_id`; the ReAct agent itself is
    memoized by (tool names, llm, thread_id) so the topology is built
    once per thread (`_get_cached_agent`).
    """
    thread_id = _thread_id_from_config(config)
    tool = make_retrieve_documents_tool(thread_id)
    from backend.llm import llm_fast
    agent = _get_cached_agent(
        [tavily_search, wikipedia_search, url_reader, datetime_tool, tool],
        llm_fast,
        prompt=RESEARCH_AGENT_PROMPT,
        thread_id=thread_id,
    )
    result = agent.invoke({"messages": state["messages"]}, config=config)
    messages = result["messages"]
    return {
        "agent_output": _output_from_messages(messages),
        "sources": extract_sources(messages),
    }


# --- Analysis agent (calculator + RAG) -------------------------------------


def analysis_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Sync ReAct agent bound to calculator + retrieve_documents.

    Same per-invocation tool factory + cached-agent pattern as
    `research_agent_node`. Returns `agent_output` only (no external
    sources expected from calculator/RAG). SYNC: see research_agent_node.
    """
    thread_id = _thread_id_from_config(config)
    tool = make_retrieve_documents_tool(thread_id)
    from backend.llm import llm_fast
    agent = _get_cached_agent(
        [calculator, code_interpreter, tool],
        llm_fast,
        prompt=ANALYSIS_AGENT_PROMPT,
        thread_id=thread_id,
    )
    result = agent.invoke({"messages": state["messages"]}, config=config)
    messages = result["messages"]
    return {
        "agent_output": _output_from_messages(messages),
        "sources": [],
    }


# --- Chat agent (RAG-aware direct LLM) ------------------------------------


def chat_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Sync ReAct agent with retrieve_documents tool — fast path for casual turns.

    Phase 7 fix: chat_agent now has access to the per-thread FAISS retriever
    so that questions about uploaded PDFs are answered from the document even
    when the router classifies the turn as 'chat'. If no index exists for this
    thread, the tool returns "No documents uploaded" and the agent falls back
    to its own knowledge — identical behaviour to the old plain-LLM path.

    Phase 2: also writes `final_response` so the conditional edge in
    `build_graph.py` can route chat turns straight to `human_review`
    without re-entering the synthesizer.
    """
    thread_id = _thread_id_from_config(config)
    rag_tool = make_retrieve_documents_tool(thread_id)
    from backend.llm import llm_fast
    agent = _get_cached_agent([rag_tool], llm_fast, prompt=CHAT_AGENT_PROMPT, thread_id=thread_id)
    result = agent.invoke({"messages": state["messages"]}, config=config)
    text = _output_from_messages(result["messages"])

    return {"agent_output": text, "final_response": text}
