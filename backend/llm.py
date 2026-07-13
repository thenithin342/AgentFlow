"""
LLM client setup for AgentFlow.

Reference: TECH_STACK.md section 2 "LLM Providers"

Fallback strategy
-----------------
Three Groq API keys are supported: GROQ_API_KEY (primary), GROQ_API_KEY_2
(secondary), GROQ_API_KEY_3 (tertiary). Every call starts with the primary key;
the secondary and tertiary keys are tried **only when the primary raises an
exception** (e.g. 429 rate-limit, 503 service error, network timeout). This is
standard `RunnableWithFallbacks` semantics — it does NOT round-robin or share
load across keys. A Gemini fallback is appended last if GOOGLE_API_KEY is set.

Lazy initialisation strategy
-----------------------------
We do NOT build ChatGroq objects at import time when the key is absent —
instantiating with ``api_key=None`` triggers network probing in some
LangChain versions and surfaces a confusing error in every test that does
not touch the LLM.

``llm_smart`` and ``llm_fast`` are resolved via module-level ``__getattr__``
(PEP 562, Python 3.7+).  The first attribute access on either name calls
``get_llm_smart()`` / ``get_llm_fast()`` and caches the real runnable back
onto the module, so every subsequent access is a plain dict lookup.
"""

import logging as _logging
import os
import sys
import threading
from typing import Any

import groq
from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import SecretStr

load_dotenv()

# We use tiktoken for a fast, reliable token count estimate.
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

class TokenBudgetWrapper(RunnableBinding):
    """Prevents silent context window overflow by bounding input tokens."""
    budget: int = 100_000

    def _check(self, input: Any) -> None:
        messages = input.get("messages", []) if isinstance(input, dict) else input
        total_text = " ".join(
            m.content if hasattr(m, "content") else str(m.get("content", ""))
            for m in (messages if isinstance(messages, list) else [messages])
        )
        tokens = len(_enc.encode(total_text, disallowed_special=()))
        if tokens > self.budget:
            _log = _logging.getLogger("agentflow.llm")
            _log.error(
                "token_budget_exceeded",
                extra={"tokens": tokens, "budget": self.budget},
            )
            raise ValueError(
                f"Context window budget exceeded (estimated {tokens:,} tokens > "
                f"{self.budget:,} limit). Please start a new conversation or shorten "
                f"your message."
            )

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self._check(input)
        return super().invoke(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self._check(input)
        return await super().ainvoke(input, config, **kwargs)

    def stream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self._check(input)
        yield from super().stream(input, config, **kwargs)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self._check(input)
        async for chunk in super().astream(input, config, **kwargs):
            yield chunk

    def bind_tools(self, *args: Any, **kwargs: Any) -> Runnable:
        """Delegate bind_tools to the underlying runnable so agents can use tools."""
        bound_with_tools = self.bound.bind_tools(*args, **kwargs)
        return self.__class__(
            bound=bound_with_tools,
            budget=self.budget,
            config=self.config,
            kwargs=self.kwargs,
        )


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _collect_groq_keys() -> list[str]:
    """Return all non-empty Groq API keys defined in the environment.

    Primary key: GROQ_API_KEY
    Secondary:   dummy1  (alias GROQ_API_KEY_2)
    Tertiary:    dummy2  (alias GROQ_API_KEY_3)

    Each is stripped of whitespace and surrounding quotes before use.
    Keys that are missing or blank are silently skipped.
    """
    candidates = [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("dummy1", ""),
        os.environ.get("dummy2", ""),
    ]
    seen: set[str] = set()
    keys: list[str] = []
    for k in candidates:
        k = k.strip().strip('"').strip("'")
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _build_groq_client(model: str, api_key: str, max_retries: int = 2) -> ChatGroq:
    """Build a single ChatGroq client for one API key."""
    return ChatGroq(
        model=model,
        temperature=0,
        api_key=SecretStr(api_key),
        max_retries=max_retries,
        timeout=60,
    )


def _build_groq_pool(model: str) -> Runnable:
    """Build a RunnableWithFallbacks that falls back across all Groq keys on exception.

    Each key gets its own ChatGroq instance. The primary key is the first
    runnable; the rest are chained as fallbacks with exceptions_to_handle=
    (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError)
    so 429 / 503 / network errors trigger the next key. Catching bare
    Exception would also mask programming bugs (TypeError, KeyError, ...)
    and silently route them to a different key ? bad for debugging.
    Falls back to Gemini at the end of the chain if GOOGLE_API_KEY is set.
    """
    keys = _collect_groq_keys()
    if not keys:
        raise RuntimeError(
            "No Groq API key found. Set GROQ_API_KEY in your .env file. "
            "Get a free key at https://console.groq.com."
        )
    clients = [_build_groq_client(model, k) for k in keys]
    fallbacks: list[Any] = list(clients[1:])
    gemini = _build_fallback()
    if gemini:
        fallbacks.append(gemini)
    if not fallbacks:
        return clients[0]
    return clients[0].with_fallbacks(
        fallbacks,
        exceptions_to_handle=(
            groq.APIConnectionError,
            groq.APITimeoutError,
            groq.RateLimitError,
            groq.APIStatusError,
        ),
    )


def _build_fallback() -> ChatGoogleGenerativeAI | None:
    google_key = os.environ.get("GOOGLE_API_KEY", "").strip().strip('"').strip("'")
    if not google_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        api_key=google_key,
        max_retries=3,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_llm_smart = None
_llm_fast = None
_llm_lock = threading.Lock()


def get_llm_smart():
    """Smart-tier pool: llama-3.3-70b-versatile across all Groq keys + Gemini fallback."""
    global _llm_smart
    if _llm_smart is None:
        with _llm_lock:
            if _llm_smart is None:
                _llm_smart = TokenBudgetWrapper(
                    bound=_build_groq_pool("llama-3.3-70b-versatile"), 
                    budget=100_000
                )
    return _llm_smart


def get_llm_fast():
    """Fast-tier pool: llama-3.1-8b-instant across all Groq keys + Gemini fallback.

    Using the supported llama-3.1-8b-instant model. Rotating across 3 keys 
    multiplies the 6K TPM limit to ~18K TPM effective.
    """
    global _llm_fast
    if _llm_fast is None:
        with _llm_lock:
            if _llm_fast is None:
                _llm_fast = TokenBudgetWrapper(
                    bound=_build_groq_pool("llama-3.1-8b-instant"), 
                    budget=100_000
                )
    return _llm_fast


# ---------------------------------------------------------------------------
# Optional fail-fast guard for scripts/CLI
# ---------------------------------------------------------------------------

_require_groq = os.environ.get("AGENTFLOW_REQUIRE_GROQ", "").strip().lower() in {
    "1", "true", "yes", "on",
}
if _require_groq and not _collect_groq_keys():
    raise RuntimeError(
        "GROQ_API_KEY is required — copy .env.example to .env and fill in your key."
    )


# ---------------------------------------------------------------------------
# Module-level lazy attributes (PEP 562)
# ---------------------------------------------------------------------------

_LAZY = {
    "llm_smart": get_llm_smart,
    "llm_fast": get_llm_fast,
}


def __getattr__(name: str):
    if name in _LAZY:
        value = _LAZY[name]()
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
