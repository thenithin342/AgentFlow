"""
Long-Term Memory (LTM) for AgentFlow.

Stores user facts, preferences, and important context across threads using a
per-user FAISS index (same embedding model as the RAG pipeline — all-MiniLM-L6-v2).

Design:
    - Facts are extracted by `memory_writer_node` after each turn by asking the
      LLM "what is worth remembering about this exchange?"
    - Each fact is stored as an embedding in a per-user-id FAISS index with
      metadata {fact, source_thread_id, timestamp}.
    - `read_ltm(user_id, query)` retrieves the top-k most relevant memories
      for the current turn's context query.
    - User identity is tracked via `user_id` passed in the API request body
      (defaults to "default" for single-user local setups).

The LTM store lives under `ltm_indexes/{user_id}/` alongside `faiss_indexes/`.

IMPORTANT — privacy note: LTM stores plaintext facts extracted from user
conversations. For a single-user local project this is fine. A multi-user
production deployment would need per-user encryption, consent flows, and a
deletion endpoint. Those are out of scope for this portfolio project.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentflow.memory.ltm")

LTM_ROOT = Path(__file__).resolve().parent.parent.parent / "ltm_indexes"
_LTM_LOCK = threading.Lock()

# Max facts retrieved per query.
LTM_TOP_K = 5
# Max facts stored total per user (older ones evicted if exceeded).
LTM_MAX_FACTS = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embeddings():
    """Reuse the same HuggingFace model as the RAG pipeline."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _mask_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


_LTM_MIGRATE_LOCK = threading.Lock()
_LTM_MIGRATED: set[str] = set()


def _ltm_dir(user_id: str) -> Path:
    """Return the hashed LTM directory for *user_id*.

    If a legacy sanitised directory (colon→underscore) exists and the hashed
    directory does not, migrates the data once (concurrency-safe) before
    returning the hashed path.
    """
    safe = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    new_dir = LTM_ROOT / safe
    if new_dir.exists():
        return new_dir
    # Check for legacy path (used before the hashing migration).
    legacy_safe = user_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    legacy_dir = LTM_ROOT / legacy_safe
    if legacy_dir.exists() and user_id not in _LTM_MIGRATED:
        with _LTM_MIGRATE_LOCK:
            if user_id not in _LTM_MIGRATED and legacy_dir.exists() and not new_dir.exists():
                try:
                    import shutil as _shutil
                    _shutil.copytree(str(legacy_dir), str(new_dir))
                    _shutil.rmtree(str(legacy_dir), ignore_errors=True)
                    logger.info("[LTM] Migrated legacy index for user %s", _mask_id(user_id))
                except Exception:
                    logger.warning("[LTM] Failed to migrate legacy index for user %s", _mask_id(user_id), exc_info=True)
                _LTM_MIGRATED.add(user_id)
    return new_dir


def _load_index(user_id: str):
    """Load the user's FAISS index, or return None if it doesn't exist yet."""
    try:
        from langchain_community.vectorstores import FAISS
        idx_dir = _ltm_dir(user_id)
        faiss_file = idx_dir / "index.faiss"
        pkl_file = idx_dir / "index.pkl"
        if faiss_file.is_file() and pkl_file.is_file():
            from backend.security import verify_file
            if not verify_file(pkl_file):
                logger.error("[LTM] Integrity check failed for user %s index.pkl", _mask_id(user_id))
                return None
            return FAISS.load_local(
                str(idx_dir),
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
    except Exception:
        logger.warning("[LTM] failed to load index for user %s", _mask_id(user_id), exc_info=True)
    return None


def _save_index(user_id: str, index) -> None:
    """Persist the FAISS index to disk."""
    idx_dir = _ltm_dir(user_id)
    idx_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(idx_dir))
    from backend.security import sign_file
    sign_file(idx_dir / "index.pkl")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_ltm(user_id: str, query: str) -> str:
    """Retrieve the top-k facts most relevant to `query` from the LTM store.

    Returns a formatted string ready to inject into an agent system prompt.
    Returns an empty string if no memories exist or retrieval fails.
    """
    try:
        with _LTM_LOCK:
            index = _load_index(user_id)
        if index is None:
            return ""
        retriever = index.as_retriever(search_kwargs={"k": LTM_TOP_K})
        docs = retriever.invoke(query)
        if not docs:
            return ""
        lines = ["## Long-term memory (facts about this user / conversation history)"]
        for doc in docs:
            fact = doc.page_content.strip()
            if fact:
                lines.append(f"- {fact}")
        return "\n".join(lines)
    except Exception:
        logger.warning("[LTM] read_ltm failed for user %s", _mask_id(user_id), exc_info=True)
        return ""


def write_ltm(user_id: str, facts: list[str], source_thread_id: str = "") -> None:
    """Store extracted facts into the user's LTM FAISS index.

    Each fact is stored as a Document with metadata including the source thread
    and timestamp, so memories can be attributed and potentially deleted later.
    """
    if not facts:
        return
        
    from backend.settings import get_settings
    settings = get_settings()
    if not getattr(settings, "ltm_enabled", True): # Optional config gate
        return
        
    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS

        ts = datetime.now(timezone.utc).isoformat()
        docs = [
            Document(
                page_content=fact.strip(),
                metadata={"source_thread_id": source_thread_id, "timestamp": ts},
            )
            for fact in facts
            if fact.strip()
        ]
        if not docs:
            return

        with _LTM_LOCK:
            index = _load_index(user_id)
            if index is None:
                index = FAISS.from_documents(docs, _get_embeddings())
            else:
                index.add_documents(docs)
                
            if index.index.ntotal > LTM_MAX_FACTS:
                all_docs = list(index.docstore._dict.values())
                all_docs.sort(key=lambda d: d.metadata.get("timestamp", ""))
                keep_docs = all_docs[-LTM_MAX_FACTS:]
                index = FAISS.from_documents(keep_docs, _get_embeddings())
                
            _save_index(user_id, index)
        logger.info("[LTM] wrote %d facts for user %s", len(docs), _mask_id(user_id))
    except Exception:
        logger.warning("[LTM] write_ltm failed for user %s", _mask_id(user_id), exc_info=True)


def extract_facts(turn_text: str, llm) -> list[str]:
    """Ask the LLM to extract memorable facts from a conversation turn.

    Returns a list of short fact strings (empty list on failure or if nothing
    memorable).
    """
    if not turn_text or not turn_text.strip():
        return []

    prompt = (
        "Extract memorable facts from this conversation turn that would be useful "
        "in future sessions with this user.\n\n"
        "Focus on:\n"
        "- User's name, occupation, or domain expertise\n"
        "- Specific preferences, goals, or constraints stated\n"
        "- Important topics the user cares about\n"
        "- Decisions or conclusions reached\n\n"
        "Format: Return ONLY a JSON array of short fact strings, one per item.\n"
        "Return [] if nothing notable is worth remembering.\n\n"
        f"Conversation turn:\n{turn_text[:2000]}\n\n"
        "Facts (JSON array):"
    )

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        text = text if isinstance(text, str) else str(text)
        # Extract JSON array from the response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            arr = json.loads(text[start : end + 1])
            return [str(f) for f in arr if isinstance(f, str) and f.strip()]
    except Exception:
        logger.debug("[LTM] fact extraction failed", exc_info=True)
    return []
