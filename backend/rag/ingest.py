"""
RAG ingestion pipeline — PDF upload to vector store.

Backend selection (Sprint 4):
    - When QDRANT_URL is set → Qdrant collection per thread (horizontally
      scalable, visible to all replicas immediately after upload).
    - When QDRANT_URL is unset → per-thread FAISS index on disk (original
      single-node behaviour, unchanged).

Embeddings: Google Generative AI `models/gemini-embedding-001` (API-based,
3072-dim). No local model is downloaded — embeddings are computed via the
Google API using the GOOGLE_API_KEY env var. This keeps memory well under
Render's 512 MB free-tier limit (the old FastEmbed ONNX model consumed
~120 MB).

Model + dimension come from settings (EMBED_MODEL / EMBED_DIM). Google
retires embedding models on a schedule — `models/text-embedding-004`
started returning `404 NOT_FOUND` for embedContent, which took down PDF
ingestion (/upload → 500) AND Long-Term Memory (silent read/write failure)
at the same time. Config, not code, is the fix for the next retirement.

Reference: DESIGN_DOC.md section 6 "RAG Pipeline", TECH_STACK.md section 4
"Retrieval / RAG".
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("agentflow.rag.ingest")

def _get_index_root() -> Path:
    """Return FAISS index root from env (FAISS_INDEX_DIR) or source-tree fallback."""
    env_val = os.environ.get("FAISS_INDEX_DIR", "").strip()
    if env_val:
        return Path(env_val)
    return Path(__file__).resolve().parent.parent.parent / "faiss_indexes"

INDEX_ROOT: Path = _get_index_root()

def _config_value(attr: str, env_name: str, default: str) -> str:
    """Read an embedding config value from Settings, then env, then default."""
    try:
        from backend.settings import get_settings
        value = getattr(get_settings(), attr, None)
        if value:
            return str(value)
    except Exception:
        logger.debug("[RAG] settings unavailable for %s; using env/default", attr)
    return os.environ.get(env_name, "").strip() or default


# Google Generative AI embeddings — API-based, zero local memory.
# Requires GOOGLE_API_KEY env var (already set in Render for the LLM).
_EMBED_MODEL = _config_value("embed_model", "EMBED_MODEL", "models/gemini-embedding-001")
_EMBED_DIM = int(_config_value("embed_dim", "EMBED_DIM", "3072"))
_EMBEDDINGS = None
_EMBEDDINGS_LOCK = threading.Lock()
# Flipped to False by warm_embeddings() when the provider probe fails, so
# /readyz reports embeddings as unready instead of lying about it.
_EMBEDDINGS_WARM = True


# Substrings that identify an embedding-provider failure inside an
# exception chain. Deliberately broad: langchain wraps google.genai errors
# in `GoogleGenerativeAIError`, so the class name alone is not reliable.
_EMBED_ERROR_MARKERS = (
    "google",
    "genai",
    "embedcontent",
    "embedding",
    "not found for api version",
    "api key not valid",
)


def _exception_chain_text(exc: BaseException) -> str:
    """Flatten an exception + its __cause__/__context__ chain into one string."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts)


def describe_embedding_failure(exc: BaseException) -> str | None:
    """Return an actionable message when *exc* came from the embedding
    provider, or None when it is an unrelated error.

    Used by callers (e.g. /upload) that would otherwise report a bare
    "internal server error" for what is actually a configuration problem.
    """
    text = _exception_chain_text(exc)
    lowered = text.lower()
    if not any(marker in lowered for marker in _EMBED_ERROR_MARKERS):
        return None
    model = _EMBED_MODEL
    if "not found for api version" in lowered or "404" in lowered:
        return (
            f"embedding model '{model}' is not available from Google (404). "
            "Set EMBED_MODEL (and EMBED_DIM) to a supported embedding model."
        )
    if "api key" in lowered or "unauthenticated" in lowered or "permission" in lowered:
        return (
            "embedding provider rejected the credential — check that "
            "GOOGLE_API_KEY is set and valid."
        )
    if "quota" in lowered or "resource_exhausted" in lowered or "429" in lowered:
        return "embedding provider quota exhausted — retry later or raise the quota."
    return f"embedding provider unavailable ({text[:200]})."

# FAISS-only caches — not used in Qdrant path
_RETRIEVERS: OrderedDict[str, Any] = OrderedDict()
_MAX_RETRIEVERS = 1000
_MAX_RETRIEVER_LOCKS = 1000
_RETRIEVER_LOCKS: OrderedDict[str, threading.Lock] = OrderedDict()
_RETRIEVER_LOCKS_GUARD = threading.Lock()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _use_qdrant() -> bool:
    """Return True when Qdrant is configured (QDRANT_URL is set)."""
    try:
        from backend.settings import get_settings
        return get_settings().use_qdrant
    except Exception:
        return False


def _rag_collection_name(thread_id: str) -> str:
    """One Qdrant collection per thread, named by SHA-256 hash."""
    sha = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    return f"rag_{sha}"


# ---------------------------------------------------------------------------
# Shared embedding model (used by both FAISS and Qdrant paths)
# ---------------------------------------------------------------------------


def warm_embeddings() -> None:
    """Verify the embedding provider is reachable and update `_EMBEDDINGS_WARM`.

    Google embeddings are API-based (no local model to load), so the only
    thing worth checking at startup is that the configured model + key
    actually work. A retired model id used to go unnoticed until the first
    upload failed with an opaque 500 — the probe turns that into a startup
    log line and an honest `/readyz`.

    Never raises: a provider outage must not prevent the API from booting.
    """
    global _EMBEDDINGS_WARM
    try:
        _get_embeddings().embed_query("agentflow embedding warm-up probe")
        _EMBEDDINGS_WARM = True
        logger.info("[RAG] embedding provider OK (model=%s, dim=%d)", _EMBED_MODEL, _EMBED_DIM)
    except Exception as exc:
        _EMBEDDINGS_WARM = False
        logger.error(
            "[RAG] embedding provider check FAILED — RAG uploads and long-term "
            "memory will not work: %s",
            describe_embedding_failure(exc) or exc,
            exc_info=True,
        )


def _get_embeddings():
    """Return a Google Generative AI embeddings client (singleton)."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    with _EMBEDDINGS_LOCK:
        if _EMBEDDINGS is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _EMBEDDINGS = GoogleGenerativeAIEmbeddings(
                model=_EMBED_MODEL,
                output_dimensionality=_EMBED_DIM,
            )
        return _EMBEDDINGS


# ---------------------------------------------------------------------------
# FAISS helpers (single-node path — unchanged)
# ---------------------------------------------------------------------------


def _index_dir(thread_id: str) -> Path:
    """Return the FAISS index directory for *thread_id*.

    Uses a SHA-256 hash so path-special characters cannot escape INDEX_ROOT.
    """
    safe = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    idx_dir = INDEX_ROOT / safe
    try:
        resolved = idx_dir.resolve()
        INDEX_ROOT.resolve()
        resolved.relative_to(INDEX_ROOT.resolve())
    except ValueError:
        raise ValueError(f"Index path escapes INDEX_ROOT: {idx_dir}")
    return idx_dir


def _faiss_index_files_valid(index_path: Path) -> bool:
    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"
    try:
        return (
            faiss_file.is_file()
            and pkl_file.is_file()
            and faiss_file.stat().st_size > 0
            and pkl_file.stat().st_size > 0
        )
    except OSError:
        return False


def _write_model_tag(index_path: Path) -> None:
    """Persist the embedding model name alongside the index."""
    (index_path / "embed_model.txt").write_text(_EMBED_MODEL, encoding="utf-8")


def _check_model_tag(index_path: Path) -> bool:
    """Return True if the stored model tag matches the configured model.

    An *untagged* index is treated as incompatible: every index written
    before the tag existed was produced by a different embedding model
    (bge-small-en-v1.5, 384-dim), so loading it and adding new vectors would
    raise a dimension error on every upload.
    """
    tag_file = index_path / "embed_model.txt"
    if not tag_file.exists():
        return False
    return tag_file.read_text(encoding="utf-8").strip() == _EMBED_MODEL


def _load_faiss_index(index_path: Path) -> FAISS:
    if not _faiss_index_files_valid(index_path):
        raise FileNotFoundError(f"incomplete FAISS index at {index_path}")
    from backend.security import sign_file, verify_file
    pkl_path = index_path / "index.pkl"
    hmac_path = index_path / "index.pkl.hmac"
    if not verify_file(pkl_path):
        if not hmac_path.exists():
            logger.info("Legacy FAISS index detected, backfilling HMAC signature")
            sign_file(pkl_path)
        else:
            raise ValueError(f"Integrity check failed for {index_path}/index.pkl")
    if not _check_model_tag(index_path):
        raise ValueError(
            f"Embedding model mismatch for index at {index_path}. "
            "Delete the index directory to rebuild with the current model."
        )
    return FAISS.load_local(
        str(index_path),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _retriever_lock(thread_id: str) -> threading.Lock:
    with _RETRIEVER_LOCKS_GUARD:
        lock = _RETRIEVER_LOCKS.get(thread_id)
        if lock is None:
            while len(_RETRIEVER_LOCKS) >= _MAX_RETRIEVER_LOCKS:
                _RETRIEVER_LOCKS.popitem(last=False)
            lock = threading.Lock()
            _RETRIEVER_LOCKS[thread_id] = lock
        else:
            _RETRIEVER_LOCKS.move_to_end(thread_id)
        return lock


# ---------------------------------------------------------------------------
# Public API — ingest_pdf
# ---------------------------------------------------------------------------


def ingest_pdf(
    file_path: str,
    thread_id: str,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Extract → chunk → embed → save to the configured vector store.

    Returns ingest stats: ``document_id``, ``source``, ``pages``, ``chunks``.

    Automatically routes to Qdrant or FAISS based on QDRANT_URL.
    """
    basename = source_name or os.path.basename(file_path) or "document.pdf"
    docs = PyPDFLoader(file_path).load()
    for doc in docs:
        doc.metadata.setdefault("source", basename)
        doc.metadata["thread_id"] = thread_id

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("PDF contains no extractable text")

    if _use_qdrant():
        try:
            _ingest_qdrant(thread_id, chunks)
        except Exception as exc:
            # An embedding failure will fail the FAISS path too (and the
            # retry doubles the provider calls), so propagate it instead of
            # pretending a fallback was attempted.
            if describe_embedding_failure(exc):
                raise
            logger.warning(
                "[RAG] Qdrant ingest failed for thread %s — falling back to FAISS.",
                thread_id[:16],
                exc_info=True,
            )
            _ingest_faiss(thread_id, chunks)
    else:
        _ingest_faiss(thread_id, chunks)

    document_id = f"{basename}:{len(chunks)}"
    return {
        "document_id": document_id,
        "source": basename,
        "pages": len(docs),
        "chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# Public API — get_retriever
# ---------------------------------------------------------------------------


def get_retriever(thread_id: str):
    """Return a LangChain retriever for the given thread.

    Routing priority:
    1. Qdrant, if QDRANT_URL is configured and Qdrant is reachable.
    2. FAISS (local), as a fallback when Qdrant raises an exception or
       when QDRANT_URL is not set.
    3. None if neither has data for this thread.
    """
    if _use_qdrant():
        try:
            return _get_retriever_qdrant(thread_id)
        except Exception:
            logger.warning(
                "[RAG] Qdrant unavailable for thread %s — falling back to FAISS.",
                thread_id[:16],
                exc_info=True,
            )
            # Fall through to FAISS
    return _get_retriever_faiss(thread_id)


# ---------------------------------------------------------------------------
# Qdrant implementations
# ---------------------------------------------------------------------------


def _ingest_qdrant(thread_id: str, chunks: list) -> None:
    """Upsert chunks into the thread's Qdrant collection."""
    try:
        from backend.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore(_rag_collection_name(thread_id))
        store.add_documents(chunks)
        logger.info(
            "[RAG/Qdrant] ingested %d chunks for thread %s",
            len(chunks),
            thread_id[:16],
        )
    except Exception:
        logger.exception("[RAG/Qdrant] ingest failed for thread %s", thread_id[:16])
        raise


def _get_retriever_qdrant(thread_id: str):
    """Return a Qdrant-backed retriever for *thread_id*."""
    from backend.vectorstore.qdrant_store import QdrantStore
    store = QdrantStore(_rag_collection_name(thread_id))
    return store.as_retriever(k=4)


# ---------------------------------------------------------------------------
# FAISS implementations (original, unchanged)
# ---------------------------------------------------------------------------


def _ingest_faiss(thread_id: str, chunks: list) -> None:
    lock = _retriever_lock(thread_id)
    with lock:
        out = _index_dir(thread_id)
        out.mkdir(parents=True, exist_ok=True)
        embeddings = _get_embeddings()
        index = None
        if _faiss_index_files_valid(out):
            try:
                index = _load_faiss_index(out)
            except (ValueError, FileNotFoundError) as exc:
                # The stored index was built with a retired/other embedding
                # model. It can never accept vectors from the current model,
                # so rebuild from these chunks instead of failing the upload.
                logger.warning(
                    "[RAG] rebuilding incompatible FAISS index for thread %s: %s",
                    thread_id[:16],
                    exc,
                )
                index = None
        if index is None:
            index = FAISS.from_documents(chunks, embeddings)
        else:
            index.add_documents(chunks)
        index.save_local(str(out))
        from backend.security import sign_file
        sign_file(out / "index.pkl")
        _write_model_tag(out)
        _RETRIEVERS.pop(thread_id, None)


def _get_retriever_faiss(thread_id: str):
    cached = _RETRIEVERS.get(thread_id)
    if cached is not None:
        _RETRIEVERS.move_to_end(thread_id)
        return cached
    lock = _retriever_lock(thread_id)
    with lock:
        cached = _RETRIEVERS.get(thread_id)
        if cached is not None:
            _RETRIEVERS.move_to_end(thread_id)
            return cached
        index_path = _index_dir(thread_id)
        index = _load_faiss_index(index_path)
        retriever = index.as_retriever(search_kwargs={"k": 4})
        while len(_RETRIEVERS) >= _MAX_RETRIEVERS:
            _RETRIEVERS.popitem(last=False)
        _RETRIEVERS[thread_id] = retriever
        return retriever
