"""
RAG ingestion pipeline — PDF upload to vector store.

Backend selection (Sprint 4):
    - When QDRANT_URL is set → Qdrant collection per thread (horizontally
      scalable, visible to all replicas immediately after upload).
    - When QDRANT_URL is unset → per-thread FAISS index on disk (original
      single-node behaviour, unchanged).

Both backends use BAAI/bge-small-en-v1.5 (FastEmbed ONNX, ~80MB resident)
so no re-embedding is needed when switching backends.

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
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("agentflow.rag.ingest")

INDEX_ROOT = Path(__file__).resolve().parent.parent.parent / "faiss_indexes"

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_EMBEDDINGS: FastEmbedEmbeddings | None = None
_EMBEDDINGS_LOCK = threading.Lock()
_EMBEDDINGS_WARM = False  # public read-only flag for /readyz short-circuit

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
    """Load the embedding model (call from FastAPI lifespan)."""
    _get_embeddings()


def _get_embeddings() -> FastEmbedEmbeddings:
    global _EMBEDDINGS, _EMBEDDINGS_WARM
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    with _EMBEDDINGS_LOCK:
        if _EMBEDDINGS is None:
            _EMBEDDINGS = FastEmbedEmbeddings(model_name=_EMBED_MODEL)
            _EMBEDDINGS_WARM = True
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
    """Return True if the stored model tag matches _EMBED_MODEL."""
    tag_file = index_path / "embed_model.txt"
    if not tag_file.exists():
        return True
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
        _ingest_qdrant(thread_id, chunks)
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

    Automatically routes to Qdrant or FAISS based on QDRANT_URL.
    """
    if _use_qdrant():
        return _get_retriever_qdrant(thread_id)
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
        if _faiss_index_files_valid(out):
            index = _load_faiss_index(out)
            index.add_documents(chunks)
        else:
            index = FAISS.from_documents(chunks, embeddings)
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
