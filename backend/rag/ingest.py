"""
RAG ingestion pipeline — PDF upload to FAISS index.

Phase 7 implementation. Extracts text from a PDF with `PyPDFLoader`, chunks it
with `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=150 — matches
DESIGN_DOC.md §6), embeds the chunks with a local sentence-transformers model
(`all-MiniLM-L6-v2`, cached under the default HuggingFace cache dir so we do
not burn Groq quota on embedding calls — see TECH_STACK.md §4), and persists
the index to `faiss_indexes/{thread_id}/`. Retrieval is scoped per thread so
one conversation's documents do not leak into another.

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

# Using FastEmbedEmbeddings (ONNX-based, ~80MB resident) instead of
# sentence-transformers (PyTorch, ~350MB) so the app fits inside
# Render / Railway free-tier 512MB containers.
_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_EMBEDDINGS: FastEmbedEmbeddings | None = None
_EMBEDDINGS_LOCK = threading.Lock()
_EMBEDDINGS_WARM = False  # public read-only flag for /readyz short-circuit
_RETRIEVERS: OrderedDict[str, Any] = OrderedDict()
_MAX_RETRIEVERS = 1000
_MAX_RETRIEVER_LOCKS = 1000
_RETRIEVER_LOCKS: OrderedDict[str, threading.Lock] = OrderedDict()
_RETRIEVER_LOCKS_GUARD = threading.Lock()


def warm_embeddings() -> None:
    """Load the embedding model (call from FastAPI lifespan)."""
    _get_embeddings()


def _get_embeddings() -> FastEmbedEmbeddings:
    global _EMBEDDINGS, _EMBEDDINGS_WARM
    # Fast path: already initialised. Avoid lock acquire on the hot path.
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    # Double-checked locking: under the lock, re-check the flag before
    # instantiating, otherwise concurrent first-time callers would all
    # load the (large) FastEmbed model and stomp each other.
    with _EMBEDDINGS_LOCK:
        if _EMBEDDINGS is None:
            _EMBEDDINGS = FastEmbedEmbeddings(model_name=_EMBED_MODEL)
            _EMBEDDINGS_WARM = True
        return _EMBEDDINGS


def _index_dir(thread_id: str) -> Path:
    """Return the index directory for *thread_id*.

    Uses a SHA-256 hash of the full thread_id so that colons, slashes,
    or other path-special characters cannot escape INDEX_ROOT, and so
    that the resulting directory name is safe on all platforms.
    The resolved path is verified to stay within INDEX_ROOT.
    """
    safe = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    idx_dir = INDEX_ROOT / safe
    # Defensive: resolve and confirm containment (guards against symlink attacks)
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
        # Legacy index without a tag — assume compatible (back-compat).
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


def ingest_pdf(
    file_path: str,
    thread_id: str,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Extract → chunk → embed → save a per-thread FAISS index.

    Returns ingest stats: ``document_id``, ``source``, ``pages``, ``chunks``.

    The caller is responsible for validating ``thread_id`` *before* scoping it
    to the user. We don't re-validate here because the index path is keyed on
    the scoped id (e.g. ``user:admin:abc``) which the raw regex doesn't accept.
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

    document_id = f"{basename}:{len(chunks)}"
    return {
        "document_id": document_id,
        "source": basename,
        "pages": len(docs),
        "chunks": len(chunks),
    }


def get_retriever(thread_id: str):
    cached = _RETRIEVERS.get(thread_id)
    if cached is not None:
        _RETRIEVERS.move_to_end(thread_id)
        return cached
    # thread_id arrives pre-scoped from the agent config (user:user:abc) and
    # the raw regex above would reject it. Callers gate this through the
    # HTTP layer / make_thread_id, so no re-validation here.
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
