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

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.validation import validate_thread_id


INDEX_ROOT = Path(__file__).resolve().parent.parent.parent / "faiss_indexes"

_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_RETRIEVERS: OrderedDict[str, Any] = OrderedDict()
_MAX_RETRIEVERS = 1000
_MAX_RETRIEVER_LOCKS = 1000
_RETRIEVER_LOCKS: OrderedDict[str, threading.Lock] = OrderedDict()
_RETRIEVER_LOCKS_GUARD = threading.Lock()


def warm_embeddings() -> None:
    """Load the embedding model (call from FastAPI lifespan)."""
    _get_embeddings()


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMBEDDINGS


def _index_dir(thread_id: str) -> Path:
    return INDEX_ROOT / thread_id


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


def _load_faiss_index(index_path: Path) -> FAISS:
    if not _faiss_index_files_valid(index_path):
        raise FileNotFoundError(f"incomplete FAISS index at {index_path}")
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
    """
    validate_thread_id(thread_id)
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
    validate_thread_id(thread_id)
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
