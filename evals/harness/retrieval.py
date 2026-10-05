"""Tier A deterministic retrieval helpers (Phase 1 — Offline Tier A).

Zero LLM / zero network: ingestion embeds via the deterministic fake
embeddings installed by ``tests/conftest.py`` (patched into
``backend.rag.ingest._get_embeddings``), and all chunk math is local.

Responsibilities:
    - extract per-page PDF text (pypdf) for a ``source_doc`` slug;
    - ingest each ``source_doc`` once into a dedicated eval thread
      (``tier-a-<source_doc>``) via ``backend.rag.ingest.ingest_pdf``;
    - MATERIALIZE chunk IDs: re-apply the exact recursive text splitter
      used by ``ingest_pdf`` to the extracted text and find the 0-based
      chunk index containing each ``relevant_text`` anchor substring
      (whitespace-normalized);
    - run top-k retrieval for the thread and compute contextual recall /
      precision against the materialized IDs.

Chunk-ID backfill: the Tier A retriever test writes the materialized
``relevant_chunk_ids`` back into ``evals/datasets/retriever.jsonl`` so
Tier B can reuse them without re-chunking.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve()
DATASETS_DIR = HERE.parent.parent / "datasets"
CORPUS_DIR = DATASETS_DIR / "corpus"
RESULTS_DIR = HERE.parent.parent / "results"

# Corpus slugs backing evals/datasets/retriever.jsonl `source_doc` values.
SOURCE_DOCS = ("agentflow_facts", "langgraph_reference", "northwind_2024")

# Must mirror backend/rag/ingest.py::ingest_pdf exactly
# (RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)).
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 150


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (PDF extraction and
    the retrieve tool both mangle newlines, so anchors are matched
    whitespace-insensitively)."""
    return re.sub(r"\s+", " ", text).strip()


def corpus_pdf_path(source_doc: str) -> Path:
    """Return the golden corpus PDF for a ``source_doc`` slug."""
    return CORPUS_DIR / f"{source_doc}.pdf"


def eval_thread_id(source_doc: str) -> str:
    """Dedicated eval thread per corpus doc (matches THREAD_ID_RE)."""
    return f"tier-a-{source_doc}"


def extract_pdf_text(source_doc: str) -> list[str]:
    """Extract raw per-page text from the corpus PDF (pypdf, local only)."""
    from pypdf import PdfReader

    reader = PdfReader(str(corpus_pdf_path(source_doc)))
    return [page.extract_text() or "" for page in reader.pages]


def materialized_chunks(source_doc: str) -> list[str]:
    """Return the whitespace-normalized chunk texts exactly as
    ``ingest_pdf`` indexed them: same ``PyPDFLoader`` page docs through
    the same ``RecursiveCharacterTextSplitter`` settings."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = PyPDFLoader(str(corpus_pdf_path(source_doc))).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    return [_norm(chunk.page_content) for chunk in splitter.split_documents(docs)]


def materialize_chunk_ids(source_doc: str, anchors: list[str]) -> list[int]:
    """Map each anchor substring to its 0-based chunk index.

    Raises ``ValueError`` when an anchor is found in zero chunks (indexing
    mangled it) or in more than one chunk (anchor is ambiguous) — both
    are dataset/corpus bugs Tier A must surface, not silently pass.
    """
    chunks = materialized_chunks(source_doc)
    ids: list[int] = []
    for anchor in anchors:
        needle = _norm(anchor)
        hits = [i for i, chunk in enumerate(chunks) if needle in chunk]
        if len(hits) != 1:
            raise ValueError(
                f"anchor {anchor!r} found in {len(hits)} chunks of "
                f"{source_doc} (expected exactly 1)"
            )
        ids.append(hits[0])
    return ids


# --- Ingest-once cache -------------------------------------------------------

_INGEST_CACHE: dict[str, dict] = {}


def ensure_ingested(source_doc: str) -> dict:
    """Ingest the corpus PDF into its eval thread (once per process).

    Embeddings come from the ``tests/conftest.py`` fake — no API keys,
    no network. Returns the ``ingest_pdf`` stats dict.
    """
    from backend.rag.ingest import _index_dir, ingest_pdf

    thread_id = eval_thread_id(source_doc)
    if thread_id in _INGEST_CACHE and _index_dir(thread_id).exists():
        return _INGEST_CACHE[thread_id]
    stats = ingest_pdf(
        str(corpus_pdf_path(source_doc)),
        thread_id,
        source_name=f"{source_doc}.pdf",
    )
    _INGEST_CACHE[thread_id] = stats
    return stats


def topk_chunk_ids(source_doc: str, query: str, k: int = 5) -> list[int]:
    """Run top-k retrieval for the thread; return materialized chunk indexes.

    Uses the thread's retriever directly (``Document`` objects carry the
    full ``page_content`` needed for chunk-index mapping — the
    ``retrieve_documents`` *tool* returns a pre-formatted string, so it is
    exercised separately for format/fallback assertions). Each retrieved
    doc must map back to exactly one indexed chunk, else ``ValueError``.
    """
    from backend.rag.ingest import get_retriever

    ensure_ingested(source_doc)
    retriever = get_retriever(eval_thread_id(source_doc))
    # The FAISS retriever defaults to k=4; Tier A evaluates top-5.
    retriever.search_kwargs = {"k": k}
    docs = retriever.invoke(query)

    chunks = materialized_chunks(source_doc)
    index_of = {chunk: i for i, chunk in enumerate(chunks)}
    ids: list[int] = []
    for doc in docs[:k]:
        needle = _norm(doc.page_content)
        if needle in index_of:
            ids.append(index_of[needle])
            continue
        hits = [i for i, chunk in enumerate(chunks) if needle in chunk or chunk in needle]
        if len(hits) != 1:
            raise ValueError(
                f"retrieved doc maps to {len(hits)} indexed chunks "
                f"of {source_doc} (expected exactly 1)"
            )
        ids.append(hits[0])
    return ids


def recall_precision(
    relevant_ids: list[int], retrieved_ids: list[int], k: int
) -> tuple[float, float]:
    """Contextual recall = |relevant ∩ top-k| / |relevant|;
    contextual precision = |relevant ∩ top-k| / k."""
    relevant = set(relevant_ids)
    retrieved = list(retrieved_ids)[:k]
    hits = len(relevant.intersection(retrieved))
    recall = hits / len(relevant) if relevant else 1.0
    precision = hits / k if k else 0.0
    return recall, precision


def write_results_report(rows: list[dict]) -> Path:
    """Write the per-row Tier A retrieval report under evals/results/
    (gitignored). Returns the report path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "tier_a_retrieval.json"
    recalls = [row["recall"] for row in rows]
    precisions = [row["precision"] for row in rows]
    payload = {
        "k": 5,
        "rows": rows,
        "aggregate": {
            "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "mean_precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "n_rows": len(rows),
        },
        "note": (
            "Tier A uses deterministic fake embeddings, so FAISS top-k "
            "ranking is arbitrary; recall/precision here are informational. "
            "The Tier A gate is 100% anchor materialization over the index. "
            "config.yaml thresholds (0.85/0.80) are enforced in Tier B "
            "with real embeddings."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def cleanup_eval_indexes() -> None:
    """Remove per-doc eval FAISS indexes (best-effort; Windows may hold
    the dir open via FAISS mmap — never fail the suite on cleanup).

    Cleans Tier A threads only (``tier-a-<source_doc>``). For Tier B
    ``tierb-retriever-*`` threads, use ``cleanup_tierb_indexes``.
    """
    import shutil

    import backend.rag.ingest as ingest_mod

    for slug in SOURCE_DOCS:
        thread_id = eval_thread_id(slug)
        ingest_mod._RETRIEVERS.pop(thread_id, None)
        index_dir = ingest_mod._index_dir(thread_id)
        if index_dir.exists():
            try:
                shutil.rmtree(index_dir)
            except PermissionError:
                pass
    _INGEST_CACHE.clear()


def cleanup_tierb_indexes(thread_prefix: str = "tierb-retriever-") -> int:
    """Remove Tier B scratch FAISS indexes (best-effort).

    Returns the number of index dirs removed.
    """
    import shutil

    import backend.rag.ingest as ingest_mod

    removed = 0
    # Scan INDEX_ROOT for dirs whose thread_id rounds-trips to a tierb prefix
    index_root = ingest_mod._get_index_root()
    if not index_root.exists():
        return 0
    for hashed_dir in index_root.iterdir():
        if not hashed_dir.is_dir():
            continue
        try:
            # Reverse: find the thread_id whose hash matches this dir name
            # Brute-force over known source_docs + prefix patterns.
            for slug in SOURCE_DOCS:
                for prefix in (f"tierb-retriever-{slug}", f"tierb-{slug}"):
                    if hashlib.sha256(prefix.encode("utf-8")).hexdigest() == hashed_dir.name:
                        ingest_mod._RETRIEVERS.pop(prefix, None)
                        try:
                            shutil.rmtree(hashed_dir)
                            removed += 1
                        except PermissionError:
                            pass
                        break
        except (ValueError, OSError):
            pass
    return removed
