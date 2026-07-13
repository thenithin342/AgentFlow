"""
Qdrant vector store adapter for AgentFlow.

Provides a thin wrapper around langchain-qdrant so that both the RAG
pipeline (rag/ingest.py) and Long-Term Memory (memory/ltm.py) can use
Qdrant when QDRANT_URL is set, without touching any of their internal
logic.

Design decisions:
  - One Qdrant collection per user (LTM) or per thread (RAG), named by
    the SHA-256 hash of the id — consistent with the current FAISS layout.
  - Vectors use the same BAAI/bge-small-en-v1.5 model (384-dim) as FAISS
    so no re-embedding is needed when switching backends.
  - Collection creation is idempotent (create_collection with
    if_not_exists=True semantics via get_or_create).
  - The Qdrant client is a module-level singleton — one connection pool
    for the whole process.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("agentflow.vectorstore.qdrant")

# Vector dimensionality for BAAI/bge-small-en-v1.5
_VECTOR_SIZE = 384

_CLIENT: Any = None          # qdrant_client.QdrantClient singleton
_CLIENT_LOCK = threading.Lock()


def _get_client():
    """Return the module-level Qdrant client, creating it on first call."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from backend.settings import get_settings
            from qdrant_client import QdrantClient

            s = get_settings()
            if not s.qdrant_url:
                raise RuntimeError(
                    "QDRANT_URL is not set — cannot create Qdrant client. "
                    "Check your .env or set the environment variable."
                )
            logger.info("[Qdrant] connecting to %s", s.qdrant_url)
            _CLIENT = QdrantClient(
                url=s.qdrant_url,
                api_key=s.qdrant_api_key or None,
                prefer_grpc=s.qdrant_prefer_grpc,
                timeout=20,
            )
        return _CLIENT


def _ensure_collection(collection_name: str) -> None:
    """Create the Qdrant collection if it does not already exist.

    Uses cosine distance to match the FAISS IndexFlatL2 behaviour at the
    semantic search level (cosine works better than L2 for normalised
    sentence-transformer embeddings).
    """
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("[Qdrant] created collection '%s'", collection_name)


class QdrantStore:
    """Thin LangChain-compatible wrapper around a single Qdrant collection.

    Usage::

        store = QdrantStore("rag_<thread_sha>")
        store.add_documents(chunks)
        docs = store.similarity_search("my query", k=4)
        retriever = store.as_retriever(k=4)
    """

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self._vc: Any = None          # langchain_qdrant.QdrantVectorStore
        self._vc_lock = threading.Lock()

    def _get_vc(self):
        """Return the LangChain QdrantVectorStore, lazily initialised."""
        if self._vc is not None:
            return self._vc
        with self._vc_lock:
            if self._vc is None:
                from langchain_qdrant import QdrantVectorStore

                _ensure_collection(self.collection_name)
                self._vc = QdrantVectorStore(
                    client=_get_client(),
                    collection_name=self.collection_name,
                    embedding=_get_embeddings(),
                )
        return self._vc

    # ------------------------------------------------------------------
    # Public API — mirrors the FAISS interface used by rag/ingest.py
    # and memory/ltm.py so they can call this transparently.
    # ------------------------------------------------------------------

    def add_documents(self, docs: list) -> None:
        """Embed and upsert documents into the collection."""
        if not docs:
            return
        vc = self._get_vc()
        vc.add_documents(docs)
        logger.debug("[Qdrant] added %d docs to '%s'", len(docs), self.collection_name)

    def similarity_search(self, query: str, k: int = 4) -> list:
        """Return the top-k most similar documents for *query*."""
        vc = self._get_vc()
        return vc.similarity_search(query, k=k)

    def as_retriever(self, k: int = 4):
        """Return a LangChain-compatible retriever."""
        vc = self._get_vc()
        return vc.as_retriever(search_kwargs={"k": k})

    def delete_collection(self) -> None:
        """Drop the entire Qdrant collection (used for thread/user cleanup)."""
        client = _get_client()
        try:
            client.delete_collection(self.collection_name)
            self._vc = None
            logger.info("[Qdrant] deleted collection '%s'", self.collection_name)
        except Exception:
            logger.warning(
                "[Qdrant] failed to delete collection '%s'",
                self.collection_name,
                exc_info=True,
            )

    def count(self) -> int:
        """Return the number of points in the collection."""
        client = _get_client()
        try:
            info = client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def delete_oldest(self, keep: int) -> None:
        """Delete the oldest points, keeping only the most recent *keep* entries.

        Points are ordered by their 'timestamp' payload field.
        Used by write_ltm to enforce LTM_MAX_FACTS.
        """
        client = _get_client()
        from qdrant_client.models import Filter, FieldCondition, MatchAny, ScrollRequest

        # Scroll all points to get their timestamps
        all_points = []
        offset = None
        while True:
            result = client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        if len(all_points) <= keep:
            return

        # Sort by timestamp ascending (oldest first)
        def _ts(pt):
            ts = (pt.payload or {}).get("timestamp", "")
            return ts

        all_points.sort(key=_ts)
        to_delete = all_points[: len(all_points) - keep]
        ids_to_delete = [pt.id for pt in to_delete]

        if ids_to_delete:
            client.delete(
                collection_name=self.collection_name,
                points_selector=ids_to_delete,
            )
            logger.debug(
                "[Qdrant] evicted %d old points from '%s'",
                len(ids_to_delete),
                self.collection_name,
            )


def _get_embeddings():
    """Reuse the same FastEmbed model as the RAG pipeline."""
    from backend.rag.ingest import _get_embeddings as _rag_embeddings
    return _rag_embeddings()
