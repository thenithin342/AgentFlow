"""Regression tests for the embedding-provider wiring.

Context: Google retired `models/text-embedding-004` — it started returning
`404 NOT_FOUND` for embedContent. Because that single model id was hardcoded
in rag/ingest.py, the failure took down *two* features at once:

  - `POST /upload` → 500 "internal server error" (embedding call failed)
  - Long-Term Memory → silently empty (read and write both swallowed it),
    so the assistant "forgot" facts the user had stated in other threads

These tests pin the behaviour that made the outage recoverable:
  - model + dimension come from config, never a hardcoded literal
  - an index built by a different/retired model is detected and rebuilt
  - provider failures are classified so /upload can return an actionable hint
"""

import shutil

from langchain_core.documents import Document

import backend.rag.ingest as ingest


def _clean(thread_id: str) -> None:
    idx = ingest._index_dir(thread_id)
    if idx.exists():
        shutil.rmtree(idx, ignore_errors=True)
    ingest._RETRIEVERS.pop(thread_id, None)


def _google_404() -> Exception:
    """Build the exception shape Google returned for the retired model."""

    class GoogleGenerativeAIError(RuntimeError):
        pass

    inner = Exception(
        "404 NOT_FOUND. {'error': {'code': 404, 'message': "
        "'models/text-embedding-004 is not found for API version v1beta, or is "
        "not supported for embedContent.'}}"
    )
    err = GoogleGenerativeAIError("Error embedding content (NOT_FOUND): 404")
    err.__cause__ = inner
    return err


# --- configuration ---------------------------------------------------------


def test_embed_config_is_read_from_settings():
    """The model/dimension must be configuration, not hardcoded literals."""
    from backend.settings import get_settings

    s = get_settings()
    assert ingest._EMBED_MODEL == s.embed_model
    assert ingest._EMBED_DIM == s.embed_dim
    assert ingest._EMBED_DIM > 0


def test_retired_model_is_not_the_default():
    """`text-embedding-004` is retired — the default must not point at it."""
    from backend.settings import Settings

    assert Settings(_env_file=None).embed_model != "models/text-embedding-004"


# --- index model tags ------------------------------------------------------


def test_model_tag_rejects_untagged_index(tmp_path):
    """Pre-tag indexes were built by a different model — treat as stale."""
    assert ingest._check_model_tag(tmp_path) is False


def test_model_tag_accepts_current_model(tmp_path):
    ingest._write_model_tag(tmp_path)
    assert ingest._check_model_tag(tmp_path) is True


def test_model_tag_rejects_other_model(tmp_path):
    (tmp_path / "embed_model.txt").write_text(
        "models/text-embedding-004", encoding="utf-8"
    )
    assert ingest._check_model_tag(tmp_path) is False


def test_ingest_faiss_rebuilds_index_from_retired_model():
    """Uploading into a thread whose index used a retired model must succeed."""
    thread = "test-embed-rebuild-001"
    _clean(thread)
    try:
        ingest._ingest_faiss(
            thread,
            [Document(page_content="alpha policy text", metadata={"source": "a.pdf"})],
        )
        # Simulate an index left behind by the retired embedding model.
        (ingest._index_dir(thread) / "embed_model.txt").write_text(
            "models/text-embedding-004", encoding="utf-8"
        )

        # Must not raise: the stale index is replaced, not merged into.
        ingest._ingest_faiss(
            thread,
            [Document(page_content="beta policy text", metadata={"source": "b.pdf"})],
        )
        assert ingest._check_model_tag(ingest._index_dir(thread)) is True

        retriever = ingest._get_retriever_faiss(thread)
        assert retriever.invoke("beta policy text"), "rebuilt index returned nothing"
    finally:
        _clean(thread)


# --- failure classification ------------------------------------------------


def test_describe_failure_flags_retired_model():
    msg = ingest.describe_embedding_failure(_google_404())
    assert msg is not None
    assert "EMBED_MODEL" in msg


def test_describe_failure_flags_bad_credential():
    exc = RuntimeError("GoogleGenerativeAIError: API key not valid. Please pass a valid API key.")
    msg = ingest.describe_embedding_failure(exc)
    assert msg is not None
    assert "GOOGLE_API_KEY" in msg


def test_describe_failure_flags_quota():
    exc = RuntimeError("GoogleGenerativeAIError: RESOURCE_EXHAUSTED quota exceeded for embedContent")
    msg = ingest.describe_embedding_failure(exc)
    assert msg is not None
    assert "quota" in msg


def test_describe_failure_ignores_unrelated_errors():
    assert ingest.describe_embedding_failure(FileNotFoundError("no such file")) is None
    assert ingest.describe_embedding_failure(KeyError("thread_id")) is None


# --- Qdrant collection dimension guard -------------------------------------
#
# A collection created for a previous embedding model has a frozen vector
# size; Qdrant rejects every upsert that doesn't match it. Left unguarded,
# LTM writes fail silently in exactly the same way the retired model did.


class _StubQdrantClient:
    def __init__(self, size):
        from types import SimpleNamespace

        self._SimpleNamespace = SimpleNamespace
        self.size = size
        self.deleted: list[str] = []
        self.created: list[tuple[str, int]] = []

    def get_collections(self):
        return self._SimpleNamespace(collections=[self._SimpleNamespace(name="rag_x")])

    def get_collection(self, name):
        vectors = self._SimpleNamespace(size=self.size)
        return self._SimpleNamespace(
            config=self._SimpleNamespace(params=self._SimpleNamespace(vectors=vectors))
        )

    def delete_collection(self, name):
        self.deleted.append(name)

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config.size))


def _patch_qdrant(monkeypatch, stub):
    import backend.vectorstore.qdrant_store as qs

    monkeypatch.setattr(qs, "_get_client", lambda: stub)
    monkeypatch.setattr(qs, "_vector_size", lambda: 3072)
    return qs


def test_qdrant_collection_recreated_when_dimension_changes(monkeypatch):
    import pytest as _pytest

    _pytest.importorskip("qdrant_client")
    stub = _StubQdrantClient(size=768)  # collection built by the old model
    qs = _patch_qdrant(monkeypatch, stub)

    qs._ensure_collection("rag_x")

    assert stub.deleted == ["rag_x"]
    assert stub.created == [("rag_x", 3072)]


def test_qdrant_collection_untouched_when_dimension_matches(monkeypatch):
    import pytest as _pytest

    _pytest.importorskip("qdrant_client")
    stub = _StubQdrantClient(size=3072)
    qs = _patch_qdrant(monkeypatch, stub)

    qs._ensure_collection("rag_x")

    assert stub.deleted == []
    assert stub.created == []
