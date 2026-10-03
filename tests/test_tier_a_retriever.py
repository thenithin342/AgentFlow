"""Tier A retriever regression (Phase 1 — Offline Tier A).

Data-driven from evals/datasets/retriever.jsonl. Deterministic: ingestion
embeds with the tests/conftest.py fake embeddings (no keys, no network).

Tier A gate: 100% anchor -> chunk materialization over the indexed chunks
(the exact-phrase recall EVAL_PLAN.md Phase 1 specifies — proves the
chunker + ingest path preserves every golden anchor verbatim). The
materialized IDs are written back into retriever.jsonl (replacing null)
so Tier B can reuse them without re-chunking.

Per-row FAISS top-5 contextual recall/precision are computed, printed,
and saved to evals/results/tier_a_retrieval.json as INFORMATIONAL ONLY:
with fake embeddings the top-k ranking is arbitrary, so the config.yaml
0.85/0.80 gates are enforced in Tier B with real embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.harness import retrieval

pytestmark = pytest.mark.tier_a

DATASET_PATH = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "retriever.jsonl"
TOP_K = 5


def _load_rows() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def backfilled_rows() -> list[dict]:
    """Ingest each source_doc once, materialize anchor chunk IDs, and write
    them back into retriever.jsonl via surgical replacement of the
    `"relevant_chunk_ids": null` sentinel (all other bytes preserved)."""
    rows = _load_rows()
    ids_by_query: dict[str, list[int]] = {}
    for row in rows:
        retrieval.ensure_ingested(row["source_doc"])
        ids_by_query[row["query"]] = retrieval.materialize_chunk_ids(
            row["source_doc"], row["relevant_text"]
        )

    raw = DATASET_PATH.read_text(encoding="utf-8")
    trailing_newline = raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == len(rows), "dataset changed mid-run — refusing to backfill"
    rewritten: list[str] = []
    for line, row in zip(lines, rows, strict=True):
        if line.strip():
            sentinel = '"relevant_chunk_ids": null'
            expected = ids_by_query[row["query"]]
            if sentinel in line:
                line = line.replace(
                    sentinel,
                    f'"relevant_chunk_ids": {json.dumps(expected)}',
                    1,
                )
            else:
                # Already backfilled by a prior run: the stored IDs must
                # equal the freshly materialized ones (determinism check).
                stored = json.loads(line)["relevant_chunk_ids"]
                assert stored == expected, (
                    f"chunk-ID drift for {row['query']!r}: "
                    f"stored {stored} != materialized {expected}"
                )
        rewritten.append(line)
    DATASET_PATH.write_text(
        "\n".join(rewritten) + ("\n" if trailing_newline else ""),
        encoding="utf-8",
    )
    for row in rows:
        row["relevant_chunk_ids"] = ids_by_query[row["query"]]
    return rows


@pytest.fixture(scope="module", autouse=True)
def _cleanup_eval_indexes():
    yield
    retrieval.cleanup_eval_indexes()


def test_anchor_materialization_complete(backfilled_rows):
    """Every golden anchor resolves to exactly one indexed chunk (Tier A
    exact-phrase recall = 100%), and the backfill round-trips through the
    dataset file byte-identically apart from the replaced sentinel."""
    assert len(backfilled_rows) == 12
    for row in backfilled_rows:
        assert row["relevant_chunk_ids"], f"empty chunk IDs for {row['query']!r}"
        assert len(row["relevant_chunk_ids"]) == len(row["relevant_text"])

    on_disk = _load_rows()
    assert len(on_disk) == len(backfilled_rows)
    for mem, disk in zip(backfilled_rows, on_disk, strict=True):
        assert disk["relevant_chunk_ids"] == mem["relevant_chunk_ids"]
        assert disk["relevant_text"] == mem["relevant_text"]
        assert disk["source_doc"] == mem["source_doc"]


def test_topk_structure_and_report(backfilled_rows, capsys):
    """Top-5 retrieval returns well-formed chunk IDs per row; per-row and
    aggregate recall/precision are reported (informational — see module
    docstring) and persisted to evals/results/tier_a_retrieval.json."""
    report: list[dict] = []
    with capsys.disabled():
        print(f"\nTier A retriever report (k={TOP_K}, fake embeddings — informational):")
        for row in backfilled_rows:
            retrieved = retrieval.topk_chunk_ids(row["source_doc"], row["query"], k=TOP_K)
            assert len(retrieved) <= TOP_K
            assert all(isinstance(i, int) and i >= 0 for i in retrieved)
            recall, precision = retrieval.recall_precision(
                row["relevant_chunk_ids"], retrieved, TOP_K
            )
            print(
                f"  recall={recall:.2f} precision={precision:.2f} "
                f"chunks={retrieved} :: {row['query'][:70]}"
            )
            report.append(
                {
                    "query": row["query"],
                    "source_doc": row["source_doc"],
                    "relevant_chunk_ids": row["relevant_chunk_ids"],
                    "retrieved_chunk_ids": retrieved,
                    "recall": recall,
                    "precision": precision,
                }
            )
    path = retrieval.write_results_report(report)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["n_rows"] == len(backfilled_rows)
    print(
        f"\nAggregate: mean_recall={payload['aggregate']['mean_recall']:.3f} "
        f"mean_precision={payload['aggregate']['mean_precision']:.3f} "
        f"({path})"
    )


def test_no_index_fallback():
    """A thread with no index returns the friendly 'no documents' message,
    not an exception."""
    from backend.graph.tools import make_retrieve_documents_tool

    tool = make_retrieve_documents_tool("tier-a-nonexistent-xyz")
    result = tool.invoke({"query": "anything"})
    assert "no documents" in result.lower(), f"Expected friendly fallback, got {result!r}"
