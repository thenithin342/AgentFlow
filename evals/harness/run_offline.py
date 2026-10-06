#!/usr/bin/env python
"""
Tier B offline evaluation CLI.

Usage:
    python evals/harness/run_offline.py <subcommand> [options]

Subcommands:
    router      Run router accuracy evaluation (fully implemented this session)
    retriever   Stub — not implemented yet
    generator   Stub — not implemented yet
    rag-qa      RAG QA triad eval (end-to-end pipeline)
    application Application G-Eval (full graph + synthesizer prebuilt)
    safety      Safety probes through the real graph (gates)
    blog        Blog writer graph turns + structure + G-Eval
    memory      LTM fact extraction + cross-thread round-trip
    baseline    Aggregate evals/results/*_tierb.json into baseline.json

Options:
    --limit N       Limit number of rows to process
    --offset N      Skip first N rows
    --resume        Resume from previous run (not yet implemented)
    --out DIR       Output directory for reports (default: evals/results/)
    --compare PATH  Compare against baseline report
    --dry-run       Print plan without making LLM calls

Environment:
    GROQ_API_KEY    Required for LLM calls
    PYTHONPATH=.    Required for imports
    QDRANT_URL=""   Clear for Tier B runs
    LANGCHAIN_TRACING_V2=false
    LANGCHAIN_API_KEY= (clear it)

Example:
    python evals/harness/run_offline.py router --limit 5
    python evals/harness/run_offline.py router
    python evals/harness/run_offline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure we can import from the project root
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Hard constraint for Tier B: local FAISS vectorstore only (never remote Qdrant)
import os

os.environ["QDRANT_URL"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)
# Tier B memory isolation: keep eval LTM writes out of the real ltm_indexes/ tree.
os.environ["LTM_INDEX_DIR"] = str(PROJECT_ROOT / "evals" / "results" / "tierb_ltm")

from evals.config import load_config
from evals.harness.metrics import (
    confusion_matrix,
    failing_rows,
    per_class_f1,
    router_accuracy,
)
from evals.harness.report import print_summary, write_report
from evals.harness.tierb_runners import (
    build_baseline,
    compare_run_after_run,
    load_application_dataset,
    load_blog_dataset,
    load_memory_dataset,
    load_safety_dataset,
    run_application_evaluation,
    run_blog_evaluation,
    run_compare_only,
    run_memory_evaluation,
    run_safety_evaluation,
    write_tierb_report,
)

# Try to import the router; fail gracefully if backend isn't available
try:
    from backend.graph.router import _route_for_message
    ROUTER_AVAILABLE = True
except Exception as e:
    ROUTER_AVAILABLE = False
    _ROUTER_IMPORT_ERROR = e


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_router_dataset() -> list[dict]:
    """Load the router evaluation dataset from evals/datasets/router.jsonl."""
    dataset_path = PROJECT_ROOT / "evals" / "datasets" / "router.jsonl"
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Router evaluation
# ---------------------------------------------------------------------------

def run_router_evaluation(
    rows: list[dict],
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Run router accuracy evaluation on the given rows.

    Args:
        rows: list of router dataset rows.
        limit: maximum number of rows to process (None = all).
        offset: number of rows to skip from the start.

    Returns:
        Results dict with predictions, accuracy, F1, confusion matrix, etc.
    """
    if not ROUTER_AVAILABLE:
        return {
            "status": "error",
            "error": f"Router not available: {_ROUTER_IMPORT_ERROR}",
        }

    # Apply offset and limit
    sliced_rows = rows[offset:]
    if limit is not None:
        sliced_rows = sliced_rows[:limit]

    predictions: list[str] = []
    timings: list[float] = []
    errors: list[Exception] = []
    # Token tracking placeholder (used when real LLM calls capture usage)
    _token_totals: dict[str, int] = {"input": 0, "output": 0, "total": 0}

    start_time = time.time()

    for i, row in enumerate(sliced_rows):
        query = row.get("input", "")
        row_start = time.time()

        try:
            prediction = _route_for_message(query)
            predictions.append(prediction)

            # Try to extract token usage from the response if available
            # (this is a bit of a hack since _route_for_message doesn't return it)
            row_time = time.time() - row_start
            timings.append(row_time)

        except Exception as e:
            errors.append(e)
            predictions.append("chat")  # fallback prediction
            timings.append(time.time() - row_start)

        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(sliced_rows):
            print(f"    Processed {i + 1}/{len(sliced_rows)} rows...", file=sys.stderr)

    wall_time = time.time() - start_time

    # Compute metrics
    accuracy = router_accuracy(sliced_rows, predictions)
    per_class = per_class_f1(sliced_rows, predictions)
    cm = confusion_matrix(sliced_rows, predictions)
    fails = failing_rows(sliced_rows, predictions)

    # Load thresholds from config
    config = load_config()
    thresholds_config = config.get("thresholds", {})
    router_threshold = thresholds_config.get("router_accuracy", 0.90)

    # Build results
    results: dict[str, Any] = {
        "name": "router",
        "judge_model": "n/a (router uses llm_fast directly)",
        "rows": [],
        "aggregate": {
            "accuracy": accuracy,
            "total_rows": len(sliced_rows),
            "correct": sum(1 for r, p in zip(sliced_rows, predictions, strict=True) if r.get("expected") == p),
            "incorrect": sum(1 for r, p in zip(sliced_rows, predictions, strict=True) if r.get("expected") != p),
        },
        "thresholds": {
            "router_accuracy": {
                "value": accuracy,
                "threshold": router_threshold,
                "passed": accuracy >= router_threshold,
            },
        },
        "operational": {
            "wall_time": wall_time,
            "timings": timings,
            "error_count": len(errors),
        },
    }

    # Add per-row results
    for row, pred in zip(sliced_rows, predictions, strict=True):
        results["rows"].append({
            "query": row.get("input", ""),
            "expected": row.get("expected", ""),
            "predicted": pred,
            "correct": row.get("expected") == pred,
            "reason": row.get("reason", ""),
        })

    # Add aggregate analysis
    results["per_class"] = per_class
    results["confusion_matrix"] = cm
    results["failures"] = fails

    # Print summary
    print_summary(results)

    return results


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_retriever_dataset() -> list[dict]:
    """Load the retriever evaluation dataset from evals/datasets/retriever.jsonl."""
    dataset_path = PROJECT_ROOT / "evals" / "datasets" / "retriever.jsonl"
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_generator_dataset() -> list[dict]:
    """Load the generator evaluation dataset from evals/datasets/generator.jsonl."""
    dataset_path = PROJECT_ROOT / "evals" / "datasets" / "generator.jsonl"
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Retriever evaluation
# ---------------------------------------------------------------------------

def run_retriever_evaluation(
    rows: list[dict],
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Run retriever quality evaluation on the given rows.

    For each row:
    - Ingest the source_doc corpus PDF into a scratch thread using real
      Google GenAI embeddings (once per source_doc — cached across rows).
    - Run top-k retrieval (k=5) for the row's query.
    - Compute recall_at_k and precision_at_k against stored relevant_chunk_ids.

    Args:
        rows: list of retriever dataset rows.
        limit: maximum number of rows to process (None = all).
        offset: number of rows to skip from the start.

    Returns:
        Results dict with per-row recall/precision, aggregate means, etc.
    """
    from backend.rag.ingest import get_retriever, ingest_pdf
    from evals.harness.metrics import aggregate_operational
    from evals.harness.retrieval import (
        _norm,
        cleanup_tierb_indexes,
        materialized_chunks,  # noqa: E402
        recall_precision,
    )

    # Apply offset and limit
    sliced_rows = rows[offset:]
    if limit is not None:
        sliced_rows = sliced_rows[:limit]

    config = load_config()
    retriever_config = config.get("retriever", {})
    k = retriever_config.get("k", 5)
    thread_prefix = retriever_config.get("thread_prefix", "tierb-retriever-")

    thresholds_config = config.get("thresholds", {})
    recall_threshold = thresholds_config.get("contextual_recall", 0.85)
    precision_threshold = thresholds_config.get("contextual_precision", 0.80)

    results_rows: list[dict] = []
    recalls: list[float] = []
    precisions: list[float] = []
    timings: list[float] = []
    errors: list[Exception] = []

    start_time = time.time()

    # Track which source_docs have been ingested (cache across rows)
    ingested_docs: set[str] = set()
    corpus_dir = PROJECT_ROOT / "evals" / "datasets" / "corpus"

    for i, row in enumerate(sliced_rows):
        query = row.get("query", "")
        source_doc = row.get("source_doc", "")
        relevant_ids = row.get("relevant_chunk_ids", [])

        row_start = time.time()

        try:
            # Ingest source_doc if not already ingested (once per doc)
            if source_doc not in ingested_docs:
                print(f"    Ingesting {source_doc} into {thread_prefix}{source_doc}...", file=sys.stderr)

                corpus_path = corpus_dir / f"{source_doc}.pdf"
                thread_id = f"{thread_prefix}{source_doc}"
                ingest_pdf(
                    str(corpus_path),
                    thread_id,
                    source_name=f"{source_doc}.pdf",
                )
                ingested_docs.add(source_doc)

            # Run retrieval
            thread_id = f"{thread_prefix}{source_doc}"
            retriever = get_retriever(thread_id)
            retriever.search_kwargs = {"k": k}
            docs = retriever.invoke(query)

            # Map retrieved docs back to materialized chunk indexes
            chunks = materialized_chunks(source_doc)
            index_of = {chunk: i for i, chunk in enumerate(chunks)}
            retrieved_ids: list[int] = []

            for doc in docs[:k]:
                needle = _norm(doc.page_content)
                if needle in index_of:
                    retrieved_ids.append(index_of[needle])
                else:
                    # Fuzzy fallback: substring match in either direction
                    hits = [
                        idx for idx, chunk in enumerate(chunks)
                        if needle in chunk or chunk in needle
                    ]
                    if len(hits) == 1:
                        retrieved_ids.append(hits[0])

            # Compute recall and precision
            recall, precision = recall_precision(relevant_ids, retrieved_ids, k)

            recalls.append(recall)
            precisions.append(precision)
            row_time = time.time() - row_start
            timings.append(row_time)

            results_rows.append({
                "query": query,
                "source_doc": source_doc,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "recall": recall,
                "precision": precision,
                "recall_threshold": recall_threshold,
                "precision_threshold": precision_threshold,
                "recall_passed": recall >= recall_threshold,
                "precision_passed": precision >= precision_threshold,
            })

        except Exception as e:
            errors.append(e)
            timings.append(time.time() - row_start)
            results_rows.append({
                "query": query,
                "source_doc": source_doc,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": [],
                "recall": 0.0,
                "precision": 0.0,
                "recall_threshold": recall_threshold,
                "precision_threshold": precision_threshold,
                "recall_passed": False,
                "precision_passed": False,
                "error": str(e),
            })

        # Progress indicator
        if (i + 1) % 5 == 0 or (i + 1) == len(sliced_rows):
            print(f"    Processed {i + 1}/{len(sliced_rows)} rows...", file=sys.stderr)

    wall_time = time.time() - start_time

    # Cleanup Tier B indexes
    print("\n    Cleaning up Tier B FAISS indexes...", file=sys.stderr)
    n_removed = cleanup_tierb_indexes(thread_prefix)
    print(f"    Removed {n_removed} Tier B index dirs", file=sys.stderr)

    # Compute aggregates
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0

    operational = aggregate_operational(
        timings=timings if timings else None,
        errors=errors if errors else None,
        total_calls=len(sliced_rows),
    )

    # Build results
    results: dict[str, Any] = {
        "name": "retriever",
        "judge_model": "n/a (no judge calls for retriever)",
        "implementation": "computed",
        "rows": results_rows,
        "aggregate": {
            "mean_recall": mean_recall,
            "mean_precision": mean_precision,
            "n_rows": len(sliced_rows),
            "n_below_threshold": sum(
                1 for r in results_rows
                if not r.get("recall_passed", False) or not r.get("precision_passed", False)
            ),
        },
        "thresholds": {
            "contextual_recall": {
                "value": mean_recall,
                "threshold": recall_threshold,
                "passed": mean_recall >= recall_threshold,
            },
            "contextual_precision": {
                "value": mean_precision,
                "threshold": precision_threshold,
                "passed": mean_precision >= precision_threshold,
            },
        },
        "operational": {
            "wall_time": wall_time,
            "tokens": {"input": 0, "output": 0, "total": 0, "calls": 0},
            **operational,
        },
    }

    # Print summary
    print_summary(results)

    return results


# ---------------------------------------------------------------------------
# Generator evaluation
# ---------------------------------------------------------------------------

def run_generator_evaluation(
    rows: list[dict],
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Run generator quality evaluation on the given rows.

    For each row:
    - Feed the row's context + query to judge_faithfulness and
      judge_answer_relevancy from evals/harness/llm_judge.py.
    - Use the row's reference as the expected answer for answer relevancy.

    Token tracking: calls ``_call_judge`` directly and records the response
    via ``JudgeTokenTracker.record`` so per-call tokens are captured.

    Args:
        rows: list of generator dataset rows.
        limit: maximum number of rows to process (None = all).
        offset: number of rows to skip from the start.

    Returns:
        Results dict with per-row scores, aggregate means, tokens, etc.
    """
    from evals.harness.llm_judge import (
        JudgeTokenTracker,
        judge_answer_relevancy,
        judge_faithfulness,
    )
    from evals.harness.metrics import aggregate_operational

    # Apply offset and limit
    sliced_rows = rows[offset:]
    if limit is not None:
        sliced_rows = sliced_rows[:limit]

    config = load_config()
    thresholds_config = config.get("thresholds", {})
    faithfulness_threshold = thresholds_config.get("faithfulness", 0.90)
    answer_relevancy_threshold = thresholds_config.get("answer_relevancy", 0.90)

    # Token tracker for judge calls
    token_tracker = JudgeTokenTracker()

    results_rows: list[dict] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    timings: list[float] = []
    errors: list[Exception] = []

    start_time = time.time()

    for i, row in enumerate(sliced_rows):
        query = row.get("query", "")
        context = row.get("context", "")
        reference = row.get("reference", "")

        row_start = time.time()

        try:
            # Judge faithfulness and answer relevancy using public API with token tracker
            faithfulness = judge_faithfulness(
                query, context, reference, token_tracker=token_tracker
            )
            answer_relevancy = judge_answer_relevancy(
                query, reference, token_tracker=token_tracker
            )

            if faithfulness is not None:
                faithfulness_scores.append(faithfulness)
            if answer_relevancy is not None:
                answer_relevancy_scores.append(answer_relevancy)

            row_time = time.time() - row_start
            timings.append(row_time)

            results_rows.append({
                "query": query,
                "context": context[:200] + "..." if len(context) > 200 else context,
                "reference": reference,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "faithfulness_threshold": faithfulness_threshold,
                "answer_relevancy_threshold": answer_relevancy_threshold,
                "faithfulness_passed": faithfulness is not None and faithfulness >= faithfulness_threshold,
                "answer_relevancy_passed": answer_relevancy is not None and answer_relevancy >= answer_relevancy_threshold,
            })

        except Exception as e:
            errors.append(e)
            timings.append(time.time() - row_start)
            results_rows.append({
                "query": query,
                "context": context[:200] + "..." if len(context) > 200 else context,
                "reference": reference,
                "faithfulness": None,
                "answer_relevancy": None,
                "faithfulness_threshold": faithfulness_threshold,
                "answer_relevancy_threshold": answer_relevancy_threshold,
                "faithfulness_passed": False,
                "answer_relevancy_passed": False,
                "error": str(e),
            })

        # Progress indicator
        if (i + 1) % 5 == 0 or (i + 1) == len(sliced_rows):
            print(f"    Processed {i + 1}/{len(sliced_rows)} rows...", file=sys.stderr)

    wall_time = time.time() - start_time

    # Compute aggregates
    mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    mean_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0.0

    operational = aggregate_operational(
        timings=timings if timings else None,
        errors=errors if errors else None,
        total_calls=len(sliced_rows),
    )

    # Build results
    results: dict[str, Any] = {
        "name": "generator",
        "judge_model": config.get("judges", {}).get("model", "unknown"),
        "implementation": "hand-rolled",
        "rows": results_rows,
        "aggregate": {
            "mean_faithfulness": mean_faithfulness,
            "mean_answer_relevancy": mean_answer_relevancy,
            "n_rows": len(sliced_rows),
            "n_below_threshold": sum(
                1 for r in results_rows
                if not r.get("faithfulness_passed", False) or not r.get("answer_relevancy_passed", False)
            ),
        },
        "thresholds": {
            "faithfulness": {
                "value": mean_faithfulness,
                "threshold": faithfulness_threshold,
                "passed": mean_faithfulness >= faithfulness_threshold,
            },
            "answer_relevancy": {
                "value": mean_answer_relevancy,
                "threshold": answer_relevancy_threshold,
                "passed": mean_answer_relevancy >= answer_relevancy_threshold,
            },
        },
        "operational": {
            "wall_time": wall_time,
            "tokens": token_tracker.to_dict(),
            **operational,
        },
    }

    # Print summary
    print_summary(results)

    return results


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_ragqa_dataset() -> list[dict]:
    """Load the RAG QA evaluation dataset from evals/datasets/rag_qa.jsonl."""
    dataset_path = PROJECT_ROOT / "evals" / "datasets" / "rag_qa.jsonl"
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# RAG QA evaluation (Tier B pipeline triad)
# ---------------------------------------------------------------------------

def _load_existing_results(output_dir: Path, name: str) -> dict | None:
    """Load existing results JSON if it exists, for --resume support."""
    json_path = output_dir / f"{name}.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _count_tavily_calls(messages: list) -> int:
    """Count tavily_search tool calls from a list of messages.

    Scans for ToolMessage entries whose name contains 'tavily'.
    """
    count = 0
    for msg in messages:
        name = getattr(msg, "name", None)
        if name and "tavily" in name.lower():
            count += 1
    return count


def _is_groq_rate_limit_error(exc: BaseException) -> bool:
    """True when *exc* looks like a Groq/provider 429 rate limit.

    Matches the exception's type name and message only — no provider SDK
    imports needed (RateLimitError etc. stringify their HTTP status).
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text and ("rate" in text or "quota" in text or "groq" in text)


def _classify_judge_failure(judge_error: str | None) -> str:
    """Map a recorded judge failure message to a row status.

    Groq 429 -> "groq_quota"; any other judge failure -> "error".
    """
    lowered = (judge_error or "").lower()
    if "429" in lowered and (
        "rate" in lowered or "quota" in lowered or "groq" in lowered
    ):
        return "groq_quota"
    return "error"


def run_ragqa_evaluation(
    rows: list[dict],
    limit: int | None = None,
    offset: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    """Run RAG QA pipeline triad evaluation on the given rows.

    For each row:
    1. Ingest the row's reference text into a scratch FAISS thread with
       real Google GenAI embeddings.
    2. Run one full graph turn (router → agent → synthesizer).
    3. Extract final_response, documents, route from the state.
    4. Score the triad: contextual_relevancy, faithfulness, answer_relevancy.

    Args:
        rows: list of rag_qa dataset rows.
        limit: maximum number of rows to process (None = all).
        offset: number of rows to skip from the start.
        resume: if True, skip rows already present in the output JSON.

    Returns:
        Results dict with per-row triad scores, route, latency, tokens, etc.
    """
    from evals.harness.llm_judge import (
        JudgeTokenTracker,
        judge_answer_relevancy,
        judge_contextual_relevancy,
        judge_faithfulness,
    )
    from evals.harness.metrics import aggregate_operational

    # Apply offset and limit
    sliced_rows = rows[offset:]
    if limit is not None:
        sliced_rows = sliced_rows[:limit]

    results_rows: list[dict] = []
    cr_scores: list[float] = []
    fth_scores: list[float] = []
    ar_scores: list[float] = []
    timings: list[float] = []
    errors: list[Exception] = []
    tavily_calls_total = 0
    status_counts: dict[str, int] = {"ok": 0, "tavily_quota": 0, "groq_quota": 0, "error": 0}

    # --resume: load existing results and skip completed rows. Only rows
    # with status "ok" count as completed — quota/error rows are retried.
    # Prior completed rows are seeded into the report so aggregates span
    # all runs (batched-over-days execution keeps one coherent artifact).
    existing = None
    completed_queries: set[str] = set()
    if resume:
        existing = _load_existing_results(
            PROJECT_ROOT / "evals" / "results", "ragqa_tierb"
        )
        if existing and "rows" in existing:
            for prev in existing["rows"]:
                if prev.get("status") != "ok":
                    continue
                completed_queries.add(prev.get("query", ""))
                results_rows.append(prev)
                status_counts["ok"] += 1
                for key, bucket in (
                    ("contextual_relevancy", cr_scores),
                    ("faithfulness", fth_scores),
                    ("answer_relevancy", ar_scores),
                ):
                    value = prev.get(key)
                    if isinstance(value, (int, float)):
                        bucket.append(float(value))
            if completed_queries:
                print(
                    f"Resuming: {len(completed_queries)} completed rows found, "
                    f"skipping them",
                    file=sys.stderr,
                )

    config = load_config()
    thresholds_config = config.get("thresholds", {})
    cr_threshold = thresholds_config.get("contextual_relevancy", 0.80)
    fth_threshold = thresholds_config.get("faithfulness", 0.90)
    ar_threshold = thresholds_config.get("answer_relevancy", 0.90)

    token_tracker = JudgeTokenTracker()

    start_time = time.time()

    # Hoisted imports
    import shutil
    import tempfile
    from pathlib import Path

    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import HumanMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Judge error classification for quota handling. Judge functions
    # return None on exception (never raise), so the recorded error is
    # inspected after each row's judge calls to detect provider 429s.
    import evals.harness.llm_judge as llm_judge_mod
    from backend.graph.build_graph import build_compiled_graph
    from backend.rag.ingest import (
        _EMBED_MODEL,
        _get_embeddings,
        _index_dir,
        get_retriever,
    )

    scratch_db = PROJECT_ROOT / "evals" / "results" / "tierb_rag.db"
    scratch_db.parent.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(sliced_rows):
        original_idx = offset + i

        # --resume: skip if this query was already processed
        if resume and row.get("query", "") in completed_queries:
            print(
                f"    Skipping row {original_idx} (already completed)",
                file=sys.stderr,
            )
            continue

        query = row.get("query", "")
        reference = row.get("reference", "")
        thread_id = f"tierb-rag-{original_idx}"

        row_start = time.time()
        row_status = "ok"
        tavily_calls = 0

        try:
            # --- Step 1: Ingest reference text into scratch FAISS thread ---
            tmp_dir = Path(tempfile.mkdtemp(prefix="ragqa_ingest_"))
            try:
                txt_path = tmp_dir / f"ref_{original_idx}.txt"
                txt_path.write_text(reference, encoding="utf-8")

                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = f"ref_{original_idx}.txt"
                    doc.metadata["thread_id"] = thread_id

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=150
                )
                chunks = splitter.split_documents(docs)

                if chunks:
                    index_dir = _index_dir(thread_id)
                    index_dir.mkdir(parents=True, exist_ok=True)
                    embeddings = _get_embeddings()

                    if not any(index_dir.iterdir()):
                        faiss_index = FAISS.from_documents(chunks, embeddings)
                    else:
                        faiss_index = FAISS.load_local(
                            str(index_dir),
                            embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        faiss_index.add_documents(chunks)

                    faiss_index.save_local(str(index_dir))
                    # Write model tag so _load_faiss_index doesn't reject this
                    # index as incompatible (untagged indexes are treated as
                    # built with a different embedding model).
                    (index_dir / "embed_model.txt").write_text(
                        _EMBED_MODEL, encoding="utf-8"
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # --- Step 2: Retrieve documents and run one full graph turn ---
            graph = build_compiled_graph(db_path=str(scratch_db))
            graph_config = {"configurable": {"thread_id": thread_id}}

            # Pre-retrieve documents from the scratch thread so the agent
            # has context available. The research/analysis agents may call
            # retrieve_documents themselves, but pre-populating ensures the
            # documents field is set for triad scoring even if the agent
            # decides not to use the tool.
            context_docs: list[str] = []
            try:
                retriever = get_retriever(thread_id)
                if retriever:
                    retrieved = retriever.invoke(query)
                    context_docs = [d.page_content.strip() for d in retrieved]
            except Exception:
                pass

            graph_start = time.time()
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "documents": context_docs,
                },
                config=graph_config,
            )
            graph_time = time.time() - graph_start

            # --- Step 3: Extract state fields ---
            final_response = result.get("final_response") or ""
            documents: list[str] = result.get("documents") or []
            route = result.get("route") or "unknown"

            all_messages = result.get("messages") or []
            tavily_calls = _count_tavily_calls(all_messages)
            tavily_calls_total += tavily_calls

            # --- Step 4: Score the triad ---
            context_text = "\n\n".join(documents) if documents else ""

            contextual_relevancy: float | None = None
            faithfulness: float | None = None
            answer_relevancy: float | None = None

            if context_text:
                contextual_relevancy = judge_contextual_relevancy(
                    query, context_text, token_tracker=token_tracker
                )

            faithfulness = judge_faithfulness(
                query, context_text, final_response,
                token_tracker=token_tracker,
            )
            answer_relevancy = judge_answer_relevancy(
                query, final_response, token_tracker=token_tracker
            )

            if contextual_relevancy is not None:
                cr_scores.append(contextual_relevancy)
            if faithfulness is not None:
                fth_scores.append(faithfulness)
            if answer_relevancy is not None:
                ar_scores.append(answer_relevancy)

            row_time = time.time() - row_start
            timings.append(row_time)

            results_rows.append({
                "query": query,
                "reference": reference,
                "route": route,
                "final_response": (
                    final_response[:300] + "..."
                    if len(final_response) > 300
                    else final_response
                ),
                "documents_count": len(documents),
                "contextual_relevancy": contextual_relevancy,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "contextual_relevancy_threshold": cr_threshold,
                "faithfulness_threshold": fth_threshold,
                "answer_relevancy_threshold": ar_threshold,
                "contextual_relevancy_passed": (
                    contextual_relevancy is not None
                    and contextual_relevancy >= cr_threshold
                ),
                "faithfulness_passed": (
                    faithfulness is not None
                    and faithfulness >= fth_threshold
                ),
                "answer_relevancy_passed": (
                    answer_relevancy is not None
                    and answer_relevancy >= ar_threshold
                ),
                "graph_time": graph_time,
                "tavily_calls": tavily_calls,
                "status": row_status,
            })

        except Exception as e:
            errors.append(e)
            timings.append(time.time() - row_start)
            row_status = "error"
            status_counts["error"] += 1
            results_rows.append({
                "query": query,
                "reference": reference,
                "route": "error",
                "final_response": "",
                "documents_count": 0,
                "contextual_relevancy": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "contextual_relevancy_threshold": cr_threshold,
                "faithfulness_threshold": fth_threshold,
                "answer_relevancy_threshold": ar_threshold,
                "contextual_relevancy_passed": False,
                "faithfulness_passed": False,
                "answer_relevancy_passed": False,
                "graph_time": 0.0,
                "tavily_calls": 0,
                "status": row_status,
                "error": str(e),
            })

        # Progress indicator
        processed = len(results_rows)
        total = len(sliced_rows)
        if processed % 5 == 0 or processed == total:
            print(f"    Processed {processed}/{total} rows...", file=sys.stderr)

        # --- Quota handling (eval-plan spec) ---
        # Tavily 429 -> mark row tavily_quota, skip triad, continue.
        # Groq 429 -> stop the run entirely (no retry).
        if row_status == "error":
            last_err = errors[-1] if errors else None
            if last_err is not None and _is_groq_rate_limit_error(last_err):
                status_counts["groq_quota"] += 1
                results_rows[-1]["status"] = "groq_quota"
                print(
                    "    Groq rate limit hit (429) - stopping run. "
                    "Use --resume to continue later.",
                    file=sys.stderr,
                )
                break
            if last_err is not None and "tavily" in str(last_err).lower() and "429" in str(last_err):
                status_counts["tavily_quota"] += 1
                results_rows[-1]["status"] = "tavily_quota"
                continue

        # --- Judge failure classification (judges never raise) ---
        if (
            row_status == "ok"
            and (
                contextual_relevancy is None
                or faithfulness is None
                or answer_relevancy is None
            )
        ):
            if _classify_judge_failure(llm_judge_mod.last_judge_error) == "groq_quota":
                status_counts["groq_quota"] += 1
                results_rows[-1]["status"] = "groq_quota"
                print(
                    "    Groq rate limit hit (429) during judging - stopping run. "
                    "Use --resume to continue later.",
                    file=sys.stderr,
                )
                break

        if row_status == "ok":
            status_counts["ok"] += 1
            time.sleep(3)

    wall_time = time.time() - start_time

    # Sort results_rows to maintain original dataset order across resume runs
    query_order = {r.get("query", ""): idx for idx, r in enumerate(rows)}
    results_rows.sort(key=lambda r: query_order.get(r.get("query", ""), 999))

    # Compute aggregates
    mean_cr = (
        sum(cr_scores) / len(cr_scores) if cr_scores else 0.0
    )
    mean_fth = (
        sum(fth_scores) / len(fth_scores) if fth_scores else 0.0
    )
    mean_ar = (
        sum(ar_scores) / len(ar_scores) if ar_scores else 0.0
    )

    operational = aggregate_operational(
        timings=timings if timings else None,
        errors=errors if errors else None,
        total_calls=len(sliced_rows),
    )

    results: dict[str, Any] = {
        "name": "ragqa",
        "judge_model": config.get("judges", {}).get("model", "unknown"),
        "implementation": "hand-rolled",
        "rows": results_rows,
        "aggregate": {
            "mean_contextual_relevancy": mean_cr,
            "mean_faithfulness": mean_fth,
            "mean_answer_relevancy": mean_ar,
            "n_rows": len(results_rows),
            "n_below_threshold": sum(
                1 for r in results_rows
                if not r.get("contextual_relevancy_passed", False)
                or not r.get("faithfulness_passed", False)
                or not r.get("answer_relevancy_passed", False)
            ),
        },
        "thresholds": {
            "contextual_relevancy": {
                "value": mean_cr,
                "threshold": cr_threshold,
                "passed": mean_cr >= cr_threshold,
            },
            "faithfulness": {
                "value": mean_fth,
                "threshold": fth_threshold,
                "passed": mean_fth >= fth_threshold,
            },
            "answer_relevancy": {
                "value": mean_ar,
                "threshold": ar_threshold,
                "passed": mean_ar >= ar_threshold,
            },
        },
        "operational": {
            "wall_time": wall_time,
            "tokens": token_tracker.to_dict(),
            "tavily_calls": tavily_calls_total,
            "status_counts": status_counts,
            **operational,
        },
    }

    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Stub subcommands
# ---------------------------------------------------------------------------

STUB_COMMANDS: set[str] = set()


def run_stub(command: str, args: argparse.Namespace) -> dict[str, Any]:
    """Return a stub result for unimplemented subcommands."""
    return {
        "status": "not_implemented",
        "command": command,
        "message": f"The '{command}' subcommand is not yet implemented. "
                   f"It will be implemented in a future session.",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    # Parent parser with global options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process",
    )
    parent.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N rows",
    )
    parent.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skips already-completed ok rows)",
    )
    parent.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for reports (default: evals/results/)",
    )
    parent.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare against baseline report",
    )
    parent.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making LLM calls",
    )

    parser = argparse.ArgumentParser(
        prog="run_offline.py",
        description="AgentFlow Tier B offline evaluation harness",
    )

    parser.add_argument(
        "--compare",
        type=Path,
        default=argparse.SUPPRESS,
        help="Compare results against a baseline report",
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation subcommand")

    # Router subcommand (inherits global options)
    subparsers.add_parser(
        "router",
        parents=[parent],
        help="Run router accuracy evaluation",
    )

    # Retriever subcommand (inherits global options)
    subparsers.add_parser(
        "retriever",
        parents=[parent],
        help="Run retriever quality evaluation",
    )

    # Generator subcommand (inherits global options)
    subparsers.add_parser(
        "generator",
        parents=[parent],
        help="Run generator quality evaluation",
    )

    # Stub subcommands (inherits global options)
    for cmd in sorted(STUB_COMMANDS):
        subparsers.add_parser(
            cmd,
            parents=[parent],
            help=f"{cmd} evaluation (not yet implemented)",
        )

    # Rag-qa subcommand (inherits global options) — implemented this session
    subparsers.add_parser(
        "rag-qa",
        parents=[parent],
        help="Run RAG QA pipeline triad evaluation (end-to-end graph runs)",
    )

    # Session 5 subcommands (implemented in harness/tierb_runners.py)
    subparsers.add_parser(
        "application",
        parents=[parent],
        help="Run application G-Eval (full graph + synthesizer prebuilt)",
    )
    subparsers.add_parser(
        "safety",
        parents=[parent],
        help="Run safety probes through the real graph (gates)",
    )
    subparsers.add_parser(
        "memory",
        parents=[parent],
        help="Run LTM fact extraction + cross-thread round-trip",
    )
    subparsers.add_parser(
        "blog",
        parents=[parent],
        help="Run blog writer graph turns + structure + G-Eval",
    )
    subparsers.add_parser(
        "baseline",
        parents=[parent],
        help="Aggregate evals/results/*_tierb.json into baseline.json",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    compare_path = getattr(args, "compare", None)

    # If no command given, show help
    if args.command is None:
        if compare_path is not None:
            out_dir = getattr(args, "out", None) or PROJECT_ROOT / "evals" / "results"
            return run_compare_only(compare_path, out_dir)
        parser.print_help()
        return 0

    # Handle --dry-run
    if args.dry_run:
        print("=== DRY RUN — No LLM calls will be made ===\n")
        print(f"Command: {args.command}")
        print(f"Limit: {args.limit if args.limit else 'all rows'}")
        print(f"Offset: {args.offset}")
        print(f"Output dir: {args.out or 'evals/results/'}")
        if args.compare:
            print(f"Compare against: {args.compare}")

        if args.command == "router":
            rows = load_router_dataset()
            sliced = rows[args.offset:]
            if args.limit:
                sliced = sliced[:args.limit]
            print(f"\nRouter dataset: {len(rows)} total rows, "
                  f"will process {len(sliced)} rows")
            print("\nPlan:")
            print("  1. Load router dataset from evals/datasets/router.jsonl")
            print("  2. For each row, call _route_for_message() via Groq")
            print("  3. Compare predictions to expected labels")
            print("  4. Compute accuracy, per-class F1, confusion matrix")
            print("  5. Write JSON + Markdown report to evals/results/")
        elif args.command == "retriever":
            rows = load_retriever_dataset()
            sliced = rows[args.offset:]
            if args.limit:
                sliced = sliced[:args.limit]
            print(f"\nRetriever dataset: {len(rows)} total rows, "
                  f"will process {len(sliced)} rows")
            print("\nPlan:")
            print("  1. Load retriever dataset from evals/datasets/retriever.jsonl")
            print("  2. For each row, ingest source_doc corpus with real embeddings")
            print("  3. Run top-k retrieval (k=5) for each query")
            print("  4. Compute recall@k and precision@k against relevant_chunk_ids")
            print("  5. Write JSON + Markdown report to evals/results/")
        elif args.command == "generator":
            rows = load_generator_dataset()
            sliced = rows[args.offset:]
            if args.limit:
                sliced = sliced[:args.limit]
            print(f"\nGenerator dataset: {len(rows)} total rows, "
                  f"will process {len(sliced)} rows")
            print("\nPlan:")
            print("  1. Load generator dataset from evals/datasets/generator.jsonl")
            print("  2. For each row, call judge_faithfulness and judge_answer_relevancy")
            print("  3. Compare scores to thresholds")
            print("  4. Write JSON + Markdown report to evals/results/")
        elif args.command == "rag-qa":
            rows = load_ragqa_dataset()
            sliced = rows[args.offset:]
            if args.limit:
                sliced = sliced[:args.limit]
            print(f"\nRAG QA dataset: {len(rows)} total rows, "
                  f"will process {len(sliced)} rows")
            if args.resume:
                print("Resume mode: skipping already-completed rows")
            print("\nPlan:")
            print("  1. Load RAG QA dataset from evals/datasets/rag_qa.jsonl")
            print("  2. For each row, ingest reference text into scratch thread")
            print("  3. Run one full graph turn (router -> agent -> synthesizer)")
            print("  4. Score triad: contextual_relevancy, faithfulness, answer_relevancy")
            print("  5. Write JSON + Markdown report to evals/results/")
            print("  6. Clean up scratch DB and FAISS indexes")
        elif args.command in {"application", "safety", "memory", "blog"}:
            loaders = {
                "application": load_application_dataset,
                "safety": load_safety_dataset,
                "memory": load_memory_dataset,
                "blog": load_blog_dataset,
            }
            rows = loaders[args.command]()
            sliced = rows[args.offset:]
            if args.limit:
                sliced = sliced[:args.limit]
            print(f"\n{args.command} dataset: {len(rows)} total rows, "
                  f"will process {len(sliced)} rows")
            print("\nPlan:")
            print("  1. Run real graph turns (synthesizer prebuilt for synthesizer rows)")
            print("  2. Score with hand-rolled judges + deterministic gates")
            print("  3. Write JSON + Markdown report to evals/results/")
        elif args.command == "baseline":
            print("\nPlan: aggregate evals/results/*_tierb.json into baseline.json")
        else:
            print("\nStatus: not_implemented (stub)")

        return 0

    # Run the evaluation
    print(f"\n{'='*60}")
    print(f"  AgentFlow Eval: {args.command}")
    print(f"{'='*60}\n")

    if args.command == "router":
        if not ROUTER_AVAILABLE:
            print(f"ERROR: Router not available: {_ROUTER_IMPORT_ERROR}", file=sys.stderr)
            print("Make sure the backend package is installed and GROQ_API_KEY is set.",
                  file=sys.stderr)
            return 1

        rows = load_router_dataset()
        print(f"Loaded {len(rows)} router evaluation rows")

        results = run_router_evaluation(
            rows,
            limit=args.limit,
            offset=args.offset,
        )

        if results.get("status") == "error":
            print(f"ERROR: {results.get('error')}", file=sys.stderr)
            return 1

        # Write reports
        output_dir = args.out or PROJECT_ROOT / "evals" / "results"
        json_path, md_path = write_report(
            name="router_tierb",
            results=results,
            output_dir=output_dir,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return 0

    elif args.command == "retriever":
        rows = load_retriever_dataset()
        print(f"Loaded {len(rows)} retriever evaluation rows")

        # Check for GOOGLE_API_KEY
        import os
        if not os.environ.get("GOOGLE_API_KEY"):
            print("WARNING: GOOGLE_API_KEY not set. Retriever evaluation requires real embeddings.",
                  file=sys.stderr)
            print("The evaluation will attempt to run but may fail if embeddings are not available.",
                  file=sys.stderr)

        results = run_retriever_evaluation(
            rows,
            limit=args.limit,
            offset=args.offset,
        )

        if results.get("status") == "error":
            print(f"ERROR: {results.get('error')}", file=sys.stderr)
            return 1

        # Write reports
        output_dir = args.out or PROJECT_ROOT / "evals" / "results"
        json_path, md_path = write_report(
            name="retriever_tierb",
            results=results,
            output_dir=output_dir,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return 0

    elif args.command == "generator":
        rows = load_generator_dataset()
        print(f"Loaded {len(rows)} generator evaluation rows")

        # Check for GROQ_API_KEY (needed for judge calls)
        import os
        if not os.environ.get("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY not set. Generator evaluation requires LLM judge calls.",
                  file=sys.stderr)
            print("The evaluation will attempt to run but may fail if the judge is not available.",
                  file=sys.stderr)

        results = run_generator_evaluation(
            rows,
            limit=args.limit,
            offset=args.offset,
        )

        if results.get("status") == "error":
            print(f"ERROR: {results.get('error')}", file=sys.stderr)
            return 1

        # Write reports
        output_dir = args.out or PROJECT_ROOT / "evals" / "results"
        json_path, md_path = write_report(
            name="generator_tierb",
            results=results,
            output_dir=output_dir,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return 0

    elif args.command == "rag-qa":
        rows = load_ragqa_dataset()
        print(f"Loaded {len(rows)} RAG QA evaluation rows")

        # Check for required API keys
        import os
        missing = []
        if not os.environ.get("GROQ_API_KEY"):
            missing.append("GROQ_API_KEY")
        if not os.environ.get("GOOGLE_API_KEY"):
            missing.append("GOOGLE_API_KEY")
        if missing:
            print(f"WARNING: {', '.join(missing)} not set. RAG QA evaluation may fail.",
                  file=sys.stderr)

        results = run_ragqa_evaluation(
            rows,
            limit=args.limit,
            offset=args.offset,
            resume=args.resume,
        )

        # Write reports
        output_dir = args.out or PROJECT_ROOT / "evals" / "results"
        json_path, md_path = write_report(
            name="ragqa_tierb",
            results=results,
            output_dir=output_dir,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Clean up scratch DB and FAISS indexes
        print("\nCleaning up scratch resources...", file=sys.stderr)
        scratch_db = PROJECT_ROOT / "evals" / "results" / "tierb_rag.db"
        if scratch_db.exists():
            try:
                scratch_db.unlink()
                print(f"  Removed scratch DB: {scratch_db}", file=sys.stderr)
            except Exception as e:
                print(f"  Could not remove scratch DB: {e}", file=sys.stderr)

        # Clean up tierb-rag-* FAISS indexes
        from evals.harness.retrieval import cleanup_tierb_indexes
        n_removed = cleanup_tierb_indexes("tierb-rag-")
        print(f"  Removed {n_removed} tierb-rag- FAISS index dirs", file=sys.stderr)

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        # Report quota state
        op = results.get("operational", {})
        sc = op.get("status_counts", {})
        if sc.get("tavily_quota", 0) > 0:
            print(f"\n[QUOTA] Tavily quota exhausted on {sc['tavily_quota']} row(s). "
                  "Remaining rows skipped.", file=sys.stderr)
        if sc.get("groq_quota", 0) > 0:
            print(f"\n[QUOTA] Groq quota exhausted on {sc['groq_quota']} row(s).", file=sys.stderr)

        return 0

    elif args.command in ("application", "safety", "memory", "blog"):
        loaders = {
            "application": load_application_dataset,
            "safety": load_safety_dataset,
            "memory": load_memory_dataset,
            "blog": load_blog_dataset,
        }
        runners = {
            "application": run_application_evaluation,
            "safety": run_safety_evaluation,
            "memory": run_memory_evaluation,
            "blog": run_blog_evaluation,
        }
        rows = loaders[args.command]()
        print(f"Loaded {len(rows)} {args.command} evaluation rows")

        results = runners[args.command](
            rows,
            limit=args.limit,
            offset=args.offset,
            resume=args.resume,
        )

        output_dir = args.out or PROJECT_ROOT / "evals" / "results"
        json_path, md_path = write_tierb_report(
            name=f"{args.command}_tierb",
            results=results,
            output_dir=output_dir,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        op = results.get("operational", {})
        if op.get("tavily_calls"):
            print(f"  Tavily search calls this run: {op['tavily_calls']}")
        graph_tokens = op.get("graph_tokens")
        if isinstance(graph_tokens, dict) and graph_tokens.get("calls"):
            print(f"  In-graph LLM tokens: {graph_tokens.get('total', 0):,} "
                  f"({graph_tokens.get('calls', 0)} calls)")

        if compare_path is not None:
            compare_run_after_run(compare_path, output_dir, args.command)

        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return 0

    elif args.command == "baseline":
        output_dir = args.out or PROJECT_ROOT / "evals" / "results"
        baseline = build_baseline(output_dir)
        out_path = output_dir / "baseline.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(baseline, handle, indent=2, ensure_ascii=False)
        print(f"Baseline written: {out_path}")
        print(f"  Runs aggregated: {', '.join(sorted(baseline['runs'].keys()))}")
        print(f"  Metrics captured: {len(baseline['metrics'])}")
        if compare_path is not None:
            run_compare_only(compare_path, output_dir)
        return 0

    elif args.command in STUB_COMMANDS:
        results = run_stub(args.command, args)
        print(json.dumps(results, indent=2))
        return 0

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
