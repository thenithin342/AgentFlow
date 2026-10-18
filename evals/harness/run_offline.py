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
    --compare-run K With --compare and no subcommand, compare only run K
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
import contextlib
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
    _build_eval_graph,
    bounded_call,
    build_baseline,
    build_provenance,
    compare_run_after_run,
    dataset_sha,
    load_application_dataset,
    load_blog_dataset,
    load_memory_dataset,
    load_safety_dataset,
    report_row_issues,
    row_identity,
    run_application_evaluation,
    run_blog_evaluation,
    run_compare_only,
    run_key_for_cli,
    run_memory_evaluation,
    run_safety_evaluation,
    seed_prior_ok,
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
            prediction = bounded_call(_route_for_message, query)
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
    from evals.harness.metrics import aggregate_operational, average_precision_at_k
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
    import uuid
    thread_prefix = retriever_config.get("thread_prefix", "tierb-retriever-") + uuid.uuid4().hex + "-"

    thresholds_config = config.get("thresholds", {})
    recall_threshold = thresholds_config.get("contextual_recall", 0.85)
    precision_threshold = thresholds_config.get("contextual_precision", 0.80)

    results_rows: list[dict] = []
    recalls: list[float] = []
    precisions: list[float] = []
    contextual_precisions: list[float] = []
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
                bounded_call(ingest_pdf,
                    str(corpus_path),
                    thread_id,
                    source_name=f"{source_doc}.pdf",
                )
                ingested_docs.add(source_doc)

            # Run retrieval
            thread_id = f"{thread_prefix}{source_doc}"
            retriever = get_retriever(thread_id)
            retriever.search_kwargs = {"k": k}
            docs = bounded_call(retriever.invoke, query)

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

            # Recall plus a rank-weighted contextual precision (AP@k). Raw
            # precision@k is kept as an informational signal but never gated:
            # with 1-2 gold chunks and k=5 it is capped at |relevant|/k and
            # cannot reach the 0.80 target (audit P2 #5).
            recall, precision = recall_precision(relevant_ids, retrieved_ids, k)
            contextual_precision = average_precision_at_k(
                set(relevant_ids), retrieved_ids, k
            )

            recalls.append(recall)
            precisions.append(precision)
            contextual_precisions.append(contextual_precision)
            row_time = time.time() - row_start
            timings.append(row_time)

            results_rows.append({
                "query": query,
                "source_doc": source_doc,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "recall": recall,
                "precision": precision,
                "contextual_precision": contextual_precision,
                "recall_threshold": recall_threshold,
                "precision_threshold": precision_threshold,
                "recall_passed": recall >= recall_threshold,
                "contextual_precision_passed": (
                    contextual_precision >= precision_threshold
                ),
            })

        except Exception as e:
            errors.append(e)
            recalls.append(0.0)
            precisions.append(0.0)
            contextual_precisions.append(0.0)
            timings.append(time.time() - row_start)
            results_rows.append({
                "query": query,
                "source_doc": source_doc,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": [],
                "recall": 0.0,
                "precision": 0.0,
                "contextual_precision": 0.0,
                "recall_threshold": recall_threshold,
                "precision_threshold": precision_threshold,
                "recall_passed": False,
                "contextual_precision_passed": False,
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
    mean_contextual_precision = (
        sum(contextual_precisions) / len(contextual_precisions)
        if contextual_precisions else 0.0
    )

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
            "mean_contextual_precision": mean_contextual_precision,
            "n_rows": len(sliced_rows),
            "n_below_threshold": sum(
                1 for r in results_rows
                if not r.get("recall_passed", False)
                or not r.get("contextual_precision_passed", False)
            ),
        },
        "thresholds": {
            "contextual_recall": {
                "value": mean_recall,
                "threshold": recall_threshold,
                "passed": mean_recall >= recall_threshold,
            },
            # ``contextual_precision`` is the rank-weighted AP@k gate: it is
            # attainable, unlike the raw precision@k it replaces (audit P2 #5).
            "contextual_precision": {
                "value": mean_contextual_precision,
                "threshold": precision_threshold,
                "passed": mean_contextual_precision >= precision_threshold,
            },
            # Raw precision@k stays visible but informational (never gated).
            "precision_at_k": {
                "value": mean_precision,
                "threshold": None,
                "passed": None,
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

def generator_answer_prompt(query: str, context: str) -> list:
    """Build the grounded-answer prompt for the component under test.

    Pure (no LLM), so it is unit-testable. Scored answers must come from this
    prompt, not from the dataset's golden ``reference``.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You are the AgentFlow answer generator. Answer the user's question "
        "using ONLY the provided context. If the context does not contain the "
        "answer, say so explicitly instead of guessing. Be concise and never "
        "invent facts."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _truncate(text: str, n: int) -> str:
    """Truncate *text* to *n* characters with an ellipsis marker."""
    text = text or ""
    return text[:n] + "..." if len(text) > n else text


def _response_text(response: Any) -> str:
    """Best-effort plain text from a chat-model response."""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def run_generator_evaluation(
    rows: list[dict],
    limit: int | None = None,
    offset: int = 0,
    *,
    calibrate_reference: bool = False,
) -> dict[str, Any]:
    """Evaluate the *generator* — not the golden references.

    For each row the component under test (``llm_fast`` + a grounded-answer
    prompt) generates an answer from the row's ``query`` + ``context``, and the
    judges score **that generated answer**. Previously the golden ``reference``
    was judged and the result was mislabelled as generator quality (audit
    item 4).

    The reference is only scored when *calibrate_reference* is set, and its
    scores are reported under ``calibration`` — never gated, never mixed into
    the generator means — so judge/prompt miscalibration stays visible.

    Failed generation or judge calls record 0.0 (never vanish from a mean) and
    are counted in ``n_generation_failures`` / ``n_judge_failures`` so a
    partially-failed run cannot report an inflated average.
    """
    from backend.llm import llm_fast
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

    # Token trackers: judge calls vs the generator call under test.
    token_tracker = JudgeTokenTracker()
    generator_tracker = JudgeTokenTracker()

    results_rows: list[dict] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    reference_faithfulness_scores: list[float] = []
    reference_answer_relevancy_scores: list[float] = []
    timings: list[float] = []
    errors: list[Exception] = []
    judge_failures = 0
    generation_failures = 0

    start_time = time.time()

    for i, row in enumerate(sliced_rows):
        query = row.get("query", "")
        context = row.get("context", "")
        reference = row.get("reference", "")

        row_start = time.time()
        truncated_context = context[:200] + "..." if len(context) > 200 else context

        try:
            # --- Step 1: generate an answer from query + context (bounded so a
            # stuck provider call cannot block the job; re-audit P1) ---
            from evals.harness.tierb_runners import (
                GRAPH_TIMEOUT_S as _GEN_TIMEOUT_S,
            )
            from evals.harness.tierb_runners import (
                JUDGE_TIMEOUT_S as _GEN_JUDGE_TIMEOUT_S,
            )
            from evals.harness.tierb_runners import _call_with_timeout as _bounded_call

            response, gen_timed_out = _bounded_call(
                llm_fast.invoke,
                generator_answer_prompt(query, context),
                timeout_s=_GEN_TIMEOUT_S,
                thread_name="tierb-generator-invoke",
            )
            if gen_timed_out:
                raise TimeoutError(
                    f"generator invoke exceeded {_GEN_TIMEOUT_S:.0f}s bound"
                )
            generator_tracker.record(response)
            answer = _response_text(response).strip()
            if not answer:
                raise ValueError("generator returned an empty answer")

            # --- Step 2: judge the generated answer (this is the component eval) ---
            faithfulness, _fth_timed_out = _bounded_call(
                judge_faithfulness,
                query,
                context,
                answer,
                token_tracker=token_tracker,
                timeout_s=_GEN_JUDGE_TIMEOUT_S,
                thread_name="tierb-judge-invoke",
            )
            answer_relevancy, _ar_timed_out = _bounded_call(
                judge_answer_relevancy,
                query,
                answer,
                token_tracker=token_tracker,
                timeout_s=_GEN_JUDGE_TIMEOUT_S,
                thread_name="tierb-judge-invoke",
            )
            if _fth_timed_out or _ar_timed_out:
                import evals.harness.llm_judge as _llm_judge_mod

                _llm_judge_mod.last_judge_error = (
                    f"TimeoutError: generator judge exceeded {_GEN_JUDGE_TIMEOUT_S:.0f}s bound"
                )

            if faithfulness is None or answer_relevancy is None:
                judge_failures += 1
            faithfulness_scores.append(float(faithfulness) if faithfulness is not None else 0.0)
            answer_relevancy_scores.append(
                float(answer_relevancy) if answer_relevancy is not None else 0.0
            )

            # --- Step 3 (opt-in): golden-reference calibration (bounded) ---
            ref_faithfulness = None
            ref_answer_relevancy = None
            if calibrate_reference:
                ref_faithfulness, _ = _bounded_call(
                    judge_faithfulness,
                    query,
                    context,
                    reference,
                    token_tracker=token_tracker,
                    timeout_s=_GEN_JUDGE_TIMEOUT_S,
                    thread_name="tierb-judge-invoke",
                )
                ref_answer_relevancy, _ = _bounded_call(
                    judge_answer_relevancy,
                    query,
                    reference,
                    token_tracker=token_tracker,
                    timeout_s=_GEN_JUDGE_TIMEOUT_S,
                    thread_name="tierb-judge-invoke",
                )
                reference_faithfulness_scores.append(
                    float(ref_faithfulness) if ref_faithfulness is not None else 0.0
                )
                reference_answer_relevancy_scores.append(
                    float(ref_answer_relevancy) if ref_answer_relevancy is not None else 0.0
                )

            row_time = time.time() - row_start
            timings.append(row_time)

            row_result: dict[str, Any] = {
                "query": query,
                "context": truncated_context,
                "reference": reference,
                "generated_answer": _truncate(answer, 600),
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "faithfulness_threshold": faithfulness_threshold,
                "answer_relevancy_threshold": answer_relevancy_threshold,
                "faithfulness_passed": faithfulness is not None and faithfulness >= faithfulness_threshold,
                "answer_relevancy_passed": answer_relevancy is not None and answer_relevancy >= answer_relevancy_threshold,
                "status": "ok" if (faithfulness is not None and answer_relevancy is not None) else "error",
            }
            if calibrate_reference:
                row_result["calibration"] = {
                    "reference_faithfulness": ref_faithfulness,
                    "reference_answer_relevancy": ref_answer_relevancy,
                }
            results_rows.append(row_result)

        except Exception as e:
            errors.append(e)
            # The row never produced a judged answer: this is a generation/setup
            # failure, not a judge failure.
            generation_failures += 1
            faithfulness_scores.append(0.0)
            answer_relevancy_scores.append(0.0)
            timings.append(time.time() - row_start)
            results_rows.append({
                "query": query,
                "context": truncated_context,
                "reference": reference,
                "generated_answer": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "faithfulness_threshold": faithfulness_threshold,
                "answer_relevancy_threshold": answer_relevancy_threshold,
                "faithfulness_passed": False,
                "answer_relevancy_passed": False,
                "status": "error",
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
        "implementation": "generated-answer + judge",
        "rows": results_rows,
        "aggregate": {
            "mean_faithfulness": mean_faithfulness,
            "mean_answer_relevancy": mean_answer_relevancy,
            "n_rows": len(results_rows),
            "n_judge_failures": judge_failures,
            "n_generation_failures": generation_failures,
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
        "calibration": {
            "enabled": calibrate_reference,
            "note": (
                "Golden-reference judge scores — judge calibration only, never "
                "a generator-quality gate."
            ),
            "mean_reference_faithfulness": (
                sum(reference_faithfulness_scores) / len(reference_faithfulness_scores)
                if reference_faithfulness_scores else None
            ),
            "mean_reference_answer_relevancy": (
                sum(reference_answer_relevancy_scores) / len(reference_answer_relevancy_scores)
                if reference_answer_relevancy_scores else None
            ),
        },
        "operational": {
            "wall_time": wall_time,
            "tokens": token_tracker.to_dict(),
            "judge_tokens": token_tracker.to_dict(),
            "generator_tokens": generator_tracker.to_dict(),
            "expected_rows": len(sliced_rows),
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

try:
    from langchain_core.embeddings import Embeddings as _EmbeddingsBase
except Exception:  # pragma: no cover - langchain_core always present in practice
    _EmbeddingsBase = object  # type: ignore[misc,assignment]


class _RetryingEmbeddings(_EmbeddingsBase):  # type: ignore[misc]
    """Bounded retry/backoff wrapper around an embeddings client.

    Subclasses LangChain's ``Embeddings`` interface so FAISS takes its
    ``embed_documents``/``embed_query`` path instead of its callable fallback
    (which raised ``TypeError: '_RetryingEmbeddings' object is not callable``
    at query time). ``__call__`` delegates to ``embed_query`` for any caller
    that still treats the embedding function as a plain callable.

    The embeddings provider intermittently fails with 503 / handshake
    timeouts; a single transient blip used to abort a whole RAG-QA row and, on
    the retriever, a whole run (audit item 6). Only raised exceptions are
    retried — a successful (or empty) response is returned as-is.
    """

    def __init__(
        self,
        inner: Any,
        attempts: int = 4,
        base_delay_s: float = 1.5,
        max_delay_s: float = 15.0,
    ) -> None:
        self._inner = inner
        self._attempts = max(1, int(attempts))
        self._base_delay = float(base_delay_s)
        self._max_delay = float(max_delay_s)
        self.attempts_used = 0
        self.retries = 0

    def _call(self, name: str, *args, **kwargs):
        delay = self._base_delay
        last_exc: BaseException | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                self.attempts_used += 1
                return getattr(self._inner, name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == self._attempts:
                    break
                self.retries += 1
                print(
                    f"    [embeddings] {name} failed (attempt {attempt}/"
                    f"{self._attempts}): {exc}; retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay = min(delay * 2, self._max_delay)
        assert last_exc is not None
        raise last_exc

    def embed_documents(self, texts, **kwargs):
        return self._call("embed_documents", texts, **kwargs)

    def embed_query(self, text, **kwargs):
        return self._call("embed_query", text, **kwargs)

    def __call__(self, text, **kwargs):
        """Delegate callable use to ``embed_query`` (FAISS fallback path)."""
        return self.embed_query(text, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._inner, name)


@contextlib.contextmanager
def _retrying_embeddings(attempts: int = 4, base_delay_s: float = 1.5):
    """Temporarily install a retrying embeddings client for the eval run.

    Swaps ``backend.rag.ingest._EMBEDDINGS`` (what ``_get_embeddings()``
    returns) for a retrying wrapper, so both PDF ingestion and query-time
    retrieval get bounded retry/backoff. Both the cached singleton *and* the
    module-level getter are patched: patching only ``_EMBEDDINGS`` still
    bypasses the wrapper whenever ``_get_embeddings()`` early-returns the
    saved value or a test fixture replaced the getter itself. Restored on
    exit.
    """
    import backend.rag.ingest as ingest_mod

    saved_singleton = ingest_mod._EMBEDDINGS
    saved_getter = ingest_mod._get_embeddings
    # Resolve the real client without going through a patched getter: when
    # the getter itself was replaced (e.g. tests/conftest.py CI fake), call
    # through to the underlying factory so the wrapper still sits in front.
    try:
        inner = saved_getter()
    except Exception:  # noqa: BLE001 - fall back to the cached value
        inner = saved_singleton
    if isinstance(inner, _RetryingEmbeddings):
        inner = inner._inner
    wrapper = _RetryingEmbeddings(inner, attempts, base_delay_s)
    ingest_mod._EMBEDDINGS = wrapper

    def _wrapped_getter():
        return wrapper

    # Keep a back-reference so nested/stacked installs unwrap correctly.
    _wrapped_getter.__wrapped__ = saved_getter  # type: ignore[attr-defined]
    ingest_mod._get_embeddings = _wrapped_getter  # type: ignore[assignment]
    try:
        yield wrapper
    finally:
        ingest_mod._EMBEDDINGS = saved_singleton
        ingest_mod._get_embeddings = saved_getter  # type: ignore[assignment]


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
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run RAG QA pipeline triad evaluation on the given rows.

    Corpus-backed: the three golden corpus PDFs are ingested through the real
    ``ingest_pdf`` pipeline into one per-run scratch index, so each query has to
    pick the right chunk out of distractors instead of being handed its own
    reference text (audit item 6). Per row:

    1. Retrieve top-k chunks from the scratch corpus index (with bounded
       retry/backoff on transient embedding failures).
    2. Run one full graph turn (router → agent → synthesizer) against a *fresh*
       checkpointer so conversation history cannot leak between rows.
    3. Extract final_response, documents, route from the state.
    4. Score the triad: contextual_relevancy, faithfulness, answer_relevancy.

    Args:
        rows: list of rag_qa dataset rows.
        limit: maximum number of rows to process (None = all).
        offset: number of rows to skip from the start.
        resume: if True, seed completed rows from *output_dir* only.
        output_dir: results directory (the run's isolated output dir).

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

    results_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "evals" / "results"
    results_rows: list[dict] = []
    cr_scores: list[float] = []
    fth_scores: list[float] = []
    ar_scores: list[float] = []
    timings: list[float] = []
    errors: list[Exception] = []
    tavily_calls_total = 0
    thread_ids: list[str] = []
    status_counts: dict[str, int] = {"ok": 0, "tavily_quota": 0, "groq_quota": 0, "error": 0}

    config = load_config()

    # --resume: seed completed rows from the *selected* output directory only,
    # and only when the prior report's provenance matches this dataset/model.
    # Stale or duplicate rows are refused rather than imported (audit item 1).
    resume_issues: list[str] = []
    completed_queries: set[str] = set()
    if resume:
        prior_rows, completed_queries, resume_issues = seed_prior_ok(
            results_dir,
            "ragqa_tierb",
            "query",
            expected_dataset_sha=dataset_sha("ragqa"),
            expected_judge_model=config.get("judges", {}).get("model"),
        )
        for issue in resume_issues:
            print(f"    [resume] {issue}", file=sys.stderr)
        selected_queries = {row.get("query", "") for row in sliced_rows}
        prior_rows = [row for row in prior_rows if row.get("query") in selected_queries]
        for prev in prior_rows:
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

    thresholds_config = config.get("thresholds", {})
    cr_threshold = thresholds_config.get("contextual_relevancy", 0.80)
    fth_threshold = thresholds_config.get("faithfulness", 0.90)
    ar_threshold = thresholds_config.get("answer_relevancy", 0.90)

    retriever_config = config.get("retriever", {})
    k = retriever_config.get("k", 5)
    retry_config = retriever_config.get("retry", {}) or {}
    retry_attempts = int(retry_config.get("attempts", 4))
    retry_base_delay = float(retry_config.get("base_delay_s", 1.5))

    token_tracker = JudgeTokenTracker()

    start_time = time.time()

    # Hoisted imports
    import shutil
    import uuid

    from langchain_core.messages import HumanMessage

    # Judge error classification for quota handling. Judge functions
    # return None on exception (never raise), so the recorded error is
    # inspected after each row's judge calls to detect provider 429s.
    import evals.harness.llm_judge as llm_judge_mod
    from backend.rag.ingest import get_retriever, ingest_pdf
    from evals.harness.retrieval import SOURCE_DOCS

    # Fresh per-run namespace: one per-run scratch corpus index plus a fresh
    # in-memory checkpointer assembled per row (see step 2 below). Fixed ids or
    # a shared checkpointer store let a re-run inherit history from a previous
    # run, contaminating the triad (audit P2 #7 / item 1).
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    index_thread_id = f"tierb-rag-{run_id}"
    thread_ids.append(index_thread_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 0: ingest the golden corpus PDFs (distractors included) ---
    # Every corpus doc lands in the same per-run index, so each query must pick
    # the right chunk among distractors instead of being handed its own
    # reference text (audit item 6).
    corpus_dir = PROJECT_ROOT / "evals" / "datasets" / "corpus"
    corpus_stats: list[dict] = []
    embedding_retries = 0
    with _retrying_embeddings(retry_attempts, retry_base_delay) as embedding_client:
        for source_doc in SOURCE_DOCS:
            pdf_path = corpus_dir / f"{source_doc}.pdf"
            print(f"    Ingesting corpus PDF {pdf_path.name}...", file=sys.stderr)
            corpus_stats.append(
                bounded_call(ingest_pdf,
                    str(pdf_path),
                    index_thread_id,
                    source_name=f"{source_doc}.pdf",
                )
            )
        embedding_retries = embedding_client.retries

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

        row_start = time.time()
        row_status = "ok"
        tavily_calls = 0

        try:
            # --- Step 1: retrieve from the corpus index (retry on 503s) ---
            context_docs: list[str] = []
            with _retrying_embeddings(retry_attempts, retry_base_delay) as retriever_embeddings:
                retriever = get_retriever(index_thread_id)
                if retriever:
                    retriever.search_kwargs = {"k": k}
                    retrieved = bounded_call(retriever.invoke, query)
                    context_docs = [d.page_content.strip() for d in retrieved]
                embedding_retries += retriever_embeddings.retries

            # --- Step 2: run one full graph turn against a fresh checkpointer.
            # A per-row InMemorySaver means a shared corpus thread_id cannot
            # leak conversation history between rows. Bounded like every other
            # graph call so a stuck turn cannot block the job (re-audit P1).
            from evals.harness.tierb_runners import (
                GRAPH_TIMEOUT_S as _GRAPH_TIMEOUT_S,
            )
            from evals.harness.tierb_runners import (
                JUDGE_TIMEOUT_S as _JUDGE_TIMEOUT_S,
            )
            from evals.harness.tierb_runners import _call_with_timeout as _bounded_call

            graph = _build_eval_graph()
            graph_config = {"configurable": {"thread_id": index_thread_id}}

            graph_start = time.time()
            result, graph_timed_out = _bounded_call(
                graph.invoke,
                {
                    "messages": [HumanMessage(content=query)],
                    "documents": context_docs,
                },
                config=graph_config,
                timeout_s=_GRAPH_TIMEOUT_S,
                thread_name="tierb-graph-invoke",
            )
            if graph_timed_out:
                raise TimeoutError(
                    f"graph invoke exceeded {_GRAPH_TIMEOUT_S:.0f}s bound"
                )
            result = result or {}
            graph_time = time.time() - graph_start

            # --- Step 3: Extract state fields ---
            final_response = result.get("final_response") or ""
            documents: list[str] = result.get("documents") or []
            route = result.get("route") or "unknown"

            all_messages = result.get("messages") or []
            tavily_calls = _count_tavily_calls(all_messages)
            tavily_calls_total += tavily_calls

            # --- Step 4: Score the triad (bounded judge calls) ---
            context_text = "\n\n".join(documents) if documents else ""

            contextual_relevancy: float | None = None
            faithfulness: float | None = None
            answer_relevancy: float | None = None
            judge_timed_out = False

            if context_text:
                contextual_relevancy, _cr_timed_out = _bounded_call(
                    judge_contextual_relevancy,
                    query,
                    context_text,
                    token_tracker=token_tracker,
                    timeout_s=_JUDGE_TIMEOUT_S,
                    thread_name="tierb-judge-invoke",
                )
                judge_timed_out = judge_timed_out or _cr_timed_out

            faithfulness, _fth_timed_out = _bounded_call(
                judge_faithfulness,
                query,
                context_text,
                final_response,
                token_tracker=token_tracker,
                timeout_s=_JUDGE_TIMEOUT_S,
                thread_name="tierb-judge-invoke",
            )
            judge_timed_out = judge_timed_out or _fth_timed_out
            answer_relevancy, _ar_timed_out = _bounded_call(
                judge_answer_relevancy,
                query,
                final_response,
                token_tracker=token_tracker,
                timeout_s=_JUDGE_TIMEOUT_S,
                thread_name="tierb-judge-invoke",
            )
            judge_timed_out = judge_timed_out or _ar_timed_out
            if judge_timed_out:
                llm_judge_mod.last_judge_error = (
                    f"TimeoutError: RAG-QA judge exceeded {_JUDGE_TIMEOUT_S:.0f}s bound"
                )

            # Failed/absent judge scores count as 0.0 rather than vanishing
            # from the mean, so a partially-failed run cannot inflate its
            # triad scores (audit P2 #6).
            cr_scores.append(
                float(contextual_relevancy) if contextual_relevancy is not None else 0.0
            )
            fth_scores.append(float(faithfulness) if faithfulness is not None else 0.0)
            ar_scores.append(float(answer_relevancy) if answer_relevancy is not None else 0.0)

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
            cr_scores.append(0.0)
            fth_scores.append(0.0)
            ar_scores.append(0.0)
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
        # Any required measurement that is missing fails the row: non-quota
        # judge failures used to leave the row marked ``ok`` with the score
        # simply absent from the aggregates (re-audit P2). Groq 429 stops the
        # run; any other missing score marks the row ``error`` so resume
        # retries it and completeness checks fail it.
        if row_status == "ok" and (
            contextual_relevancy is None
            or faithfulness is None
            or answer_relevancy is None
        ):
            if _classify_judge_failure(llm_judge_mod.last_judge_error) == "groq_quota":
                status_counts["groq_quota"] += 1
                results_rows[-1]["status"] = "groq_quota"
                row_status = "groq_quota"
                print(
                    "    Groq rate limit hit (429) during judging - stopping run. "
                    "Use --resume to continue later.",
                    file=sys.stderr,
                )
                break
            status_counts["error"] += 1
            row_status = "error"
            results_rows[-1]["status"] = "error"
            results_rows[-1]["error"] = (
                f"judge failure (missing score): {llm_judge_mod.last_judge_error or 'unknown'}"
            )
            errors.append(
                RuntimeError(
                    f"judge failure on query {query[:80]!r}: "
                    f"{llm_judge_mod.last_judge_error or 'unknown'}"
                )
            )

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

    n_judge_failures = sum(
        1 for r in results_rows if r.get("status") == "error" and "judge failure" in str(r.get("error", ""))
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
            "n_judge_failures": n_judge_failures,
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
            "run_id": run_id,
            "index_thread_id": index_thread_id,
            "results_dir": str(results_dir),
            "corpus_docs": [stat.get("source") for stat in corpus_stats],
            "corpus_chunks": sum(int(stat.get("chunks") or 0) for stat in corpus_stats),
            "embedding_retries": embedding_retries,
            "expected_rows": len(sliced_rows),
            "resume_issues": resume_issues,
            **operational,
        },
    }

    # Remove this run's scratch FAISS index (best-effort; Windows may hold the
    # files open and must not fail the run).
    import backend.rag.ingest as ingest_mod
    for tid in thread_ids:
        ingest_mod._RETRIEVERS.pop(tid, None)
        index_dir = ingest_mod._index_dir(tid)
        if index_dir.exists():
            try:
                shutil.rmtree(index_dir)
            except (PermissionError, OSError):
                pass

    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Stub subcommands
# ---------------------------------------------------------------------------

STUB_COMMANDS: set[str] = set()

# High-level plans printed by the top-level `--dry-run` (no subcommand given).
# Static text only — no dataset loading, no LLM, no network.
DRY_RUN_PLANS: tuple[tuple[str, str], ...] = (
    ("router", "score evals/datasets/router.jsonl via _route_for_message (accuracy, F1, confusion matrix)"),
    ("retriever", "embed each source_doc corpus PDF, retrieve top-k, score recall@k / precision@k"),
    ("generator", "judge evals/datasets/generator.jsonl with faithfulness + answer relevancy"),
    ("rag-qa", "ingest reference text, run one full graph turn, score the RAG triad"),
    ("application", "run real graph turns on evals/datasets/application.jsonl + G-Eval scoring"),
    ("safety", "drive evals/datasets/safety.jsonl through the real graph and run the safety gates"),
    ("memory", "LTM fact extraction + cross-thread round-trip over evals/datasets/memory.jsonl"),
    ("blog", "blog writer graph turns + structure check + G-Eval over evals/datasets/blog.jsonl"),
    ("baseline", "aggregate evals/results/*_tierb.json into baseline.json (no LLM calls)"),
)


def _expected_row_count(rows: list[dict], args: argparse.Namespace) -> int:
    """Number of dataset rows this invocation was asked to process."""
    sliced = rows[args.offset:]
    if args.limit is not None:
        sliced = sliced[: args.limit]
    return len(sliced)


def _absolute_gate_failures(results: dict[str, Any]) -> list[str]:
    """Names of ``results['thresholds']`` entries with ``passed is False``.

    Informational metrics (``passed is None``) never fail the gate. This is
    the comparison-independent absolute check so a complete run that misses
    its quality bar fails even when no ``--compare`` baseline was supplied
    (re-audit P1: ``_finish_run`` with ``passed=False`` and no compare path
    returned 0).
    """
    failures: list[str] = []
    thresholds = results.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        return failures
    for metric, check in thresholds.items():
        if isinstance(check, dict) and check.get("passed") is False:
            failures.append(str(metric))
    return sorted(failures)


def _stamp_requested_slice(results: dict, rows: list[dict], args: argparse.Namespace) -> None:
    requested = rows[args.offset:]
    if args.limit is not None:
        requested = requested[:args.limit]
    operational = results.setdefault("operational", {})
    operational["expected_rows"] = len(requested)
    operational["requested_keys"] = [row_identity(args.command, row) for row in requested]


def _finish_run(
    results: dict[str, Any],
    rows: list[dict],
    args: argparse.Namespace,
    compare_path: Path | None,
    output_dir: Path,
    *,
    enforce_absolute: bool,
) -> int:
    """Compare the run against the baseline and fail when it is incomplete.

    Returns 0 only when every requested row produced a usable measurement and
    the baseline/absolute gates passed. Missing or failed rows fail the
    command outright rather than being written as a clean-looking report
    (audit items 1 and 8).

    Absolute gates are enforced independently of whether a baseline was
    supplied: with ``enforce_absolute=True`` (the default) a failing
    ``thresholds[*].passed`` entry fails the command even when
    ``compare_path`` is None. ``--report-only`` downgrades those gates to
    informational warnings.
    """
    rc = 0
    if compare_path is not None:
        rc = compare_run_after_run(
            compare_path,
            output_dir,
            run_key_for_cli(args.command),
            enforce_absolute=enforce_absolute,
        )
    requested = rows[args.offset:]
    if args.limit is not None:
        requested = requested[:args.limit]
    issues = report_row_issues(
        {**results, "name": args.command}, _expected_row_count(rows, args),
        [row_identity(args.command, row) for row in requested],
    )
    if issues:
        print(
            f"\nERROR: {args.command} run is incomplete — refusing to present it "
            "as a valid result:",
            file=sys.stderr,
        )
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    if enforce_absolute:
        failures = _absolute_gate_failures(results)
        if failures:
            print(
                f"\nERROR: {args.command} run failed absolute quality gate(s): "
                + ", ".join(failures)
                + ". Pass --report-only to treat absolute gates as informational.",
                file=sys.stderr,
            )
            return 1
    else:
        failures = _absolute_gate_failures(results)
        if failures:
            print(
                f"\n[NOTE] {args.command} absolute gate(s) not met "
                f"(report-only): " + ", ".join(failures),
                file=sys.stderr,
            )
    return rc


def print_root_dry_run() -> None:
    """Print a high-level plan for every subcommand without running anything."""
    print("=== DRY RUN — No LLM calls will be made ===")
    print()
    print("No subcommand given — showing the plan for every available subcommand:")
    print()
    for name, plan in DRY_RUN_PLANS:
        print(f"  {name:<12} {plan}")
    print()
    print("Add a subcommand for a dataset-sized plan (e.g. `router --dry-run`),")
    print("or drop --dry-run to execute for real.")


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
        default=argparse.SUPPRESS,
        help="Print plan without making LLM calls",
    )
    parent.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Do not fail --compare on absolute quality gates, only on "
            "regressions/incompleteness (default: absolute gates are enforced)"
        ),
    )
    parent.add_argument(
        "--calibrate-reference",
        action="store_true",
        help=(
            "generator: also judge the golden reference and report it under "
            "'calibration' (judge calibration only, never gated)"
        ),
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

    # Top-level --dry-run so `run_offline.py --dry-run` (no subcommand) works.
    # SUPPRESS mirrors --compare: an explicit subcommand still supplies its own
    # default, and the bare form is read with getattr(args, "dry_run", False).
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print a high-level plan for every subcommand without running them",
    )
    # Mirrors the subcommand-level flag so `--compare ... ` (no subcommand)
    # can opt out of enforcing absolute gates for the transition period.
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Do not fail --compare on absolute quality gates",
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

    # With the top-level --compare and no subcommand, restrict the comparison to
    # a single run's metrics (e.g. `--compare baseline.json --compare-run router`).
    parser.add_argument(
        "--compare-run",
        default=None,
        help="With --compare and no subcommand, compare only this subcommand's run",
    )

    parser.set_defaults(dry_run=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    compare_path = getattr(args, "compare", None)
    # Absolute quality gates are enforced by default; --report-only downgrades
    # them to informational warnings for the baseline-refresh transition
    # (audit item 8).
    enforce_absolute = not getattr(args, "report_only", False)

    # If no command given, show help
    if args.command is None:
        if getattr(args, "dry_run", False):
            print_root_dry_run()
            return 0
        if compare_path is not None:
            # Compare against the results that sit next to the named baseline.
            # Defaulting to evals/results/ instead would silently compare a
            # different directory than the caller asked for (audit item 1).
            out_dir = getattr(args, "out", None) or Path(compare_path).parent
            if not str(out_dir):
                out_dir = PROJECT_ROOT / "evals" / "results"
            compare_run = getattr(args, "compare_run", None)
            if compare_run:
                return compare_run_after_run(
                    compare_path,
                    out_dir,
                    run_key_for_cli(compare_run),
                    enforce_absolute=enforce_absolute,
                )
            return run_compare_only(
                compare_path, out_dir, enforce_absolute=enforce_absolute
            )
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

    # Run the evaluation. ``output_dir`` is resolved once here and threaded
    # into every runner so reports are written to — and --resume rows are read
    # from — exactly the directory the caller selected (audit item 1).
    output_dir = getattr(args, "out", None) or PROJECT_ROOT / "evals" / "results"
    print(f"\n{'='*60}")
    print(f"  AgentFlow Eval: {args.command}")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir}")
    measurement_provenance = build_provenance(args.command) if args.command != "baseline" else None

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
        _stamp_requested_slice(results, rows, args)
        json_path, md_path = write_report(
            name="router_tierb",
            results=results,
            output_dir=output_dir,
            provenance=measurement_provenance,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return _finish_run(
            results,
            rows,
            args,
            compare_path,
            output_dir,
            enforce_absolute=enforce_absolute,
        )

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
        _stamp_requested_slice(results, rows, args)
        json_path, md_path = write_report(
            name="retriever_tierb",
            results=results,
            output_dir=output_dir,
            provenance=measurement_provenance,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return _finish_run(
            results,
            rows,
            args,
            compare_path,
            output_dir,
            enforce_absolute=enforce_absolute,
        )

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
            calibrate_reference=args.calibrate_reference,
        )

        if results.get("status") == "error":
            print(f"ERROR: {results.get('error')}", file=sys.stderr)
            return 1

        # Write reports
        _stamp_requested_slice(results, rows, args)
        json_path, md_path = write_report(
            name="generator_tierb",
            results=results,
            output_dir=output_dir,
            provenance=measurement_provenance,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Check for errors
        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        return _finish_run(
            results,
            rows,
            args,
            compare_path,
            output_dir,
            enforce_absolute=enforce_absolute,
        )

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
            output_dir=output_dir,
        )

        # Write reports
        _stamp_requested_slice(results, rows, args)
        json_path, md_path = write_report(
            name="ragqa_tierb",
            results=results,
            output_dir=output_dir,
            provenance=measurement_provenance,
        )

        print("\nReports written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Safety-net cleanup: the runner already removes its own per-run
        # scratch index (checkpoints are per-row in memory). This sweeps any
        # scratch DBs left behind by runs from before the in-memory switch.
        print("\nCleaning up scratch resources...", file=sys.stderr)
        results_dir = output_dir
        for scratch_db in results_dir.glob("tierb_rag_*.db"):
            try:
                scratch_db.unlink()
                print(f"  Removed scratch DB: {scratch_db}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"  Could not remove scratch DB: {e}", file=sys.stderr)
        legacy_db = results_dir / "tierb_rag.db"
        if legacy_db.exists():
            try:
                legacy_db.unlink()
                print(f"  Removed scratch DB: {legacy_db}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"  Could not remove scratch DB: {e}", file=sys.stderr)

        # Legacy candidate-name sweep; per-run ids are cleaned in the runner.
        from evals.harness.retrieval import cleanup_tierb_indexes
        owned_index = results.get("operational", {}).get("index_thread_id")
        n_removed = cleanup_tierb_indexes(owned_index) if owned_index else 0
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

        return _finish_run(
            results,
            rows,
            args,
            compare_path,
            output_dir,
            enforce_absolute=enforce_absolute,
        )

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
            output_dir=output_dir,
        )

        _stamp_requested_slice(results, rows, args)
        json_path, md_path = write_tierb_report(
            name=f"{args.command}_tierb",
            results=results,
            output_dir=output_dir,
            provenance=measurement_provenance,
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
            rc = compare_run_after_run(
                compare_path,
                output_dir,
                run_key_for_cli(args.command),
                enforce_absolute=enforce_absolute,
            )
            if rc != 0:
                return rc

        if results.get("operational", {}).get("error_count", 0) > 0:
            print("\n[NOTE] Errors occurred during evaluation. "
                  "Check the report for details.", file=sys.stderr)

        # NOTE: no second comparison here — _finish_run performs the single
        # gated comparison with the caller's report-only setting, so
        # ``--compare --report-only`` cannot fail on absolute gates via a
        # redundant default-enforcement pass (re-audit P2).
        return _finish_run(
            results,
            rows,
            args,
            None,
            output_dir,
            enforce_absolute=enforce_absolute,
        )

    elif args.command == "baseline":
        try:
            baseline = build_baseline(output_dir)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        out_path = output_dir / "baseline.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(baseline, handle, indent=2, ensure_ascii=False)
        print(f"Baseline written: {out_path}")
        print(f"  Runs aggregated: {', '.join(sorted(baseline['runs'].keys()))}")
        print(f"  Metrics captured: {len(baseline['metrics'])}")
        if compare_path is not None:
            rc = run_compare_only(
                compare_path, output_dir, enforce_absolute=enforce_absolute
            )
            if rc != 0:
                return rc
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
