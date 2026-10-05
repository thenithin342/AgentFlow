#!/usr/bin/env python
"""
Tier B offline evaluation CLI.

Usage:
    python evals/harness/run_offline.py <subcommand> [options]

Subcommands:
    router      Run router accuracy evaluation (fully implemented this session)
    retriever   Stub — not implemented yet
    generator   Stub — not implemented yet
    rag-qa      Stub — not implemented yet
    application Stub — not implemented yet
    safety      Stub — not implemented yet
    blog        Stub — not implemented yet
    memory      Stub — not implemented yet

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

from evals.config import load_config
from evals.harness.metrics import (
    confusion_matrix,
    failing_rows,
    per_class_f1,
    router_accuracy,
)
from evals.harness.report import print_summary, write_report

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
# Stub subcommands
# ---------------------------------------------------------------------------

STUB_COMMANDS = {
    "rag-qa",
    "application",
    "safety",
    "blog",
    "memory",
}


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
        help="Resume from previous run (not yet implemented)",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # If no command given, show help
    if args.command is None:
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
