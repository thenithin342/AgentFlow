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
# Stub subcommands
# ---------------------------------------------------------------------------

STUB_COMMANDS = {
    "retriever",
    "generator",
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
