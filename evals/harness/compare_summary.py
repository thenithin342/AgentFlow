#!/usr/bin/env python3
"""Publish a Tier B baseline-comparison report to the GitHub Actions Job Summary.

Reads the JSON emitted by ``run_offline.py --compare`` (``compare.json`` or a
per-subcommand ``compare_<run>.json``), renders a Markdown report, appends it to
``$GITHUB_STEP_SUMMARY``, emits ``::error::`` annotations for every regression,
and mirrors the report to ``evals/results/compare_summary.md`` so the artifact
bundle carries a readable copy. Exits non-zero when the comparison reports a
regression, so it can double as a local gate.

Usage:
    python evals/harness/compare_summary.py --subcommand all
    python evals/harness/compare_summary.py --subcommand router
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.harness.tierb_runners import (  # noqa: E402
    _git_sha,
    emit_regression_annotations,
    render_compare_markdown,
    run_key_for_cli,
    write_step_summary,
)


def locate_compare_file(results_dir: Path, subcommand: str) -> Path | None:
    """Return the comparison JSON for *subcommand*, else ``compare.json``.

    Single-subcommand runs write ``compare_<run_key>.json`` (e.g. router ->
    ``compare_router.json``, rag-qa -> ``compare_ragqa.json``); full runs write
    ``compare.json``.
    """
    candidates: list[Path] = []
    if subcommand and subcommand != "all":
        candidates.append(results_dir / f"compare_{run_key_for_cli(subcommand)}.json")
        candidates.append(results_dir / f"compare_{subcommand}.json")
    candidates.append(results_dir / "compare.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_summary.py",
        description="Render a Tier B baseline comparison into the Actions Job Summary",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "evals" / "results",
        help="Directory holding compare*.json (default: evals/results)",
    )
    parser.add_argument(
        "--subcommand",
        default="all",
        help="Subcommand whose comparison to publish (default: all)",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Git SHA for the header (default: $GITHUB_SHA, then git rev-parse)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Explicit step-summary file to append to (default: $GITHUB_STEP_SUMMARY)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)

    compare_path = locate_compare_file(results_dir, args.subcommand)
    if compare_path is None:
        print(
            f"::error::No comparison file found in {results_dir}; "
            "failing the regression gate.",
            file=sys.stderr,
        )
        return 1

    with open(compare_path, encoding="utf-8") as handle:
        compare = json.load(handle)

    git_sha = args.git_sha or os.environ.get("GITHUB_SHA") or _git_sha()
    markdown = render_compare_markdown(compare, git_sha=git_sha)

    # Mirror to a Markdown artifact so a failing run is readable without JSON.
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "compare_summary.md").write_text(markdown, encoding="utf-8")

    if not write_step_summary(markdown, args.summary_file):
        # Outside CI (no $GITHUB_STEP_SUMMARY): print so local runs still see it.
        try:
            print(markdown)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            print(markdown.encode(encoding, errors="replace").decode(encoding, errors="replace"))

    emit_regression_annotations(compare)

    print(f"Comparison source: {compare_path}")
    if not compare.get("metric_deltas"):
        print("::error::No comparable metrics found in comparison report; failing gate.", file=sys.stderr)
        return 1
    if compare.get("clean"):
        print("[PASS] No regressions vs baseline.")
        return 0
    print(f"[FAIL] {len(compare.get('regressions', []))} regression(s) detected vs baseline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
