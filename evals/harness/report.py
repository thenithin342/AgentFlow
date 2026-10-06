"""
Report writer for Tier B batch harness results.

Writes JSON + Markdown sibling reports to evals/results/.
Provides comparison between baseline and current runs.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default output directory for reports
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _get_git_sha() -> str:
    """Get the current git commit SHA, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _format_threshold_check(name: str, value: float, threshold: float) -> dict:
    """Format a single threshold check result."""
    return {
        "metric": name,
        "value": value,
        "threshold": threshold,
        "passed": value >= threshold,
    }


def _format_latency(latency: dict) -> dict:
    """Ensure latency dict has expected keys."""
    if not latency:
        return {}
    return {
        "mean": latency.get("mean", 0.0),
        "p95": latency.get("p95", 0.0),
        "min": latency.get("min", 0.0),
        "max": latency.get("max", 0.0),
    }


def write_report(
    name: str,
    results: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write JSON + Markdown sibling reports.

    Args:
        name: run name (e.g. "router", "retriever").
        results: results dict with the schema described below.
        output_dir: directory to write reports to (defaults to evals/results/).

    Returns:
        Tuple of (json_path, md_path).

    Report JSON schema:
    {
        "name": "...",
        "timestamp": "...",
        "git_sha": "...",
        "judge_model": "...",
        "rows": [...],
        "aggregate": {...},
        "thresholds": {key: {value, threshold, passed}},
        "operational": {p95, mean_latency, tokens, success_rate, error_rate}
    }
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _now_iso()
    git_sha = _get_git_sha()

    # Build the full report object
    report: dict[str, Any] = {
        "name": name,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "judge_model": results.get("judge_model", "unknown"),
        "rows": results.get("rows", []),
        "aggregate": results.get("aggregate", {}),
        "thresholds": results.get("thresholds", {}),
        "operational": results.get("operational", {}),
    }
    for extra_key in ("per_class", "confusion_matrix", "failures"):
        if extra_key in results:
            report[extra_key] = results[extra_key]

    # Write JSON
    json_path = output_dir / f"{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Write Markdown
    md_path = output_dir / f"{name}.md"
    _write_markdown(md_path, report)

    return json_path, md_path


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a Markdown summary report."""
    lines: list[str] = []

    lines.append(f"# {report['name'].title()} Evaluation Report")
    lines.append("")
    lines.append(f"**Timestamp:** {report['timestamp']}")
    lines.append(f"**Git SHA:** `{report['git_sha'][:8]}`")
    lines.append(f"**Judge Model:** {report['judge_model']}")
    lines.append("")

    # Aggregate metrics
    aggregate = report.get("aggregate", {})
    if aggregate:
        lines.append("## Aggregate Metrics")
        lines.append("")
        for key, value in aggregate.items():
            if isinstance(value, float):
                lines.append(f"- **{key}:** {value:.4f}")
            else:
                lines.append(f"- **{key}:** {value}")
        lines.append("")

    # Threshold checks
    thresholds = report.get("thresholds", {})
    if thresholds:
        lines.append("## Threshold Checks")
        lines.append("")
        all_passed = True
        for name, check in thresholds.items():
            status = "PASS" if check.get("passed") else "FAIL"
            if not check.get("passed"):
                all_passed = False
            lines.append(f"- [{status}] **{name}** = {check.get('value', 'N/A'):.4f} (threshold: {check.get('threshold', 'N/A')})")
        lines.append("")
        if all_passed:
            lines.append("**All thresholds passed.**")
        else:
            lines.append("**Some thresholds failed.**")
        lines.append("")

    # Per-class metrics (for router)
    rows = report.get("rows", [])
    per_class = report.get("per_class")
    if not per_class and rows:
        for r in rows:
            if "per_class" in r:
                per_class = r["per_class"]
                break
    if per_class:
        lines.append("## Per-Class F1 Scores")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|-----|---------|")
        for label, metrics in sorted(per_class.items()):
            lines.append(
                f"| {label} | {metrics.get('precision', 0):.3f} | "
                f"{metrics.get('recall', 0):.3f} | "
                f"{metrics.get('f1', 0):.3f} | "
                f"{metrics.get('support', 0)} |"
            )
        lines.append("")

    # Confusion matrix (for router)
    cm = report.get("confusion_matrix")
    if not cm and rows:
        for r in rows:
            if "confusion_matrix" in r:
                cm = r["confusion_matrix"]
                break
    if cm:
        lines.append("## Confusion Matrix")
        lines.append("")
        labels = sorted(cm.keys())
        lines.append("| Actual | " + " | ".join(labels) + " |")
        lines.append("|-------|" + "|".join(["---"] * len(labels)) + "|")
        for actual in labels:
            counts = [str(cm[actual].get(pred, 0)) for pred in labels]
            lines.append(f"| {actual} | " + " | ".join(counts) + " |")
        lines.append("")

    # Failing rows (for router)
    failing = report.get("failures")
    if not failing and rows:
        for r in rows:
            if r.get("failures"):
                failing = r["failures"]
                break
    if failing:
        lines.append(f"## Failing Rows ({len(failing)})")
        lines.append("")
        for i, fail in enumerate(failing, 1):
            lines.append(f"{i}. **Query:** {fail.get('query', '')[:80]}")
            lines.append(f"   - Expected: `{fail.get('expected', '')}`")
            lines.append(f"   - Predicted: `{fail.get('predicted', '')}`")
            if fail.get("reason"):
                lines.append(f"   - Reason: {fail['reason'][:100]}")
            lines.append("")
        lines.append("")

    # Operational metrics
    operational = report.get("operational", {})
    if operational:
        lines.append("## Operational Metrics")
        lines.append("")
        if "latency" in operational:
            lat = operational["latency"]
            lines.append(f"- **Mean latency:** {lat.get('mean', 0):.3f}s")
            lines.append(f"- **P95 latency:** {lat.get('p95', 0):.3f}s")
            lines.append(f"- **Min latency:** {lat.get('min', 0):.3f}s")
            lines.append(f"- **Max latency:** {lat.get('max', 0):.3f}s")
        if "tokens" in operational:
            tokens = operational["tokens"]
            lines.append(f"- **Input tokens:** {tokens.get('input', 0):,}")
            lines.append(f"- **Output tokens:** {tokens.get('output', 0):,}")
            lines.append(f"- **Total tokens:** {tokens.get('total', 0):,}")
            lines.append(f"- **LLM calls:** {tokens.get('calls', 0)}")
        if "success_rate" in operational:
            lines.append(f"- **Success rate:** {operational['success_rate']:.2%}")
        if "error_rate" in operational:
            lines.append(f"- **Error rate:** {operational['error_rate']:.2%}")
        if "error_count" in operational:
            lines.append(f"- **Error count:** {operational['error_count']}")
        lines.append("")

    # Raw rows (for debugging)
    if rows:
        lines.append("## Raw Results")
        lines.append("")
        sample = rows[0]
        if "recall" in sample and "precision" in sample:
            lines.append("| # | Query | Recall | Precision | Status |")
            lines.append("|---|-------|--------|-----------|--------|")
            for i, row in enumerate(rows, 1):
                query = str(row.get("query", row.get("input", "")))[:50]
                rec = f"{row.get('recall', 0.0):.2f}"
                prec = f"{row.get('precision', 0.0):.2f}"
                passed = row.get("recall_passed", False) and row.get("precision_passed", False)
                status = "PASS" if passed else "FAIL"
                lines.append(f"| {i} | {query} | {rec} | {prec} | {status} |")
        elif (
            "contextual_relevancy" in sample
            and "faithfulness" in sample
            and "answer_relevancy" in sample
        ):
            lines.append("| # | Query | Route | Ctx Rel | Faithfulness | Ans Rel | Status |")
            lines.append("|---|-------|-------|---------|--------------|---------|--------|")
            for i, row in enumerate(rows, 1):
                query = str(row.get("query", row.get("input", "")))[:40]
                route = row.get("route", "")
                cr_val = f"{row.get('contextual_relevancy', 0.0):.2f}" if row.get("contextual_relevancy") is not None else "N/A"
                f_val = f"{row.get('faithfulness', 0.0):.2f}" if row.get("faithfulness") is not None else "N/A"
                ar_val = f"{row.get('answer_relevancy', 0.0):.2f}" if row.get("answer_relevancy") is not None else "N/A"
                passed = (
                    row.get("contextual_relevancy_passed", False)
                    and row.get("faithfulness_passed", False)
                    and row.get("answer_relevancy_passed", False)
                )
                status = "PASS" if passed else "FAIL"
                if row.get("status") in ("groq_quota", "tavily_quota", "error"):
                    status = row.get("status").upper()
                lines.append(f"| {i} | {query} | {route} | {cr_val} | {f_val} | {ar_val} | {status} |")
        elif "faithfulness" in sample and "answer_relevancy" in sample:
            lines.append("| # | Query | Faithfulness | Relevancy | Status |")
            lines.append("|---|-------|--------------|-----------|--------|")
            for i, row in enumerate(rows, 1):
                query = str(row.get("query", row.get("input", "")))[:50]
                f_val = f"{row.get('faithfulness', 0.0):.2f}" if row.get("faithfulness") is not None else "N/A"
                r_val = f"{row.get('answer_relevancy', 0.0):.2f}" if row.get("answer_relevancy") is not None else "N/A"
                passed = row.get("faithfulness_passed", False) and row.get("answer_relevancy_passed", False)
                status = "PASS" if passed else "FAIL"
                lines.append(f"| {i} | {query} | {f_val} | {r_val} | {status} |")
        else:
            lines.append("| # | Query | Expected | Predicted | Correct |")
            lines.append("|---|-------|----------|-----------|---------|")
            for i, row in enumerate(rows, 1):
                query = str(row.get("query", row.get("input", "")))[:50]
                expected = row.get("expected", "")
                predicted = row.get("predicted", "")
                correct = "Y" if expected == predicted else "N"
                lines.append(f"| {i} | {query} | {expected} | {predicted} | {correct} |")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by AgentFlow eval harness*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compare_reports(baseline: Path, current: Path) -> dict[str, Any]:
    """Compare two report JSON files and return delta summary.

    Args:
        baseline: path to baseline report JSON.
        current: path to current report JSON.

    Returns:
        Dict with per-metric deltas and regression flags.
    """
    if not baseline.exists() or not current.exists():
        return {"error": "One or both report files do not exist"}

    with open(baseline, encoding="utf-8") as f:
        base = json.load(f)
    with open(current, encoding="utf-8") as f:
        curr = json.load(f)

    deltas: dict[str, Any] = {
        "baseline_file": str(baseline),
        "current_file": str(current),
        "baseline_timestamp": base.get("timestamp", ""),
        "current_timestamp": curr.get("timestamp", ""),
        "metric_deltas": {},
        "regressions": [],
    }

    # Compare aggregate metrics
    base_agg = base.get("aggregate", {})
    curr_agg = curr.get("aggregate", {})

    for key in set(base_agg.keys()) | set(curr_agg.keys()):
        base_val = base_agg.get(key)
        curr_val = curr_agg.get(key)
        if isinstance(base_val, (int, float)) and isinstance(curr_val, (int, float)):
            delta = curr_val - base_val
            deltas["metric_deltas"][key] = {
                "baseline": base_val,
                "current": curr_val,
                "delta": delta,
                "regression": delta < 0,  # lower is worse for most metrics
            }
            if delta < 0:
                deltas["regressions"].append(f"{key}: {base_val:.4f} → {curr_val:.4f} ({delta:+.4f})")

    # Compare threshold checks
    base_thresh = base.get("thresholds", {})
    curr_thresh = curr.get("thresholds", {})

    for key in set(base_thresh.keys()) | set(curr_thresh.keys()):
        base_check = base_thresh.get(key, {})
        curr_check = curr_thresh.get(key, {})
        base_passed = base_check.get("passed", False)
        curr_passed = curr_check.get("passed", False)
        if base_passed and not curr_passed:
            deltas["regressions"].append(f"Threshold regression: {key} now failing")

    return deltas


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    name = results.get("name", "unknown")
    print(f"\n{'='*60}")
    print(f"  {name.title()} Evaluation Summary")
    print(f"{'='*60}")

    aggregate = results.get("aggregate", {})
    if aggregate:
        print("\n  Aggregate Metrics:")
        for key, value in aggregate.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    thresholds = results.get("thresholds", {})
    if thresholds:
        print("\n  Threshold Checks:")
        all_passed = True
        for metric, check in sorted(thresholds.items()):
            status = "PASS" if check.get("passed") else "FAIL"
            if not check.get("passed"):
                all_passed = False
            print(f"    [{status}] {metric}: {check.get('value', 'N/A'):.4f} (threshold: {check.get('threshold', 'N/A')})")
        print()
        if all_passed:
            print("  [PASS] All thresholds passed")
        else:
            print("  [FAIL] Some thresholds failed")

    operational = results.get("operational", {})
    if operational:
        print("\n  Operational Metrics:")
        if "latency" in operational:
            lat = operational["latency"]
            print(f"    Mean latency: {lat.get('mean', 0):.3f}s")
            print(f"    P95 latency: {lat.get('p95', 0):.3f}s")
        if "tokens" in operational:
            tokens = operational["tokens"]
            print(f"    Input tokens: {tokens.get('input', 0):,}")
            print(f"    Output tokens: {tokens.get('output', 0):,}")
            print(f"    Total tokens: {tokens.get('total', 0):,}")
            print(f"    LLM calls: {tokens.get('calls', 0)}")
        if "success_rate" in operational:
            print(f"    Success rate: {operational['success_rate']:.2%}")
        if "error_count" in operational:
            print(f"    Errors: {operational['error_count']}")

    # Failing rows
    rows = results.get("rows", [])
    for row in rows:
        if row.get("failures"):
            print(f"\n  Failing Rows: {len(row['failures'])}")
            for fail in row["failures"][:5]:  # Show first 5
                print(f"    - '{fail.get('query', '')[:60]}...'")
                print(f"      Expected: {fail.get('expected')} | Predicted: {fail.get('predicted')}")
            if len(row["failures"]) > 5:
                print(f"    ... and {len(row['failures']) - 5} more")
            break

    print(f"\n{'='*60}\n")
