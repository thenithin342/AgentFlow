"""Feedback ↔ judge correlation.

Correlates human thumbs up/down feedback (the ``user_feedback`` table or a JSON
export) with offline judge verdicts, to test whether the LLM judges agree with
real users:

* **thumbs-up agreement** — share of up-votes whose route's offline judge
  verdict passed;
* **thumbs-down failure recall** — share of down-votes whose route's offline
  judge verdict failed (i.e. dissatisfaction the judges actually caught).

Live feedback carries ``thread_id``/``message_id``/``route`` but no dataset-row
key, so correlation is at **route granularity** against the Tier-B report for
that route (``evals/results/<dataset>.json``). Every input is a plain file so
the module runs fully offline and deterministically in Tier A.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

# Defensive: allow importing this module without the repo root on sys.path
# (e.g. `python evals/online/feedback_correlation.py` run directly).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.harness.tierb_runners import _FAILED_ROW_STATUSES  # noqa: E402

DEFAULT_RESULTS_DIR = HERE.parent / "results"

# Human/`route` value → Tier-B report stem in evals/results/.
DEFAULT_ROUTE_MAP: dict[str, str] = {
    "chat": "generator_tierb",
    "research": "ragqa_tierb",
    "analysis": "application_tierb",
    "blog": "blog_tierb",
}

# Per-row "passed" flags by report name — a row passes when every listed flag
# is present and truthy. Reports without flags fall back to none (unknown).
_ROUTE_PASS_FIELDS: dict[str, tuple[str, ...]] = {
    "generator_tierb": ("faithfulness_passed", "answer_relevancy_passed"),
    "ragqa_tierb": (
        "contextual_relevancy_passed",
        "faithfulness_passed",
        "answer_relevancy_passed",
    ),
    "application_tierb": ("passed",),
    "blog_tierb": ("passed",),
}


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_route_map(path: str | Path | None = None) -> dict[str, str]:
    """Load an optional ``{route: report_stem}`` override map."""
    if path is None:
        return dict(DEFAULT_ROUTE_MAP)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("route map must be a JSON object of {route: report_stem}")
    return {str(k): str(v) for k, v in data.items()}


def load_feedback_from_json(path: str | Path) -> list[dict[str, Any]]:
    """Load feedback records from a JSON export (list, or ``{feedback|rows|records}``)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("feedback", "rows", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("feedback JSON must be a list or contain a 'feedback'/'rows' list")
    return [dict(row) for row in data if isinstance(row, dict)]


def load_feedback_from_db(
    path: str | Path, *, table: str = "user_feedback"
) -> list[dict[str, Any]]:
    """Load feedback records from the SQLite ``user_feedback`` table."""
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"SELECT thread_id, message_id, rating, comment, route, created_at FROM {table}"
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def normalize_feedback(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only up/down ratings; carry the (optional) route through."""
    out: list[dict[str, Any]] = []
    for row in records:
        rating = row.get("rating")
        if rating not in ("up", "down"):
            continue
        route = row.get("route")
        out.append({"rating": rating, "route": route if route else None})
    return out


def _row_passed(report_name: str, row: dict[str, Any]) -> bool | None:
    fields = _ROUTE_PASS_FIELDS.get(report_name)
    if not fields:
        return None
    values = [row.get(field) for field in fields]
    if any(value is None for value in values):
        return None
    return all(bool(value) for value in values)


def load_judge_verdicts(
    results_dir: str | Path, route_map: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build a per-route judge verdict from each route's Tier-B report.

    ``verdict`` is True when the report's threshold checks all pass (falling
    back to a 0.5 pass-rate rule when no thresholds are recorded). ``pass_rate``
    is the fraction of measured rows whose per-row verdict passed.
    """
    route_map = route_map or DEFAULT_ROUTE_MAP
    results_dir = Path(results_dir)
    verdicts: dict[str, dict[str, Any]] = {}
    for route, report_name in route_map.items():
        path = results_dir / f"{report_name}.json"
        if not path.exists():
            verdicts[route] = {
                "dataset": report_name,
                "available": False,
                "n_total": 0,
                "n_measured": 0,
                "pass_rate": None,
                "verdict": None,
            }
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        rows = [r for r in (data.get("rows") or []) if isinstance(r, dict)]
        measured = [r for r in rows if r.get("status") not in _FAILED_ROW_STATUSES]
        flags = [p for p in (_row_passed(report_name, r) for r in measured) if p is not None]
        pass_rate = (sum(1 for f in flags if f) / len(flags)) if flags else None

        thresholds = data.get("thresholds") or {}
        checks = [c for c in thresholds.values() if isinstance(c, dict) and "passed" in c]
        if checks:
            verdict: bool | None = all(bool(c.get("passed")) for c in checks)
        elif pass_rate is not None:
            verdict = pass_rate >= 0.5
        else:
            verdict = None
        verdicts[route] = {
            "dataset": report_name,
            "available": True,
            "n_total": len(rows),
            "n_measured": len(measured),
            "pass_rate": pass_rate,
            "verdict": verdict,
        }
    return verdicts


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def correlate(
    feedback: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compute up-vote agreement and down-vote failure recall vs judge verdicts."""
    per_route: dict[str, dict[str, Any]] = {}
    matched = 0
    for row in feedback:
        route = row.get("route")
        if route is None:
            continue  # no route -> cannot be correlated to a judge verdict
        bucket = per_route.setdefault(route, {"up": 0, "down": 0})
        bucket[row["rating"]] += 1

    up_total = sum(1 for r in feedback if r["rating"] == "up")
    down_total = sum(1 for r in feedback if r["rating"] == "down")
    up_matched = 0
    down_matched = 0
    up_agree = 0
    down_fail = 0

    for row in feedback:
        info = verdicts.get(row.get("route"))
        if not info or info.get("verdict") is None:
            continue
        matched += 1
        if row["rating"] == "up":
            up_matched += 1
            if info["verdict"] is True:
                up_agree += 1
        else:
            down_matched += 1
            if info["verdict"] is False:
                down_fail += 1

    for route, bucket in per_route.items():
        info = verdicts.get(route)
        bucket["up_rate"] = _rate(bucket["up"], bucket["up"] + bucket["down"])
        bucket["judge_dataset"] = info.get("dataset") if info else None
        bucket["judge_verdict"] = info.get("verdict") if info else None
        bucket["judge_pass_rate"] = info.get("pass_rate") if info else None

    unmatched_routes = sorted(
        r for r in per_route if r not in verdicts or verdicts[r].get("verdict") is None
    )
    notes: list[str] = []
    if unmatched_routes:
        notes.append(f"routes without a judge verdict: {', '.join(unmatched_routes)}")

    return {
        "n_feedback": len(feedback),
        "n_up": up_total,
        "n_down": down_total,
        "n_matched": matched,
        "n_unmatched": len(feedback) - matched,
        "up_agreement_rate": _rate(up_agree, up_matched),
        "down_failure_recall": _rate(down_fail, down_matched),
        "per_route": per_route,
        "notes": notes,
    }


def build_correlation_report(
    *,
    feedback: list[dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    route_map: dict[str, str] | None = None,
    feedback_source: str = "",
    scores_source: str = "",
    min_per_route: int = 1,
) -> dict[str, Any]:
    """Assemble the (redact-free) correlation report dict."""
    core = correlate(feedback, verdicts)
    notes = list(core["notes"])
    if min_per_route > 1:
        thin = [
            route
            for route, bucket in core["per_route"].items()
            if (bucket["up"] + bucket["down"]) < min_per_route
        ]
        if thin:
            notes.append(f"routes with < {min_per_route} rating(s): {', '.join(sorted(thin))}")
    missing_reports = [r for r, info in verdicts.items() if not info.get("available")]
    if missing_reports:
        notes.append(f"missing judge reports for routes: {', '.join(sorted(missing_reports))}")
    return {
        "name": "feedback_correlation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feedback_source": feedback_source,
        "scores_source": scores_source,
        "route_map": dict(route_map or DEFAULT_ROUTE_MAP),
        "verdicts": verdicts,
        **core,
        "notes": notes,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Feedback ↔ Judge Correlation",
        "",
        f"- Generated: {report.get('timestamp')}",
        f"- Git SHA: `{report.get('git_sha')}`",
        f"- Feedback source: `{report.get('feedback_source')}`",
        f"- Scores source: `{report.get('scores_source')}`",
        "",
        "## Agreement",
        "",
        "| Signal | Value |",
        "|---|---|",
        f"| Ratings | {report.get('n_feedback')} "
        f"({report.get('n_up')} up / {report.get('n_down')} down) |",
        f"| Matched to a judge verdict | {report.get('n_matched')} |",
        f"| Thumbs-up agreement with pass | {report.get('up_agreement_rate')} |",
        f"| Thumbs-down failure recall | {report.get('down_failure_recall')} |",
        "",
        "## Per route",
        "",
        "| Route | Up | Down | Up rate | Judge | Verdict | Pass rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for route, bucket in sorted(report.get("per_route", {}).items()):
        lines.append(
            f"| {route} | {bucket['up']} | {bucket['down']} | {bucket.get('up_rate')} | "
            f"{bucket.get('judge_dataset')} | {bucket.get('judge_verdict')} | "
            f"{bucket.get('judge_pass_rate')} |"
        )
    notes = report.get("notes") or []
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    lines.append("")
    return "\n".join(lines)


def write_correlation_report(
    report: dict[str, Any], output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Write ``feedback_correlation.json`` + ``.md``; return both paths."""
    out = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "feedback_correlation.json"
    md_path = out / "feedback_correlation.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_render_markdown(report))
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedback_correlation.py",
        description="Correlate human thumbs up/down feedback with offline judge scores",
    )
    parser.add_argument("--feedback", type=Path, default=None, help="Feedback JSON export")
    parser.add_argument("--db", type=Path, default=None, help="SQLite DB with the user_feedback table")
    parser.add_argument("--table", default="user_feedback", help="Feedback table name")
    parser.add_argument(
        "--scores", type=Path, default=DEFAULT_RESULTS_DIR, help="Tier-B results directory"
    )
    parser.add_argument("--route-map", type=Path, default=None, help="JSON {route: report_stem}")
    parser.add_argument(
        "--min-per-route", type=int, default=1, help="Flag routes with fewer ratings (default: 1)"
    )
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: evals/results/)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    route_map = load_route_map(args.route_map)

    records: list[dict[str, Any]] = []
    source = ""
    if args.feedback:
        records = load_feedback_from_json(args.feedback)
        source = str(args.feedback)
    elif args.db:
        records = load_feedback_from_db(args.db, table=args.table)
        source = f"{args.db}#{args.table}"
    else:
        print("No --feedback or --db given; emitting an empty correlation report.", file=sys.stderr)

    feedback = normalize_feedback(records)
    verdicts = load_judge_verdicts(args.scores, route_map)
    report = build_correlation_report(
        feedback=feedback,
        verdicts=verdicts,
        route_map=route_map,
        feedback_source=source,
        scores_source=str(args.scores),
        min_per_route=args.min_per_route,
    )
    json_path, md_path = write_correlation_report(report, args.out)
    print(
        f"Feedback vs judge correlation: {report['n_feedback']} ratings, "
        f"up-agreement={report['up_agreement_rate']}, "
        f"down-failure-recall={report['down_failure_recall']}"
    )
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
