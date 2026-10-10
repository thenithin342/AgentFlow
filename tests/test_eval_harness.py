"""Tier A unit tests for the offline evaluation harness (Phase 3).

Everything here is deterministic and offline — pure scoring functions,
judge-response parsing, token accounting, report writers and the CLI
argument parser. No LLM calls, no network, no dataset ingestion.

The harness modules that actually contact a model (``judge_faithfulness``
and friends) are never exercised here; only their pure helpers are.

Marked ``tier_a`` so the fast CI gate (``pytest -m "tier_a"``) picks them
up and the merge-blocking job catches harness regressions for free.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.harness import metrics
from evals.harness.compare_summary import locate_compare_file
from evals.harness.llm_judge import (
    JudgeTokenTracker,
    _extract_tokens,
    _parse_json_from_response,
)
from evals.harness.mocks import FakeLLMResponse
from evals.harness.report import compare_reports, write_report
from evals.harness.tierb_runners import (
    compare_run_after_run,
    compare_to_baseline,
    emit_regression_annotations,
    render_compare_markdown,
    run_compare_only,
    run_key_for_cli,
    write_step_summary,
)

pytestmark = pytest.mark.tier_a

SUBCOMMANDS = [
    "router",
    "retriever",
    "generator",
    "rag-qa",
    "application",
    "safety",
    "memory",
    "blog",
    "baseline",
]


@pytest.fixture(scope="module")
def run_offline_module():
    """Import ``evals.harness.run_offline`` with its env side effects isolated.

    ``run_offline.py`` mutates the process environment at import time as part
    of its Tier B hygiene (clears ``LANGCHAIN_API_KEY``, blanks ``QDRANT_URL``,
    pins ``LTM_INDEX_DIR``). Snapshot and restore those keys so importing it
    from a unit test cannot leak settings into the rest of the Tier A suite.
    """
    env_keys = ("QDRANT_URL", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LTM_INDEX_DIR")
    saved = {key: os.environ.get(key) for key in env_keys}

    import evals.harness.run_offline as module

    yield module

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# evals/harness/metrics.py
# ---------------------------------------------------------------------------

def test_router_accuracy_perfect_partial_and_empty():
    rows = [{"expected": "chat"}, {"expected": "research"}, {"expected": "blog"}]
    assert metrics.router_accuracy(rows, ["chat", "research", "blog"]) == 1.0
    assert metrics.router_accuracy(rows, ["chat", "chat", "blog"]) == pytest.approx(2 / 3)
    assert metrics.router_accuracy([], []) == 0.0


def test_per_class_f1_balanced_and_missed_class():
    rows = [{"expected": "a"}, {"expected": "a"}, {"expected": "b"}, {"expected": "b"}]
    scores = metrics.per_class_f1(rows, ["a", "a", "b", "b"])
    assert set(scores) == {"a", "b"}
    for label in ("a", "b"):
        assert scores[label]["precision"] == 1.0
        assert scores[label]["recall"] == 1.0
        assert scores[label]["f1"] == 1.0
        assert scores[label]["support"] == 2

    # A label that is only ever *predicted* has zero support and zero recall,
    # while the class it stole a prediction from drops to recall 0.5.
    scores = metrics.per_class_f1([{"expected": "a"}, {"expected": "a"}], ["b", "a"])
    assert scores["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert scores["a"]["precision"] == 1.0
    assert scores["a"]["recall"] == 0.5
    assert scores["a"]["f1"] == pytest.approx(2 / 3)


def test_per_class_f1_empty_input_is_empty_dict():
    assert metrics.per_class_f1([], []) == {}


def test_confusion_matrix_dimensions_and_tallies():
    rows = [{"expected": "a"}, {"expected": "b"}, {"expected": "a"}]
    matrix = metrics.confusion_matrix(rows, ["a", "b", "b"])
    assert set(matrix) == {"a", "b"}
    assert all(set(counts) == {"a", "b"} for counts in matrix.values())
    assert matrix["a"] == {"a": 1, "b": 1}
    assert matrix["b"] == {"a": 0, "b": 1}
    assert metrics.confusion_matrix([], []) == {}


def test_recall_and_precision_at_k_edges():
    # Empty relevant set is the vacuous-truth case: perfect recall.
    assert metrics.recall_at_k(set(), ["x"], 5) == 1.0
    # k smaller than the relevant set only credits the hits inside the window.
    assert metrics.recall_at_k({1, 2, 3}, [1, 4, 2], 2) == pytest.approx(1 / 3)
    # k larger than the ranked list still counts every hit.
    assert metrics.recall_at_k({1, 2}, [1, 2, 3], 10) == 1.0

    assert metrics.precision_at_k({1}, [1, 2], 5) == pytest.approx(0.5)
    assert metrics.precision_at_k({1}, [2, 3], 2) == 0.0
    assert metrics.precision_at_k({1}, [], 5) == 0.0
    assert metrics.precision_at_k({1}, [1, 2], 0) == 0.0


def test_aggregate_operational_latency_cost_and_errors():
    agg = metrics.aggregate_operational(
        timings=[1.0, 2.0, 3.0, 4.0],
        costs=[0.5, 1.5],
        errors=[ValueError("boom")],
        total_calls=4,
    )
    assert agg["latency"]["mean"] == pytest.approx(2.5)
    assert agg["latency"]["min"] == 1.0
    assert agg["latency"]["max"] == 4.0
    assert agg["latency"]["p95"] == pytest.approx(3.85)
    assert agg["cost"]["total"] == pytest.approx(2.0)
    assert agg["success_rate"] == pytest.approx(0.75)
    assert agg["error_rate"] == pytest.approx(0.25)
    assert agg["error_count"] == 1


def test_aggregate_operational_defaults_without_calls():
    agg = metrics.aggregate_operational()
    assert "latency" not in agg
    assert "cost" not in agg
    assert agg["success_rate"] == 1.0
    assert agg["error_rate"] == 0.0
    assert agg["error_count"] == 0


def test_failing_rows_filters_on_mismatch():
    rows = [
        {"input": "q1", "expected": "chat", "reason": "greeting"},
        {"input": "q2", "expected": "research"},
        {"input": "q3", "expected": "blog"},
    ]
    failures = metrics.failing_rows(rows, ["chat", "chat", "blog"])
    assert failures == [
        {"query": "q2", "expected": "research", "predicted": "chat", "reason": ""},
    ]
    assert metrics.failing_rows([{"input": "q", "expected": "chat"}], ["chat"]) == []


# ---------------------------------------------------------------------------
# evals/harness/llm_judge.py
# ---------------------------------------------------------------------------

def test_extract_tokens_prefers_usage_metadata():
    response = FakeLLMResponse(content="ok", usage={"input": 100, "output": 50})
    assert _extract_tokens(response) == {"input": 100, "output": 50, "total": 150}


def test_extract_tokens_falls_back_to_response_metadata():
    # FakeLLMResponse without `usage` exposes usage_metadata=None, so the
    # extractor must read response_metadata["usage"] (the ChatGroq shape).
    response = FakeLLMResponse(
        content="ok",
        response_metadata={"usage": {"input_tokens": 7, "output_tokens": 3}},
    )
    assert _extract_tokens(response) == {"input": 7, "output": 3, "total": 10}

    # OpenAI-style prompt/completion keys are the configured secondary names.
    alt = SimpleNamespace(
        response_metadata={"usage": {"prompt_tokens": 12, "completion_tokens": 4}}
    )
    assert _extract_tokens(alt) == {"input": 12, "output": 4, "total": 16}
    assert _extract_tokens(SimpleNamespace()) == {"input": 0, "output": 0, "total": 0}


def test_parse_json_from_response_clean_fenced_and_inline():
    assert _parse_json_from_response('{"score": 0.8}') == {"score": 0.8}
    fenced = '```json\n{"score": 0.5, "reasoning": "relevant"}\n```'
    assert _parse_json_from_response(fenced) == {"score": 0.5, "reasoning": "relevant"}
    inline = 'Sure, here is the verdict: {"score": 0.25} and that is all.'
    assert _parse_json_from_response(inline) == {"score": 0.25}


def test_parse_json_from_response_invalid_returns_none():
    assert _parse_json_from_response("") is None
    assert _parse_json_from_response("no json object in this text") is None


def test_judge_token_tracker_accumulates_and_reports():
    tracker = JudgeTokenTracker()
    tracker.record(FakeLLMResponse(content="a", usage={"input": 10, "output": 5}))
    tracker.record(FakeLLMResponse(content="b", usage={"input": 1, "output": 2}))
    assert tracker.to_dict() == {"input": 11, "output": 7, "total": 18, "calls": 2}

    direct = JudgeTokenTracker()
    direct.record_direct({"input": 3, "output": 4})
    direct.record_direct({"input": 1, "output": 1})
    assert direct.input_tokens == 4
    assert direct.output_tokens == 5
    assert direct.total_tokens == 9
    assert direct.calls == 2


def test_judge_token_tracker_is_thread_safe():
    tracker = JudgeTokenTracker()
    response = FakeLLMResponse(content="x", usage={"input": 2, "output": 3})

    def _worker() -> None:
        for _ in range(50):
            tracker.record(response)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tracker.calls == 200
    assert tracker.total_tokens == 200 * 5


# ---------------------------------------------------------------------------
# evals/harness/report.py
# ---------------------------------------------------------------------------

def test_write_report_writes_json_and_markdown_siblings(tmp_path):
    results = {
        "name": "router",
        "judge_model": "test-judge",
        "rows": [{"query": "hi", "expected": "chat", "predicted": "chat"}],
        "aggregate": {"accuracy": 1.0},
        "thresholds": {"router_accuracy": {"value": 1.0, "threshold": 0.9, "passed": True}},
        "operational": {"wall_time": 0.01},
    }
    json_path, md_path = write_report("unit_router", results, output_dir=tmp_path)

    assert json_path == tmp_path / "unit_router.json"
    assert md_path == tmp_path / "unit_router.md"
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["name"] == "unit_router"
    assert payload["aggregate"]["accuracy"] == 1.0
    assert payload["thresholds"]["router_accuracy"]["passed"] is True
    assert "timestamp" in payload and "git_sha" in payload

    markdown = md_path.read_text(encoding="utf-8")
    assert "Unit_Router Evaluation Report" in markdown
    assert "router_accuracy" in markdown


def test_compare_reports_flags_aggregate_and_threshold_regressions(tmp_path):
    baseline = {
        "name": "baseline",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "aggregate": {"accuracy": 0.90, "n_rows": 10},
        "thresholds": {"faithfulness": {"passed": True}},
    }
    current = {
        "name": "current",
        "timestamp": "2026-01-02T00:00:00+00:00",
        "aggregate": {"accuracy": 0.80, "n_rows": 10},
        "thresholds": {"faithfulness": {"passed": False}},
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")

    deltas = compare_reports(baseline_path, current_path)

    accuracy = deltas["metric_deltas"]["accuracy"]
    assert accuracy["delta"] == pytest.approx(-0.10)
    assert accuracy["regression"] is True
    # A flat metric is not a regression.
    assert deltas["metric_deltas"]["n_rows"]["regression"] is False
    assert any(item.startswith("accuracy") for item in deltas["regressions"])
    assert any("faithfulness" in item for item in deltas["regressions"])


def test_compare_reports_missing_file_returns_error(tmp_path):
    result = compare_reports(tmp_path / "nope.json", tmp_path / "also_nope.json")
    assert "error" in result


# ---------------------------------------------------------------------------
# evals/harness/tierb_runners.py — baseline comparison semantics
# ---------------------------------------------------------------------------

def test_compare_to_baseline_higher_is_better_regression():
    baseline = {"name": "baseline", "timestamp": "t0", "metrics": {"router_tierb.router_accuracy": 0.95}}
    current = {"router_tierb.router_accuracy": 0.80}

    compare = compare_to_baseline(baseline, current)

    info = compare["metric_deltas"]["router_tierb.router_accuracy"]
    assert info["delta"] == pytest.approx(-0.15)
    assert info["regression"] is True
    assert compare["clean"] is False
    assert "router_tierb.router_accuracy" in compare["regressions"]

    # A drop inside the -0.05 tolerance band is not a regression.
    within = compare_to_baseline(
        {"metrics": {"generator_tierb.faithfulness": 0.90}},
        {"generator_tierb.faithfulness": 0.87},
    )
    assert within["metric_deltas"]["generator_tierb.faithfulness"]["regression"] is False
    assert within["clean"] is True


def test_compare_to_baseline_lower_is_better_semantics():
    # Toxicity / leakage metrics improve when they go *down*, so a rise is the
    # regression and a fall is clean.
    worse = compare_to_baseline(
        {"metrics": {"safety_tierb.toxicity_max": 0.05}},
        {"safety_tierb.toxicity_max": 0.30},
    )
    assert worse["metric_deltas"]["safety_tierb.toxicity_max"]["regression"] is True
    assert worse["clean"] is False

    better = compare_to_baseline(
        {"metrics": {"safety_tierb.leakage_pii_hits": 2.0}},
        {"safety_tierb.leakage_pii_hits": 0.0},
    )
    assert better["metric_deltas"]["safety_tierb.leakage_pii_hits"]["regression"] is False
    assert better["clean"] is True


def test_compare_to_baseline_ignores_non_numeric_and_new_metrics():
    compare = compare_to_baseline(
        {"metrics": {"run.metric": 1.0}},
        {"run.metric": "n/a", "brand.new": 0.5},
    )
    assert compare["metric_deltas"] == {}
    assert compare["regressions"] == []
    assert compare["clean"] is True


# ---------------------------------------------------------------------------
# evals/harness/run_offline.py — CLI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_build_parser_accepts_every_subcommand(run_offline_module, command):
    args = run_offline_module.build_parser().parse_args([command])
    assert args.command == command


def test_build_parser_parses_global_options(run_offline_module):
    parser = run_offline_module.build_parser()
    args = parser.parse_args(
        ["router", "--limit", "5", "--offset", "2", "--resume", "--dry-run",
         "--compare", "evals/results/baseline.json"]
    )
    assert args.limit == 5
    assert args.offset == 2
    assert args.resume is True
    assert args.dry_run is True
    assert args.compare == Path("evals/results/baseline.json")


def test_build_parser_subcommand_defaults(run_offline_module):
    args = run_offline_module.build_parser().parse_args(["safety"])
    assert args.limit is None
    assert args.offset == 0
    assert args.resume is False
    assert args.dry_run is False
    assert args.compare is None


def test_top_level_dry_run_parses_without_subcommand(run_offline_module):
    args = run_offline_module.build_parser().parse_args(["--dry-run"])
    assert args.command is None
    assert args.dry_run is True


def test_main_top_level_dry_run_prints_plan_for_every_subcommand(run_offline_module, capsys):
    rc = run_offline_module.main(["--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    for command in SUBCOMMANDS:
        assert command in out


def test_main_subcommand_dry_run_prints_dataset_plan(run_offline_module, capsys):
    rc = run_offline_module.main(["router", "--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "router.jsonl" in out
    assert "will process" in out


def test_dry_run_flag_before_subcommand(run_offline_module, capsys):
    args = run_offline_module.build_parser().parse_args(["--dry-run", "router"])
    assert args.command == "router"
    assert args.dry_run is True

    rc = run_offline_module.main(["--dry-run", "router"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "router.jsonl" in out


def test_build_parser_parses_top_level_compare_run(run_offline_module):
    args = run_offline_module.build_parser().parse_args(
        ["--compare", "evals/results/baseline.json", "--compare-run", "router"]
    )
    assert args.command is None
    assert args.compare == Path("evals/results/baseline.json")
    assert args.compare_run == "router"


# ---------------------------------------------------------------------------
# evals/harness/tierb_runners.py — Job Summary rendering (Phase 3 Session 3)
# ---------------------------------------------------------------------------

def _compare_payload(clean: bool) -> dict:
    """Minimal compare.json payload with one flat and one regressing metric."""
    return {
        "baseline_file": "baseline",
        "baseline_timestamp": "2026-01-01T00:00:00+00:00",
        "current_timestamp": "2026-01-02T00:00:00+00:00",
        "metric_deltas": {
            "router.router_accuracy": {
                "baseline": 0.9773,
                "current": 0.9773,
                "delta": 0.0,
                "regression": False,
            },
            "safety.scope_refusals": {
                "baseline": 0.75,
                "current": 0.75 if clean else 0.70,
                "delta": 0.0 if clean else -0.05,
                "regression": not clean,
            },
        },
        "regressions": [] if clean else ["safety.scope_refusals"],
        "clean": clean,
    }


def test_render_compare_markdown_header_and_table():
    markdown = render_compare_markdown(_compare_payload(clean=True), git_sha="abc1234")
    assert "| Metric | Baseline | Current | Delta | Status |" in markdown
    assert "| router.router_accuracy | 0.9773 | 0.9773 | +0.0000 | ✅ OK |" in markdown
    assert "**Metrics evaluated:** 2" in markdown
    assert "**Passing:** 2 | **Regressions:** 0" in markdown
    assert "abc1234" in markdown
    assert "No regressions vs baseline." in markdown
    assert "REGRESSION" not in markdown


def test_render_compare_markdown_marks_regression_and_counts():
    markdown = render_compare_markdown(_compare_payload(clean=False))
    assert "| safety.scope_refusals | 0.7500 | 0.7000 | -0.0500 | ❌ REGRESSION |" in markdown
    assert "**Passing:** 1 | **Regressions:** 1" in markdown
    assert "1 regression(s) detected" in markdown


def test_emit_regression_annotations_prints_one_error_per_regression(capsys):
    annotations = emit_regression_annotations(_compare_payload(clean=False))
    assert annotations == [
        "::error::Regression detected in metric: safety.scope_refusals (delta: -0.0500)"
    ]
    out = capsys.readouterr().out
    assert "::error::Regression detected in metric: safety.scope_refusals" in out


def test_run_key_for_cli_maps_rag_qa_only():
    assert run_key_for_cli("rag-qa") == "ragqa"
    assert run_key_for_cli("router") == "router"
    assert run_key_for_cli("application") == "application"


def test_write_step_summary_appends_and_adds_trailing_newline(tmp_path):
    summary = tmp_path / "summary.md"
    assert write_step_summary("hello", str(summary)) is True
    assert write_step_summary("# second", str(summary)) is True
    assert summary.read_text(encoding="utf-8") == "hello\n# second\n"


def test_write_step_summary_returns_false_without_target(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert write_step_summary("# nowhere to go") is False


def test_write_step_summary_uses_github_env(monkeypatch, tmp_path):
    summary = tmp_path / "env_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert write_step_summary("body") is True
    assert summary.read_text(encoding="utf-8") == "body\n"


def test_compare_run_after_run_writes_single_run_file(tmp_path):
    baseline = {
        "name": "baseline",
        "timestamp": "t0",
        "metrics": {"router.router_accuracy": 0.9, "safety.scope_refusals": 1.0},
    }
    (tmp_path / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (tmp_path / "router_tierb.json").write_text(
        json.dumps({"thresholds": {"router_accuracy": {"value": 0.9}}}), encoding="utf-8"
    )

    rc = compare_run_after_run(tmp_path / "baseline.json", tmp_path, "router")

    assert rc == 0
    payload = json.loads((tmp_path / "compare_router.json").read_text(encoding="utf-8"))
    # Only the router run's metrics are compared, never the whole baseline.
    assert set(payload["metric_deltas"]) == {"router.router_accuracy"}
    assert payload["clean"] is True


def test_compare_run_after_run_returns_nonzero_on_regression(tmp_path):
    baseline = {
        "name": "baseline",
        "timestamp": "t0",
        "metrics": {"router.router_accuracy": 0.99},
    }
    (tmp_path / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (tmp_path / "router_tierb.json").write_text(
        json.dumps({"thresholds": {"router_accuracy": {"value": 0.5}}}), encoding="utf-8"
    )

    rc = compare_run_after_run(tmp_path / "baseline.json", tmp_path, "router")

    assert rc == 1
    payload = json.loads((tmp_path / "compare_router.json").read_text(encoding="utf-8"))
    assert payload["clean"] is False


def test_locate_compare_file_prefers_single_run_then_falls_back(tmp_path):
    assert locate_compare_file(tmp_path, "all") is None

    (tmp_path / "compare.json").write_text("{}", encoding="utf-8")
    assert locate_compare_file(tmp_path, "all") == tmp_path / "compare.json"
    assert locate_compare_file(tmp_path, "router") == tmp_path / "compare.json"

    (tmp_path / "compare_router.json").write_text("{}", encoding="utf-8")
    assert locate_compare_file(tmp_path, "router") == tmp_path / "compare_router.json"


def test_locate_compare_file_maps_rag_qa_to_ragqa(tmp_path):
    (tmp_path / "compare_ragqa.json").write_text("{}", encoding="utf-8")
    assert locate_compare_file(tmp_path, "rag-qa") == tmp_path / "compare_ragqa.json"


def test_compare_run_after_run_returns_nonzero_when_no_metrics_found(tmp_path):
    baseline = {"metrics": {"router.router_accuracy": 0.99}}
    (tmp_path / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    rc = compare_run_after_run(tmp_path / "baseline.json", tmp_path, "router")
    assert rc == 1


def test_run_compare_only_returns_nonzero_when_no_metrics_found(tmp_path):
    baseline = {"metrics": {"router.router_accuracy": 0.99}}
    (tmp_path / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    rc = run_compare_only(tmp_path / "baseline.json", tmp_path)
    assert rc == 1


