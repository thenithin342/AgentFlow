"""Deterministic Tier A tests for evals/online/feedback_correlation.py."""

import json
import sqlite3

import pytest

from evals.online import feedback_correlation as fc

pytestmark = pytest.mark.tier_a


def _write_report(path, *, rows, thresholds):
    path.write_text(
        json.dumps({"name": path.stem, "rows": rows, "thresholds": thresholds}),
        encoding="utf-8",
    )


@pytest.fixture
def results_dir(tmp_path):
    """A minimal Tier-B results tree for the four route-mapped datasets."""
    d = tmp_path / "results"
    d.mkdir()
    _write_report(
        d / "generator_tierb.json",
        rows=[
            {"status": "ok", "faithfulness_passed": True, "answer_relevancy_passed": True},
            {"status": "ok", "faithfulness_passed": False, "answer_relevancy_passed": True},
            {"status": "error", "faithfulness_passed": None, "answer_relevancy_passed": None},
        ],
        thresholds={"faithfulness": {"value": 0.5, "threshold": 0.9, "passed": False}},
    )
    _write_report(
        d / "application_tierb.json",
        rows=[{"status": "ok", "passed": True}, {"status": "ok", "passed": True}],
        thresholds={"g_eval_composite": {"value": 4.5, "threshold": 4.0, "passed": True}},
    )
    _write_report(
        d / "ragqa_tierb.json",
        rows=[
            {
                "status": "ok",
                "contextual_relevancy_passed": True,
                "faithfulness_passed": True,
                "answer_relevancy_passed": True,
            }
        ],
        thresholds={"faithfulness": {"value": 0.95, "threshold": 0.9, "passed": True}},
    )
    _write_report(
        d / "blog_tierb.json",
        rows=[{"status": "ok", "passed": True}],
        thresholds={"g_eval_composite": {"value": 4.6, "threshold": 4.0, "passed": True}},
    )
    return d


# --- inputs -----------------------------------------------------------------


def test_load_feedback_from_json_wrapped(tmp_path):
    p = tmp_path / "fb.json"
    p.write_text(json.dumps({"feedback": [{"rating": "up", "route": "chat"}]}), encoding="utf-8")
    assert fc.load_feedback_from_json(p) == [{"rating": "up", "route": "chat"}]


def test_load_feedback_from_json_rejects_bad_shape(tmp_path):
    p = tmp_path / "fb.json"
    p.write_text(json.dumps(42), encoding="utf-8")
    with pytest.raises(ValueError):
        fc.load_feedback_from_json(p)


def test_normalize_feedback_filters_invalid_ratings():
    rows = [
        {"rating": "up", "route": "chat"},
        {"rating": "sideways", "route": "chat"},
        {"rating": "down", "route": None},
        {"rating": "down", "route": ""},
    ]
    assert fc.normalize_feedback(rows) == [
        {"rating": "up", "route": "chat"},
        {"rating": "down", "route": None},
        {"rating": "down", "route": None},
    ]


def test_load_feedback_from_db(tmp_path):
    db = tmp_path / "fb.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE user_feedback (thread_id TEXT, message_id TEXT, rating TEXT, "
        "comment TEXT, route TEXT, created_at REAL)"
    )
    conn.execute("INSERT INTO user_feedback VALUES ('t', 'm', 'up', NULL, 'chat', 1.0)")
    conn.commit()
    conn.close()
    rows = fc.load_feedback_from_db(db)
    assert rows[0]["rating"] == "up"
    assert rows[0]["route"] == "chat"


# --- verdicts ---------------------------------------------------------------


def test_load_judge_verdicts(results_dir):
    verdicts = fc.load_judge_verdicts(results_dir)
    assert verdicts["chat"]["dataset"] == "generator_tierb"
    assert verdicts["chat"]["verdict"] is False  # threshold failed
    assert verdicts["chat"]["pass_rate"] == 0.5  # one of two ok rows passed
    assert verdicts["analysis"]["verdict"] is True
    assert verdicts["analysis"]["pass_rate"] == 1.0
    assert verdicts["research"]["verdict"] is True
    assert verdicts["blog"]["verdict"] is True


def test_load_judge_verdicts_marks_missing_reports(tmp_path):
    verdicts = fc.load_judge_verdicts(tmp_path)
    assert all(info["available"] is False for info in verdicts.values())
    assert all(info["verdict"] is None for info in verdicts.values())


# --- correlation ------------------------------------------------------------


def test_correlate_agreement_and_failure_recall():
    verdicts = {
        "analysis": {"dataset": "application_tierb", "verdict": True, "pass_rate": 1.0},
        "chat": {"dataset": "generator_tierb", "verdict": False, "pass_rate": 0.5},
    }
    feedback = [
        {"rating": "up", "route": "analysis"},
        {"rating": "up", "route": "analysis"},
        {"rating": "down", "route": "chat"},
        {"rating": "down", "route": "chat"},
    ]
    out = fc.correlate(feedback, verdicts)
    assert out["up_agreement_rate"] == 1.0
    assert out["down_failure_recall"] == 1.0
    assert out["n_matched"] == 4
    assert out["n_unmatched"] == 0
    assert out["per_route"]["analysis"]["up_rate"] == 1.0
    assert out["per_route"]["chat"]["judge_verdict"] is False


def test_correlate_disagreement_and_unmatched():
    verdicts = {"chat": {"dataset": "generator_tierb", "verdict": False, "pass_rate": 0.0}}
    feedback = [
        {"rating": "up", "route": "chat"},  # disagrees with a failing judge
        {"rating": "down", "route": "blog"},  # no verdict for blog -> unmatched
    ]
    out = fc.correlate(feedback, verdicts)
    assert out["up_agreement_rate"] == 0.0
    assert out["down_failure_recall"] is None  # no matched down-votes
    assert out["n_matched"] == 1
    assert out["n_unmatched"] == 1
    assert any("blog" in note for note in out["notes"])


def test_build_correlation_report_notes_missing_reports(tmp_path):
    verdicts = fc.load_judge_verdicts(tmp_path)  # all missing
    report = fc.build_correlation_report(
        feedback=[{"rating": "up", "route": "chat"}],
        verdicts=verdicts,
        scores_source=str(tmp_path),
        min_per_route=5,
    )
    assert report["name"] == "feedback_correlation"
    assert report["n_feedback"] == 1
    assert report["up_agreement_rate"] is None
    assert any("missing judge reports" in note for note in report["notes"])
    assert any("< 5" in note for note in report["notes"])


def test_write_correlation_report(results_dir, tmp_path):
    verdicts = fc.load_judge_verdicts(results_dir)
    report = fc.build_correlation_report(feedback=[], verdicts=verdicts)
    json_path, md_path = fc.write_correlation_report(report, tmp_path / "out")
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["name"] == "feedback_correlation"
    assert "Feedback" in md_path.read_text(encoding="utf-8")


# --- CLI --------------------------------------------------------------------


def test_load_route_map_default_and_override(tmp_path):
    assert fc.load_route_map() == fc.DEFAULT_ROUTE_MAP
    override = tmp_path / "map.json"
    override.write_text(json.dumps({"chat": "custom_tierb"}), encoding="utf-8")
    assert fc.load_route_map(override) == {"chat": "custom_tierb"}


def test_build_parser_defaults():
    args = fc.build_parser().parse_args([])
    assert args.table == "user_feedback"
    assert args.scores == fc.DEFAULT_RESULTS_DIR
    assert args.min_per_route == 1


def test_main_writes_report(tmp_path, results_dir):
    fb = tmp_path / "fb.json"
    fb.write_text(
        json.dumps([{"rating": "up", "route": "analysis"}, {"rating": "down", "route": "chat"}]),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = fc.main(["--feedback", str(fb), "--scores", str(results_dir), "--out", str(out)])
    assert rc == 0
    data = json.loads((out / "feedback_correlation.json").read_text(encoding="utf-8"))
    assert data["n_feedback"] == 2
    assert data["up_agreement_rate"] == 1.0
    assert data["down_failure_recall"] == 1.0
