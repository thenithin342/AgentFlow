"""LangSmith Online Evaluation test for AgentFlow.

Runs live LangSmith evaluation using evals/online/langsmith_runner.py
against the real router target and golden datasets.

Run with:
    pytest tests/test_eval.py
or
    python tests/test_eval.py
"""

from __future__ import annotations

import os

import pytest

from evals.online.langsmith_runner import (
    evaluate_router_match,
    run_langsmith_evaluation,
    target_router,
)


@pytest.mark.eval
@pytest.mark.skipif(
    os.environ.get("LANGCHAIN_API_KEY") in (None, "dummy_key_for_ci", ""),
    reason="No LangSmith API key",
)
def test_langsmith_evaluation():
    """Online test running router evaluation in LangSmith."""
    experiment_results = run_langsmith_evaluation(
        dataset="router",
        target=target_router,
        evaluators=[evaluate_router_match],
        prefix="AgentFlow-Router-Online",
    )
    if experiment_results is None:
        pytest.skip("LangSmith evaluation skipped: client unavailable")

    results_list = list(experiment_results)
    if not results_list:
        return

    total_score = 0.0
    count = 0
    for r in results_list:
        if isinstance(r, dict):
            results = r.get("evaluation_results", {}).get("results", [])
        else:
            eval_res = getattr(r, "evaluation_results", None) or {}
            results = eval_res.get("results", []) if isinstance(eval_res, dict) else []

        for metric in results:
            if getattr(metric, "key", "") == "router_accuracy":
                total_score += getattr(metric, "score", 0.0)
                count += 1

    if count > 0:
        avg_score = total_score / count
        assert avg_score >= 0.8, (
            f"Evaluation failed: Average router_accuracy was {avg_score:.2f} (expected >= 0.8)"
        )


if __name__ == "__main__":
    if os.environ.get("LANGCHAIN_API_KEY") not in (None, "dummy_key_for_ci", ""):
        test_langsmith_evaluation()
