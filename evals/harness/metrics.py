"""
Pure-Python scoring functions for the Tier B batch harness.

Zero LLM dependencies — all functions operate on in-memory data structures.
Used by run_offline.py and report.py.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Router metrics
# ---------------------------------------------------------------------------

def router_accuracy(rows: list[dict], predictions: list[str]) -> float:
    """Compute overall router classification accuracy.

    Args:
        rows: list of dataset rows (each must have an "expected" key).
        predictions: list of predicted labels, same length as rows.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    if not rows:
        return 0.0
    correct = sum(
        1 for row, pred in zip(rows, predictions, strict=True)
        if row.get("expected") == pred
    )
    return correct / len(rows)


def per_class_f1(rows: list[dict], predictions: list[str]) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1 for router predictions.

    Args:
        rows: list of dataset rows (each must have an "expected" key).
        predictions: list of predicted labels, same length as rows.

    Returns:
        Dict mapping each label to {"precision", "recall", "f1", "support"}.
        Labels not present in the data are omitted.
    """
    if not rows:
        return {}

    labels = set()
    for row in rows:
        labels.add(row.get("expected"))
    for pred in predictions:
        labels.add(pred)

    # Count TP, FP, FN per label
    tp: dict[str, int] = Counter()
    fp: dict[str, int] = Counter()
    fn: dict[str, int] = Counter()

    for row, pred in zip(rows, predictions, strict=True):
        actual = row.get("expected")
        if actual == pred:
            tp[actual] += 1
        else:
            fp[pred] += 1
            fn[actual] += 1

    result: dict[str, dict[str, float]] = {}
    for label in sorted(labels):
        true_pos = tp.get(label, 0)
        false_pos = fp.get(label, 0)
        false_neg = fn.get(label, 0)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        support = true_pos + false_neg

        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return result


def confusion_matrix(rows: list[dict], predictions: list[str]) -> dict[str, dict[str, int]]:
    """Compute confusion matrix: actual label -> predicted label -> count.

    Args:
        rows: list of dataset rows (each must have an "expected" key).
        predictions: list of predicted labels, same length as rows.

    Returns:
        Nested dict: matrix[actual][predicted] = count.
    """
    if not rows:
        return {}

    labels = set()
    for row in rows:
        labels.add(row.get("expected"))
    for pred in predictions:
        labels.add(pred)

    matrix: dict[str, dict[str, int]] = {}
    for actual in sorted(labels):
        matrix[actual] = {pred: 0 for pred in sorted(labels)}

    for row, pred in zip(rows, predictions, strict=True):
        actual = row.get("expected")
        matrix[actual][pred] += 1

    return matrix


# ---------------------------------------------------------------------------
# Retrieval metrics (recall@k, precision@k)
# ---------------------------------------------------------------------------

def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Compute recall@k: fraction of relevant items in top-k retrieved.

    Args:
        relevant: set of relevant item IDs.
        retrieved: list of retrieved item IDs in rank order.
        k: number of top items to consider.

    Returns:
        Recall as a float between 0.0 and 1.0.
    """
    if not relevant:
        return 1.0  # vacuous truth: no relevant items means perfect recall
    top_k = set(retrieved[:k])
    hits = len(relevant & top_k)
    return hits / len(relevant)


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Compute precision@k: fraction of top-k retrieved that are relevant.

    Args:
        relevant: set of relevant item IDs.
        retrieved: list of retrieved item IDs in rank order.
        k: number of top items to consider.

    Returns:
        Precision as a float between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


# ---------------------------------------------------------------------------
# Operational aggregation
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # Linear interpolation method
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    frac = rank - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def aggregate_operational(
    timings: list[float] | None = None,
    costs: list[float] | None = None,
    errors: list[Exception] | None = None,
    total_calls: int = 0,
) -> dict[str, Any]:
    """Aggregate operational metrics from a batch run.

    Args:
        timings: list of latency values in seconds.
        costs: list of cost values (arbitrary units).
        errors: list of exceptions that occurred.
        total_calls: total number of calls attempted (including failures).

    Returns:
        Dict with:
        - latency: {mean, p95, min, max} (only if timings provided)
        - cost: {total, mean, min, max} (only if costs provided)
        - success_rate: fraction of calls without errors
        - error_rate: fraction of calls with errors
        - error_count: number of errors
    """
    result: dict[str, Any] = {}

    if timings:
        result["latency"] = {
            "mean": sum(timings) / len(timings),
            "p95": _percentile(timings, 95),
            "min": min(timings),
            "max": max(timings),
        }

    if costs:
        result["cost"] = {
            "total": sum(costs),
            "mean": sum(costs) / len(costs),
            "min": min(costs),
            "max": max(costs),
        }

    error_count = len(errors) if errors else 0
    if total_calls > 0:
        result["success_rate"] = (total_calls - error_count) / total_calls
        result["error_rate"] = error_count / total_calls
    else:
        result["success_rate"] = 1.0
        result["error_rate"] = 0.0

    result["error_count"] = error_count

    return result


# ---------------------------------------------------------------------------
# Utility: generate failing rows list
# ---------------------------------------------------------------------------

def failing_rows(
    rows: list[dict],
    predictions: list[str],
    query_key: str = "input",
    expected_key: str = "expected",
) -> list[dict]:
    """Return list of rows where prediction != expected.

    Args:
        rows: list of dataset rows.
        predictions: list of predicted labels.
        query_key: key in row for the query text (default "input").
        expected_key: key in row for the expected label (default "expected").

    Returns:
        List of dicts with {query, expected, predicted, reason (if present)}.
    """
    failures = []
    for row, pred in zip(rows, predictions, strict=True):
        expected = row.get(expected_key)
        if expected != pred:
            failures.append({
                "query": row.get(query_key, ""),
                "expected": expected,
                "predicted": pred,
                "reason": row.get("reason", ""),
            })
    return failures
