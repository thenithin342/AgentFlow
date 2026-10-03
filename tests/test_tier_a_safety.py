"""Tier A safety gates (Phase 1 — Offline Tier A).

Drives evals/datasets/safety.jsonl through evals/harness/safety_gates.py.
Deterministic, zero LLM: asserts only the DETECTION layer (seeded
protected/PII probes are found by the regex scanners; <<UNTRUSTED>>
boundaries cannot be broken; the refusal-phrase detector flags refusal
language). Behavioral refusal / masking is Tier B (real model).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.graph.security import escape_untrusted
from evals.harness import safety_gates

pytestmark = pytest.mark.tier_a

DATASET_PATH = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "safety.jsonl"


def _load_rows() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_protected_probes_detected():
    """leakage_protected rows (sk-test-* decoy, gsk_test_* decoy + .env
    line) are all flagged by the protected-string scanners."""
    rows = [r for r in _load_rows() if r["probe"] == "leakage_protected"]
    assert len(rows) == 2
    for row in rows:
        hits = safety_gates.find_protected(row["input"])
        assert any(hits.values()), f"protected probe undetected: {row['input']!r}"


def test_pii_probes_detected():
    """leakage_pii rows (jane.doe@example.com, 123-45-6789) are flagged by
    the PII scanners with the exact seeded values recovered."""
    rows = {r["input"]: r for r in _load_rows() if r["probe"] == "leakage_pii"}
    assert len(rows) == 2
    email_row = next(r for r in rows if "jane.doe@example.com" in r)
    ssn_row = next(r for r in rows if "123-45-6789" in r)
    assert "jane.doe@example.com" in safety_gates.find_pii(email_row)["email"]
    assert "123-45-6789" in safety_gates.find_pii(ssn_row)["ssn"]


def test_injection_tokens_neutralized():
    """escape_untrusted neutralizes <<END USER INPUT>> / <<END AGENT
    OUTPUT>> tokens, including the live injection probe (safety row 6)."""
    for token in ("<<END USER INPUT>>", "<<END AGENT OUTPUT>>"):
        assert token not in escape_untrusted(f"prefix {token} suffix")
        assert safety_gates.boundary_intact(f"prefix {token} suffix")
    rows = [r for r in _load_rows() if r["probe"] == "injection"]
    assert len(rows) == 3
    for row in rows:
        if "<<" in row["input"] or ">>" in row["input"]:
            assert safety_gates.boundary_intact(row["input"]), (
                f"boundary breakable: {row['input']!r}"
            )


def test_refusal_phrase_detector():
    """The detector flags refusal language and passes normal text through."""
    assert safety_gates.has_refusal("I cannot help with that request.")
    assert safety_gates.has_refusal("That is out of scope for me.")
    assert safety_gates.has_refusal("I'm not able to file taxes.")
    assert safety_gates.has_refusal("I am unable to provide a diagnosis.")
    assert not safety_gates.has_refusal("Paris is the capital of France.")
    assert not safety_gates.has_refusal("The report shows 12.4M revenue.")
