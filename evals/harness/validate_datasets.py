"""Phase 0 dataset validator (stdlib only — no pytest, no third-party deps).

For each ``evals/datasets/*.jsonl`` checks:
    - every non-blank line parses as JSON (``json.loads``)
    - ``kind`` equals the file stem (e.g. router.jsonl -> ``"router"``)
    - all REQUIRED_FIELDS for that kind are present
    - no duplicate dedupe key (first present of query / input / topic / turn)
    - retriever rows carry ``relevant_chunk_ids`` as null (Phase 0 seed) or
      a list of non-negative int chunk indexes (Phase 1+ backfill,
      materialized from ``relevant_text`` anchors — see
      evals/harness/retrieval.py)
    - safety rows carry an ``expected.behavior`` in the allowed enum

Also does a naive ``thresholds:`` existence check on ``evals/config.yaml``
(PyYAML if available, else a line scan). A missing config file or a missing
``thresholds:`` key makes validation FAIL with a nonzero exit.

Exit non-zero on any dataset violation or failed config check. Usage:
    python evals/harness/validate_datasets.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
DATASETS_DIR = HERE.parent.parent / "datasets"
CONFIG_PATH = HERE.parent.parent / "config.yaml"

# File stem -> required fields (``kind`` itself is checked separately).
REQUIRED_FIELDS = {
    "router": {"input", "expected", "reason", "kind"},
    "retriever": {"query", "source_doc", "relevant_text", "relevant_chunk_ids", "kind"},
    "rag_qa": {"query", "thread", "reference", "expected_facts", "kind"},
    "generator": {"query", "context", "reference", "kind"},
    "application": {"query", "route", "rubric", "kind"},
    "blog": {"topic", "rubric", "expected_structure", "kind"},
    "memory": {"turn", "expected_facts", "expected_entities", "kind"},
    "safety": {"input", "probe", "expected", "kind"},
}

# First-present key used for duplicate detection.
DEDUPE_KEYS = ("query", "input", "topic", "turn")

SAFETY_BEHAVIORS = {"refuse", "mask", "no_leak", "non_toxic"}


def _dedupe_key(row: dict):
    """Return ``(key, value)`` for the first dedupe candidate that is a
    non-empty string, else None. Values must be strings before they can be
    used in ``seen`` membership — unhashable JSON arrays/objects are reported
    by ``_invalid_dedupe_value`` and never reach the set."""
    for key in DEDUPE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return (key, value)
    return None


def _invalid_dedupe_value(row: dict):
    """Return ``(key, value)`` for the first dedupe candidate that is present
    but not a non-empty string (e.g. ``[]`` / ``{}`` / numbers), else None."""
    for key in DEDUPE_KEYS:
        value = row.get(key)
        if value is not None and not (isinstance(value, str) and value.strip()):
            return (key, value)
    return None


def validate_file(path: Path) -> tuple:
    """Validate one JSONL file; return (count, [error strings])."""
    stem = path.stem
    errors = []
    count = 0
    seen = {}
    required = REQUIRED_FIELDS.get(stem)
    if required is None:
        return 0, [f"{path.name}: unknown dataset kind {stem!r} (no REQUIRED_FIELDS entry)"]
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path.name}:{lineno}: row is not a JSON object")
                continue
            count += 1
            if row.get("kind") != stem:
                errors.append(
                    f"{path.name}:{lineno}: kind={row.get('kind')!r}, expected {stem!r}"
                )
            missing = required - set(row.keys())
            if missing:
                errors.append(f"{path.name}:{lineno}: missing fields: {sorted(missing)}")
            key = _dedupe_key(row)
            bad = _invalid_dedupe_value(row)
            if bad is not None:
                bad_key, bad_value = bad
                errors.append(
                    f"{path.name}:{lineno}: dedupe key {bad_key!r} must be a "
                    f"non-empty string, got {bad_value!r}"
                )
            elif key is None:
                errors.append(f"{path.name}:{lineno}: no dedupe key {DEDUPE_KEYS}")
            elif key in seen:
                errors.append(
                    f"{path.name}:{lineno}: duplicate {key[0]}={key[1]!r} "
                    f"(first seen line {seen[key]})"
                )
            else:
                seen[key] = lineno
            if stem == "retriever" and "relevant_chunk_ids" in row:
                ids = row["relevant_chunk_ids"]
                if ids is not None and not (
                    isinstance(ids, list)
                    and ids
                    and all(
                        isinstance(i, int) and not isinstance(i, bool) and i >= 0
                        for i in ids
                    )
                ):
                    errors.append(
                        f"{path.name}:{lineno}: relevant_chunk_ids must be null "
                        f"or a non-empty list of chunk indexes, got {ids!r}"
                    )
            if stem == "safety":
                behavior = (row.get("expected") or {}).get("behavior") if isinstance(
                    row.get("expected"), dict
                ) else None
                if behavior not in SAFETY_BEHAVIORS:
                    errors.append(
                        f"{path.name}:{lineno}: expected.behavior={behavior!r}, "
                        f"must be one of {sorted(SAFETY_BEHAVIORS)}"
                    )
    return count, errors


def check_config_thresholds() -> bool:
    """Naive ``thresholds:`` existence check; returns False when the config
    file is missing or has no ``thresholds:`` key (main() then fails)."""
    if not CONFIG_PATH.exists():
        print(f"WARNING: config not found at {CONFIG_PATH}")
        return False
    try:
        import yaml  # type: ignore

        with open(CONFIG_PATH, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "thresholds" in data:
            print(f"config check: top-level 'thresholds:' present ({CONFIG_PATH.name})")
            return True
        print(f"WARNING: no top-level 'thresholds:' key in {CONFIG_PATH}")
        return False
    except ImportError:
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            found = any(line.strip().startswith("thresholds:") for line in handle)
        if found:
            print(f"config check: 'thresholds:' line found ({CONFIG_PATH.name}, line scan)")
            return True
        print(f"WARNING: no 'thresholds:' line in {CONFIG_PATH} (line scan, PyYAML absent)")
        return False


def main() -> int:
    files = sorted(DATASETS_DIR.glob("*.jsonl"))
    if not files:
        print(f"ERROR: no *.jsonl files in {DATASETS_DIR}")
        return 1
    total = 0
    all_errors = []
    for path in files:
        count, errors = validate_file(path)
        total += count
        status = "OK" if not errors else f"{len(errors)} ERROR(S)"
        print(f"{path.name}: {count} rows — {status}")
        all_errors.extend(errors)
    config_ok = check_config_thresholds()
    if all_errors:
        print(f"\n{len(all_errors)} violation(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    if not config_ok:
        print(
            f"\nERROR: {CONFIG_PATH.name} is missing or has no 'thresholds:' "
            "key — validation failed."
        )
        return 1
    print(f"\nAll datasets valid: {len(files)} files, {total} rows total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
