# Eval fixes: independent re-audit, 2026-10-11

Recommendation: fix the remaining harness defects and product safety/cost failures before starting Phase 4. The patch improves generator evaluation, corpus ingestion, output-directory isolation, retrieval scoring, and workflow failure propagation, but does not yet make the results reliable enough for release decisions.

This audit reviewed the current uncommitted files, ran deterministic checks, and reproduced failure paths without provider calls. It did not change application or harness code or refresh the baseline. The previous live-run evidence is in EVAL_RUN_2026-10-10.md; that run predates these fixes and must not be treated as validation of the updated generator or RAG implementation.

## Remaining findings

| Priority | Location | Finding and required correction |
|---|---|---|
| P1 | evals/harness/run_offline.py:799 | `_RetryingEmbeddings` is not a LangChain `Embeddings` subclass. FAISS chooses its callable fallback, producing `TypeError: '_RetryingEmbeddings' object is not callable` at query time. Inherit the interface and exercise the installed wrapper through a real FAISS query. The autouse test getter returns the fake directly, bypassing `_EMBEDDINGS`, so the corpus test masks this failure. |
| P1 | evals/harness/run_offline.py:1393 | `_finish_run` enforces absolute gates only through comparison. A complete run with `passed=False`, `enforce_absolute=True`, and no `--compare` returns zero. Enforce absolute thresholds independently of whether a baseline was supplied; preserve the explicit report-only option. |
| P1 | evals/harness/tierb_runners.py:302 | Resume checks dataset hash and optionally judge name, but ignores recorded configuration/code provenance and accepts an absent judge name. Changed prompts, generation models, or code can reuse historical scores and stamp the resulting report with current provenance. Validate a fingerprint of relevant code, configuration, models, and corpus before accepting rows. Git HEAD alone cannot identify these uncommitted changes. |
| P1 | evals/harness/tierb_runners.py:610 | Timeout protection is used by safety only. Application, blog (1243), and RAG-QA (run_offline.py:1095) still invoke the graph directly. Those jobs can remain blocked indefinitely. Apply bounded invocation consistently, including synthesizer-only generation and provider/judge calls. The acknowledged orphaned daemon-thread limitation also remains. |
| P1 | evals/harness/tierb_runners.py:1471 | Compare-only checks trust aggregate counts/error rate, without validating expected row count, row statuses, or provenance. A one-row RAG report declaring 25 expected rows and passing metric values was reported `complete=True`, `clean=True`, `releasable=True`. Reuse strict row/identity/provenance validation in comparison. |
| P2 | evals/harness/run_offline.py:1233 | Non-quota judge failures leave RAG rows marked `ok`; safety has the same gap at tierb_runners.py:850. Missing scores are absent from the score aggregates, and completeness checks accept those rows. Mark any required missing judge measurement as failed and include a judge-failure count; resume must retry it. |
| P2 | evals/harness/tierb_runners.py:1506 | Baseline creation silently skips missing/unreadable reports and does not reject partial/error/stale runs. A directory with only RAG produced a baseline covering only RAG, narrowing future comparison scope. Require all intended families and validated provenance/completeness before writing an accepted baseline. |
| P2 | evals/harness/run_offline.py:2002 | Application/safety/memory/blog perform an extra comparison using default absolute enforcement before `_finish_run` receives the report-only flag. Thus `--compare --report-only` still fails on absolute gates in those commands. Remove the redundant comparison or pass the same enforcement setting. |

## Reproduction evidence

- Installed FAISS `_embed_query` with the retry wrapper around a deterministic `Embeddings` instance: `TypeError: '_RetryingEmbeddings' object is not callable`.
- `_finish_run` with one successful row and a failed accuracy threshold, absolute enforcement enabled, no comparison path: exit `0`.
- `seed_prior_ok` with matching dataset/judge but old config/code hashes: old row accepted, no issues reported.
- Comparison of a one-row report with `operational.expected_rows=25`: complete, clean, and releasable all true.
- `report_row_issues` with `status=ok` and missing faithfulness: no issues.
- `build_baseline` with seven absent families: produced a one-family baseline.

## Phase status and decision

The evaluation roadmap contains five phases, numbered 0 through 4. Documentation checks off Phases 0-3: four of five implemented. Phase 4's online runner, API suite, monitors, and feedback remain unchecked; `evals/online/` contains only `__init__.py`.

For acceptance, distinguish implementation from verification: foundations and Tier A are established; Tier B and its nightly gate exist but still have blocking correctness gaps. There is no defensible current full quality pass for the revised harness. Calling four phases operationally complete would overstate the evidence.

Fix the P1 findings first, then missing judge handling, baseline acceptance, and report-only consistency. Preserve the historical baseline until a clean full eight-family run has trustworthy provenance and complete measurements. Resolve the previously measured scope refusal/PII failures and investigate the 29,018-token single application probe before treating the gate as releasable. Historical results are evidence of these risks, not newly measured post-fix performance.

## Verification

- Dataset validator: passed, eight files and 146 rows.
- Focused harness Tier A suite: 83 passed; one pytest cache permission warning, no test failures.
- Full Tier A suite: 146 passed, 95 deselected, one LangSmith deprecation warning.
- Fresh post-fix full live Tier B batch: not performed in this re-audit. The confirmed RAG wrapper failure would make that batch invalid; deterministic mocks do not calibrate model quality.
