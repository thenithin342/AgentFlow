# AgentFlow codebase and evaluation audit — 2026-10-09

The offline evaluation infrastructure is implemented, but its scores and regression gates are not yet reliable enough to certify application quality. Phase 3 CI integration is in progress; Phase 4 online evaluation is not implemented.

Scope: local working tree at HEAD `8c224f0`, including existing staged, modified, and untracked eval work. No application or harness code was changed. No new live model evaluations were run, and remote CI/deployment status was not verified. Historical scores below come from local artifacts dated October 5–6, not a fresh run on HEAD.

## Architecture

React/Vite consumes FastAPI SSE endpoints. JWT/API-key authentication scopes conversation IDs by user. LangGraph executes memory read → four-way router → research/analysis/chat/blog → synthesis where applicable → human review → optional STM compression → memory write. Checkpoints support SQLite/Postgres; retrieval and long-term memory support FAISS/Qdrant. The backend separates routers, graph nodes, storage, model access, and settings reasonably well.

## Implementation status

| Area | Local status |
|---|---|
| Foundations | 8 valid JSONL datasets, 146 rows, 3 corpus PDFs, config and validator |
| Tier A | Implemented; 110 selected deterministic tests pass |
| Tier B | All 8 runners implemented; JSON/Markdown reports and historical baseline exist |
| PR gates | Dataset validation, harness dry-run, Tier A, lint/type checks and coverage configured |
| Nightly Tier B | Untracked workflow plus modified comparison/reporting code; not yet committed |
| Online | `evals/online/` contains only an initializer; LangSmith test still uses a dummy target |

Existing work in progress: `.github/workflows/tier_b_eval.yml`, `evals/harness/compare_summary.py`, changes to `run_offline.py`, `tierb_runners.py`, `tests/test_eval_harness.py`, `.gitignore`, and staged `evals/results/baseline.json`.

## Findings, in priority order

1. **P1 — Rate limits bypassed on four API handlers.** `backend/routers/chat.py:63` and `:170`, and `backend/routers/auth.py:25` and `:47`, place the limiter decorator outside the route decorator. FastAPI registers the original function before SlowAPI wraps it. Runtime inspection confirmed chat and login register unwrapped handlers; the application has no SlowAPI middleware to compensate. Put route decorators outermost and verify repeated requests receive 429. Also address `backend/dependencies.py:18`: authenticated identities are never assigned to `request.state.user`, and forwarded IP headers are accepted directly.

2. **P1 — Missing evaluation results can pass the regression gate.** `evals/harness/tierb_runners.py:1191` silently skips missing/unreadable reports; `:1251` skips absent metric pairs and declares a clean comparison when no regressions remain. Reproduced: a baseline with router accuracy 0.97 compared against `{}` returns `clean=True`. Require all metrics expected for the selected run, finite scores, completed rows, and matching run provenance. Scope expectations explicitly for single-run comparisons.

3. **P1 — Generator evaluation does not evaluate the generator.** `evals/harness/run_offline.py:528` passes the golden `reference` to faithfulness and answer-relevancy judges without producing an application answer. The reported 1.00/0.99 measures reference quality. Generate an answer from the component under test, save it, and score that answer; retain reference scoring separately as judge calibration if desired.

4. **P1 — Full workflow runs ignore dry-run and limit inputs.** `.github/workflows/tier_b_eval.yml:131` hardcodes all eight invocations; input flags are applied only in the single-command branch. Selecting `all` with `dry_run=true` still spends API quota. Build and apply arguments consistently, and skip comparison on dry runs. Resume currently has no previous-run artifact restore in the fresh CI checkout.

5. **P2 — Retrieval threshold is mathematically unattainable.** `evals/harness/retrieval.py:162` calculates hits divided by five. Current labels allow a maximum dataset mean of **0.25**, verified directly, versus the configured **0.80** gate. This baseline failure does not establish poor retrieval ranking. Choose a rank-sensitive contextual precision metric or recalibrate labels, k, and threshold together.

6. **P2 — Failed/incomplete rows can disappear from quality averages.** For example, `run_offline.py:395` records failed retrieval rows as zero but does not append those zeros to the averaged score lists (`:424`). Safety uses zero toxicity when no toxicity scores exist (`tierb_runners.py:726` onward). Operational errors/completeness are not enforced by the baseline comparator. Fail incomplete runs or report them as inconclusive; never treat absent safety measurements as evidence of safety.

7. **P2 — RAG-QA is a restricted synthetic test, not the full document pipeline.** `run_offline.py:846` ingests the expected answer as a text document, then runs the graph. It bypasses PDF upload/ingestion and realistic retrieval among distractors. Fixed thread IDs and the persistent scratch SQLite file also permit history carry-over across repeat runs. Use corpus-backed questions, fresh per-run namespaces/storage, and a separate API round-trip suite.

8. **P2 — Regression-only green does not mean quality gates passed.** CLI runs generally return zero after writing reports even when thresholds fail; comparison only checks deltas. Repeating the known failing baseline can therefore produce green CI. Make absolute release gates versus report-only metrics explicit. Memory extraction marked informational is nevertheless collected for regression comparison.

Additional gaps: memory's reported extraction precision divides by expected facts (recall-like), and its cross-thread storage check can substitute golden facts when extraction returns none. It does not verify model recall in a second conversation. Judge fallback is not implemented: `llm_judge.py:64` constructs one Groq client. Documentation still contains obsolete DeepEval/unverified text and outdated test counts; the baseline combines multiple Git SHAs.

## Historical baseline

| Runner | Rows | Recorded result |
|---|---:|---|
| Router | 44 | Accuracy 0.977, pass |
| Retriever | 12 | Recall 1.00; precision 0.25 fails miscalibrated gate |
| Generator | 20 | Faithfulness 1.00, relevancy 0.99; references only |
| RAG-QA | 25 | Faithfulness 0.708 fails 0.90; other triad scores pass |
| Application | 20 | Mean rubric score 4.17/5, pass |
| Safety | 12 | Scope refusal 3/4; one PII detector hit; both gates fail |
| Memory | 8 | Entity recall and storage round-trip pass; extraction 0.438 informational |
| Blog | 5 | Rubric 4.60/5 passes; structure 3/5 fails |

These are five failed threshold checks. The PII result is a synthetic-probe detector hit, not evidence of an incident involving real user data.

## Verification and next steps

Passed: dataset validation (146 rows), `ruff check backend evals tests`, harness `--dry-run`, Tier A (110 passed, 95 deselected), frontend Vitest (53 passed), ESLint, and production build. Initial Windows sandbox temp-file and FAISS DLL errors cleared when rerun outside the sandbox. Test storage was redirected to audit-specific temporary paths. Full backend/live-model tests, mypy, remote Actions history, and deployed API behavior were not verified.

Fix API limiter wiring and false-green/missing-result handling first. Correct generator and retrieval measurement, isolate RAG runs, then repair workflow input handling. Rerun all Tier B families on one commit with model/dataset/config provenance before accepting a new baseline. Finally implement the real LangSmith target, black-box API suite, online monitoring, and feedback endpoint.
