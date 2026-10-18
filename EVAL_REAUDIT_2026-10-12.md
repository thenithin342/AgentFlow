# Independent re-audit — 2026-10-12

Recommendation: finish the remaining harness fixes before a full live Tier B batch or Phase 4. The claim that all eight findings are closed is not supported by the current code.

## Verification

- Full deterministic Tier A: **144 passed, 9 failed, 95 deselected**, one LangSmith deprecation warning. The diagnostic retry completed tests in 6.33 seconds after imports. The first collection attempt stalled and was stopped; the suite is executable in this environment.
- Dataset validator: all eight files, 146 rows, passed.
- `git diff --check`: passed (line-ending warnings only).
- No application/harness code changed; no baseline refreshed; no fresh live model batch run.

## Confirmed remaining findings

1. **P1 — Fresh memory reports always fail freshness validation.** `evals/harness/tierb_runners.py:1850` prefers the report's top-level judge name and compares it with the configured judge. The memory runner emits `n/a (llm_fast fact extraction)` at line 1425. Reproduction with current provenance and eight complete rows flagged memory stale against `openai/gpt-oss-20b`. Memory resume uses the same incompatible expectation. Validate judge identity only for families with a judge, and record the actual extraction model separately. This blocks a successful full comparison and baseline even with perfect measurements.

2. **P1 — Comparison and baseline still accept stale datasets and absent provenance.** At `tierb_runners.py:1833`, config/code checks only run when the old hash exists; dataset SHA is never checked. A 44-row router report with an explicitly incorrect dataset SHA returned complete/clean/releasable true. Removing provenance and judge identity entirely also returned all three true. Eight one-row reports, one per family, with no provenance were accepted by `build_baseline`. Require all provenance fields, compare current dataset/corpus/model identities, and require full dataset coverage for accepted baselines. Keep partial diagnostic runs distinct from full quality evidence.

3. **P1 — Row counts do not prove coverage.** `tierb_runners.py:1713` trusts aggregate.n_rows rather than enforcing actual row length and dataset identities. Forty-four copies of the same router row passed comparison. Check actual row count, distinct identities, and membership in the requested dataset slice; reject duplicates and omissions. The previous count defect is only partially closed.

4. **P1 — CI report-only transition is broken by the new standalone gates.** `.github/workflows/tier_b_eval.yml:161` builds EXTRA without `--report-only`; the input `enforce_absolute_gates=false` is used only by the later comparison step (line 215). Every run now fails standalone absolute gates in `_finish_run`, so the evaluation step leaves the whole workflow red despite the transition switch. Pass report-only to every runner when absolute enforcement is disabled, while retaining completeness and regression failure behavior.

5. **P2 — Fingerprint omits generation-defining files and runtime settings.** `tierb_runners.py:191` omits `backend/graph/agents.py`, `blog_agent.py`, tools, RAG ingest, memory code, graph security, and the corpus PDFs. Generation model names are selected from runtime settings/environment, also unrecorded. Changing an agent prompt, corpus, or GROQ_FAST_MODEL can still reuse old rows with a matching fingerprint. Fingerprint relevant source/corpus files and record nonsecret effective model/backend/settings identities. Do not hash credentials.

6. **P2 — Timeout coverage remains incomplete.** Graph/judge/extraction calls are bounded, but retriever ingestion/query (`run_offline.py:347,360`), RAG ingestion/query (`run_offline.py:1138,1168`), and memory round-trip write/read (`tierb_runners.py:1234,1235`) remain synchronous provider paths. Retry attempts only bound the number of failures, not elapsed time for a stuck call. Bound these paths or configure and verify provider deadlines. Orphaned daemon workers still continue and can mutate shared counters/state after timeout; this documented limitation remains.

## What the patch does fix

- The retry wrapper now implements LangChain Embeddings; the corpus test exercises the wrapper.
- Standalone absolute gates are checked, and direct command report-only behavior is implemented.
- Resume now checks the hashes it records; missing judge scores are rejected for applicable rows.
- Graph/judge/extraction calls gained timeout wrappers.
- The redundant application/safety/memory/blog comparison was removed.
- Baseline construction rejects absent families and several incomplete/stale cases.

These improvements do not close the identity, full-coverage, memory-family, and CI semantics gaps above.

## Failed Tier A tests

- test_compare_run_after_run_writes_single_run_file
- test_collect_run_metrics_report_flags_empty_and_unhealthy (duplicate empty_runs entry)
- test_seed_prior_ok_refuses_rows_from_a_different_judge_model
- test_seed_prior_ok_dedupes_completed_rows_and_skips_failed
- test_seed_prior_ok_without_expectations_keeps_legacy_behaviour
- test_run_compare_only_report_only_downgrades_absolute_gates
- test_compare_run_after_run_enforces_absolute_gates
- test_main_compare_reads_the_baselines_own_directory
- test_main_writes_reports_to_the_selected_out_dir

Several fixtures need valid current provenance/measurements to isolate their intended behavior. Update those fixtures and assertions; do not weaken production validation to accept invalid reports merely to retain historical tests. The zero-row report test also needs its intended report-only/gating semantics made explicit.

## Next decision

Keep Phase 4 pending. Fix the confirmed defects, make all Tier A tests green, and verify memory and CI report-only behavior end to end. Then run a small live smoke across all eight families, followed by one complete isolated batch with unchanged code/configuration. Refresh the baseline only from validated full coverage. Earlier product PII/refusal failures and high application token usage still require post-fix empirical checks.
