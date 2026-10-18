# Eval harness fixes — 2026-10-12

Implemented the six remaining findings from EVAL_REAUDIT_2026-10-12.md. Application quality remains an empirical question; this change makes the harness suitable for a fresh validation run.

## Changes

- Unjudged router, retrieval and memory families use effective model/embedding settings for freshness. Memory comparison and resume no longer compare the extractor's n/a label with an unrelated LLM judge.
- Comparison requires dataset/config/code/model provenance and generation metadata. Missing provenance and changed datasets, prompts, corpus PDFs or effective models are rejected. Code fingerprints include backend source, harness source, configuration and corpus PDFs; credentials are never hashed or reported.
- Completeness checks use actual rows, reject aggregate count mismatches and duplicate identities, validate requested dataset slice identities, and require finite measurements. Full baseline creation requires all eight full datasets. Partial smoke reports can be compared but cannot become an accepted baseline.
- The CLI records requested slice identities and captures provenance before model execution. Resume imports only rows in the requested slice.
- CI forwards report-only to every runner when enforce_absolute_gates=false. Absolute threshold failures become informational, while incomplete results and regressions remain failures.
- Router, ingestion, retrieval and memory write/read calls have bounded harness deadlines, alongside the existing graph/judge/extraction bounds. Retrieval scratch namespaces are unique; RAG cleanup targets only its owned index.
- Tests now use valid current-provenance fixtures rather than weakening validation for historical fixtures. New tests exercise stale/missing provenance, duplicates, full baseline coverage, unjudged families, memory resume, deadlines, all eight CLI paths, workflow execution and the rag-qa alias.

## Verification

- Full Tier A run: **175 passed, 95 deselected**; one LangSmith deprecation warning.
- Focused harness run: **112 passed** before the final RAG alias regression was added.
- Final alias/workflow targeted check: **2 passed, 111 deselected**.
- Ruff across backend, tests, harness, datasets and online package: passed.
- Dataset validator: eight datasets, 146 rows, passed.
- Git diff whitespace check: passed; existing CRLF normalization warnings only.
- All five workflow shell blocks passed bash syntax validation. Executed workflow simulation confirmed all eight runners receive report-only; a failed memory runner still executes blog and leaves exit code 1.
- No fresh live-model Tier B batch was performed, and the historical baseline was not overwritten.

## Remaining operating limits

Timed-out daemon workers can continue until provider/network timeouts; the harness does not cancel external calls. Their results are failures, so they cannot yield an accepted complete baseline. Historical reports without the new provenance must be regenerated. Prior PII/refusal failures and application token usage need a fresh live batch before a release or Phase 4 readiness claim.
