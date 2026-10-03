# AgentFlow evals/

LLM evaluation package for AgentFlow (companion to `EVAL_PLAN.md`).
It covers every LLM surface: router, Research/Analysis/Chat agents, Blog writer,
Synthesizer, RAG pipeline, LTM/STM memory, and the end-to-end graph + `/chat` API.

Layout follows `EVAL_PLAN.md §5`:

```text
evals/
├── README.md              # this file
├── config.yaml            # per-metric thresholds (EVAL_PLAN.md §7)
├── datasets/              # JSONL golden datasets (the core artifact)
│   ├── corpus/            # golden retrieval corpus PDFs + build_corpus.py
│   ├── router.jsonl       # 4-way route labels (accuracy/F1)
│   ├── retriever.jsonl    # queries + golden relevant_text anchors + materialized relevant_chunk_ids
│   ├── rag_qa.jsonl       # doc-QA pairs → pipeline triad
│   ├── generator.jsonl    # {query, context, reference} → faithfulness/relevancy
│   ├── application.jsonl  # queries + G-Eval rubrics per skill
│   ├── blog.jsonl         # blog topics + structural expectations
│   ├── memory.jsonl       # LTM fact / STM entity checks
│   └── safety.jsonl       # scope, injection, leakage, toxicity probes
├── harness/
│   └── validate_datasets.py  # Phase 0 stdlib-only dataset validator
├── online/                # Phase 4: LangSmith runner, API black-box suite, monitors
└── results/               # generated reports (gitignored)
```

## Tiers

| Tier | What runs | Cost | When |
|---|---|---|---|
| **Offline Tier A — deterministic** | Mocked/fake LLMs, recorded tool outputs, deterministic fake embeddings; parsing, orchestration, tool selection, retrieval structure, schema validation, safety regex gates | Zero API cost | CI on every push/PR — blocks merge (Phase 1) |
| **Offline Tier B — real-model batch** | Same golden datasets through the real Groq/Gemini stack; router accuracy, retrieval/generation triad (DeepEval), G-Eval, safety, operational KPIs | Free-tier quota (see budget below) | Local / scheduled nightly CI, report-only → then gated (Phase 2) |
| **Online — live** | LangSmith experiments on the deployed stack, black-box API suites, traffic replay, latency/reliability/cost monitors, user feedback | Free-tier quota + deployed infra | Scheduled cron + live (Phase 4) |

## How to run (Phase 0)

No pytest tests this phase — `pytest.ini` `testpaths` is pinned to `tests/` intentionally.
Validation is a standalone stdlib-only script:

```bash
# 1. Build the golden corpus PDFs (deterministic — regenerates every run)
python evals/datasets/corpus/build_corpus.py

# 2. Validate all datasets (exits non-zero on any violation)
python evals/harness/validate_datasets.py

# 3. Lint the eval package
ruff check evals/
```

`validate_datasets.py` checks: every line is valid JSON, `kind` matches the file,
all required fields are present, and no duplicate `query`/`input` values.
It also does a naive stdlib-only `thresholds:` existence check on `config.yaml`.

## How to run (Phase 1 — Tier A)

Tier A is a zero-LLM, zero-network pytest suite under `tests/test_tier_a_*.py`
(21 tests, marked `tier_a`, collected by the default CI run `-m "not eval"`):

```bash
python -m pytest tests/test_tier_a_*.py -q
```

What it gates: 100% retriever anchor → chunk materialization (12/12 rows), top-k
structure + no-index fallback, safety regex/boundary gates, synthesizer `Sources:`
rules + `<<UNTRUSTED>>` escaping, blog JSON schema, memory triggers, graph topology.
Per-row retriever recall/precision are saved to `evals/results/tier_a_retrieval.json`
as **informational only** — with the deterministic fake embeddings the FAISS top-k
ranking is arbitrary, so the real 0.85/0.80 gates run in Tier B with real embeddings.

Local-dev note: if your `.env` sets `LANGCHAIN_TRACING_V2=true` with a
`LANGCHAIN_API_KEY`, Tier A tests pass but the process lingers ~30 s at exit while
the LangSmith tracer flushes (same issue class as `ci.yml`'s eval-test comment; CI
is unaffected because it has no key). For a clean local exit:

```bash
# PowerShell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGCHAIN_API_KEY=''
python -m pytest tests/test_tier_a_*.py -q
```

## Quota budget

Free-tier caps and the estimated Tier-B calls per full run (one dataset row ≈
one LLM call unless noted):

| Provider | Quota | Dataset | Rows (Phase 0 seed) | Est. calls/run |
|---|---|---|---|---|
| Groq | 100K tokens/day | router.jsonl | 40+ | ~40 classify calls |
| Groq | 100K tokens/day | generator.jsonl | 12+ | ~12 judge calls (faithfulness + relevancy ≈ 24) |
| Groq | 100K tokens/day | rag_qa.jsonl | 15+ | ~15 e2e turns + ~45 triad judge calls |
| Groq | 100K tokens/day | application.jsonl | 12+ | ~12 runs + ~36 G-Eval judge calls |
| Groq | 100K tokens/day | blog.jsonl | 5+ | ~5 runs + ~10 judge calls |
| Groq | 100K tokens/day | memory.jsonl | 8+ | ~8 extract/summarise calls |
| Groq | 100K tokens/day | safety.jsonl | 12+ | ~12 probe calls |
| Tavily | 1K searches/mo | research + blog runs | ~20 search-backed runs | ~20–60 searches (1–3 per run; reuse cached outputs where possible) |
| Google | 1M tokens/day | embeddings (ingest + LTM) | 3 corpus PDFs + memory | negligible (short docs) |

Total ≈ 130–190 LLM calls/run — sized to fit within one Groq daily budget
(~1 full Tier-B run/day). Tavily usage stays well under 1K/mo at nightly cadence.

Note: token cost accounting uses LLM `usage_metadata` (see `backend/llm.py`
`TokenBudgetWrapper`); judge prompts are versioned and judge models pinned in
`config.yaml` so score drift ≠ behavioral drift.

## Judges / DeepEval

Decision (2026-09-14): **DeepEval** (`deepeval>=4.2,<5`, isolated in
`requirements-eval.txt`) is the Tier-B judge framework. Judges run on **Groq via an
OpenAI-compatible endpoint** (no OpenAI key needed); Gemini is the fallback.

Why DeepEval over RAGAS:
- First-class **G-Eval** implementation (matches the Application metric block in
  EVAL_PLAN.md §3.4) plus the full RAG triad — contextual recall / precision,
  faithfulness, answer relevancy — and pytest-style ergonomics
  (`assert_test`, `deepeval test run`).
- Brings its own LLM layer (openai SDK) with **zero langchain and zero litellm
  dependencies** — decoupled from this stack's pinned langchain versions (the
  exact failure point that broke ragas). Verified against PyPI 2026-09-14.
- Install footprint note: deepeval 4.2.x caps `rich<15` and adds
  pytest-xdist / pytest-rerunfailures / pytest-repeat — use a **fresh venv**,
  never the project's main venv.
- RAGAS was rejected after a real install test: `ragas 0.4.3` resolves but
  `import ragas` fails against `langchain_community 0.4.2` / `langchain_core 1.4.8`
  (`ragas/llms/base.py` imports `ChatVertexAI`, which the 0.4.x sunset line removed).
  The install also downgraded `tenacity 9.1.4 -> 8.5.0` and `rich 15.0.0 -> 13.9.4`
  in a shared venv (both restored afterwards).

Status: **UNVERIFIED — the spike is the Phase 2 entry gate.** Before the Tier-B
harness is written, a fresh venv must: install this file, score ~3 rows from each
of `rag_qa.jsonl` / `generator.jsonl` / `application.jsonl` / `safety.jsonl` with
Groq as the judge, and confirm no conflict with the `tenacity` / `rich` pins. If
the spike fails, the hand-rolled judges below are the fallback (no new
dependencies); `config.yaml` thresholds are identical either way.

Judges are LLMs too — judge prompts are versioned and judge models pinned in
`config.yaml` so score drift ≠ behavioural drift.

Fallback (hand-rolled judges): if the DeepEval spike fails, implement judge prompts
directly in `evals/harness/llm_judge.py` (stdlib + langchain only):

- faithfulness: claim-split the answer, verdict per claim against context
  (supported / unsupported), score = supported / total.
- answer/contextual relevancy: 1–5 rubric prompt, normalised to 0–1.
- contextual recall/precision: anchor-substring hit rate over retrieved chunks
  (exact-phrase, deterministic — Tier A already).
- G-Eval: correctness / completeness / style 1–5 rubric per skill, composite mean.
- toxicity: 0–1 judge score (lower = safer).

This fallback needs no new dependencies and preserves the `config.yaml`
thresholds unchanged.
