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

## How to run (Phase 2 — Tier B)

Tier B runs the golden datasets through the real Groq/Gemini stack. Always run
from the repo root with `PYTHONPATH=.`; never under pytest.

```bash
# Env hygiene — the harness also forces these itself
export QDRANT_URL="" LANGCHAIN_TRACING_V2=false
unset LANGCHAIN_API_KEY

# Validate the datasets first (stdlib only; exits 0)
python evals/harness/validate_datasets.py

# One subcommand per metric family. --limit N smoke-tests, --resume continues
# after a quota stop (only rows still in "ok" status are skipped).
python evals/harness/run_offline.py router
python evals/harness/run_offline.py retriever
python evals/harness/run_offline.py generator
python evals/harness/run_offline.py rag-qa --resume
python evals/harness/run_offline.py application --resume
python evals/harness/run_offline.py safety
python evals/harness/run_offline.py memory
python evals/harness/run_offline.py blog

# Aggregate every *_tierb.json into a baseline, then diff runs against it
python evals/harness/run_offline.py baseline
python evals/harness/run_offline.py --compare evals/results/baseline.json
python evals/harness/run_offline.py application --compare evals/results/baseline.json
```

`--compare` prints a per-metric delta (`current - baseline`) and flags a
regression when a higher-is-better metric drops by more than 0.05 (or a
lower-is-better metric such as toxicity/leakage rises by more than 0.05). Exit
code is 0 on a clean diff, 1 when a regression is flagged.

### Observed — full run 2026-10-06 (judge `openai/gpt-oss-20b` on Groq)

| Subcommand | Rows | Wall | LLM calls | Tokens | Tavily | Gate |
|---|---|---|---|---|---|---|
| router | 44 | 213 s | 44 classify | n/a | 0 | accuracy 0.977 ✅ |
| retriever | 12 | 14 s | 0 (embeddings only) | n/a | 0 | recall 1.00 ✅ / precision 0.25 ❌ |
| generator | 20 | 86 s | 40 judge | 20.4 K | 0 | faithfulness 1.00 ✅ / relevancy 0.99 ✅ |
| rag-qa | 25 | 372 s | 24 judge | 12.1 K | 0 | ctx-rel 0.952 ✅ / faith 0.708 ❌ / relevancy 0.948 ✅ |
| application | 20 | 232 s | 27 graph + 20 judge | 54.9 K + 5.3 K | 10–30 | G-Eval 4.17 ✅ |
| safety | 12 | 197 s | 39 graph + 12 judge | 28.9 K + 4.6 K | 2 | scope 0.75 ❌ / PII 1 ❌ / protected 0 ✅ / tox 0 ✅ |
| memory | 8 | 28 s | 8 extract | 2.5 K | 0 | entity 1.00 ✅ / round-trip ✅ (extract precision 0.44, informational) |
| blog | 5 | 209 s | 23 graph + 5 judge | 37.3 K + 11.6 K | 16 | G-Eval 4.60 ✅ / structure 0.60 ❌ |

Notes from that run:

- **Groq free tier is ~200 K tokens/day per org.** The primary key exhausted
  mid-run and the judge client (single key, no fallback) stopped `application`
  at 15/20 rows. `--resume` completed it with a secondary key. The graph LLM
  pool falls back across `GROQ_API_KEY` / `_2` / `_3`, but the judge only reads
  `GROQ_API_KEY`, so rotate that env var before a run when the primary is spent.
- **Tavily counting** uses a LangChain `on_tool_start` callback (sub-agent tool
  messages are not propagated to the parent graph state). On `--resume` the
  seeded rows keep their recorded count, but the run total only covers the
  resumed slice (application: 20 searches in run 1 + 10 in the resume = 30).
- **Blog is the expensive path** — 3–4 Tavily searches and 5–9 K tokens per row.

Implementation notes:

- The real-graph runners (`application`, `safety`, `blog`) compile the graph
  with an `InMemorySaver`, not the sync `SqliteSaver`: the blog node's
  `agent.ainvoke` cannot run under a sync checkpointer (the ReAct sub-agent
  inherits the parent checkpointer). Persistence is therefore per-run, which is
  fine for single-turn evals.
- Eval LTM writes are redirected to `evals/results/tierb_ltm/` via
  `LTM_INDEX_DIR`, so the real `ltm_indexes/` tree is never touched.
- Safety probes run through the real graph with a 30 s per-call bound. Scope
  refusal reuses the shared `has_refusal`; the harness folds typographic
  apostrophes (U+2019) first so `"I can’t"` is not a false negative.
- Memory extraction precision is reported but not gated — the golden
  `expected_facts` are paraphrases, so a literal substring metric is
  informational. Entity recall and the cross-thread round-trip are gated.

## Quota budget

Free-tier caps and the estimated Tier-B calls per full run (one dataset row ≈
one LLM call unless noted):

| Provider | Quota | Dataset | Rows | Calls (observed 2026-10-06) |
|---|---|---|---|---|
| Groq | 200K tokens/day | router.jsonl | 44 | 44 classify |
| Groq | 200K tokens/day | generator.jsonl | 20 | 40 judge |
| Groq | 200K tokens/day | rag_qa.jsonl | 25 | 24 triad judge |
| Groq | 200K tokens/day | application.jsonl | 20 | 27 graph + 20 G-Eval judge |
| Groq | 200K tokens/day | blog.jsonl | 5 | 23 graph + 5 G-Eval judge |
| Groq | 200K tokens/day | memory.jsonl | 8 | 8 extract |
| Groq | 200K tokens/day | safety.jsonl | 12 | 39 graph + 12 toxicity judge |
| Tavily | 1K searches/mo | research + blog runs | 45 search-backed rows | ~48 searches observed (research ≈ 6/run, blog ≈ 3–4/run) |
| Google | 1M tokens/day | embeddings (ingest + LTM) | 3 corpus PDFs + memory | negligible (short docs) |

Total ≈ 240 LLM calls/run, ~180K Groq tokens — sized to fit within one daily budget
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
