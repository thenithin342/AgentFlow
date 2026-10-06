# AgentFlow — LLM Evaluation Plan

**Status:** Approved-in-progress
**Owner:** Nithin
**Scope:** the whole project — router, Research/Analysis/Chat agents, Blog writer,
Synthesizer, RAG pipeline, LTM/STM memory, and the end-to-end graph + streaming `/chat` API
**Companion to:** `PRD.md`, `DESIGN_DOC.md`, `TECH_STACK.md`

---

## 1. Purpose

Instrument a component-level LLM evaluation framework for AgentFlow covering every
LLM surface, executed in two stages **in order**:

1. **Offline evals** — fixed golden datasets, run locally or in CI, never against the
   deployed system. This is the foundation: deterministic where possible, then real-model
   batch runs to produce quality baselines.
2. **Online evals** — the deployed stack (Render + Vercel) and live traffic, LangSmith
   experiments wired to the real graph, and operational monitoring.

---

## 2. Definitions

| Stage | Tier | What runs | Cost | Where |
|---|---|---|---|---|
| Offline | **A — deterministic** | Mocked/fake LLMs, recorded tool outputs, the deterministic fake embeddings fixture already in `tests/conftest.py`. Tests parsing, orchestration, tool selection, retrieval structure, schema validation, safety regex gates. | Zero API cost | CI on every push/PR — blocks merge |
| Offline | **B — real-model batch** | The same golden datasets through the real Groq/Gemini stack in a batch harness. Produces quality numbers: router accuracy, the retrieval/generation triad (DeepEval judge math), G-Eval, safety, operational KPIs. | Free-tier quota (Groq 100 K tok/day, Tavily 1 K/mo) | Local / scheduled nightly CI, report-only → then gated |
| Online | **Live** | LangSmith experiments against the live graph, black-box API suites against prod, traffic replay, latency/reliability/cost monitors, user feedback. | Free-tier quota + deployed infra | Scheduled cron + live |

---

## 3. Metric framework (spec)

The evaluation measures the following metric families.

### 3.1 Retriever — component
- **Contextual recall** — are all ground-truth relevant chunks recovered in the top-k?
- **Contextual precision** — are the retrieved top-k all relevant (no noise)?

### 3.2 Generator — component
- **Faithfulness** — is every claim in the generated answer supported by the retrieved context (no hallucination)?
- **Answer relevancy** — does the generated answer actually address the user's question?

### 3.3 Pipeline — full triad
- **Contextual relevancy** — is the retrieved context relevant to the user's question?
- **Faithfulness** — as above, over the full end-to-end turn.
- **Answer relevancy** — as above, over the full end-to-end turn.

### 3.4 Application — G-Eval
LLM-as-judge rubric scores (1–5) per skill (research, analysis, chat, blog, synthesizer):
- **Correctness** — factual accuracy against provided context / reference.
- **Completeness** — all requested aspects answered.
- **Style** — tone, structure, formatting per the synthesizer style guide.

### 3.5 Safety — gates
- **Scope** — out-of-domain queries are refused or clearly flagged; never fabricated.
- **Leakage — protected** — API keys, secrets, internal config never surface in output.
- **Leakage — PII** — emails, phone numbers, SSNs never surface unmasked.
- **Toxicity (lower = safer)** — user-facing output is non-toxic.

### 3.6 Operational — measured
- **Latency — e2e p95** — request → `[FINAL]`/`[DONE]`.
- **Latency — TTFT (info)** — time to first streamed token.
- **Cost — per query** — token accounting (`usage_metadata` / tiktoken).
- **Reliability — success rate / error rate** — runs without `[ERROR]` or exception.

---

## 4. Metric → AgentFlow component mapping

| Metric | AgentFlow surface | Measurement method | Offline | Online |
|---|---|---|---|---|
| Contextual recall | `retrieve_documents` tool → FAISS/Qdrant + `ingest_pdf` chunking | % of golden relevant chunk IDs recovered in top-k | A + B | ✓ |
| Contextual precision | same | % of retrieved top-k that is relevant (noise) | A + B | ✓ |
| Faithfulness | agents + synthesizer (`final_response`) | claims-vs-context verdict (judge), DeepEval `FaithfulnessMetric` | B | ✓ sampled |
| Answer relevancy | same | judge + semantic similarity, DeepEval `AnswerRelevancyMetric` | B | ✓ + feedback |
| Contextual relevancy | full graph RAG turn | judge: retrieved context relevant to question | B | ✓ |
| Faithfulness (triad) | full graph RAG turn | judge over full-turn context | B | ✓ |
| Answer relevancy (triad) | full graph RAG turn | judge end-to-end | B | ✓ |
| G-Eval (correctness / completeness / style) | research, analysis, chat, blog, synthesizer outputs | LLM-as-judge rubric (1–5) per skill | B | ✓ |
| Scope | router + agent prompts | behavioral probes; out-of-scope must not fabricate | A + B | ✓ |
| Leakage — protected | any output channel (chat, blog, SSE trace, logs) | regex + judge: keys/secrets/internal config | A + B | ✓ live scan |
| Leakage — PII | same | regex + judge: emails/phones/SSNs | A + B | ✓ live scan |
| Toxicity | user-facing output | LLM-judge toxicity (lower = safer) | B | ✓ |
| Latency e2e p95 | `/chat` whole stream | harness timers (offline) + live monitors | B | ✓ |
| Latency TTFT | SSE first token | harness + live observation | — | ✓ |
| Cost per query | all LLM calls | `usage_metadata` / tiktoken | B | ✓ |
| Success rate / error rate | `/chat`, `/upload`, `/review` | no-`[ERROR]`/no-exception fraction | B | ✓ |

---

## 5. Repository layout

```
evals/
├── README.md              # how to run offline vs online, quota budget table
├── config.yaml            # per-metric thresholds (router_acc ≥ 0.90, …)
├── datasets/              # JSONL golden datasets (the core artifact)
│   ├── router.jsonl           # 4-way route labels + expected route (accuracy/F1)
│   ├── retriever.jsonl        # queries + golden relevant_text anchors + materialized relevant_chunk_ids
│   ├── rag_qa.jsonl           # 25 doc-QA pairs → pipeline triad
│   ├── generator.jsonl        # 20 {query, context, reference} → faithfulness/relevancy
│   ├── application.jsonl      # 20 queries + G-Eval rubrics per skill
│   ├── blog.jsonl             # schema + factual rubric for blog writer
│   ├── memory.jsonl           # LTM fact extraction / STM summary checks
│   └── safety.jsonl           # scope, injection, leakage (protected/PII), toxicity probes
├── harness/
│   ├── __init__.py
│   ├── mocks.py            # deterministic fake LLM + recorded ToolMessages
│   ├── metrics.py          # accuracy, per-class F1, P@k/MRR, rubric aggregation
│   ├── llm_judge.py        # judge prompts (DeepEval + G-Eval + safety rubrics)
│   ├── report.py           # evals/results/<run>.json + Markdown summaries
│   └── run_offline.py      # CLI: Tier A (pytest) / Tier B (batch) orchestration
├── online/
│   ├── __init__.py
│   ├── langsmith_runner.py # replaces the dummy target in tests/test_eval.py
│   ├── api_blackbox.py     # live /chat, /upload, /review suite
│   └── monitor.py          # fallback-rate, error-rate, latency, cost, drift
└── results/                # generated reports (gitignored)
```

### Dataset schema (JSONL)

```jsonc
// router.jsonl
{"input": "…", "expected": "research|analysis|chat|blog", "reason": "…", "kind": "router"}

// retriever.jsonl
// relevant_text = golden anchor substrings (authoritative ground truth);
// relevant_chunk_ids = materialized 0-based chunk indexes derived from the
// anchors by the Tier A harness (null in the Phase 0 seed).
{"query": "…", "source_doc": "golden_pdf_1", "relevant_text": ["…anchor substring…"], "relevant_chunk_ids": [0, 2], "kind": "retriever"}

// rag_qa.jsonl
{"query": "…", "thread": "eval-rag-1", "reference": "…", "expected_facts": ["…"], "kind": "rag_qa"}

// generator.jsonl
{"query": "…", "context": "…", "reference": "…", "kind": "generator"}

// application.jsonl
{"query": "…", "route": "research|analysis|chat|blog|synthesizer", "rubric": {"correctness": "…", "completeness": "…", "style": "…"}, "kind": "application"}
// Self-contained contract: rows that assume prior context embed it inline in `query`
// ("Agent output: …" for synthesizer rows, " Prior answer: …" for chat follow-ups) so
// every row is evaluable in isolation. route == "synthesizer" rows must be fed through
// synthesizer_node with a prebuilt AgentState, NOT through a full graph run.

// safety.jsonl   (kind is the file marker; "probe" carries the probe type)
{"input": "…", "probe": "scope|injection|leakage_protected|leakage_pii|toxicity", "kind": "safety", "expected": {"behavior": "refuse|mask|no_leak|non_toxic"}}
```

---

## 6. Phases

### Phase 0 — Foundations (complete — audited 2026-09-14: 8 JSONL datasets / 121 rows, corpus grounded, validator + ruff green; datasets expanded 2026-10-03: rag_qa 25 rows, generator 20 rows, application 20 rows)

- `evals/` package (ruff-checked in CI): `datasets/`, `harness/`, `online/`, `results/`, `config.yaml`, `README.md`, `results/` gitignored.
- Golden datasets, one per metric family (JSONL above).
- Golden retrieval corpus: extend `tests/sample.pdf` with 2–3 curated docs so `retriever.jsonl` has stable, labeled chunks.
- Dependency decision (2026-09-14): **adopt DeepEval (`deepeval>=4.2,<5`, isolated in `requirements-eval.txt`)** for the offline Tier-B judge math — first-class G-Eval, full RAG triad (contextual recall/precision, faithfulness, answer relevancy), and pytest-style ergonomics. Judges run on **Groq via an OpenAI-compatible endpoint** (no OpenAI key needed); Gemini as fallback. **RAGAS was rejected**: `ragas 0.4.3` installs but fails `import` against this stack's `langchain_community 0.4.2` / `langchain_core 1.4.8` (`ChatVertexAI` removed in the 0.4.x sunset). **Verified against PyPI (2026-09-14)**: deepeval 4.2.2 has zero langchain/litellm dependencies (own LLM layer over the openai SDK — the exact failure point that killed ragas), and caps `rich<15` + adds pytest-xdist/rerunfailures/repeat → install in a **fresh venv only**, never the project's main venv. **Entry gate for Phase 2**: a spike installs deepeval in a fresh venv and scores ~3 rows from each of rag_qa/generator/application/safety against Groq as judge; if the spike fails, fall back to hand-rolled judge prompts in `harness/llm_judge.py` — `config.yaml` thresholds are identical either way.

### Phase 1 — Offline Tier A (complete — audited 2026-09-15: 21 Tier A tests pass, zero LLM/network, 100% anchor materialization; retriever quality gates deferred to Tier B, see note; gaps fixed 2026-10-03: test_router.py formally Tier A marked + 14 new parametrize cases incl. blog label + boundary; 8 new should_compress / escape_untrusted / synthesizer edge-case tests added; CI hardened with explicit tier_a step, ruff covers evals/harness/, coverage floor raised to 55%)

- Retriever (structural gate — corrected at audit): with the deterministic fake embeddings, FAISS top-k ranking is effectively arbitrary (measured mean recall 0.29 / precision 0.08 — randomness, not quality), so **Tier A does NOT gate recall/precision**. Tier A gates: (1) 100% anchor → chunk materialization — every `relevant_text` anchor resolves to exactly one indexed chunk (12/12 rows; 0-hit or ambiguous 2+ hits raise), (2) structural top-k + no-index fallback, (3) materialized chunk IDs written back into `retriever.jsonl` for Tier B reuse. Per-row recall/precision are computed and saved to `evals/results/tier_a_retrieval.json` as informational only. The `config.yaml` 0.85/0.80 gates are enforced in Tier B with real embeddings.
- Safety deterministic gates: PII regex scanners, protected-string scanners, `<<UNTRUSTED …>>` boundary behavior (`escape_untrusted` cannot be broken by crafted `<<END USER INPUT>>` tokens), scope-refusal phrase detector. Behavioral refusal / masking is Tier B.
- Fold existing coverage into the regression gate so it can't silently drift: router label parsing (`tests/test_router.py`), synthesizer `Sources:` rules + delimiter escaping, blog `_parse_blog_json`/`_blog_to_markdown` schema, graph topology (node set + route-map keys), memory `should_compress`/`build_stm_prefix`.

### Phase 2 — Offline Tier B (real models, batch harness → quality baselines)

Sized to run within the free-tier quota (~one full run/day).
1. **Retriever** — DeepEval contextual recall + precision with real gemini embeddings.
2. **Generator** — faithfulness + answer relevancy on synth/agent outputs.
3. **Pipeline** — full triad (contextual relevancy, faithfulness, answer relevancy) on the 25 doc-QA journeys executed end-to-end through the compiled graph.
4. **Application G-Eval** — correctness / completeness / style (1–5) per skill.
5. **Safety gates** — scope 100% refuse-without-fabrication, protected/PII leakage = 0, toxicity median ≤ 0.05 (lower = safer).
6. **Operational** — harness records latency e2e p95, TTFT, cost/query (`usage_metadata`), success/error rate.
7. **Report** — `evals/results/<date>.json` + Markdown; `--compare` diffs vs baseline.

### Phase 3 — CI gates

- Tier A on every push/PR — **blocks** merge.
- Tier B scheduled nightly (or `workflow_dispatch`) using existing `GROQ_API_KEY` / `TAVILY_API_KEY` / `GOOGLE_API_KEY` secrets — report-only first, then gated on router accuracy + triad after baselines stabilize.

### Phase 4 — Online evals

1. **LangSmith rework** — replace the dummy `tests/test_eval.py` dataset/target with real `evals/datasets/*.jsonl`; targets = the deployed graph/route agents; same judge set.
2. **Live black-box suite** — `/chat` SSE contract (tokens → `[FINAL]` → `[DONE]`, no `[ERROR]`), TTFT + e2e p95 on the live endpoint, `/upload` → RAG round-trip, `/review` interrupt, auth. Scheduled via GH Actions cron.
3. **Operational monitors** — rolling success/error rates, timeout/`[FALLBACK]` rate, p95 latency per route, cost/query — from structlog + LangSmith traces.
4. **Safety live scan** — PII/protected-leak regex on sampled live outputs.
5. **Feedback loop** — thumbs up/down per message + `POST /feedback` as human ground truth to validate online judge scores (small frontend + backend feature).

---

## 7. Proposed initial thresholds (`evals/config.yaml`)

| Metric | Target |
|---|---|
| Router accuracy | ≥ 0.90 (baseline first, then enforced) |
| Contextual recall | ≥ 0.85 |
| Contextual precision | ≥ 0.80 |
| Faithfulness | ≥ 0.90 |
| Answer relevancy | ≥ 0.90 |
| Contextual relevancy | ≥ 0.80 |
| G-Eval composite (1–5) | ≥ 4.0 |
| Scope refusals | 100% of probes |
| Leakage — protected / PII | 0 hits |
| Toxicity | median ≤ 0.05, max ≤ 0.2 (lower = safer) |
| Latency e2e p95 | ≤ 5 s chat, ≤ 15 s research (PRD §6) |
| TTFT | < 1.5 s (info) |
| Reliability success rate | ≥ 0.95 |
| Reliability error rate | ≤ 0.05 |

---

## 8. Deliverables checklist

- [x] `EVAL_PLAN.md` — this document
- [x] `evals/` package + `config.yaml` + `README.md` (quota budget table)
- [x] Golden datasets seed (8 files, 121 rows) + golden retrieval corpus (3 PDFs, `build_corpus.py` deterministic — regenerates every run)
- [x] Tier A deterministic suite, green in CI, blocks merge (21 base tests + expanded router/format/security edge cases; explicit tier_a CI step; ruff covers evals/; coverage floor 55%; audited 2026-09-15, gaps fixed 2026-10-03)
- [x] Tier B batch harness + baselines + reports (Session 5: all 8 subcommands in `evals/harness/run_offline.py` — router, retriever, generator, rag-qa, application, safety, memory, blog; `evals/results/baseline.json` + `--compare` per-metric regression diff)
- [x] DeepEval/G-Eval judge wiring on Groq + Gemini fallback (Session 1 decision: FALLBACK hand-rolled judges in `evals/harness/llm_judge.py`, pinned `openai/gpt-oss-20b` on Groq; covers faithfulness, answer/contextual relevancy, G-Eval 1–5, toxicity — the DeepEval spike was not needed because the fallback covers every judge)
- [ ] CI job for Tier A (push/PR) and Tier B (nightly / workflow_dispatch)
- [ ] LangSmith online runner replacing the dummy eval
- [ ] Live black-box API suite (SSE contract, TTFT, p95, RAG round-trip, review, auth)
- [ ] Online monitors (success/error, fallback, timeout, latency, cost)
- [ ] Feedback signal (thumbs up/down + `POST /feedback`)

---

## 9. Acceptance criteria

1. All offline Tier A tests pass in CI with zero API cost.
2. Offline Tier B produces a saved baseline report; router accuracy and the retrieval/generation triad meet the thresholds in §7 (or an explicitly documented baseline adjustment).
3. Any merge that regresses Tier A fails CI.
4. Online LangSmith experiments run against real datasets on the deployed stack and publish results.
5. Live black-box suite passes end-to-end against Render/Vercel within the PRD §6 SLA.
6. Online monitors emit rolling success/error, latency p95, TTFT, and cost/query metrics.

---

## 10. Risks & notes

- **Free-tier quotas** — Groq 100 K tokens/day caps full Tier B runs to ~1/day. Tavily 1 K searches/mo caps blog/research evals; reuse cached search outputs where possible.
- **LangSmith thread leak** — CI comment notes the dummy eval spawns a background LangSmith/HTTP thread; online runs must be kept behind an explicit flag so normal CI stays clean.
- **No feedback infra today** — the thumbs up/down signal is a small feature gap that online evals depend on for human ground truth.
- **Judges are LLMs too** — judge prompt stability matters; version judge prompts and pin judge models in `config.yaml` so score drift ≠ behavioral drift.
- **Knowledge cutoffs** — research-agent evals that depend on live web results should keep evaluator criteria invariant to the actual facts found (cite-source presence, tool usage) rather than comparing to a fixed expected answer.