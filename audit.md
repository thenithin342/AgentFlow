# AgentFlow — Complete Technical Audit
### Senior Engineer / ML Intern / Open Source Evaluation

---

## Phase 1 — Project Understanding

### What the Project Does

**AgentFlow** is a full-stack, production-grade **multi-agent AI assistant** built on LangGraph. It receives user messages, classifies intent via an LLM router, dispatches to one of four specialized agents (Research, Analysis, Chat, Blog Writer), optionally synthesizes and refines the response with a second LLM pass, and streams tokens to the browser in real time over SSE.

### Main Objective

Demonstrate every key skill that distinguishes a production AI engineer: stateful graph orchestration, multi-agent coordination, RAG, human-in-the-loop, durable persistence, streaming, authentication, and deployment — all wired together in a single coherent system.

### Technologies Used

| Layer | Technology |
|---|---|
| **Graph / Orchestration** | LangGraph 1.x (`StateGraph`, `interrupt`, `AsyncSqliteSaver`, `AsyncPostgresSaver`) |
| **LLM** | Groq (llama-3.3-70b + llama-3.1-8b), Google Gemini 2.0 Flash (fallback) |
| **RAG** | FAISS + FastEmbed (`BAAI/bge-small-en-v1.5`) + PyPDF |
| **Memory** | Per-user FAISS LTM store + rolling STM compression |
| **Backend** | FastAPI + Uvicorn/Gunicorn + aiosqlite + structlog + slowapi |
| **Auth** | JWT (HS256, PyJWT) + bcrypt + static API key fallback |
| **Frontend** | React 18 + Vite + react-markdown + rehype-sanitize |
| **Deployment** | Docker (multi-stage, tini, non-root), Render, Railway, Vercel (frontend) |
| **CI** | GitHub Actions (pytest) |
| **Observability** | LangSmith tracing, structured JSON logs, `/healthz` + `/readyz` probes |

### Overall Architecture

```
Browser (React + Vite)
    │ JWT Bearer  │ SSE stream
    ▼             ▼
FastAPI (main.py)
    │  JWT auth  │  rate-limit  │  CORS
    ▼
LangGraph StateGraph (async, AsyncSqliteSaver / AsyncPostgresSaver)
    │
    ├─► memory_reader_node   ← FAISS LTM per-user
    │
    ├─► router_node          ← llama-3.3-70b few-shot classifier
    │
    ├─► research_agent_node  ← ReAct: tavily + wikipedia + url_reader + RAG
    ├─► analysis_agent_node  ← ReAct: calculator + code_interpreter + RAG
    ├─► chat_agent_node      ← ReAct: RAG only (fast path)
    └─► blog_writer_node     ← ReAct: tavily + RAG → structured JSON blog
            │
            ▼
        synthesizer_node     ← llama-3.3-70b polish + citations
            │
            ▼
        human_review_node    ← interrupt() gate (optional)
            │
            ├─► stm_compressor_node  ← rolling context compression
            └─► memory_writer_node   ← FAISS LTM write
                    │
                    ▼
                END → SSE [DONE] → browser
```

### Data Flow

1. User POSTs `/chat` → JWT validated → `_config_for(user, thread_id)` scopes the LangGraph config
2. `graph.astream_events()` is called; FastAPI streams `on_chat_model_stream` events as SSE tokens to the browser
3. Internally: `memory_reader` injects LTM context → `router` classifies → agent runs ReAct loop → `synthesizer` polishes → `human_review` optionally pauses → `memory_writer` extracts and stores facts
4. After the stream ends, the API reads the final snapshot and emits `[FINAL:…]`, `[SOURCES:N]`, `[DONE]`

### Strengths

- **LangGraph 1.x mastery** — correct use of `interrupt()`, `Command(resume=…)`, `astream_events(v2)`, conditional edges, per-thread checkpointing
- **Security depth** — SSRF-protected `url_reader` (DNS pinning + redirect limits), HMAC-signed FAISS pickles, SHA-256 hashed index directories, prompt-injection barriers (`<<UNTRUSTED>>` markers)
- **Production signals** — dual health probes, structured JSON logs, rate limiting (slowapi), non-root Docker user, tini PID-1
- **Resilience** — triple Groq key fallback + Gemini last-resort, router falls back to `chat` on exception
- **Memory system** — both STM (rolling compression) and LTM (per-user FAISS), a level of sophistication most projects skip entirely
- **Testing surface** — 11 test files covering API shape contracts, graph integration, memory, tools, router
- **Documentation** — comprehensive README, DESIGN_DOC, PRD, TECH_STACK, DEPLOYMENT docs

### Weaknesses

- `main.py` is 1082 lines — a monolith that should be decomposed into routers
- User store is a flat JSON file — not scalable, no locking under concurrent writes
- No integration tests with a real LLM (all mocked)
- No frontend unit/component tests whatsoever
- CI pipeline is minimal (no linting, no type-check, no coverage, no caching)
- LTM eviction uses FAISS docstore private `_dict` attribute (fragile internals)
- `code_interpreter` uses `multiprocessing` — broken on some cloud platforms (no `fork`)
- No token-budget enforcement (users can send very long conversations)
- `agentflow.db` committed to the repo (contains real checkpoint data)
- No refresh tokens — 24h JWT expires without a graceful re-login flow

---

## Phase 2 — Code Review

### Issue 1: Monolithic `main.py` (1082 lines)

**Problem:** All route handlers, middleware, helpers, and data-access logic live in one file. Finding, testing, or extending any single endpoint requires navigating a 1000-line file.

**Why it matters:** Violates the Single Responsibility Principle. Makes PR reviews slow. A bug in one helper can silently affect unrelated endpoints.

**How to fix:** Decompose into FastAPI `APIRouter` modules:
```
backend/
  routers/
    auth.py       ← /auth/login
    chat.py       ← /chat, /review/{id}
    threads.py    ← /threads, /threads/{id}/state, /threads/{id}/history
    upload.py     ← /upload
    health.py     ← /healthz, /readyz
  dependencies.py ← shared Depends (require_user, get_graph)
```

**Priority:** High

---

### Issue 2: JSON File User Store (auth.py)

**Problem:** `data/users.json` is read and written with no inter-process locking. On a multi-worker Gunicorn deploy two workers can race and corrupt the file.

**Why it matters:** Data loss / authentication failure in production. Even SQLite would be an improvement.

**How to fix:**
```python
# Immediate: use a filelock
import filelock

def _save_users(settings, users):
    path = _users_file(settings)
    with filelock.FileLock(str(path) + ".lock"):
        path.write_text(json.dumps(payload, indent=2))
```

Or migrate to a proper user table in the existing SQLite/Postgres checkpointer DB.

**Priority:** Critical (for multi-worker deploys)

---

### Issue 3: LTM Accesses FAISS Private Internal (`docstore._dict`)

**Problem:** `ltm.py` line 192: `all_docs = list(index.docstore._dict.values())` — this is a private attribute of `InMemoryDocstore`. It will break on any langchain-community version that renames it.

**How to fix:**
```python
# Use the public API
ids = list(index.index_to_docstore_id.values())
all_docs = [index.docstore.search(i) for i in ids]
```

**Priority:** High

---

### Issue 4: `code_interpreter` Uses `multiprocessing` — Broken on Some Platforms

**Problem:** `multiprocessing.Process` uses `fork` by default on Linux but `spawn` on macOS and Windows. On Railway/Render (Linux) `fork` inside a running asyncio event loop is dangerous and can cause hangs. On macOS it errors silently.

**Why it matters:** Production code interpreter silently fails on some cloud platforms.

**How to fix:** Explicitly set `multiprocessing.set_start_method("spawn")` at startup, or replace with a subprocess-based isolation using `concurrent.futures.ProcessPoolExecutor` with `mp_context=multiprocessing.get_context("spawn")`.

**Priority:** High

---

### Issue 5: Duplicate URL-Extraction Logic

**Problem:** `_extract_sources` is defined twice — identically in `agents.py` (lines 58-73) and `blog_agent.py` (lines 96-105). Same regex, same logic.

**How to fix:** Move to `backend/graph/messages.py` or a shared `utils.py` and import in both places.

**Priority:** Medium

---

### Issue 6: `_collect_groq_keys()` duplicated between `llm.py` and the settings system

**Problem:** `llm.py` reads Groq keys directly from `os.environ` via `_collect_groq_keys()` instead of going through the already-typed `settings` object. This bypasses `pydantic-settings` validation and makes testing harder.

**How to fix:**
```python
def _collect_groq_keys(settings=None) -> list[str]:
    if settings is None:
        settings = get_settings()
    return [k for k in [settings.groq_api_key, settings.groq_api_key_2,
                         settings.groq_api_key_3] if k and k.strip()]
```

**Priority:** Medium

---

### Issue 7: `build_graph.py` Module-Level Side Effect (DB Connection at Import)

**Problem:** `_DEFAULT_DB_PATH` is read at import time and `get_default_graph()` lazily opens a SQLite connection. If a test imports `build_graph` before setting the env var, it silently uses the production DB path.

**How to fix:** The lazy init already handles this reasonably, but the module-level `graph = _GraphProxy()` is created at import and the test suite patches around it. A cleaner pattern: make `get_default_graph()` accept settings injection.

**Priority:** Low

---

### Issue 8: Frontend — `App.jsx` is 2350 Lines (87KB)

**Problem:** The entire frontend lives in a single component file. This mixes state management, API calls, rendering, SSE parsing, authentication, routing, and UI logic.

**Why it matters:** Impossible to test individual components. Any change risks a regression anywhere in the file. Extremely slow to onboard contributors.

**How to fix:** Split into proper component modules:
```
src/
  components/
    Chat/
      ChatInput.jsx
      MessageList.jsx
      MessageBubble.jsx
    Sidebar/
      ThreadList.jsx
      ThreadItem.jsx
    Blog/
      BlogViewer.jsx
    Review/
      ReviewPanel.jsx
  hooks/
    useSSE.js
    useChat.js
    useThreads.js
  api/
    client.js
  store/
    chatStore.js
```

**Priority:** High

---

### Issue 9: No Input Token Budget / Context Window Guard

**Problem:** There is no enforcement of how many tokens accumulate in `state["messages"]` before being sent to the LLM. A long conversation can silently push over the model's context window (8k for llama-3.1-8b-instant), causing a 400 from Groq and a 500 to the user.

**How to fix:** Implement a token estimator before each LLM call and trim or summarize messages if the budget is exceeded. STM compression already exists but only fires every N turns — it doesn't guard against a single large message burst.

**Priority:** High

---

### Issue 10: `ensure_admin()` Uses `logging` (stdlib) Instead of `structlog`

**Problem:** `auth.py` imports and uses `logging.getLogger()` directly instead of the configured structlog logger used everywhere else. Log format and routing differ from the rest of the app.

**Priority:** Low

---

## Phase 3 — Architecture Review

### Current Architecture Assessment

The architecture is **sound and well-conceived** for a portfolio / startup MVP. The choice of LangGraph StateGraph is appropriate — it gives you durable state, conditional routing, and human-in-the-loop "for free" compared to rolling your own. The separation of concerns across modules (graph/agents, graph/tools, memory/ltm, rag/ingest) is clean.

### What's Good

- Clear node responsibility: each node has exactly one job
- Checkpointer abstraction works: same graph topology compiles against SQLite (dev) or Postgres (prod)
- The security module (`HMAC-signed pickles`, `SSRF-hardened url_reader`) shows production thinking

### What Needs Work

```
Current:
Browser
  ↓ SSE/HTTP
FastAPI (1 monolithic main.py)
  ↓
LangGraph (sqlite file / postgres)
  ↓
Tools (FAISS local disk, multiprocessing sandbox)
  ↓
LLMs (Groq API → Gemini fallback)

Issues:
- main.py = god module
- User store = flat JSON (not DB-backed)
- FAISS on local disk = no horizontal scaling
- code_interpreter = no container isolation
```

```
Recommended:
Browser
  ↓ SSE/HTTP
FastAPI (decomposed routers + shared deps)
  ↓
LangGraph (Postgres checkpointer)
  ↓
┌───────────────────────────────────────────┐
│  Tool Services                             │
│  - RAG: Qdrant / Pinecone (not local disk) │
│  - Code: Docker exec or E2B sandbox        │
│  - Search: Tavily (keep as-is)             │
└───────────────────────────────────────────┘
  ↓
LLMs (Groq → Gemini → OpenAI fallback)
```

### Scalability Gap

The current design cannot scale horizontally because:
1. FAISS indexes live on local disk — Worker B cannot read Worker A's index
2. LTM FAISS also on disk — same problem
3. `_AGENT_CACHE` is process-local — compiled agents must be rebuilt per process
4. User store is a JSON file — race conditions under concurrent writers

**Production path:** Replace FAISS with Qdrant (self-hosted) or Pinecone (managed). Replace JSON user store with a Postgres `users` table. Use a shared Postgres checkpointer.

---

## Phase 4 — Is This Project Resume Worthy?

### Scores (1–10)

| Dimension | Score | Reasoning |
|---|---|---|
| **Technical Depth** | 8/10 | LangGraph 1.x mastery, dual memory system, HMAC FAISS, SSRF mitigation — genuinely non-trivial |
| **Originality** | 7/10 | Multi-agent + STM/LTM combination is uncommon in portfolios; blog agent adds differentiation |
| **ML/AI Complexity** | 8/10 | ReAct agents, RAG pipeline, STM compression, LTM fact extraction, multi-model fallback chain |
| **Software Engineering Quality** | 7/10 | Strong backend, typed settings, structlog — dragged down by monolithic main.py and App.jsx |
| **Production Readiness** | 6/10 | Docker, health probes, rate limiting, JWT — but JSON user store and local FAISS cap it |
| **Resume Value** | 9/10 | Hits almost every LLM engineering buzzword with real implementations, not just wrappers |
| **Interview Value** | 9/10 | Every component is an interview question generator |
| **Overall** | **7.7/10** | Strong — would pass the resume screen at most top AI companies |

### Would this impress...

| Audience | Verdict |
|---|---|
| **Recruiters** | ✅ Yes — keywords: LangGraph, RAG, multi-agent, streaming, FastAPI, Docker |
| **Hiring Managers** | ✅ Yes — they'll see you know the full stack, not just the prompt |
| **ML Engineer interviews** | ✅ Yes — RAG, embeddings, retrieval, memory are all here |
| **LLM Engineer interviews** | ✅ Yes — LangGraph internals, astream_events, interrupt/resume, prompt injection defenses |
| **Software Engineer interviews** | ⚠️ Conditional — would impress on depth but raise flags on the 1000-line main.py and 87KB App.jsx |

---

## Phase 5 — Missing Features

### Priority Order

| Priority | Feature | Gap |
|---|---|---|
| 🔴 Critical | **Horizontal scaling for FAISS** | Local disk indexes block multi-worker/multi-node deploys |
| 🔴 Critical | **Proper user DB** | JSON file is not safe under concurrent writes |
| 🟠 High | **Refresh tokens** | JWT expires after 24h; no silent refresh flow exists |
| 🟠 High | **Frontend component split** | App.jsx (87KB) has no tests and is unnavigable |
| 🟠 High | **Token budget enforcement** | No guard against context window overflow |
| 🟠 High | **Frontend tests** | Zero component or hook tests |
| 🟡 Medium | **User management API** | No endpoint to create/list/delete users (admin only) |
| 🟡 Medium | **LangSmith evaluation dataset** | Great for demonstrating eval discipline |
| 🟡 Medium | **Prometheus metrics endpoint** | `/metrics` for Grafana integration |
| 🟡 Medium | **Docker Compose for full-stack local dev** | Frontend container missing from compose |
| 🟡 Medium | **Coverage report in CI** | No `--cov` flag, no badge |
| 🟡 Medium | **Linting / type-check in CI** | No `ruff`, `mypy`, or `eslint` in CI |
| 🟢 Low | **Dark mode on frontend** | Already has CSS vars; just needs a toggle |
| 🟢 Low | **CONTRIBUTING.md** | Missing |
| 🟢 Low | **Streaming for blog agent** | Blog agent does not stream tokens (blocks until done) |
| 🟢 Low | **PDF multi-file support** | Only one PDF accumulation per thread; no per-document delete |

---

## Phase 6 — Production Readiness (1 Million Users)

### What Would Break First

1. **Local FAISS indexes** — the very first horizontal pod would fail to find the index written by pod 1. Must move to Qdrant/Pinecone or a shared NFS volume (NFS is still a single point of failure).

2. **SQLite checkpointer** — file-locked; single writer. Switch to Postgres immediately (the code already supports it via `POSTGRES_CONN_STRING`).

3. **JSON user store** — race condition under concurrent logins. Migrate to a `users` table in the existing Postgres instance.

4. **Sync agent nodes blocking the event loop** — all four agent nodes (research, analysis, chat, blog) are marked `SYNC on purpose`. At 1M users this becomes a bottleneck: every `asyncio.to_thread` call consumes a thread-pool slot. Consider rewriting agents as async with `arun` instead of `invoke`.

5. **`multiprocessing` code interpreter** — spawning a process per code-interpreter invocation at scale is expensive. Replace with a persistent pool (e.g., `concurrent.futures.ProcessPoolExecutor` with a max) or the E2B cloud sandbox API.

### Production Architecture at Scale

```
┌─────────────────────────────────────────────────────────────┐
│ CDN (Cloudflare)                                             │
└─────────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────┐
│ Load Balancer (nginx / ALB)  │
└──────────────────────────────┘
            ↓
┌────────────────────────────────────────────┐
│ FastAPI pods (K8s, 4+ replicas)            │
│ - AsyncPostgresSaver (shared checkpoints)  │
│ - JWT validation stateless                 │
└────────────────────────────────────────────┘
            ↓
┌──────────────────────────┐    ┌────────────────────────────┐
│ Postgres (primary +      │    │ Qdrant / Pinecone           │
│ read replicas)           │    │ (vector store, all indexes)│
│ - checkpoints table      │    └────────────────────────────┘
│ - users table            │
└──────────────────────────┘
            ↓
┌────────────────────────────────┐
│ Groq API (rate: 3-key pool)    │
│ → Gemini fallback              │
└────────────────────────────────┘
            ↓
┌────────────────────────────┐   ┌───────────────────────────┐
│ Prometheus + Grafana       │   │ Sentry (error reporting)   │
│ (LangSmith for LLM traces) │   └───────────────────────────┘
└────────────────────────────┘
```

### Specific Recommendations

| Area | Recommendation |
|---|---|
| **Caching** | Add Redis for router decision caching (identical messages within a thread often get same route) |
| **Rate limiting** | Move slowapi rate limit state to Redis (currently in-process, resets on restart) |
| **Background jobs** | Move LTM fact extraction to a Celery/ARQ background task (it adds latency to every turn) |
| **Streaming** | Add a WebSocket upgrade path as an alternative to SSE for bidirectional control |
| **CDN** | Serve the React build from a CDN; API only handles compute |

---

## Phase 7 — Interview Preparation Guide

### Elevator Pitch (30 seconds)

> "AgentFlow is a production multi-agent AI assistant. It uses LangGraph to route user queries to specialized agents — Research, Analysis, Chat, and Blog Writer — each running a ReAct loop with tools like web search and PDF retrieval. It streams token-by-token to the browser, supports human-in-the-loop review, and maintains both short-term and long-term memory across sessions. The whole thing is deployed with Docker and FastAPI."

---

### Two-Minute Recruiter Explanation

AgentFlow solves the problem of a one-size-fits-all chatbot by routing every user message to the most appropriate specialist. When you ask a research question, a dedicated Research Agent searches the web and your uploaded documents. When you upload a PDF and ask to analyze it, the Analysis Agent runs a sandboxed Python calculator and semantic search over your document. For casual conversation, a fast Chat Agent responds directly without the overhead of web search. And if you need a blog post written, a Blog Writer researches the topic and produces structured, SEO-ready content.

All of this is orchestrated by LangGraph — a state machine framework that persists every step to a database, so a conversation can be paused, resumed, or replayed at any point. I added JWT authentication, real-time token streaming, rate limiting, and Docker packaging because those are the difference between a demo and a product.

---

### Five-Minute Technical Walkthrough

**1. The Router (router.py)**
Every user message goes first to the router node, which calls `llama-3.3-70b-versatile` with a few-shot system prompt. The prompt has 20+ labeled examples covering edge cases like "summarize my PDF" (→ analysis) vs "find the latest papers on X" (→ research). The router returns a single word; the `_parse_label()` function uses regex with word boundaries to extract it safely even if the LLM adds punctuation. On any LLM error, it falls back to "chat" rather than 500-ing.

**2. The Agents (agents.py)**
Each agent is a LangGraph `create_react_agent` — a prebuilt ReAct loop that calls tools until it has enough information to answer. I cache these compiled agents with an LRU-128 `OrderedDict` keyed on `(tool names, model, prompt hash, thread_id)`. Without the `thread_id` in the cache key, a graph compiled for Thread A would be reused for Thread B and the `retrieve_documents` closure would search the wrong FAISS index — a cross-user data leak.

**3. RAG Pipeline (rag/ingest.py)**
PDF → PyPDFLoader → RecursiveCharacterTextSplitter (800 chars, 150 overlap) → FastEmbed BAAI/bge-small-en-v1.5 embeddings (ONNX-based, 80MB vs PyTorch's 350MB) → FAISS index. The index directory is the SHA-256 hash of the scoped thread_id so colons and slashes in the thread namespace can't escape the index root. Every pickle file is HMAC-signed with a 32-byte secret to prevent malicious deserialization attacks.

**4. Memory System (memory/ltm.py, memory/stm.py)**
Two tiers: Short-term memory compresses the oldest messages into a summary every N turns so the context window doesn't overflow. Long-term memory extracts memorable facts from each turn ("user is a ML engineer interested in transformers") via an LLM call, stores them as FAISS embeddings in a per-user index, and retrieves the top-k most relevant facts at the start of each new conversation.

**5. Human-in-the-Loop (graph/human_review.py)**
LangGraph's `interrupt()` is a cooperative yield — it snapshots the graph state and raises `GraphInterrupt`. The frontend receives `[INTERRUPT]` in the SSE stream, shows the draft, and lets the user approve or edit. On submission, the API calls `graph.ainvoke(Command(resume=<value>), config)` which injects the value into the paused `interrupt()` call and the graph continues from exactly where it stopped.

**6. Streaming (main.py)**
`graph.astream_events(version="v2")` emits fine-grained events for every LLM token, tool call start/end, and node transition. I filter these to `on_chat_model_stream` events from the visible nodes and format them as SSE `data:` lines. Control events (`[NODE_START]`, `[TOOL_START]`, `[DONE]`) are JSON-free text markers that the frontend parses with a simple prefix switch.

---

### Architecture Explanation

**Why LangGraph over raw LangChain?** LangGraph gives you a proper state machine with durable persistence. Without it, you'd need to manage checkpointing, interrupt/resume, and state merging yourself — that's months of work. The checkpointer abstraction also lets you swap SQLite for Postgres without changing any business logic.

**Why Groq?** The llama-3.1-8b-instant model on Groq achieves ~800 tokens/second — fast enough for real-time streaming to feel snappy. The 70b model is slower but smarter, used only for routing and synthesis where quality matters more than speed.

**Why FAISS over Pinecone?** For a portfolio project: zero cost, no external dependency, runs fully offline. For production: I'd switch to Qdrant or Pinecone because local FAISS doesn't survive horizontal scaling — covered in the design doc.

**Why FastEmbed over sentence-transformers?** FastEmbed uses ONNX Runtime, not PyTorch. That cuts the container from ~350MB to ~80MB, which is the difference between fitting in Render's free tier and being rejected.

---

### Tradeoffs

| Decision | Why | Alternative |
|---|---|---|
| SQLite default | Zero config for local dev | Postgres (supported via env var) |
| Sync agent nodes | pytest drives graph synchronously; async in FastAPI via `to_thread` | Full async agent nodes (future work) |
| JSON user store | Simple, no extra DB schema | Postgres users table (better for prod) |
| FAISS local disk | Free, offline, fast | Qdrant/Pinecone (horizontal scale) |
| JWT with 24h TTL | Simple, stateless | Shorter TTL + refresh tokens |
| llm_fast for routing | Cheaper, fast | llm_smart gives better routing accuracy at higher cost |

---

## Phase 8 — Interview Questions

### Easy

**Q: What is LangGraph and why did you use it?**
A: LangGraph is a stateful graph orchestration framework built on LangChain. I used it because it provides durable checkpointing (every node transition is persisted), conditional routing between nodes, and a native `interrupt()` mechanism for human-in-the-loop workflows. Building those from scratch would take weeks.

Follow-up: *How does it differ from vanilla LangChain LCEL?*

---

**Q: What is RAG and how is it implemented here?**
A: Retrieval-Augmented Generation — instead of relying on the LLM's static training data, we retrieve relevant context from an external knowledge base first. Here: PDF → chunk → embed → FAISS index. At query time, embed the user's question, find the top-4 similar chunks, inject them into the agent's context window.

Follow-up: *What chunk size did you choose and why?*

---

**Q: How does authentication work?**
A: JWT HS256. On `/auth/login`, we bcrypt-verify the password against the stored hash and issue a signed token with a 24h TTL. Every subsequent request carries `Authorization: Bearer <token>`. The `require_user` FastAPI dependency verifies the signature and extracts the username. There's also a static API key fallback for service accounts.

---

### Medium

**Q: Explain the agent caching mechanism and why thread_id is in the cache key.**
A: `create_react_agent` compiles a new `CompiledStateGraph` — it's expensive. I cache them in an LRU `OrderedDict` keyed on `(tool names, model, prompt hash, thread_id)`. The thread_id is critical because the `retrieve_documents` tool is a closure over a specific FAISS index directory (keyed by thread_id). Without it in the cache key, a cached agent compiled for Thread A would search Thread A's documents when invoked for Thread B — a data leak.

---

**Q: What is the STM/LTM memory system?**
A: Two-tier: **Short-term memory (STM)** compresses the oldest messages every N turns into a summary SystemMessage to prevent context window overflow. It keeps the `STM_KEEP_RECENT` most recent messages verbatim and replaces older ones with the summary. **Long-term memory (LTM)** uses an LLM to extract memorable facts from each turn ("user is building a SaaS startup"), stores them as FAISS vector embeddings per user, and retrieves the top-k most relevant facts at the start of each new conversation to personalize responses.

---

**Q: How does human-in-the-loop work technically?**
A: LangGraph's `interrupt()` is called inside `human_review_node`. It saves the graph state to the checkpointer and raises `GraphInterrupt` — similar to a coroutine `yield`. The API layer catches the resulting `__interrupt__` in the snapshot and sends `[INTERRUPT]` as an SSE event. The frontend shows the draft response with approve/edit buttons. When the user submits, the API calls `graph.ainvoke(Command(resume=<value>), config)`, which resumes from the `interrupt()` call with the provided value, and the graph continues to the memory nodes and END.

---

**Q: What SSRF mitigations are in the `url_reader` tool?**
A: Three layers: (1) DNS resolution followed by validation that the resolved IP is globally routable — prevents SSRF to `169.254.0.0/16` (AWS metadata), `10.0.0.0/8`, etc. (2) DNS pinning — connect to the validated IP directly, not re-resolving, to prevent DNS rebinding attacks. (3) Redirect validation — maximum 3 redirects, only `http`/`https` schemes allowed.

---

### Hard

**Q: Why are agent nodes synchronous even though the API is async? Isn't that a problem?**
A: The pytest suite drives the graph synchronously via `.invoke()`. LangGraph 1.x won't dispatch an async node from a sync driver. The FastAPI server uses `astream_events()` (async), which runs sync nodes in `asyncio.to_thread()` — they execute in the default thread pool, never blocking the event loop. This is the standard LangGraph pattern: sync nodes work everywhere, async nodes only work in async contexts. The tradeoff is thread pool saturation at high concurrency.

---

**Q: How does the dual checkpointer setup avoid a "two connections to the same SQLite file" problem?**
A: During the FastAPI lifespan, an `AsyncSqliteSaver` is compiled into the app's graph and the connection is held open for the process lifetime. The sync `_GraphProxy` (used by tests) lazily creates a separate `SqliteSaver` connection to the same file. SQLite's WAL mode handles concurrent reads but serializes writes. The `_invoke_lock` RLock in `_GraphProxy` serializes all sync `.invoke()` calls so cursor state in the single sync connection never interleaves. The async server and sync tests use completely separate connection objects and are therefore safe.

---

**Q: Walk me through what happens when a user sends a message while a human_review interrupt is pending on their thread.**
A: The `/chat` endpoint calls `graph.astream_events(input_state, config)` where `input_state` has the new `HumanMessage`. LangGraph resumes the graph from the pending interrupt if the config's `thread_id` matches the interrupted state. The `interrupt()` call inside `human_review_node` receives the new message content as its resume value. If that value is the `APPROVE_SENTINEL`, the draft is kept; otherwise the new message string becomes the final response. This is actually a potential UX bug — if the user sends a new message while review is pending, it's interpreted as an edit, not a new conversation turn.

---

### System Design

**Q: Design AgentFlow to handle 100,000 concurrent users.**

Answer:
- Replace local FAISS with Qdrant (self-hosted cluster) or Pinecone — shared vector store across all nodes
- Use Postgres checkpointer (already supported) with a connection pool (pgBouncer)
- Migrate user store to a `users` table in Postgres
- Move rate-limit counters to Redis so state is shared across pods
- Kubernetes: 10+ FastAPI pod replicas behind an ALB, HPA on CPU/memory
- Move LTM fact extraction to an async background task (Celery + Redis broker) — decoupled from the hot request path
- Add a Redis LRU cache for router decisions (same message → same route within N seconds)
- Use CDN for React static assets; API handles only compute

---

### Behavioral

**Q: What was the hardest technical problem you solved in this project?**
Suggested answer: The agent cache cross-thread data leak. Initially I cached React agents by `(tool names, model, prompt)`. When Thread B's request hit the cache and got Thread A's compiled agent, the `retrieve_documents` closure searched Thread A's FAISS index. The result was silent document cross-contamination between users. I caught it in a test by running two threads with different PDFs and verifying retrieval isolation. The fix was including `thread_id` in the cache key.

**Q: How did you decide on the system architecture?**
Emphasize: Started with the simplest thing (single LLM call), added routing when the "do everything" prompt became unreliable, added persistence when I needed conversation history, added memory when I noticed the agent forgetting things from prior sessions.

---

### LLM/ML Specific

**Q: What embedding model did you use and why?**
A: BAAI/bge-small-en-v1.5 via FastEmbed. ONNX-based so it runs without PyTorch — ~80MB resident vs ~350MB for sentence-transformers. Performs comparably to `all-MiniLM-L6-v2` on most English retrieval tasks. Fits in Render's 512MB free-tier container.

**Q: What chunk size and overlap did you choose for RAG and why?**
A: 800 characters, 150 overlap. 800 chars is roughly one dense paragraph — large enough to contain a complete thought, small enough that the top-k=4 retrieved chunks fit comfortably in the agent's 8k context window (4 × 800 = 3200 chars + prompt overhead ≈ 5k tokens). 150 char overlap prevents a sentence split at a chunk boundary from losing its context.

**Q: How would you evaluate retrieval quality?**
A: Build a golden dataset: 20 questions with known answers for a sample PDF. Compute retrieval recall@k (did the right chunk appear in the top-k results?), MRR (mean reciprocal rank), and end-to-end answer accuracy using an LLM-as-judge. Track these metrics in LangSmith as a regression suite.

---

## Phase 9 — README Assessment

The existing README is **above average** — it has badges, a feature table, a quick start, an architecture diagram, and deployment docs. Areas to improve:

- **No GIF/video demo** — recruiters spend 30 seconds; a looping GIF of the streaming UI is worth 500 words
- **No screenshot** — architecture SVG is there but no UI screenshot
- **API docs** — the FastAPI `/docs` page exists but the README doesn't link to it
- **No test badge** — add `[![Tests](https://github.com/…/ci/badge.svg)](…/actions)` once CI is passing consistently
- **"AgentFlow" all over but no deployed demo link** — if it's live anywhere, that link should be at the top

---

## Phase 10 — Presentation Outline

### Slide 1: Problem Statement
Most AI chatbots are single-LLM wrappers: one model, one prompt, one failure mode. They lack specialized reasoning for research vs analysis vs casual conversation, have no memory across sessions, and provide no mechanism for a human to verify AI output before it reaches users.

*Speaker notes: Open with a failure mode — show a generic chatbot confidently hallucinating a statistic.*

### Slide 2: Motivation
AI engineers are differentiated not by knowing how to call an API, but by building systems that are stateful, resilient, observable, and safe. This project demonstrates all four.

### Slide 3: Our Solution — AgentFlow
A graph-based multi-agent system that routes queries to specialist agents, maintains two-tier memory, streams results in real time, and enforces a human-in-the-loop checkpoint before responses reach users.

### Slide 4: Architecture Diagram
*[Show the node diagram from README]*
Walk through: Router → Agent → Synthesizer → Human Review → Memory Write → END

### Slide 5: Technology Stack
| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph 1.x | Durable state, interrupt/resume |
| LLM | Groq (llama-3.x) | 800 tok/s streaming |
| RAG | FAISS + FastEmbed | Zero-cost, offline-capable |
| Backend | FastAPI + Uvicorn | Async streaming native |
| Frontend | React + Vite | Fast build, SSE support |

### Slide 6: Demo Flow
1. Open app → log in → start new thread
2. Ask "What are the latest breakthroughs in quantum computing?" → watch routing → research agent fires → Tavily results stream in → synthesizer polishes
3. Upload PDF → ask "Summarize this document" → analysis agent retrieves chunks → answers with citations
4. Enable review mode → agent produces draft → approve/edit flow demonstrated

### Slide 7: Challenges
- **DNS rebinding in url_reader** — required custom HTTP connection classes
- **Agent cache cross-thread data leak** — discovered via test; fixed with thread_id in cache key
- **FAISS pickle deserialization attack surface** — mitigated with HMAC signatures
- **Sync nodes in async context** — LangGraph pattern: sync for tests, `asyncio.to_thread` in prod

### Slide 8: Results
- Full streaming latency: ~300ms to first token (Groq, 8b model)
- Research agent: 3-5 tool calls, ~5-10s end-to-end
- Blog writer: 2-3 tavily searches, structured JSON output, ~15-20s
- Memory recall: LTM retrieves user preferences from prior sessions with >80% subjective relevance

### Slide 9: Future Work
- Vector DB migration (Qdrant) for horizontal scaling
- OpenAI GPT-4o as additional LLM option
- Voice input integration (Whisper)
- Agent evaluation dataset with LangSmith CI eval
- Multi-tenant isolation with row-level security in Postgres

---

## Phase 11 — Project Report Summary

### Abstract
AgentFlow is a production-grade multi-agent AI assistant implemented as a LangGraph `StateGraph` with four specialized agent nodes, a two-tier memory system, RAG over uploaded documents, and real-time SSE token streaming. The system demonstrates stateful graph orchestration, human-in-the-loop review, durable persistence, and production-readiness features including JWT authentication, rate limiting, and Docker deployment.

### System Design
The graph topology is a directed acyclic graph with one conditional branch point (the router). State is a TypedDict with `add_messages` reducer semantics. The checkpointer (SQLite or Postgres) persists the full state after every node execution, enabling crash recovery, conversation replay, and the `interrupt()`/`Command(resume=...)` pattern for human review.

### Key Algorithms
1. **ReAct Agent Loop**: Think → Act (tool call) → Observe (tool result) → repeat until confident → Answer
2. **LTM Retrieval**: Cosine similarity search in FAISS vector space → top-k most semantically similar past facts
3. **STM Compression**: When `turn_count % STM_WINDOW == 0`, LLM summarizes older messages into a SystemMessage prefix
4. **HMAC Pickle Verification**: SHA-256 HMAC with a 32-byte random secret prevents pickle deserialization attacks on loaded FAISS indexes

---

## Phase 12 — GitHub Profile Review

### Current State

| Check | Status |
|---|---|
| README | ✅ Excellent |
| LICENSE | ✅ MIT |
| .gitignore | ✅ Present |
| CI/CD | ⚠️ Exists but minimal |
| CONTRIBUTING.md | ❌ Missing |
| CHANGELOG.md | ✅ Present |
| Issues / PR templates | ❌ Missing |
| Releases | ❌ No tagged releases |
| Commit quality | ⚠️ Unknown (diff_output.txt committed) |
| `agentflow.db` committed | 🔴 CRITICAL — contains real checkpoint data |

### Critical Fix: Remove the DB File

The `agentflow.db` (4.3MB) and `test_agentflow.db` (1MB) are committed to the repository. These contain real LangGraph checkpoint data. Even if the data is innocuous, committed DB files are an anti-pattern — they grow unboundedly and can contain PII.

**Fix:**
```bash
git filter-repo --path agentflow.db --invert-paths
git filter-repo --path test_agentflow.db --invert-paths
echo "*.db" >> .gitignore
```

Also: `diff_output.txt` (157KB) should not be committed.

---

## Phase 13 — Roadmap

### Quick Wins (1 day)

- [ ] Remove `agentflow.db`, `test_agentflow.db`, `diff_output.txt` from git history
- [ ] Add `*.db` and `*.txt` build artifacts to `.gitignore`
- [ ] Add `pytest --cov=backend --cov-report=term-missing` to CI
- [ ] Add `ruff check .` to CI
- [ ] Fix the duplicate `_extract_sources` in `blog_agent.py`
- [ ] Add a looping GIF demo to the README
- [ ] Tag v0.9.0 release on GitHub

### Small Improvements (1 week)

- [ ] Decompose `main.py` into `routers/` modules
- [ ] Decompose `App.jsx` into component files with hooks
- [ ] Migrate LTM eviction from private `docstore._dict` to public FAISS API
- [ ] Add `filelock` to user JSON store to prevent concurrent write corruption
- [ ] Add `--workers` env var to Docker CMD
- [ ] Add refresh token endpoint (`/auth/refresh`)
- [ ] Add `mypy --strict` check on backend to CI
- [ ] Write 5 frontend component tests with Vitest

### Medium Improvements (2–3 weeks)

- [ ] Migrate user store to a `users` table in the Postgres/SQLite checkpointer DB
- [ ] Add a `/admin/users` management API (create, list, delete users)
- [ ] Add token budget enforcement (count tokens before LLM call, trim if over limit)
- [ ] Add a `multiprocessing.get_context("spawn")` fix for code interpreter
- [ ] Add Prometheus `/metrics` endpoint with request count, latency histogram, LLM call count
- [ ] Build a LangSmith golden evaluation dataset (20 Q&A pairs per agent type)
- [ ] Add integration tests for the full RAG pipeline with a real PDF

### Major Features (1 month)

- [ ] Migrate FAISS to Qdrant (enables horizontal scaling)
- [ ] Add OpenAI GPT-4o as a third LLM provider option
- [ ] Add voice input (Whisper transcription before the chat endpoint)
- [ ] Add per-document delete endpoint for the RAG pipeline
- [ ] Add a user dashboard (token usage, memory facts viewer, document list)
- [ ] Implement WebSocket upgrade path as an alternative to SSE

### Resume Boosters

- [ ] Record a 2-minute demo video and link it at the top of the README
- [ ] Deploy to a public URL (not behind auth) for live demo
- [ ] Write a blog post on "Building a Multi-Agent System with LangGraph" and link it
- [ ] Open-source any reusable components (e.g., the SSRF-hardened url_reader as a pip package)
- [ ] Add a "Used in X projects" or "X GitHub stars" badge once the repo gains traction

### Production-Level Features

- [ ] Kubernetes Helm chart
- [ ] Database migration system (Alembic for Postgres)
- [ ] Row-level security in Postgres for multi-tenant isolation
- [ ] Redis-backed rate limiter (replace in-process slowapi)
- [ ] Circuit breaker pattern for Groq API calls (beyond current fallback chain)

---

## Summary Score Card

| Phase | Finding |
|---|---|
| Architecture | LangGraph StateGraph with clean node separation; correct async/sync dual-path |
| Code Quality | Strong backend, but main.py and App.jsx are monoliths that hurt maintainability |
| Security | Above average — HMAC, SSRF, prompt injection, JWT, bcrypt, constant-time compare |
| Testing | Good API contract tests; zero frontend tests; no LLM eval dataset |
| CI/CD | Exists but bare-minimum; no lint, no type-check, no coverage, no caching |
| Deployment | Production-ready Docker, Render/Railway configs, health probes |
| Documentation | Excellent — README, DESIGN_DOC, PRD, TECH_STACK, DEPLOYMENT, CHANGELOG |
| Overall | **7.7/10 — Strong portfolio project, ready for ML/LLM engineering interviews** |
