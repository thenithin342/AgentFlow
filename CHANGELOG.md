# Changelog

All notable changes to **AgentFlow** are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning: [Semantic Versioning](https://semver.org/)

---

## [1.0.0] — 2026-07-13 🚀

### Summary
**Production release.** The system is horizontally scalable, fully authenticated, and ships with a complete admin UI, Qdrant-backed vector search, and a polished React frontend. Every component runs on free-tier or local compute.

### Added
- **Qdrant Cloud integration** — optional vector store for both RAG (per-thread) and Long-Term Memory (per-user). Falls back to local FAISS when `QDRANT_URL` is unset, so existing single-node deployments are unaffected
- **`backend/vectorstore/qdrant_store.py`** — thin LangChain-compatible adapter: `add_documents`, `similarity_search`, `as_retriever`, `delete_collection`, `delete_oldest` (LRU eviction)
- **Qdrant in `/readyz`** — readiness probe now reports `"qdrant": true/false` when Qdrant is configured
- **Qdrant service in `docker-compose.yml`** — `--profile qdrant` starts a local Qdrant container with a persistent volume; backend wires `QDRANT_URL` automatically
- **Full Admin CRUD API**
  - `GET /admin/users` — list all registered users
  - `POST /admin/users` — create a new user (admin only)
  - `DELETE /admin/users/{username}` — delete a user; admin account is protected from deletion
  - `PUT /admin/users/{username}/password` — change any user's password
- **Admin UI — Actions column** — "Change Password" (inline form) and "Delete" (with confirm dialog) per-user row; Delete hidden for the `admin` account
- **Auto-migration** — on first boot after upgrade, `users.json` is transparently migrated to the `users` SQLite table with zero downtime
- `CHANGELOG.md` — this file

### Changed
- **User store migrated from JSON file to SQLite** — eliminates the `filelock` race condition under concurrent logins; user records now live in `agentflow.db` alongside checkpoints
- **`/auth/refresh`** — now validates token against the SQLite user table instead of loading `users.json`
- **`backend/auth.py`** — full rewrite: async CRUD helpers (`db_get_user`, `db_create_user`, `db_delete_user`, `db_update_password`, `db_list_users`), `init_user_table_async` for lifespan wiring
- **`backend/main.py`** — lifespan calls `init_user_table_async` and `ensure_admin` in both SQLite and Postgres branches; exposes `app.state.user_db_conn`
- **`backend/memory/ltm.py`** — auto-selects Qdrant or FAISS via `_use_qdrant()`; original FAISS code preserved as private functions
- **`backend/rag/ingest.py`** — auto-selects Qdrant or FAISS for `ingest_pdf` and `get_retriever`; FAISS caches only active on FAISS path
- **`requirements.txt`** — added `qdrant-client>=1.9.0`, `langchain-qdrant>=0.1.3`
- **`docker-compose.yml`** — added `qdrant` profile, `qdrant_data` volume, wired `QDRANT_URL`/`QDRANT_API_KEY` into backend service

### Security
- Admin endpoints protected by `require_admin` dependency — 403 if JWT sub ≠ `ADMIN_USERNAME`
- `DELETE /admin/users/admin` rejected with HTTP 400 to prevent admin lockout
- Usernames validated against `^[a-zA-Z0-9_.-]{3,64}$` before any DB operation

---

## [0.9.0] — 2026-07-11

### Summary
Long-Term Memory, Blog Agent, Admin panel, monitoring, and deployment hardening.

### Added
- **Long-Term Memory (LTM)** — per-user FAISS index; `write_ltm` extracts facts via LLM post-synthesis; `read_ltm` retrieves relevant memories and injects them into the system prompt
- **Memory writer node** — LangGraph node that runs post-synthesis and persists salient facts asynchronously
- **Blog Agent** — dedicated `blog_agent` node; router dispatches `blog` intent; produces structured Markdown posts
- **Admin panel** — initial `GET /admin/users` + `POST /admin/users` endpoints with frontend UI
- **Prometheus metrics** — `prometheus-fastapi-instrumentator`; `/metrics` endpoint for Grafana scraping
- **Structured logging** — `structlog` JSON logs with `request_id`, `duration_ms`, `status_code` on every request
- **HMAC index signing** — FAISS `index.pkl` signed with per-deploy secret to detect tampering
- **Rate limiting** — `slowapi` middleware; 5/min for auth, 60/min for chat
- **Docker support** — `Dockerfile` + `docker-compose.yml` with Postgres profile; health checks on all services
- **LangSmith tracing** — opt-in via `LANGSMITH_API_KEY`

### Changed
- Frontend split into `ChatPage`, `BlogPage`, `AdminPage` with tab navigation
- `/readyz` expanded to check graph, DB, and embeddings with per-check timing
- Settings consolidated into `backend/settings.py` Pydantic model

---

## [0.8.0] — 2026-07-07

### Summary
JWT authentication, token streaming, RAG pipeline, and observability.

### Added
- **JWT authentication** — `python-jose`; login returns `access_token`; all non-public routes require `Authorization: Bearer`
- **Token streaming** — `astream_events(version="v2")` → FastAPI `StreamingResponse` SSE
- **RAG pipeline** — `PyPDFLoader` → `RecursiveCharacterTextSplitter(800, 150)` → `BAAI/bge-small-en-v1.5` (FastEmbed) → per-thread FAISS
- **`POST /upload`** — multipart PDF upload; 50 MB cap; returns `{document_id, pages, chunks}`
- **React frontend** — SSE reader, PDF upload, agent trace badges, human-review panel, thread sidebar
- **Multi-key Groq pool** — `GROQ_API_KEY_2/3` in `RunnableWithFallbacks` chain
- **Gemini 2.0 Flash fallback** — last resort when all Groq keys rate-limit
- **Observability** — `/health` per-component timing, `X-Request-ID` middleware
- `pyproject.toml`, `LICENSE`, GitHub Actions workflow

### Fixed
- Replaced dangerous `.innerHTML` with `rehype-sanitize` for React markdown rendering
- Standardised thread-locking across LangGraph checkpoint layer
- `asyncio.wait_for` added to `ainvoke` calls
- SQL queries in `list_threads` properly grouped and concurrency-limited

### Changed
- Embeddings migrated from `all-MiniLM-L6-v2` (HuggingFace) to `BAAI/bge-small-en-v1.5` (FastEmbed ONNX) — 40% faster warm-up

---

## [0.7.0] — 2026-06-28

### Summary
Human-in-the-loop review and Synthesizer.

### Added
- **Human Review node** — `interrupt()` pauses execution; `POST /review/{thread_id}` resumes with `approve` or `edit`
- **Synthesizer node** — `llama-3.3-70b` polishes raw agent output with prompt-injection `<<UNTRUSTED>>` barriers
- **Execution trace panel** — frontend shows node-by-node progress in real time

---

## [0.6.0] — 2026-06-20

### Summary
Three specialist agents with tool use.

### Added
- **Research Agent** — ReAct loop: `tavily_search` + `retrieve_documents`; `llama-3.1-8b-instant`
- **Analysis Agent** — ReAct loop: sandboxed AST calculator + `retrieve_documents`
- **Chat Agent** — RAG-aware fast path; skips Synthesizer for low latency
- **AST calculator** — custom evaluator; rejects names/calls/attributes; caps length, depth, exponents
- **LRU agent cache** — `OrderedDict(maxlen=128)` keyed on `(tool_names, model, prompt_hash, thread_id)`

---

## [0.5.0] — 2026-06-15

### Summary
Conditional router and SQLite checkpointing.

### Added
- **Router node** — `llama-3.3-70b` with few-shot classification → `research | analysis | chat | blog`
- **Conditional edges** — `route_query` reads `state["route"]` and dispatches to the correct agent
- **`AsyncSqliteSaver`** — checkpoints every node transition; sessions survive restarts
- **`GET /threads/{thread_id}/state`** — introspect graph state
- **`GET /threads`** — list all threads from checkpoint table

---

## [0.1.0] — 2026-06-10

### Summary
Initial proof-of-concept: single-node LangGraph graph.

### Added
- `AgentState` TypedDict — `messages`, `route`, `agent_output`, `sources`, `user_id`
- Single `chat_node` wired into a `StateGraph`
- FastAPI skeleton with `POST /chat`
- `.env.example`, `requirements.txt`, `pytest.ini`, `conftest.py`
- MIT `LICENSE`

---

[1.0.0]: https://github.com/thenithin342/AgentFlow/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/thenithin342/AgentFlow/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/thenithin342/AgentFlow/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/thenithin342/AgentFlow/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/thenithin342/AgentFlow/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/thenithin342/AgentFlow/compare/v0.1.0...v0.5.0
[0.1.0]: https://github.com/thenithin342/AgentFlow/releases/tag/v0.1.0
