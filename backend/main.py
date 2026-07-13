"""
FastAPI application entry point — v0.9.0 (ship-ready).

Exposes the AgentFlow graph over HTTP. Endpoints map to DESIGN_DOC.md
section 7:

  POST /auth/login                 — exchange username/password for a JWT
  POST /chat                       — stream LLM tokens for a given thread_id
  POST /upload                     — ingest a PDF into a thread's FAISS index
  GET  /threads                    — list all threads
  GET  /threads/{thread_id}/state  — snapshot of the graph state
  GET  /threads/{thread_id}/history — full message history
  GET  /threads/{thread_id}/blog   — latest blog post
  POST /review/{thread_id}         — resume from the human_review interrupt
  DELETE /threads/{thread_id}      — delete a thread + FAISS index
  GET  /healthz                    — liveness probe
  GET  /readyz                     — readiness probe (graph + DB + embeddings)

Production-readiness changes (v0.9.0):
  - Typed config (pydantic-settings) instead of scattered os.environ.get()
  - JWT auth (HS256) per-user, with static API key as dev fallback
  - Postgres checkpointer when POSTGRES_CONN_STRING is set; SQLite otherwise
  - LangSmith env wiring at startup
  - Structured logging via structlog (JSON in prod, pretty in dev)
  - Rate limiting (slowapi) per-IP on /chat and /upload
  - /readyz separated from /healthz for proper k8s-style probes
  - [REASONING] SSE events emitted when an LLM produces a content chunk
    (the chat_model_stream hook already fires; we now stamp reasoning
    markers on the active node so the frontend can render a live rail)

The lifespan compiles a fresh async graph with the appropriate
checkpointer. The checkpointer is held open for the process lifetime;
on Postgres that means we need a real connection pool, on SQLite the
file is locked for the process.
"""

from __future__ import annotations

import asyncio
import os
import time as _time
import uuid as _uuid_mod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.logging_config import configure_logging, get_logger
from backend.settings import get_settings
from prometheus_fastapi_instrumentator import Instrumentator

configure_logging()
logger = get_logger("agentflow.api")
settings = get_settings()

if settings.langchain_tracing_v2:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    if settings.langchain_project:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
    if settings.langchain_endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langchain_endpoint)
    logger.info(
        "langsmith_tracing_enabled",
        project=settings.langchain_project,
    )

from slowapi.errors import RateLimitExceeded

from backend.auth import PUBLIC_PATHS, ensure_admin, init_user_table_async
from backend.dependencies import limiter
from backend.graph.build_graph import builder
from backend.rag.ingest import warm_embeddings
from backend.routers import admin, auth, chat, health, threads, upload

# ---------------------------------------------------------------------------
# Lifespan: async graph + checkpointer + admin bootstrap
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compile a fresh async graph with the configured checkpointer.

    Behaviour:
      - If `POSTGRES_CONN_STRING` is set → AsyncPostgresSaver (production)
      - Otherwise → AsyncSqliteSaver against CHECKPOINT_DB_PATH (dev)
    Both paths run `setup()` so the schema is ready before the first request.
    """
    # Bootstrap the admin user (no-op after first run).
    try:
        ensure_admin(settings)
    except Exception:
        logger.warning("admin_bootstrap_failed", exc_info=True)

    if settings.use_postgres:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        logger.info(
            "checkpointer_opening",
            backend="postgres",
        )
        assert settings.postgres_conn_string is not None
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(
            settings.postgres_conn_string
        )
        checkpointer = await checkpointer_cm.__aenter__()
        try:
            await checkpointer.setup()
        except Exception:
            logger.warning("checkpointer_setup_postgres_failed", exc_info=True)
            raise
        try:
            app.state.checkpointer_cm = checkpointer_cm
            app.state.checkpointer = checkpointer
            app.state.db_conn = getattr(checkpointer, "conn", None)
            app.state.checkpointer_kind = "postgres"

            # ---- User table bootstrap (Sprint 4) ----
            # For Postgres we use a separate aiosqlite connection for the users
            # table so we don't depend on the psycopg connection type.
            # If a native Postgres user table is needed later, swap this out.
            import aiosqlite as _aiosqlite
            _user_conn = await _aiosqlite.connect(settings.checkpoint_db_path, timeout=5.0)
            await init_user_table_async(_user_conn)
            app.state.user_db_conn = _user_conn
            ensure_admin(settings)  # migrates JSON + creates bootstrap admin
            # ---- end user table bootstrap ----

            app.state.graph = builder.compile(checkpointer=checkpointer)
            app.state.graph.name = "AgentFlow"
            await asyncio.to_thread(warm_embeddings)
            logger.info(
                "startup_ok",
                backend="postgres",
                cors_origins=settings.cors_origins_list,
            )
            try:
                yield
            finally:
                logger.info("shutting_down", backend="postgres")
                await _user_conn.close()
                await checkpointer_cm.__aexit__(None, None, None)
        except Exception:
            logger.exception("startup_failed", backend="postgres")
            raise
    else:
        logger.info(
            "checkpointer_opening",
            backend="sqlite",
            path=settings.checkpoint_db_path,
        )
        async with aiosqlite.connect(settings.checkpoint_db_path, timeout=5.0) as conn:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            checkpointer = AsyncSqliteSaver(conn) # type: ignore[assignment]
            try:
                app.state.checkpointer = checkpointer
                # Expose the connection so /threads can query the checkpoints
                # table directly. See postgres branch above for the why.
                app.state.db_conn = conn
                app.state.checkpointer_kind = "sqlite"

                # ---- User table bootstrap (Sprint 4) ----
                await init_user_table_async(conn)
                app.state.user_db_conn = conn  # same conn — users table is in agentflow.db
                ensure_admin(settings)  # migrates JSON + creates bootstrap admin
                # ---- end user table bootstrap ----

                app.state.graph = builder.compile(checkpointer=checkpointer)
                app.state.graph.name = "AgentFlow"
                await asyncio.to_thread(warm_embeddings)
                logger.info(
                    "startup_ok",
                    backend="sqlite",
                    path=settings.checkpoint_db_path,
                    cors_origins=settings.cors_origins_list,
                )
            except Exception:
                logger.exception("startup_failed", backend="sqlite")
                raise
            try:
                yield
            finally:
                logger.info("shutting_down", backend="sqlite")
                await conn.close()


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
)

# Prometheus metrics — instrument AFTER CORS so middleware wraps correctly.
# exclude_paths avoids measuring health-check noise.
Instrumentator(
    excluded_handlers=["/healthz", "/readyz", "/metrics", "/favicon.ico"],
    body_handlers=[],  # do NOT buffer response bodies — avoids BaseHTTPMiddleware body-loss bug
).instrument(app).expose(app, include_in_schema=False)

app.state.limiter = limiter


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Bind a request_id to the structlog context for the duration of the
    request, then echo it back as a response header.
    """
    req_id = request.headers.get("X-Request-ID", _uuid_mod.uuid4().hex)
    structlog.contextvars.bind_contextvars(request_id=req_id, path=request.url.path)
    start = _time.perf_counter()
    status_code = 500  # default if the handler raises before binding `response`
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception("request_unhandled_error")
        raise
    finally:
        duration_ms = round((_time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request_completed",
            method=request.method,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        structlog.contextvars.clear_contextvars()
        # `response` may be unbound if the handler raised; only stamp the
        # header when we actually have one to return.
        try:
            response.headers["X-Request-ID"] = req_id
        except UnboundLocalError:
            pass


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded"},
    )


# Public-path allowlist (no auth required). Anything not listed here is
# gated by `require_user` via Depends() on the endpoint signature.
PUBLIC_PATHS.update(
    {
        "/healthz",
        "/readyz",
        "/auth/login",
        "/auth/refresh",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(threads.router)
