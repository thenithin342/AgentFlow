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
import hashlib
import hmac
import json
import os
import shutil
import tempfile
import time as _time
import uuid as _uuid_mod
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Literal, Optional

import aiosqlite
import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from backend.settings import get_settings
from backend.logging_config import configure_logging, get_logger

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

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.auth import (
    CurrentUser,
    PUBLIC_PATHS,
    authenticate_user,
    ensure_admin,
    issue_token,
    make_thread_id,
    require_user,
)
from backend.constants import (
    MAX_MESSAGE_CHARS,
    MAX_UPLOAD_BYTES,
    SSE_TOKEN_NODES,
    TRACE_STREAM_NODES,
)
from backend.graph.build_graph import builder
from backend.graph.human_review import APPROVE_SENTINEL
from backend.graph.messages import _msg_type, content_to_str, is_human_message
from backend.logging_config import configure_logging, get_logger
from backend.rag.ingest import ingest_pdf, warm_embeddings
from backend.rag.ingest import _EMBEDDINGS_WARM  # flag set true after first FastEmbed load
from backend.settings import Settings, get_settings
from backend.validation import validate_thread_id


# Langsmith was moved above

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Per-thread upload lock + per-IP auth-fail bucket
# ---------------------------------------------------------------------------

_upload_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
_upload_locks_guard = asyncio.Lock()
_failed_auths: dict[str, dict[str, float]] = defaultdict(
    lambda: {"count": 0.0, "last": _time.time()}
)


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
            # Expose the underlying connection so /threads can query the
            # checkpoints table directly. langgraph.checkpoint.alist requires
            # `configurable.thread_id` even for global scans, which would
            # force us to enumerate per thread — N round trips. Going direct
            # to SQL is one query, indexed by the thread_id column.
            app.state.db_conn = getattr(checkpointer, "conn", None)
            app.state.checkpointer_kind = "postgres"
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

            checkpointer = AsyncSqliteSaver(conn)
            try:
                app.state.checkpointer = checkpointer
                # Expose the connection so /threads can query the checkpoints
                # table directly. See postgres branch above for the why.
                app.state.db_conn = conn
                app.state.checkpointer_kind = "sqlite"
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
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    review_required: bool = False
    user_id: str = "default"  # carried through; auth user is the source of truth

    @field_validator("thread_id")
    @classmethod
    def thread_id_valid(cls, v: str) -> str:
        return validate_thread_id(v)

    @field_validator("message")
    @classmethod
    def message_within_limit(cls, v: str) -> str:
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(
                f"message too long (max {MAX_MESSAGE_CHARS} characters)"
            )
        return v


class ReviewRequest(BaseModel):
    action: Literal["approve", "edit"]
    edited_response: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkpoint_id_to_iso(checkpoint_id: str | None) -> str | None:
    """Best-effort conversion of a LangGraph checkpoint_id to ISO8601 UTC."""
    if not checkpoint_id:
        return None
    if "T" in checkpoint_id and len(checkpoint_id) >= 19:
        try:
            normalized = checkpoint_id.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized[:26])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    try:
        u = _uuid_mod.UUID(checkpoint_id)
        if u.version == 1:
            t = int(u.time)
            dt = datetime(1582, 10, 15, tzinfo=timezone.utc) + timedelta(
                microseconds=t / 10
            )
            return dt.astimezone().isoformat().replace("+00:00", "Z")
    except (ValueError, AttributeError):
        pass
    return None


def _serialize_state_values(values: dict) -> dict:
    """Convert an AgentState.values dict into a JSON-safe dict."""
    out: dict = {}
    for k, v in values.items():
        if hasattr(v, "model_dump"):
            out[k] = v.model_dump()
        elif isinstance(v, list):
            out[k] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in v
            ]
        else:
            out[k] = v
    return out


def _config_for(user: CurrentUser, thread_id: str) -> dict:
    """Standard LangGraph RunnableConfig, scoped to the current user."""
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    scoped = make_thread_id(user, thread_id)
    return {"configurable": {"thread_id": scoped, "user_id": user.username}}


async def _get_upload_lock(thread_id: str) -> asyncio.Lock:
    async with _upload_locks_guard:
        if thread_id not in _upload_locks:
            if len(_upload_locks) >= 1000:
                # Evict the oldest unlocked entry. Snapshot the items so
                # we iterate a stable copy (the dict may be mutated by
                # other coroutines waiting on _upload_locks_guard).
                evicted = False
                for tid, lk in list(_upload_locks.items()):
                    if not lk.locked():
                        _upload_locks.pop(tid, None)
                        evicted = True
                        break
                if not evicted:
                    raise HTTPException(
                        status_code=503,
                        detail="too many concurrent uploads; retry shortly",
                    )
            _upload_locks[thread_id] = asyncio.Lock()
        _upload_locks.move_to_end(thread_id)
        return _upload_locks[thread_id]


def _sse(payload: str) -> bytes:
    """Format a single Server-Sent Event chunk."""
    if not isinstance(payload, str):
        payload = str(payload)
    if "\n" not in payload:
        return f"data: {payload}\n\n".encode("utf-8")
    body = "".join(f"data: {line}\n" for line in payload.split("\n"))
    return f"{body}\n".encode("utf-8")


def _snapshot_has_interrupt(snap) -> bool:
    if snap is None:
        return False
    if getattr(snap, "interrupts", None):
        return True
    for t in getattr(snap, "tasks", None) or []:
        if getattr(t, "interrupts", None):
            return True
    return False


def _iter_interrupts(snap):
    if snap is None:
        return
    for intr in getattr(snap, "interrupts", None) or ():
        yield intr
    for task in getattr(snap, "tasks", None) or ():
        for intr in getattr(task, "interrupts", None) or ():
            yield intr


def _extract_interrupt_draft(snap) -> str:
    for intr in _iter_interrupts(snap):
        value = getattr(intr, "value", None)
        if isinstance(value, dict) and value.get("draft") is not None:
            return str(value["draft"])
        if isinstance(value, str) and value.strip():
            return value
    values = (snap.values or {}) if snap else {}
    return str(values.get("final_response") or "")


def _serialize_interrupt(snap) -> dict | None:
    if not _snapshot_has_interrupt(snap):
        return None
    return {"pending": True, "draft": _extract_interrupt_draft(snap)}


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/login")
@limiter.limit(f"{settings.rate_limit_auth_per_minute}/minute")
async def login(req: LoginRequest, request: Request) -> dict:
    """Exchange username/password for a JWT.

    Per-IP rate limiting is applied at the dependency layer (see
    `_login_limiter` below). Failed logins never leak which field
    was wrong — we always say "invalid credentials".
    """
    if len(req.password) > 1024 or len(req.username) > 64:
        raise HTTPException(status_code=400, detail="invalid credentials")

    user = authenticate_user(settings, req.username, req.password)
    if not user:
        hashed_username = hashlib.sha256(req.username.encode()).hexdigest()[:16]
        logger.warning("login_failed", username=hashed_username)
        raise HTTPException(status_code=401, detail="invalid credentials")

    token = issue_token(settings, user.username)
    logger.info("login_ok", username=user.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.jwt_access_ttl_minutes * 60,
    }


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — process is up and serving HTTP.

    Intentionally cheap: no DB or LLM calls. Used by orchestrators to
    decide whether to restart the pod.
    """
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness probe — graph compiled, DB reachable, embeddings warm.

    Used by orchestrators to decide whether to send traffic. Returns
    503 if any subsystem is unhealthy.
    """
    timings: dict = {}

    t0 = _time.perf_counter()
    graph_ok = getattr(app.state, "graph", None) is not None
    timings["graph"] = round(_time.perf_counter() - t0, 4)

    db_ok = False
    t1 = _time.perf_counter()
    try:
        if settings.use_postgres:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(settings.postgres_conn_string)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                db_ok = True
            finally:
                await engine.dispose()
        else:
            async with aiosqlite.connect(settings.checkpoint_db_path, timeout=2.0) as db:
                await db.execute("SELECT 1")
                db_ok = True
    except Exception:
        logger.exception("readyz_db_failed")
    timings["db"] = round(_time.perf_counter() - t1, 4)

    embeddings_ok = False
    t2 = _time.perf_counter()
    try:
        # Skip the (heavy) warm_embeddings call on subsequent probes once the
        # FastEmbed model is loaded. The flag is set inside warm_embeddings /
        # _get_embeddings, so the first /readyz still pays the load cost and
        # later probes only see a dict lookup. /livez is unaffected.
        if _EMBEDDINGS_WARM:
            embeddings_ok = True
        else:
            await asyncio.to_thread(warm_embeddings)
            embeddings_ok = _EMBEDDINGS_WARM
    except Exception:
        logger.exception("readyz_embeddings_failed")
    timings["embeddings"] = round(_time.perf_counter() - t2, 4)

    ready = graph_ok and db_ok and embeddings_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "degraded",
            "graph": graph_ok,
            "db": db_ok,
            "embeddings": embeddings_ok,
            "backend": "postgres" if settings.use_postgres else "sqlite",
            "timings": timings,
        },
    )


# ---------------------------------------------------------------------------
# /chat — SSE token stream + reasoning trace
# ---------------------------------------------------------------------------


@app.post("/chat")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(
    request: Request,
    req: ChatRequest,
    user: CurrentUser = Depends(require_user),
) -> StreamingResponse:
    """Stream LLM tokens + reasoning trace for `req.message`.

    Event shapes (all `data:` lines):
      [TOOL_START:<tool_name>]
      [NODE_START:<node>|t=<iso8601>]
      [NODE_END:<node>]
      <token text>
      [SOURCES:<n>]
      [FINAL:<json-encoded text>]
      [DONE] | [INTERRUPT] | [ERROR]
    """
    config = _config_for(user, req.thread_id)
    input_state = {
        "messages": [HumanMessage(content=req.message)],
        "review_required": req.review_required,
    }
    graph = app.state.graph

    async def event_stream() -> AsyncIterator[bytes]:
        active_node: dict[str, Optional[str]] = {"name": None}
        try:
            async for event in graph.astream_events(
                input_state, config=config, version="v2"
            ):
                node = event.get("metadata", {}).get("langgraph_node", "")

                if event.get("event") == "on_tool_start":
                    tool_name = (
                        event.get("name")
                        or event.get("data", {}).get("name")
                        or "tool"
                    )
                    yield _sse(f"[TOOL_START:{tool_name}]")
                    continue

                if event.get("event") == "on_chain_start" and node in TRACE_STREAM_NODES:
                    active_node["name"] = node
                    ts = (
                        datetime.now(timezone.utc)
                        .isoformat(timespec="milliseconds")
                        .replace("+00:00", "Z")
                    )
                    yield _sse(f"[NODE_START:{node}|t={ts}]")
                    continue

                if event.get("event") == "on_chain_end" and node in TRACE_STREAM_NODES:
                    if active_node["name"] == node:
                        active_node["name"] = None
                    yield _sse(f"[NODE_END:{node}]")
                    continue

                if event.get("event") != "on_chat_model_stream":
                    continue
                if node not in SSE_TOKEN_NODES:
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk is None:
                    continue
                text = content_to_str(getattr(chunk, "content", None))
                if text:
                    # Yield the token to the visible draft. (A prior draft
                    # of this code also emitted a [REASONING:node|text]
                    # event here, but no frontend consumer rendered the
                    # reasoning text — and the sseParser fell through to
                    # kind=token, which concatenated the marker into the
                    # visible response. The live trace is driven entirely
                    # by [NODE_START]/[NODE_END] events, so we just yield
                    # the token.)
                    yield _sse(text)
        except Exception:
            logger.exception("chat_failed", thread_id=req.thread_id, user=user.username)
            yield _sse("[ERROR] internal server error")
            return

        try:
            snap = await graph.aget_state(config)
        except Exception:
            logger.exception("post_stream_aget_state_failed")
            yield _sse("[ERROR] internal server error")
            return

        if _snapshot_has_interrupt(snap):
            logger.info("thread_interrupt", thread_id=req.thread_id, user=user.username)
            yield _sse("[INTERRUPT]")
        else:
            sources = (snap.values or {}).get("sources") or []
            yield _sse(f"[SOURCES:{len(sources)}]")
            final_text = (snap.values or {}).get("final_response") or ""
            if final_text:
                yield _sse(f"[FINAL:{json.dumps(final_text)}]")
            yield _sse("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# /upload — PDF ingest
# ---------------------------------------------------------------------------


@app.post("/upload")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def upload(
    request: Request,
    thread_id: str = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_user),
) -> dict:
    """Ingest `file` (PDF) into the per-thread FAISS index for `thread_id`."""
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file must be a .pdf")

    cl_header = file.headers.get("content-length") if file.headers else None
    if cl_header is not None:
        try:
            cl = int(cl_header)
        except (TypeError, ValueError):
            cl = -1
        if cl > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file too large (>{MAX_UPLOAD_BYTES} bytes)",
            )

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

            def _copy_bounded(src, dst):
                total = 0
                chunk_size = 64 * 1024
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError(f"file too large (>{MAX_UPLOAD_BYTES} bytes)")
                    dst.write(chunk)

            try:
                await asyncio.to_thread(_copy_bounded, file.file, tmp)
            except ValueError as e:
                raise HTTPException(status_code=413, detail=str(e))

        with open(tmp_path, "rb") as f:
            head = f.read(4)
        if head[:4] != b"%PDF":
            raise HTTPException(
                status_code=400, detail="file is not a valid PDF (bad magic bytes)"
            )

        # Scope the FAISS thread_id to the user, just like the graph state.
        scoped = make_thread_id(user, thread_id)
        lock = await _get_upload_lock(scoped)
        graph = app.state.graph
        config = _config_for(user, thread_id)
        async with lock:
            stats = await asyncio.to_thread(
                ingest_pdf,
                tmp_path,
                scoped,
                source_name=file.filename,
            )
            try:
                snap = await graph.aget_state(config)
                existing = list((snap.values or {}).get("documents") or [])
                doc_id = stats["document_id"]
                if doc_id not in existing:
                    existing.append(doc_id)
                await graph.aupdate_state(config, {"documents": existing})
            except Exception:
                logger.exception(
                    "upload_state_update_failed", thread_id=scoped
                )
        return {
            "status": "indexed",
            "thread_id": thread_id,
            **stats,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("upload_failed", thread_id=thread_id, user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Thread inspection endpoints
# ---------------------------------------------------------------------------


@app.get("/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    user: CurrentUser = Depends(require_user),
) -> dict:
    graph = app.state.graph
    config = _config_for(user, thread_id)
    try:
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("get_state_failed", thread_id=thread_id, user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")
    values = snap.values if snap else {}
    return {
        "thread_id": thread_id,
        "values": _serialize_state_values(values),
        "interrupt": _serialize_interrupt(snap),
    }


@app.get("/threads")
async def list_threads(
    user: CurrentUser = Depends(require_user),
) -> dict:
    """List threads belonging to the current user.

    We query the checkpointer's underlying connection directly rather than
    calling `graph.checkpointer.alist({})`. The langgraph `alist` contract
    requires `configurable.thread_id` even for global scans; the SQL
    implementation in `langgraph/checkpoint/sqlite/utils.py:95` raises
    `KeyError: 'thread_id'` when it's missing. A per-thread `alist` would
    also be N round trips. Going direct: one query against the indexed
    `thread_id` column, filter on the per-user prefix in SQL.
    """
    scoped_prefix = f"user:{user.username}:"
    db_conn = getattr(app.state, "db_conn", None)
    kind = getattr(app.state, "checkpointer_kind", None)
    if db_conn is None or kind is None:
        return {"threads": []}

    # `checkpoints` is the langgraph table for both backends. Schema:
    #   thread_id TEXT, checkpoint_id TEXT, parent_checkpoint_id TEXT, ...
    # The most recent checkpoint per thread is the "last seen" timestamp.
    sql = (
        "SELECT thread_id, checkpoint_id FROM checkpoints "
        "WHERE thread_id LIKE ? ESCAPE '\\' "
        "ORDER BY thread_id, checkpoint_id DESC"
    )
    # LIKE wildcards: the prefix is fixed (alnum + colons + dots), so it's
    # safe — no user input. We escape '%' and '_' just in case a future
    # username is allowed to contain them.
    like_pattern = scoped_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    sql_args: tuple = (like_pattern,)

    threads: list[dict] = []
    seen: set[str] = set()
    try:
        if kind == "sqlite":
            async with db_conn.execute(sql, sql_args) as cur:
                async for row in cur:
                    tid = row[0]
                    if tid in seen:
                        continue  # keep the first (most recent) per thread
                    seen.add(tid)
                    threads.append({
                        "thread_id": tid[len(scoped_prefix):],
                        "last_seen": _checkpoint_id_to_iso(row[1]),
                    })
        elif kind == "postgres":
            # Use psycopg3 cursor API (%s placeholders, execute+fetchall).
            # db_conn is an asyncpg or psycopg3 connection; try psycopg3 first.
            pg_sql = sql.replace("?", "%s").replace("ESCAPE '\\'", "")
            execute = getattr(db_conn, "execute", None)
            if execute is None:
                logger.warning("list_threads_postgres_api_unavailable", user=user.username)
                return {"threads": []}
            cur = await execute(pg_sql, sql_args)
            rows = await cur.fetchall()
            for row in rows:
                tid = row[0]
                if tid in seen:
                    continue
                seen.add(tid)
                threads.append({
                    "thread_id": tid[len(scoped_prefix):],
                    "last_seen": _checkpoint_id_to_iso(row[1]),
                })
        else:
            return {"threads": []}
    except Exception:
        logger.exception("list_threads_failed", user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")

    # Sort by recency (most recent first) and cap at 100.
    threads.sort(key=lambda t: t["last_seen"] or "", reverse=True)
    return {"threads": threads[:100]}


@app.get("/threads/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    user: CurrentUser = Depends(require_user),
) -> dict:
    graph = app.state.graph
    config = _config_for(user, thread_id)
    try:
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("get_history_failed", thread_id=thread_id, user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")

    values = snap.values if snap else {}
    route = values.get("route")
    documents = values.get("documents") or []
    messages = []

    if "messages" in values:
        for m in values["messages"]:
            role = "user" if is_human_message(m) else "agent"
            agent = getattr(m, "name", None) or None
            if agent == "chat":
                agent = "chat_agent"
            msg_obj = {
                "id": m.id if hasattr(m, "id") else None,
                "role": role,
                "text": content_to_str(m.content if hasattr(m, "content") else m),
            }
            if agent:
                msg_obj["agent"] = agent
            elif role == "agent" and route in {"research", "analysis", "chat"}:
                msg_obj["agent"] = (
                    f"{route}_agent" if route != "chat" else "chat_agent"
                )
            if role == "agent":
                msg_obj["meta"] = f"{msg_obj.get('agent') or 'agent'} · history"
            messages.append(msg_obj)

    interrupt = _serialize_interrupt(snap)
    if interrupt and interrupt.get("pending"):
        draft = interrupt.get("draft") or ""
        if not messages or messages[-1].get("role") != "review":
            messages.append({
                "id": None,
                "role": "review",
                "text": draft,
                "meta": "human_review · pending",
            })

    return {
        "thread_id": thread_id,
        "messages": messages,
        "interrupt": interrupt,
        "route": route,
        "documents": documents,
    }


@app.post("/review/{thread_id}")
async def review(
    thread_id: str,
    req: ReviewRequest,
    user: CurrentUser = Depends(require_user),
) -> dict:
    from langgraph.types import Command

    config = _config_for(user, thread_id)
    graph = app.state.graph

    if req.action == "approve":
        resume_value = APPROVE_SENTINEL
    else:
        if not req.edited_response:
            raise HTTPException(
                status_code=400,
                detail="edited_response is required when action='edit'",
            )
        resume_value = req.edited_response

    try:
        snap = await graph.aget_state(config)
        if not _snapshot_has_interrupt(snap):
            raise HTTPException(status_code=409, detail="no pending interrupt for this thread")
        await asyncio.wait_for(
            graph.ainvoke(Command(resume=resume_value), config=config), timeout=60.0
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("review_failed", thread_id=thread_id, user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")

    return {"status": "resumed", "thread_id": thread_id}


@app.get("/threads/{thread_id}/blog")
async def get_thread_blog(
    thread_id: str,
    user: CurrentUser = Depends(require_user),
) -> dict:
    graph = app.state.graph
    config = _config_for(user, thread_id)
    try:
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("get_blog_failed", thread_id=thread_id, user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")
    values = snap.values if snap else {}
    return {
        "thread_id": thread_id,
        "blog_output": values.get("blog_output"),
    }


@app.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    user: CurrentUser = Depends(require_user),
) -> dict:
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scoped = make_thread_id(user, thread_id)
    deleted_checkpoints = 0
    deleted_faiss = False

    try:
        if settings.use_postgres:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(
                settings.postgres_conn_string
            ) as cp:
                await cp.setup()
                # adelete_thread is a LangGraph store primitive.
                if hasattr(cp, "adelete_thread"):
                    deleted_checkpoints = await cp.adelete_thread(scoped)
                else:
                    # Fallback: clear via raw connection.
                    await cp.conn.execute("DELETE FROM checkpoints WHERE thread_id = $1", scoped)
                    await cp.conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = $1", scoped)
                    await cp.conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", scoped)
                    deleted_checkpoints = 1
        else:
            async with aiosqlite.connect(settings.checkpoint_db_path, timeout=5.0) as db:
                cursor = await db.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (scoped,)
                )
                deleted_checkpoints = cursor.rowcount
                await db.commit()
    except Exception:
        logger.exception("delete_thread_db_failed", thread_id=scoped)
        raise HTTPException(status_code=500, detail="internal server error")

    try:
        from backend.rag.ingest import INDEX_ROOT
        idx_path = INDEX_ROOT / scoped
        if idx_path.is_dir():
            await asyncio.to_thread(shutil.rmtree, str(idx_path))
            deleted_faiss = True
    except Exception:
        logger.warning("delete_thread_faiss_failed", thread_id=scoped)

    logger.info(
        "thread_deleted",
        thread_id=scoped,
        user=user.username,
        deleted_checkpoints=deleted_checkpoints,
        deleted_faiss=deleted_faiss,
    )
    return {
        "status": "deleted",
        "thread_id": thread_id,
        "deleted_checkpoints": deleted_checkpoints,
        "deleted_faiss": deleted_faiss,
    }
