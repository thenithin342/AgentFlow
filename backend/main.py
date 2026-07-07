"""
FastAPI application entry point — Phase 8a.

Exposes the AgentFlow graph (Phases 1-7) over HTTP. Four endpoints map 1:1 to
DESIGN_DOC.md section 7:

  POST /chat                       — stream LLM tokens for a given thread_id
  POST /upload                     — ingest a PDF into a thread's FAISS index
  GET  /threads/{thread_id}/state  — snapshot of the graph state
  POST /review/{thread_id}         — resume from the human_review interrupt

The app owns its own async-flavored graph instance built in a `lifespan`
context. The lifespan-compiled graph is the only one the server uses;
all endpoints read `request.app.state.graph`. The sync module-level
`build_graph.graph` stays untouched so the pytest suite keeps working.

THREAD_ID IS REQUIRED for every endpoint that touches the graph — pass it in
the request body or URL path. The same thread_id passed to /chat and /upload
shares state (RAG indexes are keyed by it too).

Reference: DESIGN_DOC.md section 7 "API Design (FastAPI)".
"""

import asyncio
import hmac
import json
import logging
import os
import shutil
import tempfile
import uuid as _uuid_mod
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Literal, Optional

import aiosqlite
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, field_validator

from backend.constants import (
    MAX_MESSAGE_CHARS,
    MAX_UPLOAD_BYTES,
    SSE_TOKEN_NODES,
    TRACE_STREAM_NODES,
)
from backend.graph.build_graph import builder
from backend.graph.human_review import APPROVE_SENTINEL
from backend.graph.messages import content_to_str, is_human_message, _msg_type
from backend.rag.ingest import ingest_pdf, warm_embeddings
from backend.validation import THREAD_ID_RE, validate_thread_id


logger = logging.getLogger("agentflow.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


# --- Config -----------------------------------------------------------------

_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "agentflow.db")
# Comma-separated env var; whitespace tolerated. Default keeps dev working
# against the Vite server. Production: set CORS_ORIGINS=https://app.example.com
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:5174"
    ).split(",")
    if o.strip()
]
# Hard cap on /upload body size. PyPDF + sentence-transformers both allocate
# proportional to input; 50 MB is a generous single-doc limit. Requests with
# Content-Length over this are rejected at the header level (413) before any
# body is buffered to disk.
# Hard cap on /chat message body length (chars). Prevents unbounded prompt
# allocation and runaway token usage from a single request.
# Optional bearer token. When set, every route except /health requires
# `Authorization: Bearer <key>` or `X-API-Key: <key>`.
_API_KEY = os.environ.get("AGENTFLOW_API_KEY", "").strip()
# Per-thread serialization for concurrent /upload calls against the same
# FAISS index. Created lazily on first use; entries are never removed (one
# Lock object per thread_id for the process lifetime is bounded by the
# number of distinct thread_ids ever seen, which is finite in practice).
_upload_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
_upload_locks_guard = asyncio.Lock()

from collections import defaultdict
import time
# Token bucket for failed auths per IP: decays 1 token / 10s. Max 10 tokens.
_failed_auths = defaultdict(lambda: {"count": 0.0, "last": time.time()})


# --- Lifespan: async graph + checkpointer ----------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compile a fresh async-flavored graph with AsyncSqliteSaver.

    AsyncSqliteSaver.__init__() calls asyncio.get_running_loop() — it MUST be
    instantiated inside a running event loop, so it cannot live at module
    load. The same comment block in backend/graph/build_graph.py:91-105
    prescribes this exact pattern.
    """
    logger.info("[AgentFlow] opening async SQLite at %s", _DB_PATH)
    async with aiosqlite.connect(_DB_PATH, timeout=5.0) as conn:
        checkpointer = AsyncSqliteSaver(conn)
        try:
            app.state.graph = builder.compile(checkpointer=checkpointer)
            app.state.graph.name = "AgentFlow"
            await asyncio.to_thread(warm_embeddings)
            logger.info(
                "[AgentFlow] graph compiled OK; async checkpointer on %s; "
                "CORS allow: %s",
                _DB_PATH,
                _CORS_ORIGINS,
            )
        except Exception:
            logger.exception("[AgentFlow] startup failed — graph not compiled")
            raise
        try:
            yield
        finally:
            logger.info("[AgentFlow] shutting down — closing async checkpointer")
            await conn.close()


# --- App --------------------------------------------------------------------

app = FastAPI(title="AgentFlow", version="0.8.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", _uuid_mod.uuid4().hex)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

@app.middleware("http")
async def optional_api_key_auth(request: Request, call_next):
    """Enforce AGENTFLOW_API_KEY when configured. /health stays public."""
    if not _API_KEY or request.url.path == "/health":
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    state = _failed_auths[client_ip]
    now = time.time()
    elapsed = now - state["last"]
    state["count"] = max(0.0, state["count"] - elapsed / 10.0)
    state["last"] = now
    
    if state["count"] > 10.0:
        return JSONResponse(status_code=429, content={"detail": "Too many failed attempts"})

    auth = request.headers.get("Authorization", "")
    header_key = request.headers.get("X-API-Key", "")
    token = ""
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    elif header_key:
        token = header_key.strip()
    if not token or not hmac.compare_digest(token, _API_KEY):
        state["count"] += 1.0
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


# --- Pydantic request models -----------------------------------------------


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    review_required: bool = False
    # Optional user_id for LTM scoping. Defaults to "default" for single-user
    # local setups — every thread from the same user shares LTM this way.
    user_id: str = "default"

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

    @field_validator("user_id")
    @classmethod
    def user_id_safe(cls, v: str) -> str:
        # Sanitise: only alphanumeric, dash, underscore, dot; max 64 chars
        safe = "".join(c for c in v if c.isalnum() or c in "-_.")
        return safe[:64] or "default"


class ReviewRequest(BaseModel):
    action: Literal["approve", "edit"]
    edited_response: Optional[str] = None


# --- Helpers ----------------------------------------------------------------


def _message_content_to_str(content) -> str:
    """Backward-compatible alias for content_to_str."""
    return content_to_str(content)


def _checkpoint_id_to_iso(checkpoint_id: str | None) -> str | None:
    """Best-effort conversion of a LangGraph checkpoint_id to ISO8601 UTC.

    Returns None when the id is not a recognized timestamp-bearing format so
    clients can render a neutral placeholder instead of Invalid Date.
    """
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
            return dt.isoformat().replace("+00:00", "Z")
    except (ValueError, AttributeError):
        pass
    return None


def _serialize_state_values(values: dict) -> dict:
    """Convert an AgentState.values dict into a JSON-safe dict.

    BaseMessage objects (HumanMessage, AIMessage, ToolMessage, …) are
    Pydantic v2 models — `model_dump()` round-trips them. Other fields
    (route, agent_output, sources, documents, review_required, final_response)
    are already JSON-safe primitives.
    """
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


def _config_for(thread_id: str, user_id: str = "default") -> dict:
    """Standard LangGraph RunnableConfig for a given thread_id."""
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


async def _get_upload_lock(thread_id: str) -> asyncio.Lock:
    async with _upload_locks_guard:
        if thread_id not in _upload_locks:
            # O(1) eviction strategy
            while len(_upload_locks) >= 1000:
                # Find an unlocked lock by checking the oldest items
                # We do at most 1000 iterations in the worst case (all locked), but O(1) space and fast
                for _ in range(len(_upload_locks)):
                    old_thread_id, old_lock = next(iter(_upload_locks.items()))
                    if not old_lock.locked():
                        _upload_locks.pop(old_thread_id)
                        break
                    else:
                        # Move to end so we check the next oldest
                        _upload_locks.move_to_end(old_thread_id)
                else:
                    logger.warning("[AgentFlow] Pathological lock congestion: max upload locks reached and none are free")
                    raise HTTPException(
                        status_code=503,
                        detail="too many concurrent uploads; retry shortly",
                    )
            _upload_locks[thread_id] = asyncio.Lock()
        _upload_locks.move_to_end(thread_id)
        return _upload_locks[thread_id]


def _sse(payload: str) -> bytes:
    """Format a single Server-Sent Event chunk.

    Splits multi-line payloads into one SSE frame per line so that:
    - The LLM's own newlines (token boundaries) are preserved
    - The `[DONE]`/`[INTERRUPT]`/`[ERROR]` sentinels stay on their own frames
    - Per the SSE spec, each `data:` field is a line; multi-line `data:`
      values must be split, and the event is terminated by a blank line.
    """
    if not isinstance(payload, str):
        payload = str(payload)
    if "\n" not in payload:
        return f"data: {payload}\n\n".encode("utf-8")
    body = "".join(f"data: {line}\n" for line in payload.split("\n"))
    return f"{body}\n".encode("utf-8")


def _snapshot_has_interrupt(snap) -> bool:
    """Return True if the graph snapshot has a pending human_review interrupt.

    LangGraph 1.x exposes pending interrupts in two places:
      1. `StateSnapshot.interrupts` (top-level tuple of Interrupt objects)
      2. `StateSnapshot.tasks[i].interrupts` (per-task tuple)
    We check both because the layout differs by version.
    """
    if snap is None:
        return False
    if getattr(snap, "interrupts", None):
        return True
    tasks = getattr(snap, "tasks", None) or []
    for t in tasks:
        if getattr(t, "interrupts", None):
            return True
    return False


def _iter_interrupts(snap):
    """Yield Interrupt objects from a StateSnapshot (version-tolerant)."""
    if snap is None:
        return
    top = getattr(snap, "interrupts", None) or ()
    for intr in top:
        yield intr
    for task in getattr(snap, "tasks", None) or ():
        for intr in getattr(task, "interrupts", None) or ():
            yield intr


def _extract_interrupt_draft(snap) -> str:
    """Best-effort draft text from a pending human_review interrupt."""
    for intr in _iter_interrupts(snap):
        value = getattr(intr, "value", None)
        if isinstance(value, dict) and value.get("draft") is not None:
            return str(value["draft"])
        if isinstance(value, str) and value.strip():
            return value
    values = (snap.values or {}) if snap else {}
    return str(values.get("final_response") or "")


def _serialize_interrupt(snap) -> dict | None:
    """JSON-safe interrupt summary for GET /threads/{id}/state."""
    if not _snapshot_has_interrupt(snap):
        return None
    return {"pending": True, "draft": _extract_interrupt_draft(snap)}


async def _ensure_checkpoint_tables(db, graph=None) -> None:
    """Create LangGraph checkpoint tables if this DB has never seen a graph run."""
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
    )
    if await cursor.fetchone() is not None:
        return
    graph = graph or getattr(app.state, "graph", None)
    checkpointer = getattr(graph, "checkpointer", None) if graph else None
    setup = getattr(checkpointer, "setup", None)
    if callable(setup):
        maybe = setup()
        if hasattr(maybe, "__await__"):
            await maybe
        return
    await db.execute(
        "CREATE TABLE IF NOT EXISTS checkpoints ("
        "thread_id TEXT NOT NULL, "
        "checkpoint_ns TEXT NOT NULL DEFAULT '', "
        "checkpoint_id TEXT NOT NULL, "
        "parent_checkpoint_id TEXT, "
        "type TEXT, "
        "checkpoint BLOB, "
        "metadata BLOB, "
        "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
    )
    await db.commit()


# --- Routes -----------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Liveness probe — graph, DB, and embeddings readiness."""
    timings = {}
    
    t0 = time.perf_counter()
    graph_ok = getattr(app.state, "graph", None) is not None
    timings["graph"] = round(time.perf_counter() - t0, 4)
    
    db_ok = False
    t1 = time.perf_counter()
    try:
        async with aiosqlite.connect(_DB_PATH, timeout=2.0) as db:
            await db.execute("SELECT 1")
            db_ok = True
    except Exception:
        logger.exception("[AgentFlow] health DB check failed")
    timings["db"] = round(time.perf_counter() - t1, 4)
        
    embeddings_ok = False
    t2 = time.perf_counter()
    try:
        await asyncio.to_thread(warm_embeddings)
        embeddings_ok = True
    except Exception:
        logger.exception("[AgentFlow] health embeddings check failed")
    timings["embeddings"] = round(time.perf_counter() - t2, 4)
        
    return {
        "status": "ok" if graph_ok and db_ok else "degraded",
        "graph": graph_ok,
        "db": db_ok,
        "embeddings": embeddings_ok,
        "timings": timings,
    }


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream LLM tokens for `req.message` on `req.thread_id` as SSE.

    Each token is one `data: <token>\n\n` event. The stream ends with a
    terminal sentinel — `data: [DONE]\n\n` on a clean run, or
    `data: [INTERRUPT]\n\n` if the graph paused at human_review (only
    possible when `review_required=True`).
    """
    config = _config_for(req.thread_id, req.user_id)
    input_state = {
        "messages": [HumanMessage(content=req.message)],
        "review_required": req.review_required,
    }
    graph = app.state.graph

    async def event_stream() -> AsyncIterator[bytes]:
        # `astream_events` is an async generator. The version="v2" schema
        # emits `on_chat_model_stream` once per token from chat models
        # (including the ReAct agent and the synthesizer).
        try:
            async for event in graph.astream_events(
                input_state, config=config, version="v2"
            ):
                node = event.get("metadata", {}).get("langgraph_node", "")

                if event.get("event") == "on_tool_start":
                    tool_name = event.get("name") or event.get("data", {}).get("name") or "tool"
                    yield _sse(f"[TOOL_START:{tool_name}]")
                    continue

                if event.get("event") == "on_chain_start" and node in TRACE_STREAM_NODES:
                    # Phase 2: append server wall-clock as `t=<ISO8601>`. New
                    # frontends split on `|` and use the timestamp to compute
                    # per-node latency on `NODE_END`; old frontends ignore
                    # anything past the first `:` and still work.
                    ts = (
                        datetime.now(timezone.utc)
                        .isoformat(timespec="milliseconds")
                        .replace("+00:00", "Z")
                    )
                    yield _sse(f"[NODE_START:{node}|t={ts}]")
                    continue

                if event.get("event") == "on_chain_end" and node in TRACE_STREAM_NODES:
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
                    yield _sse(text)
        except Exception:  # noqa: BLE001
            # astream_events closes cleanly when the graph pauses at an
            # interrupt — it does NOT raise GraphInterrupt to the consumer.
            # The interrupt is exposed via the snapshot at
            # graph.get_state(config).tasks[*].interrupts. The except branch
            # here only fires for real errors. We log the full traceback
            # server-side but mask the message from the wire — exceptions
            # can carry stack hints, file paths, or LLM provider details
            # we don't want to surface to a browser tab.
            logger.exception("[AgentFlow] /chat failed for %s", req.thread_id)
            yield _sse("[ERROR] internal server error")
            return

        # Post-stream interrupt check. astream_events may end cleanly
        # either because the graph reached END (normal) or because the
        # human_review node called interrupt() and the run paused. The
        # snapshot tells us which.
        try:
            snap = await graph.aget_state(config)
        except Exception:  # noqa: BLE001
            logger.exception("[AgentFlow] post-stream aget_state failed")
            yield _sse("[ERROR] internal server error")
            return
        if _snapshot_has_interrupt(snap):
            logger.info(
                "[AgentFlow] thread %s hit interrupt; sending sentinel",
                req.thread_id,
            )
            yield _sse("[INTERRUPT]")
        else:
            # Phase 2: emit citation count so the frontend can render
            # source chips. The synthesizer + research agent already
            # populated `state["sources"]`; the snapshot exposes the
            # final value. astream_events itself does not surface
            # per-node return values, so this is the only place to
            # read the count. `sources` may be None for chat-only
            # turns — normalize to [].
            sources = (snap.values or {}).get("sources") or []
            n = len(sources)
            yield _sse(f"[SOURCES:{n}]")
            # Emit [FINAL:<text>] as a fallback for turns where the active agent
            # used blocking invoke() instead of streaming (e.g. chat_agent via
            # LangGraph tool-calling agent). The frontend uses this only when
            # its draft buffer is empty (meaning no on_chat_model_stream tokens
            # were received). For synthesizer turns the draft already has all
            # the text, so the FINAL sentinel is safely ignored.
            final_text = (snap.values or {}).get("final_response") or ""
            if final_text:
                import json as _json
                yield _sse(f"[FINAL:{_json.dumps(final_text)}]")
            yield _sse("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable proxy buffering so tokens flush as they're produced.
            # nginx ignores Cache-Control: no-cache on its own — X-Accel-Buffering
            # is the explicit opt-out.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/upload")
async def upload(thread_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    """Ingest `file` (PDF) into the per-thread FAISS index for `thread_id`.

    Streaming: we copy the upload straight to disk via `shutil.copyfileobj`
    rather than buffering the whole body in memory — a 200 MB PDF would
    otherwise pin 200 MB of RSS until the file object is GC'd. The ingest
    step (PyPDF + FAISS embedding) runs on a worker thread via
    `asyncio.to_thread` because PyPDFLoader and the sentence-transformers
    encoder are both sync, and they together run for seconds; calling
    them on the event loop would stall every other request.

    Hardening (Phase 8 review):
      - Content-Length check (413) before any body I/O — bounded RSS.
      - Magic-byte `%PDF` check (400) after the temp file is written —
        filename-extension is a lie; the first 4 bytes are ground truth.
      - Per-thread `asyncio.Lock` around the ingest call — two concurrent
        uploads for the same thread would race on the FAISS dir and
        corrupt the index. Cross-thread uploads do not block each other.
    """
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file must be a .pdf")

    # Pre-flight size cap. Reject oversized requests at the header level
    # before copying any body to disk. Content-Length is advisory
    # (Transfer-Encoding: chunked can omit it) — for chunked uploads we
    # rely on the per-thread lock + the PyPDF loader to refuse to allocate
    # past memory limits. The first line of defense is the header.
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

    # Write to a temp file, ingest, always clean up. We cannot stream
    # directly to PyPDFLoader — it needs a real file path.
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            # Stream the upload to disk in 64 KB chunks while tracking the
            # running byte total. This is the real size enforcement: the
            # Content-Length header is advisory and chunked uploads omit it
            # entirely, so shutil.copyfileobj would happily buffer an
            # arbitrarily large body. We stop as soon as we exceed the cap.
            def _copy_bounded(src, dst):
                total = 0
                chunk_size = 64 * 1024  # 64 KB
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

        # Magic-byte check. Filename extension is user-controlled; the
        # first 4 bytes are the only honest signal. `%PDF` is the PDF
        # spec's mandatory magic number (ISO 32000-1 §7.5.2).
        with open(tmp_path, "rb") as f:
            head = f.read(4)
        if head[:4] != b"%PDF":
            raise HTTPException(
                status_code=400, detail="file is not a valid PDF (bad magic bytes)"
            )

        lock = await _get_upload_lock(thread_id)
        graph = app.state.graph
        config = _config_for(thread_id)
        async with lock:
            stats = await asyncio.to_thread(
                ingest_pdf,
                tmp_path,
                thread_id,
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
                    "[AgentFlow] failed to update documents for %s", thread_id
                )
        return {
            "status": "indexed",
            "thread_id": thread_id,
            **stats,
        }
    except HTTPException:
        # Let 413 (oversized body) and 400 (bad magic bytes) pass through
        # unchanged. Without this explicit re-raise they would be caught and
        # masked by the broad `except Exception` clause below.
        raise
    except ValueError as exc:
        # `ingest_pdf` raises ValueError when the allowlist fails (it
        # can't here — we already checked — but defensively surface as 400).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("[AgentFlow] /upload failed for %s", thread_id)
        raise HTTPException(status_code=500, detail="internal server error")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # best-effort cleanup


@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str) -> dict:
    """Snapshot of the current graph state for `thread_id`.

    Returns `{"thread_id": ..., "values": {<serialized state>}}`. Messages
    are serialized via Pydantic v2 `model_dump()`.
    """
    graph = app.state.graph
    config = _config_for(thread_id)
    try:
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("[AgentFlow] /threads/%s/state failed", thread_id)
        raise HTTPException(status_code=500, detail="internal server error")
    values = snap.values if snap else {}
    return {
        "thread_id": thread_id,
        "values": _serialize_state_values(values),
        "interrupt": _serialize_interrupt(snap),
    }


@app.get("/threads")
async def list_threads() -> dict:
    """List all distinct thread_ids with last update, message preview, and turn count."""
    threads = []
    try:
        async with aiosqlite.connect(_DB_PATH, timeout=5.0) as db:
            await _ensure_checkpoint_tables(db, getattr(app.state, "graph", None))
            cursor = await db.execute(
                "SELECT c.thread_id, c.checkpoint_id, c.checkpoint "
                "FROM checkpoints c "
                "INNER JOIN ("
                "  SELECT thread_id, max(checkpoint_id) AS last_chk "
                "  FROM checkpoints GROUP BY thread_id"
                ") latest ON c.thread_id = latest.thread_id AND c.checkpoint_id = latest.last_chk "
                "ORDER BY latest.last_chk DESC LIMIT 100"
            )
            rows = await cursor.fetchall()

        # Build base entries from the DB rows.
        entries = []
        for thread_id, checkpoint_id, checkpoint_blob in rows:
            entries.append({
                "thread_id": thread_id,
                "last_seen": _checkpoint_id_to_iso(checkpoint_id),
                "preview": None,
                "turn_count": None,
                "route": None,
                "_has_blob": bool(checkpoint_blob),
            })

        # Enrich entries with live state (route, turn_count, preview) by calling
        # aget_state concurrently for all threads that have checkpoint data.
        # LangGraph's AsyncSqliteSaver uses JSON/msgpack — NOT pickle — so we
        # must use the graph's public API rather than deserialising the raw blob.
        graph = getattr(app.state, "graph", None)
        if graph is not None:
            sem = asyncio.Semaphore(10)
            
            async def _fetch_entry_meta(entry: dict) -> dict:
                if not entry["_has_blob"]:
                    return entry
                async with sem:
                    try:
                        cfg = {"configurable": {"thread_id": entry["thread_id"]}}
                        snap = await graph.aget_state(cfg)
                        if snap and snap.values:
                            cv = snap.values
                            entry["route"] = cv.get("route")
                            entry["turn_count"] = cv.get("turn_count")
                            msgs = cv.get("messages") or []
                            for m in reversed(msgs):
                                if is_human_message(m):
                                    text = content_to_str(
                                        m.content if hasattr(m, "content") else m
                                    )
                                    entry["preview"] = text[:120] + ("…" if len(text) > 120 else "")
                                    break
                            if entry["turn_count"] is None:
                                entry["turn_count"] = sum(
                                    1 for m in msgs if is_human_message(m)
                                )
                    except Exception:
                        pass  # best-effort; entry stays with None fields
                    return entry

            entries = list(await asyncio.gather(*(_fetch_entry_meta(e) for e in entries)))

        # Strip the internal _has_blob key before returning.
        for e in entries:
            e.pop("_has_blob", None)
        threads = entries

    except Exception:
        logger.exception("[AgentFlow] /threads failed")
        raise HTTPException(status_code=500, detail="internal server error")
    return {"threads": threads}


@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str) -> dict:
    """Fetch the full human/agent message history for a thread."""
    graph = app.state.graph
    config = _config_for(thread_id)
    try:
        # aget_state gives us the current snapshot
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("[AgentFlow] /threads/%s/history failed", thread_id)
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
                "text": _message_content_to_str(
                    m.content if hasattr(m, "content") else m
                ),
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
async def review(thread_id: str, req: ReviewRequest) -> dict:
    """Resume from a human_review interrupt.

    Feeds the user's decision into the pending `interrupt()` call via
    `Command(resume=...)` — the LangGraph 1.x native resume primitive
    (same pattern used in `tests/test_graph.py`). This does NOT mutate
    state directly; it unblocks the node, which then either preserves
    the existing draft (approve branch) or overwrites it with the edit.

    Contract (matches `backend/graph/human_review.py`):
    - `action=approve` → `Command(resume="approve")`. The node returns `{}`
      when `human_input == APPROVE_SENTINEL`, so the draft in
      `final_response` is preserved verbatim.
    - `action=edit` → `Command(resume=edited_response)`. The node returns
      `{"final_response": <edit>}` for any non-sentinel value.
    - The resumed run drains to END. This call BLOCKS until completion;
      response is plain JSON.
    """
    from langgraph.types import Command

    config = _config_for(thread_id)
    graph = app.state.graph

    if req.action == "approve":
        resume_value = APPROVE_SENTINEL  # non-guessable UUID token
    else:  # edit
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
        logger.exception("[AgentFlow] /review failed for %s", thread_id)
        raise HTTPException(status_code=500, detail="internal server error")

    return {"status": "resumed", "thread_id": thread_id}


@app.get("/threads/{thread_id}/blog")
async def get_thread_blog(thread_id: str) -> dict:
    """Return the latest structured blog_output for a thread.

    Returns `{thread_id, blog_output}` where `blog_output` is the dict
    produced by blog_writer_node, or null if the thread has no blog post.
    """
    graph = app.state.graph
    config = _config_for(thread_id)
    try:
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("[AgentFlow] /threads/%s/blog failed", thread_id)
        raise HTTPException(status_code=500, detail="internal server error")
    values = snap.values if snap else {}
    blog_output = values.get("blog_output")
    return {
        "thread_id": thread_id,
        "blog_output": blog_output,
    }


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    """Delete all checkpoint rows and FAISS index for a thread.

    Removes the thread from the conversation list. This is irreversible.
    """
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    deleted_checkpoints = 0
    deleted_faiss = False

    try:
        async with aiosqlite.connect(_DB_PATH, timeout=5.0) as db:
            await _ensure_checkpoint_tables(db, getattr(app.state, "graph", None))
            cursor = await db.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
            deleted_checkpoints = cursor.rowcount
            await db.commit()
    except Exception:
        logger.exception("[AgentFlow] DELETE /threads/%s failed (DB)", thread_id)
        raise HTTPException(status_code=500, detail="internal server error")

    # Remove the FAISS index directory for this thread
    try:
        from backend.rag.ingest import INDEX_ROOT
        import shutil as _shutil
        idx_path = INDEX_ROOT / thread_id
        if idx_path.is_dir():
            await asyncio.to_thread(_shutil.rmtree, str(idx_path), ignore_errors=True)
            deleted_faiss = True
    except Exception:
        logger.warning("[AgentFlow] DELETE /threads/%s: FAISS cleanup failed", thread_id)

    logger.info(
        "[AgentFlow] deleted thread %s: %d checkpoints, faiss=%s",
        thread_id,
        deleted_checkpoints,
        deleted_faiss,
    )
    return {
        "status": "deleted",
        "thread_id": thread_id,
        "deleted_checkpoints": deleted_checkpoints,
        "deleted_faiss": deleted_faiss,
    }
