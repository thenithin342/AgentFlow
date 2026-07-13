from __future__ import annotations

import asyncio
import time as _time

import aiosqlite
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import backend.rag.ingest as ingest_module
from backend.logging_config import get_logger
from backend.rag.ingest import warm_embeddings
from backend.settings import get_settings

logger = get_logger("agentflow.health")
settings = get_settings()

router = APIRouter(tags=["health"])

@router.get("/healthz")
async def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok"}

@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness probe — checks graph, DB, embeddings, and Qdrant (if configured)."""
    timings: dict = {}

    # --- Graph ---
    t0 = _time.perf_counter()
    graph_ok = getattr(request.app.state, "graph", None) is not None
    timings["graph"] = round(_time.perf_counter() - t0, 4)

    # --- Database ---
    db_ok = False
    t1 = _time.perf_counter()
    try:
        if settings.use_postgres:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine

            assert settings.postgres_conn_string is not None
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

    # --- Embeddings ---
    embeddings_ok = False
    t2 = _time.perf_counter()
    try:
        if ingest_module._EMBEDDINGS_WARM:
            embeddings_ok = True
        else:
            await asyncio.to_thread(warm_embeddings)
            embeddings_ok = ingest_module._EMBEDDINGS_WARM
    except Exception:
        logger.exception("readyz_embeddings_failed")
    timings["embeddings"] = round(_time.perf_counter() - t2, 4)

    # --- Qdrant (Sprint 4) ---
    qdrant_ok: bool | None = None   # None = not configured
    if settings.use_qdrant:
        qdrant_ok = False
        t3 = _time.perf_counter()
        try:
            from backend.vectorstore.qdrant_store import _get_client
            client = await asyncio.to_thread(_get_client)
            await asyncio.to_thread(client.get_collections)
            qdrant_ok = True
        except Exception:
            logger.exception("readyz_qdrant_failed")
        timings["qdrant"] = round(_time.perf_counter() - t3, 4)

    # Overall readiness — Qdrant failure is only fatal when Qdrant is configured
    qdrant_required_ok = (qdrant_ok is None) or (qdrant_ok is True)
    ready = graph_ok and db_ok and embeddings_ok and qdrant_required_ok

    response_body: dict = {
        "status": "ok" if ready else "degraded",
        "graph": graph_ok,
        "db": db_ok,
        "embeddings": embeddings_ok,
        "backend": "postgres" if settings.use_postgres else "sqlite",
        "timings": timings,
    }
    if qdrant_ok is not None:
        response_body["qdrant"] = qdrant_ok

    return JSONResponse(
        status_code=200 if ready else 503,
        content=response_body,
    )
