from __future__ import annotations

import asyncio
import os
import tempfile
from collections import OrderedDict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.auth import CurrentUser, make_thread_id, require_user
from backend.constants import MAX_UPLOAD_BYTES
from backend.dependencies import config_for, limiter
from backend.logging_config import get_logger
from backend.rag.ingest import ingest_pdf
from backend.settings import get_settings
from backend.validation import validate_thread_id

logger = get_logger("agentflow.upload")
settings = get_settings()

router = APIRouter(tags=["upload"])

_upload_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
_upload_locks_guard = asyncio.Lock()

async def get_upload_lock(thread_id: str) -> asyncio.Lock:
    async with _upload_locks_guard:
        if thread_id not in _upload_locks:
            if len(_upload_locks) >= 1000:
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

@router.post("/upload")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def upload(
    request: Request,
    thread_id: str = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_user),
) -> dict:
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

    tmp_path = None
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

        scoped = make_thread_id(user, thread_id)
        lock = await get_upload_lock(scoped)
        graph = request.app.state.graph
        config = config_for(user, thread_id)
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
                logger.exception("upload_state_update_failed", thread_id=scoped)
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
