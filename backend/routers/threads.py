from __future__ import annotations

import asyncio
import shutil

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request

from backend.auth import CurrentUser, make_thread_id, require_user
from backend.dependencies import (
    checkpoint_id_to_iso,
    config_for,
    serialize_interrupt,
    serialize_state_values,
)
from backend.graph.messages import content_to_str, is_human_message
from backend.logging_config import get_logger
from backend.settings import get_settings
from backend.validation import validate_thread_id

logger = get_logger("agentflow.threads")
settings = get_settings()

router = APIRouter(tags=["threads"])

@router.get("/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    request: Request,
    user: CurrentUser = Depends(require_user),
) -> dict:
    graph = request.app.state.graph
    config = config_for(user, thread_id)
    try:
        snap = await graph.aget_state(config)
    except Exception:
        logger.exception("get_state_failed", thread_id=thread_id, user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")
    values = snap.values if snap else {}
    return {
        "thread_id": thread_id,
        "values": serialize_state_values(values),
        "interrupt": serialize_interrupt(snap),
    }

@router.get("/threads")
async def list_threads(
    request: Request,
    user: CurrentUser = Depends(require_user),
) -> dict:
    scoped_prefix = f"user:{user.username}:"
    db_conn = getattr(request.app.state, "db_conn", None)
    kind = getattr(request.app.state, "checkpointer_kind", None)
    if db_conn is None or kind is None:
        return {"threads": []}

    sql = (
        "SELECT thread_id, checkpoint_id FROM checkpoints "
        "WHERE thread_id LIKE ? ESCAPE '\\' "
        "ORDER BY thread_id, checkpoint_id DESC"
    )
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
                        continue
                    seen.add(tid)
                    threads.append({
                        "thread_id": tid[len(scoped_prefix):],
                        "last_seen": checkpoint_id_to_iso(row[1]),
                    })
        elif kind == "postgres":
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
                    "last_seen": checkpoint_id_to_iso(row[1]),
                })
        else:
            return {"threads": []}
    except Exception:
        logger.exception("list_threads_failed", user=user.username)
        raise HTTPException(status_code=500, detail="internal server error")

    threads.sort(key=lambda t: t["last_seen"] or "", reverse=True)
    return {"threads": threads[:100]}

@router.get("/threads/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    request: Request,
    user: CurrentUser = Depends(require_user),
) -> dict:
    graph = request.app.state.graph
    config = config_for(user, thread_id)
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

    interrupt = serialize_interrupt(snap)
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

@router.get("/threads/{thread_id}/blog")
async def get_thread_blog(
    thread_id: str,
    request: Request,
    user: CurrentUser = Depends(require_user),
) -> dict:
    graph = request.app.state.graph
    config = config_for(user, thread_id)
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

@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    request: Request,
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

            assert settings.postgres_conn_string is not None
            async with AsyncPostgresSaver.from_conn_string(
                settings.postgres_conn_string
            ) as cp:
                await cp.setup()
                if hasattr(cp, "adelete_thread"):
                    await cp.adelete_thread(scoped)
                    deleted_checkpoints = 1
                else:
                    conn = cp.conn
                    if hasattr(conn, "execute"):
                        await conn.execute("DELETE FROM checkpoints WHERE thread_id = $1", scoped) # type: ignore
                        await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = $1", scoped) # type: ignore
                        await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", scoped) # type: ignore
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
