from __future__ import annotations

import uuid as _uuid_mod
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.auth import CurrentUser, make_thread_id
from backend.validation import validate_thread_id

limiter = Limiter(key_func=get_remote_address)

def config_for(user: CurrentUser, thread_id: str) -> dict:
    """Standard LangGraph RunnableConfig, scoped to the current user."""
    try:
        validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    scoped = make_thread_id(user, thread_id)
    return {"configurable": {"thread_id": scoped, "user_id": user.username}}

def sse(payload: str | bytes) -> bytes:
    """Format a single Server-Sent Event chunk."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        payload = str(payload)
    if "\n" not in payload:
        return f"data: {payload}\n\n".encode()
    body = "".join(f"data: {line}\n" for line in payload.split("\n"))
    return f"{body}\n".encode()

def snapshot_has_interrupt(snap) -> bool:
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

def serialize_interrupt(snap) -> dict | None:
    if not snapshot_has_interrupt(snap):
        return None
    return {"pending": True, "draft": _extract_interrupt_draft(snap)}

def serialize_state_values(values: dict) -> dict:
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

def checkpoint_id_to_iso(checkpoint_id: str | None) -> str | None:
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
