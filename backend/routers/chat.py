from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, field_validator

from backend.auth import CurrentUser, require_user
from backend.constants import MAX_MESSAGE_CHARS, SSE_TOKEN_NODES, TRACE_STREAM_NODES
from backend.dependencies import (
    config_for,
    limiter,
    snapshot_has_interrupt,
    sse,
)
from backend.graph.human_review import APPROVE_SENTINEL
from backend.graph.messages import content_to_str
from backend.logging_config import get_logger
from backend.settings import get_settings
from backend.validation import validate_thread_id

logger = get_logger("agentflow.chat")
settings = get_settings()

router = APIRouter(tags=["chat"])

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
            raise ValueError(f"message too long (max {MAX_MESSAGE_CHARS} characters)")
        return v

class ReviewRequest(BaseModel):
    action: Literal["approve", "edit"]
    edited_response: str | None = None

@router.post("/chat")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(
    request: Request,
    req: ChatRequest,
    user: CurrentUser = Depends(require_user),
) -> StreamingResponse:
    config = config_for(user, req.thread_id)
    input_state = {
        "messages": [HumanMessage(content=req.message)],
        "review_required": req.review_required,
    }
    graph = request.app.state.graph

    async def event_stream() -> AsyncIterator[bytes]:
        active_node: dict[str, str | None] = {"name": None}
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
                    yield sse(f"[TOOL_START:{tool_name}]")
                    continue

                if event.get("event") == "on_chain_start" and node in TRACE_STREAM_NODES:
                    active_node["name"] = node
                    ts = (
                        datetime.now(timezone.utc)
                        .isoformat(timespec="milliseconds")
                        .replace("+00:00", "Z")
                    )
                    yield sse(f"[NODE_START:{node}|t={ts}]")
                    continue

                if event.get("event") == "on_chain_end" and node in TRACE_STREAM_NODES:
                    if active_node["name"] == node:
                        active_node["name"] = None
                    yield sse(f"[NODE_END:{node}]")
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
                    yield sse(text)
        except Exception:
            logger.exception("chat_failed", thread_id=req.thread_id, user=user.username)
            yield sse("[ERROR] internal server error")
            return

        try:
            snap = await graph.aget_state(config)
        except Exception:
            logger.exception("post_stream_aget_state_failed")
            yield sse("[ERROR] internal server error")
            return

        if snapshot_has_interrupt(snap):
            logger.info("thread_interrupt", thread_id=req.thread_id, user=user.username)
            yield sse("[INTERRUPT]")
        else:
            sources = (snap.values or {}).get("sources") or []
            yield sse(f"[SOURCES:{len(sources)}]")
            final_text = (snap.values or {}).get("final_response") or ""
            if final_text:
                yield sse(f"[FINAL:{json.dumps(final_text)}]")
            yield sse("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

@router.post("/review/{thread_id}")
async def review(
    thread_id: str,
    req: ReviewRequest,
    request: Request,
    user: CurrentUser = Depends(require_user),
) -> dict:
    from langgraph.types import Command

    config = config_for(user, thread_id)
    graph = request.app.state.graph

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
        if not snapshot_has_interrupt(snap):
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
