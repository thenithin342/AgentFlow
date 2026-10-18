"""User feedback endpoint — closed-loop human ground truth for online evals.

``POST /feedback`` records a thumbs up/down (plus an optional comment, route,
and client telemetry) for a single assistant turn. Rows land in the shared
user DB (``user_feedback`` table) so ``evals/online/feedback_correlation.py``
can later correlate human ratings with offline judge scores.

Decoration order: ``@router.post`` is written OUTERMOST with
``@limiter.limit`` directly beneath it (innermost, closest to the handler).
``APIRouter.post`` registers the exact function object it receives, so if
``@limiter.limit`` sat above it the route would be registered *unwrapped* and
the limit would be silently skipped. This order is slowapi's documented
FastAPI usage and the ``Annotated[..., Body()]`` model is still parsed
correctly (slowapi's ``functools.wraps`` preserves the signature).
"""

from __future__ import annotations

import time as _time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.auth import CurrentUser, require_user
from backend.dependencies import limiter
from backend.logging_config import get_logger
from backend.settings import get_settings
from backend.validation import validate_thread_id

logger = get_logger("agentflow.feedback")
settings = get_settings()

router = APIRouter(tags=["feedback"])

_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS user_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT,
    thread_id  TEXT,
    message_id TEXT,
    rating     TEXT,
    comment    TEXT,
    route      TEXT,
    created_at REAL
);
"""


async def init_feedback_table_async(conn) -> None:
    """Create the ``user_feedback`` table using an aiosqlite connection."""
    await conn.execute(_CREATE_FEEDBACK_TABLE)
    await conn.commit()


class FeedbackRequest(BaseModel):
    thread_id: str
    message_id: str = Field(..., min_length=1, max_length=128)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1024)
    route: Literal["research", "analysis", "chat", "blog"] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("thread_id")
    @classmethod
    def thread_id_valid(cls, v: str) -> str:
        return validate_thread_id(v)


def _get_user_conn(request: Request):
    """Get the shared user DB connection from app state."""
    conn = getattr(request.app.state, "user_db_conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="user store not available")
    return conn


@router.post("/feedback")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def submit_feedback(
    req: Annotated[FeedbackRequest, Body()],
    request: Request,
    user: CurrentUser = Depends(require_user),
) -> dict:
    """Record a single thumbs up/down for an assistant turn."""
    conn = _get_user_conn(request)
    try:
        cur = await conn.execute(
            "INSERT INTO user_feedback "
            "(user_id, thread_id, message_id, rating, comment, route, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user.username,
                req.thread_id,
                req.message_id,
                req.rating,
                req.comment,
                req.route,
                _time.time(),
            ),
        )
        await conn.commit()
    except Exception as exc:  # noqa: BLE001 - surface as a recorded failure, not 500 traceback
        logger.exception("feedback_insert_failed", user=user.username)
        raise HTTPException(status_code=500, detail="could not record feedback") from exc

    logger.info(
        "feedback_recorded",
        user=user.username,
        thread_id=req.thread_id,
        rating=req.rating,
        route=req.route,
    )
    return {"status": "recorded", "id": getattr(cur, "lastrowid", None)}
