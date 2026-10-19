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

import json
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
    user_id    TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    message_id TEXT NOT NULL,
    rating     TEXT NOT NULL,
    comment    TEXT,
    route      TEXT,
    metadata   TEXT,
    created_at REAL NOT NULL
);
"""

# One rating per (user, message): re-rating a turn updates the existing row
# instead of inserting a duplicate, so correlation counts stay honest.
_CREATE_FEEDBACK_UNIQUE_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_feedback_user_message "
    "ON user_feedback(user_id, message_id);"
)

_UPSERT_FEEDBACK = """
INSERT INTO user_feedback
    (user_id, thread_id, message_id, rating, comment, route, metadata, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(user_id, message_id) DO UPDATE SET
    rating     = excluded.rating,
    comment    = excluded.comment,
    route      = excluded.route,
    metadata   = excluded.metadata,
    created_at = excluded.created_at;
"""


async def init_feedback_table_async(conn) -> None:
    """Create/migrate the ``user_feedback`` table using an aiosqlite connection.

    The table predates the ``metadata`` column, so older DBs get an additive
    ``ALTER TABLE`` (SQLite cannot add a UNIQUE constraint after the fact, hence
    the separate unique index which works for both fresh and existing tables).
    """
    await conn.execute(_CREATE_FEEDBACK_TABLE)
    async with conn.execute("PRAGMA table_info(user_feedback)") as cur:
        columns = {row[1] for row in await cur.fetchall()}
    if "metadata" not in columns:
        await conn.execute("ALTER TABLE user_feedback ADD COLUMN metadata TEXT")
    await conn.execute(_CREATE_FEEDBACK_UNIQUE_INDEX)
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
    metadata = json.dumps(req.metadata, ensure_ascii=False) if req.metadata is not None else None
    try:
        await conn.execute(
            _UPSERT_FEEDBACK,
            (
                user.username,
                req.thread_id,
                req.message_id,
                req.rating,
                req.comment,
                req.route,
                metadata,
                _time.time(),
            ),
        )
        await conn.commit()
        # lastrowid is unreliable across an ON CONFLICT update, so read the id back.
        async with conn.execute(
            "SELECT id FROM user_feedback WHERE user_id = ? AND message_id = ?",
            (user.username, req.message_id),
        ) as cur:
            row = await cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - surface as a recorded failure, not 500 traceback
        await conn.rollback()
        logger.exception("feedback_insert_failed", user=user.username)
        raise HTTPException(status_code=500, detail="could not record feedback") from exc

    logger.info(
        "feedback_recorded",
        user=user.username,
        thread_id=req.thread_id,
        rating=req.rating,
        route=req.route,
    )
    return {"status": "recorded", "id": row[0] if row else None}
