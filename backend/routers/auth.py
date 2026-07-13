from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.auth import authenticate_user, issue_token
from backend.dependencies import limiter
from backend.logging_config import get_logger
from backend.settings import Settings, get_settings

logger = get_logger("agentflow.auth")
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
@limiter.limit(f"{settings.rate_limit_auth_per_minute}/minute")
async def login(req: LoginRequest, request: Request, settings: Settings = Depends(get_settings)) -> dict:
    """Exchange username/password for a JWT."""
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


@router.post("/refresh")
@limiter.limit(f"{settings.rate_limit_auth_per_minute}/minute")
async def refresh(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    """Issue a fresh JWT using a recently expired one.

    This allows the frontend to silently renew a token right before (or shortly after)
    it expires without forcing the user to re-authenticate.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing token")

    token = auth_header[7:].strip()

    from backend.auth import verify_token, db_get_user
    username = verify_token(settings, token, ignore_expiration=True)
    if not username:
        raise HTTPException(status_code=401, detail="invalid token")

    # Ensure the user still exists in the DB (not deleted/disabled)
    conn = getattr(request.app.state, "user_db_conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="user store not available")
    user = await db_get_user(conn, username)
    if user is None:
        raise HTTPException(status_code=401, detail="user no longer exists")

    new_token = issue_token(settings, username)
    logger.info("token_refreshed", username=username)

    return {
        "access_token": new_token,
        "token_type": "bearer",
        "expires_in": settings.jwt_access_ttl_minutes * 60,
    }
