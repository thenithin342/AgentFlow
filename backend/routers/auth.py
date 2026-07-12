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
