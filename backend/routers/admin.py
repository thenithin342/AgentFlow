"""
Admin router — user management endpoints.

All endpoints require the request to be authenticated as the configured
admin user (ADMIN_USERNAME, default: "admin").

Endpoints:
    GET    /admin/users                         — list all users
    POST   /admin/users                         — create a user
    DELETE /admin/users/{username}              — delete a user
    PUT    /admin/users/{username}/password     — change a user's password
"""

from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.auth import (
    CurrentUser,
    _validate_username,
    db_create_user,
    db_delete_user,
    db_list_users,
    db_update_password,
    hash_password,
    require_user,
)
from backend.logging_config import get_logger
from backend.settings import Settings, get_settings

logger = get_logger("agentflow.admin")
router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Dependency: require admin
# ---------------------------------------------------------------------------


def require_admin(
    current_user: CurrentUser = Depends(require_user),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    """Dependency to ensure the current user is the configured admin."""
    if current_user.username != settings.admin_username:
        logger.warning(
            "admin_access_denied",
            username=current_user.username,
            reason="not_admin_user",
        )
        raise HTTPException(status_code=403, detail="admin privileges required")
    return current_user


def _get_user_conn(request: Request):
    """Get the shared user DB connection from app state."""
    conn = getattr(request.app.state, "user_db_conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="user store not available")
    return conn


# ---------------------------------------------------------------------------
# Response / request models
# ---------------------------------------------------------------------------


class UserResponse(BaseModel):
    username: str
    created_at: float


class CreateUserRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., min_length=6, max_length=1024)


class ChangePasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6, max_length=1024)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    request: Request,
    settings: Settings = Depends(get_settings),
    _admin: CurrentUser = Depends(require_admin),
) -> List[UserResponse]:
    """List all registered users (Admin only)."""
    conn = _get_user_conn(request)
    users = await db_list_users(conn)
    return [UserResponse(username=u.username, created_at=u.created_at) for u in users]


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    req: CreateUserRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    _admin: CurrentUser = Depends(require_admin),
) -> UserResponse:
    """Create a new user (Admin only)."""
    try:
        safe_username = _validate_username(req.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    conn = _get_user_conn(request)
    try:
        new_user = await db_create_user(
            conn,
            username=safe_username,
            password_hash=hash_password(req.password),
            created_at=time.time(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    logger.info("user_created", admin=_admin.username, new_user=safe_username)
    return UserResponse(username=new_user.username, created_at=new_user.created_at)


@router.delete("/users/{username}", status_code=204)
async def delete_user(
    username: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    _admin: CurrentUser = Depends(require_admin),
) -> None:
    """Delete a user by username (Admin only).

    You cannot delete the admin account itself.
    """
    try:
        safe_username = _validate_username(username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if safe_username == settings.admin_username:
        raise HTTPException(status_code=400, detail="cannot delete the admin account")

    conn = _get_user_conn(request)
    deleted = await db_delete_user(conn, safe_username)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"user '{safe_username}' not found")

    logger.info("user_deleted", admin=_admin.username, deleted_user=safe_username)
    # 204 No Content — return None implicitly


@router.put("/users/{username}/password", response_model=UserResponse)
async def change_password(
    username: str,
    req: ChangePasswordRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    _admin: CurrentUser = Depends(require_admin),
) -> UserResponse:
    """Change a user's password (Admin only).

    Works for any user including the admin themselves.
    """
    try:
        safe_username = _validate_username(username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    conn = _get_user_conn(request)
    updated = await db_update_password(conn, safe_username, hash_password(req.new_password))
    if not updated:
        raise HTTPException(status_code=404, detail=f"user '{safe_username}' not found")

    logger.info("password_changed", admin=_admin.username, target_user=safe_username)

    # Return the user record for confirmation
    from backend.auth import db_get_user
    rec = await db_get_user(conn, safe_username)
    if rec is None:
        raise HTTPException(status_code=404, detail="user not found after update")
    return UserResponse(username=rec.username, created_at=rec.created_at)
