"""
JWT authentication for AgentFlow.

Two layers, in order of preference:

1. **JWT bearer token** — production path. /auth/login exchanges a
   username/password for a signed JWT; subsequent requests carry it as
   `Authorization: Bearer <token>`. Tokens are HS256-signed with a
   shared secret loaded from settings.

2. **Static API key** — dev/CI fallback. When `AGENTFLOW_API_KEY` is
   set, callers can use `Authorization: Bearer <key>` or
   `X-API-Key: <key>` and skip the login flow. Useful for service
   accounts and quick smoke tests.

Per-user thread isolation: every graph call gets `user_id` baked into
the RunnableConfig so LTM is scoped per user. `thread_id` is namespaced
as `<user_id>:<thread_id>` so two users can never collide on a thread
even if they pick the same id.

The user store is intentionally a tiny JSON file on disk
(`data/users.json`). Good enough for a self-hosted single-VM deploy.
Swap for Postgres / a real auth provider when you outgrow it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import bcrypt

from backend.settings import Settings, get_settings


_bearer = HTTPBearer(auto_error=False)


def hash_password(plain: str) -> str:
    """bcrypt hash with a 72-byte safety cap (bcrypt's hard limit)."""
    return bcrypt.hashpw(plain.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# User store (JSON file)
# ---------------------------------------------------------------------------


@dataclass
class UserRecord:
    username: str
    password_hash: str
    created_at: float


def _users_file(settings: Settings) -> Path:
    return Path(settings.data_dir) / "users.json"


def _load_users(settings: Settings) -> dict[str, UserRecord]:
    """Load user records from disk. Missing file → empty store."""
    path = _users_file(settings)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        u: UserRecord(username=u, password_hash=h["password_hash"], created_at=h.get("created_at", 0.0))
        for u, h in raw.items()
    }


def _save_users(settings: Settings, users: dict[str, UserRecord]) -> None:
    path = _users_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        u: {"password_hash": rec.password_hash, "created_at": rec.created_at}
        for u, rec in users.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_admin(settings: Settings) -> None:
    """Create the bootstrap admin user if no users exist.

    Idempotent. If `ADMIN_PASSWORD` is unset in dev we generate a random
    one and print it to the server log so the operator can copy it.
    """
    users = _load_users(settings)
    if users:
        return
    pwd = settings.admin_password or secrets.token_urlsafe(16)
    users[settings.admin_username] = UserRecord(
        username=settings.admin_username,
        password_hash=hash_password(pwd),
        created_at=time.time(),
    )
    _save_users(settings, users)
    if not settings.admin_password:
        import logging
        logging.getLogger("agentflow.auth").warning(
            "[AgentFlow] created bootstrap admin user '%s' with random password: %s "
            "(set ADMIN_PASSWORD env to pin this)",
            settings.admin_username,
            pwd,
        )


def authenticate_user(settings: Settings, username: str, password: str) -> Optional[UserRecord]:
    users = _load_users(settings)
    rec = users.get(username)
    if not rec:
        return None
    if not verify_password(password, rec.password_hash):
        return None
    return rec


# ---------------------------------------------------------------------------
# JWT issuance / verification
# ---------------------------------------------------------------------------


# Ephemeral secret used when JWT_SECRET is unset. Module-level so it is
# stable for the lifetime of the process — generating it fresh on every
# call would mean tokens signed in one request can't be verified in the
# next (the comment below was wrong about "per-process" being enforced
# by `secrets.token_bytes`, which generates a new value each call).
_EPHEMERAL_SECRET: bytes = secrets.token_bytes(32)


def _signing_secret(settings: Settings) -> bytes:
    if settings.jwt_secret:
        return settings.jwt_secret.encode("utf-8")
    # Dev fallback: ephemeral secret, stable for this process. Tokens
    # invalidate on restart, which is the right behaviour — never ship
    # without JWT_SECRET.
    import logging
    logging.getLogger("agentflow.auth").warning(
        "[AgentFlow] JWT_SECRET not set — using ephemeral random key. "
        "All tokens will invalidate on restart. Set JWT_SECRET in production."
    )
    return _EPHEMERAL_SECRET


def issue_token(settings: Settings, username: str) -> str:
    """Return a signed JWT for `username`."""
    now = int(time.time())
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + settings.jwt_access_ttl_minutes * 60,
        "iss": settings.app_name,
    }
    return jwt.encode(payload, _signing_secret(settings), algorithm=settings.jwt_algorithm)


def verify_token(settings: Settings, token: str) -> Optional[str]:
    """Return the username embedded in a valid JWT, or None."""
    try:
        payload = jwt.decode(
            token,
            _signing_secret(settings),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
            issuer=settings.app_name,
        )
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return str(sub) if sub else None


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


@dataclass
class CurrentUser:
    username: str
    # Source of the credential — useful for logging / rate-limit bucketing.
    source: str  # "jwt" | "api_key" | "public"


def require_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    """FastAPI dependency: validate the bearer token / API key.

    Resolution order:
      1. JWT bearer (production path)
      2. AGENTFLOW_API_KEY static key (dev / service accounts)
      3. Public routes only (paths in `PUBLIC_PATHS`) — see main.py wiring.

    Raises 401 if neither matches.
    """
    # Public routes — let them through with a synthetic identity.
    if request.url.path in PUBLIC_PATHS:
        return CurrentUser(username="anonymous", source="public")

    token = creds.credentials if creds else None

    if token:
        # Try JWT first.
        username = verify_token(settings, token)
        if username:
            return CurrentUser(username=username, source="jwt")

        # Then the static API key (constant-time compare).
        if settings.agentflow_api_key and hmac.compare_digest(
            token, settings.agentflow_api_key
        ):
            return CurrentUser(username="api_key", source="api_key")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Public path allowlist — populated by main.py to avoid a circular import.
PUBLIC_PATHS: set[str] = set()


def make_thread_id(user: CurrentUser, thread_id: str) -> str:
    """Scope a thread_id to a user so two users can't collide.

    Format: `user:<username>:<thread_id>`. The `user:` prefix keeps the
    namespace distinct from any future tenant_id scheme.
    """
    # Sanitise the input to keep the composed id safe to use in file
    # paths and SQLite keys.
    safe = "".join(c for c in thread_id if c.isalnum() or c in "-_.")
    return f"user:{user.username}:{safe}"
