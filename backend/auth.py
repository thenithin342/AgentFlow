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

User store (Sprint 4):
  Migrated from data/users.json → a `users` table in the same
  SQLite/Postgres database used by the LangGraph checkpointer.
  This eliminates the filelock race condition and enables multi-replica
  deploys. On first startup, any existing users.json is automatically
  migrated and left in place as a backup (not deleted).
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.settings import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger("agentflow.auth")

# Username charset: letters/digits/._- only, 1..64 chars.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def _validate_username(username: str) -> str:
    """Raise ValueError if username is not a safe identifier."""
    if not isinstance(username, str) or not _USERNAME_RE.fullmatch(username):
        got = username if isinstance(username, str) else type(username).__name__
        raise ValueError(f"invalid username: must match [A-Za-z0-9._-]{{1,64}} (got {got!r})")
    return username


def hash_password(plain: str) -> str:
    """bcrypt hash with a 72-byte safety cap (bcrypt's hard limit)."""
    return bcrypt.hashpw(plain.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# User record
# ---------------------------------------------------------------------------


@dataclass
class UserRecord:
    username: str
    password_hash: str
    created_at: float


# ---------------------------------------------------------------------------
# SQLite user table — schema + helpers
# ---------------------------------------------------------------------------

_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);
"""


def init_user_table_sync(db_path: str) -> None:
    """Create the users table synchronously (called from lifespan before async loop)."""
    import sqlite3
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute(_CREATE_USERS_TABLE)
        conn.commit()


async def init_user_table_async(conn) -> None:
    """Create the users table using an aiosqlite connection."""
    await conn.execute(_CREATE_USERS_TABLE)
    await conn.commit()


# ---------------------------------------------------------------------------
# Async user CRUD (used by routers)
# ---------------------------------------------------------------------------


async def db_get_user(conn, username: str) -> UserRecord | None:
    """Fetch a single user by username. Returns None if not found."""
    async with conn.execute(
        "SELECT username, password_hash, created_at FROM users WHERE username = ?",
        (username,),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    return UserRecord(username=row[0], password_hash=row[1], created_at=row[2])


async def db_list_users(conn) -> list[UserRecord]:
    """Return all users ordered by created_at."""
    async with conn.execute(
        "SELECT username, password_hash, created_at FROM users ORDER BY created_at"
    ) as cur:
        rows = await cur.fetchall()
    return [UserRecord(username=r[0], password_hash=r[1], created_at=r[2]) for r in rows]


async def db_create_user(conn, username: str, password_hash: str, created_at: float) -> UserRecord:
    """Insert a new user. Raises ValueError if username already exists."""
    try:
        await conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, created_at),
        )
        await conn.commit()
    except Exception as exc:
        # sqlite3.IntegrityError if duplicate — translate to ValueError for callers
        if "UNIQUE" in str(exc) or "unique" in str(exc):
            raise ValueError(f"user '{username}' already exists") from exc
        raise
    return UserRecord(username=username, password_hash=password_hash, created_at=created_at)


async def db_delete_user(conn, username: str) -> bool:
    """Delete a user by username. Returns True if deleted, False if not found."""
    cur = await conn.execute("DELETE FROM users WHERE username = ?", (username,))
    await conn.commit()
    return cur.rowcount > 0


async def db_update_password(conn, username: str, new_hash: str) -> bool:
    """Update a user's password hash. Returns True if updated, False if not found."""
    cur = await conn.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (new_hash, username),
    )
    await conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Sync wrappers (used by ensure_admin + authenticate_user at startup)
# ---------------------------------------------------------------------------


def _sync_get_user(db_path: str, username: str) -> UserRecord | None:
    import sqlite3
    with sqlite3.connect(db_path, timeout=10) as conn:
        cur = conn.execute(
            "SELECT username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return UserRecord(username=row[0], password_hash=row[1], created_at=row[2])


def _sync_count_users(db_path: str) -> int:
    import sqlite3
    with sqlite3.connect(db_path, timeout=10) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users")
        return cur.fetchone()[0]


def _sync_upsert_user(db_path: str, rec: UserRecord) -> None:
    import sqlite3
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (rec.username, rec.password_hash, rec.created_at),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# JSON → SQLite migration (one-shot, runs at startup)
# ---------------------------------------------------------------------------


def _migrate_json_to_db(settings: Settings) -> None:
    """If data/users.json exists and the DB users table is empty, migrate all
    JSON users into SQLite. The JSON file is kept as a backup (not deleted).
    This is idempotent — safe to call on every startup.
    """
    json_path = Path(settings.data_dir) / "users.json"
    if not json_path.exists():
        return

    db_path = settings.checkpoint_db_path
    if _sync_count_users(db_path) > 0:
        # DB already has users — migration already done or users were created
        # directly in the DB. Don't overwrite.
        return

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[auth] Could not read users.json for migration: %s", exc)
        return

    migrated = 0
    import sqlite3
    with sqlite3.connect(db_path, timeout=10) as conn:
        for username, data in raw.items():
            try:
                safe = _validate_username(username)
            except ValueError:
                logger.warning("[auth] Skipping invalid username during migration: %r", username)
                continue
            conn.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (safe, data.get("password_hash", ""), data.get("created_at", time.time())),
            )
            migrated += 1
        conn.commit()

    logger.info("[auth] Migrated %d user(s) from users.json → SQLite users table", migrated)


# ---------------------------------------------------------------------------
# Bootstrap admin
# ---------------------------------------------------------------------------


def ensure_admin(settings: Settings) -> None:
    """Create the bootstrap admin user if no users exist in the DB.

    Idempotent. If `ADMIN_PASSWORD` is unset in dev we generate a random
    one and print it to the server log so the operator can copy it.
    """
    # First: run the one-shot JSON → DB migration
    _migrate_json_to_db(settings)

    db_path = settings.checkpoint_db_path
    if _sync_count_users(db_path) > 0:
        return

    if not settings.admin_password and settings.is_production:
        raise RuntimeError("ADMIN_PASSWORD must be set in production environment")

    pwd = settings.admin_password or secrets.token_urlsafe(16)
    admin_user = _validate_username(settings.admin_username)
    rec = UserRecord(
        username=admin_user,
        password_hash=hash_password(pwd),
        created_at=time.time(),
    )
    _sync_upsert_user(db_path, rec)

    if not settings.admin_password:
        logger.warning(
            "[AgentFlow] created bootstrap admin user '%s' with random password: %s "
            "(set ADMIN_PASSWORD env to pin this)",
            settings.admin_username,
            pwd,
        )


# ---------------------------------------------------------------------------
# Auth functions
# ---------------------------------------------------------------------------


def authenticate_user(settings: Settings, username: str, password: str) -> UserRecord | None:
    """Validate credentials against the SQLite users table.

    Rejects malformed usernames before hitting the DB (defence in depth).
    """
    try:
        safe_username = _validate_username(username)
    except ValueError:
        return None
    rec = _sync_get_user(settings.checkpoint_db_path, safe_username)
    if not rec:
        return None
    if not verify_password(password, rec.password_hash):
        return None
    return rec


# ---------------------------------------------------------------------------
# JWT issuance / verification
# ---------------------------------------------------------------------------

# Ephemeral secret — stable for the lifetime of this process. Used when
# JWT_SECRET is not set (dev convenience only).
_EPHEMERAL_SECRET: bytes = secrets.token_bytes(32)


def _signing_secret(settings: Settings) -> bytes:
    if settings.jwt_secret:
        return settings.jwt_secret.encode("utf-8")
    logger.warning(
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


def verify_token(settings: Settings, token: str, ignore_expiration: bool = False) -> str | None:
    """Return the username embedded in a valid JWT, or None."""
    try:
        payload = jwt.decode(
            token,
            _signing_secret(settings),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"], "verify_exp": not ignore_expiration},
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
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
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
    if not token:
        token = request.headers.get("X-API-Key")

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
    safe_user = _validate_username(user.username)
    safe = "".join(c for c in thread_id if c.isalnum() or c in "-_.")
    if not safe:
        raise ValueError("thread_id contains no safe characters")
    return f"user:{safe_user}:{safe}"
