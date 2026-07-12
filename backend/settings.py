"""
Typed configuration for AgentFlow.

All runtime config is loaded from environment variables (or a .env file
during local dev). Pydantic-settings gives us a single typed object
that every module can import safely — no scattered os.environ.get()
calls, no surprises in production.

Why centralize:
  - One place to see what env vars the app actually consumes
  - Validation at startup (missing JWT secret = fail fast, not 500
    on first request)
  - Easy to override in tests with `Settings(_env_file=None, ...)`

Reference: DESIGN_DOC.md section 8 "Configuration"
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "AgentFlow"
    app_version: str = "0.9.0"
    environment: Literal["dev", "staging", "production"] = "dev"
    debug: bool = False

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    timeout_seconds: int = 120

    # ---- CORS ----
    # Comma-separated origins. "*" allowed only in dev.
    cors_origins: str = "http://localhost:5173,http://localhost:5174"

    @field_validator("cors_origins")
    @classmethod
    def _strip_cors(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _validate_cors_for_prod(self):
        if self.environment == "production":
            for origin in self.cors_origins.split(","):
                if origin.strip() == "*":
                    raise ValueError("Wildcard CORS origins ('*') are not allowed in production")
        return self

    # ---- Persistence ----
    # If `postgres_conn_string` is set we use PostgresSaver; otherwise we
    # fall back to SQLite for local dev. The path is also used for the
    # FAISS HMAC secret file location, so it must always resolve.
    postgres_conn_string: str | None = None
    checkpoint_db_path: str = "agentflow.db"
    data_dir: str = "data"  # holds FAISS indexes + LTM indexes

    # ---- Auth ----
    # JWT settings. Secret is REQUIRED in production; if absent and
    # `environment != "production"`, a random ephemeral key is generated
    # at startup (dev convenience — all tokens invalidate on restart).
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_minutes: int = 60 * 24  # 24h
    # When set, all requests require Authorization: Bearer <key>.
    # Used as a single-user dev fallback before JWT is wired up.
    agentflow_api_key: str | None = None
    # Bootstrap admin credentials — created on first run if no users exist.
    # Change `admin_password` immediately after first deploy.
    admin_username: str = "admin"
    admin_password: str | None = None

    # ---- Rate limiting ----
    rate_limit_per_minute: int = 60
    rate_limit_auth_per_minute: int = 10

    # ---- LangSmith / observability ----
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "AgentFlow"
    langchain_endpoint: str | None = None

    # ---- LLM providers ----
    groq_api_key: str | None = None
    groq_api_key_2: str | None = None
    groq_api_key_3: str | None = None
    google_api_key: str | None = None
    tavily_api_key: str | None = None
    agentflow_require_groq: bool = False

    # ---- Derived helpers ----
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_postgres(self) -> bool:
        return bool(self.postgres_conn_string and self.postgres_conn_string.strip())

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    Cached so we only parse env once per process. Tests that need to
    override settings should clear the cache or instantiate a fresh
    Settings object directly.
    """
    s = Settings()
    if s.is_production and not s.jwt_secret:
        raise RuntimeError(
            "JWT_SECRET is required when ENVIRONMENT=production. "
            "Set it to a long random string (e.g. `openssl rand -hex 32`)."
        )
    return s
