"""
Structured logging for AgentFlow.

Wraps structlog with a JSON renderer for production and a console
renderer for dev. Every log line carries a `request_id` when the
caller is inside a request context (set by the middleware in main.py).

Why structlog:
  - JSON in prod → ingest into any log aggregator (Loki, Datadog, …)
  - Pretty console in dev → readable
  - Bound context propagates through `await` / `with` without plumbing
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from backend.settings import get_settings


_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent global logging setup. Safe to call from lifespan."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    settings = get_settings()
    is_dev = not settings.is_production

    # stdlib root logger
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    # structlog: shared processors, then per-renderer
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if is_dev:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name or "agentflow")
