"""Regression guard for slowapi rate limiting.

``APIRouter.post``/``get`` register the exact function object they receive, so
``@limiter.limit`` must be written *beneath* ``@router.*`` (innermost, applied
first). If the order is flipped the route is registered with the raw,
unwrapped handler while the wrapped handler is bound to the module-level name —
the route never calls the limiter and the limit is silently skipped.

These tests pin that ordering so it cannot silently regress again.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import backend.routers.auth as auth_mod
import backend.routers.chat as chat_mod
import backend.routers.feedback as feedback_mod
import backend.routers.upload as upload_mod
from backend.dependencies import limiter
from backend.main import app
from backend.settings import get_settings

pytestmark = pytest.mark.tier_a

_MODULES = {
    "auth": auth_mod,
    "chat": chat_mod,
    "feedback": feedback_mod,
    "upload": upload_mod,
}

_EXPECTED_LIMITED_ROUTES = [
    ("auth", "/auth/login"),
    ("auth", "/auth/refresh"),
    ("chat", "/chat"),
    ("chat", "/review/{thread_id}"),
    ("upload", "/upload"),
    ("feedback", "/feedback"),
]


@pytest.fixture(autouse=True)
def _reset_limiter():
    try:
        limiter.reset()
    except Exception:  # noqa: BLE001 - storage reset is best-effort
        pass
    yield
    try:
        limiter.reset()
    except Exception:  # noqa: BLE001
        pass


@pytest.mark.parametrize(("mod_name", "path"), _EXPECTED_LIMITED_ROUTES)
def test_rate_limited_route_is_registered_wrapped(mod_name, path):
    module = _MODULES[mod_name]
    route = next(
        (r for r in module.router.routes if getattr(r, "path", None) == path), None
    )
    assert route is not None, f"{path} is not registered on the {mod_name} router"

    endpoint = route.endpoint
    assert getattr(endpoint, "__wrapped__", None) is not None, (
        f"{path} is registered unwrapped — @limiter.limit must sit BELOW "
        "@router.* (innermost) or the limit is silently skipped"
    )
    assert endpoint is getattr(module, endpoint.__name__), (
        f"{path} route got a different object than the module symbol — "
        "decorator order is wrong"
    )


async def test_login_enforces_rate_limit(auth_headers):
    """Bad credentials return 401 up to the limit, then 429 — never unlimited."""
    limit = get_settings().rate_limit_auth_per_minute
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        statuses = []
        for _ in range(limit + 1):
            r = await ac.post(
                "/auth/login", json={"username": "tester", "password": "wrong"}
            )
            statuses.append(r.status_code)

    assert statuses[-1] == 429, statuses
    assert statuses.count(429) == 1, statuses
    assert set(statuses[:-1]) == {401}, statuses
