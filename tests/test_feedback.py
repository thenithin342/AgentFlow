"""Unit tests for the user feedback endpoint (``backend/routers/feedback.py``).

Deterministic and offline: the shared user DB connection is a per-test
aiosqlite file and the graph is never touched. Covers persistence, validation,
auth, the 503 guard, and slowapi rate limiting.
"""

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

from backend.dependencies import limiter
from backend.main import app
from backend.routers.feedback import init_feedback_table_async
from backend.settings import get_settings

pytestmark = pytest.mark.tier_a

_VALID = {"thread_id": "thread-1", "message_id": "msg-1", "rating": "up"}


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Start every test with a clean slowapi storage so rate counting is
    independent of other tests in the session."""
    try:
        limiter.reset()
    except Exception:  # noqa: BLE001 - storage reset is best-effort
        pass
    yield
    try:
        limiter.reset()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
async def feedback_db(tmp_path, monkeypatch):
    """A per-test aiosqlite connection with the user_feedback table."""
    db_path = str(tmp_path / "feedback.db")
    conn = await aiosqlite.connect(db_path)
    await init_feedback_table_async(conn)
    monkeypatch.setattr(app.state, "user_db_conn", conn, raising=False)
    yield conn
    await conn.close()


@pytest.fixture
async def client(feedback_db, auth_headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers=auth_headers,
    ) as ac:
        yield ac


async def test_submit_feedback_persists_row(client, feedback_db):
    r = await client.post(
        "/feedback",
        json={**_VALID, "route": "chat", "comment": "great answer", "metadata": {"model": "x"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "recorded"
    assert isinstance(body["id"], int)

    async with feedback_db.execute(
        "SELECT user_id, thread_id, message_id, rating, comment, route "
        "FROM user_feedback WHERE id = ?",
        (body["id"],),
    ) as cur:
        row = await cur.fetchone()
    assert row == ("tester", "thread-1", "msg-1", "up", "great answer", "chat")


async def test_submit_feedback_optional_fields_default_to_null(client, feedback_db):
    r = await client.post("/feedback", json=_VALID)
    assert r.status_code == 200, r.text
    row_id = r.json()["id"]
    async with feedback_db.execute(
        "SELECT comment, route FROM user_feedback WHERE id = ?", (row_id,)
    ) as cur:
        row = await cur.fetchone()
    assert row == (None, None)


async def test_submit_feedback_requires_auth(auth_headers):
    """No credentials → 401 (route is not in PUBLIC_PATHS)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/feedback", json=_VALID)
    assert r.status_code == 401, r.text


async def test_submit_feedback_rejects_invalid_thread_id(client):
    r = await client.post(
        "/feedback", json={**_VALID, "thread_id": "../../etc/passwd"}
    )
    assert r.status_code == 422, r.text
    assert "thread_id" in r.text


@pytest.mark.parametrize("rating", ["sideways", "UP", "", "1"])
async def test_submit_feedback_rejects_bad_rating(client, rating):
    r = await client.post("/feedback", json={**_VALID, "rating": rating})
    assert r.status_code == 422, r.text


async def test_submit_feedback_rejects_long_comment(client):
    r = await client.post("/feedback", json={**_VALID, "comment": "x" * 1025})
    assert r.status_code == 422, r.text


async def test_submit_feedback_rejects_unknown_route(client):
    r = await client.post("/feedback", json={**_VALID, "route": "unknown"})
    assert r.status_code == 422, r.text


async def test_submit_feedback_rejects_empty_message_id(client):
    r = await client.post("/feedback", json={**_VALID, "message_id": ""})
    assert r.status_code == 422, r.text


async def test_submit_feedback_503_when_user_store_missing(auth_headers, monkeypatch):
    monkeypatch.setattr(app.state, "user_db_conn", None, raising=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=auth_headers
    ) as ac:
        r = await ac.post("/feedback", json=_VALID)
    assert r.status_code == 503, r.text


async def test_submit_feedback_rate_limited(client):
    """Exceeding the per-minute limit returns 429."""
    limit = get_settings().rate_limit_per_minute
    statuses = []
    for _ in range(limit + 1):
        r = await client.post("/feedback", json=_VALID)
        statuses.append(r.status_code)
    assert 200 in statuses
    assert statuses[-1] == 429, statuses
    assert "rate limit" in r.text.lower()
