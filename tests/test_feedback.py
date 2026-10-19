"""Unit tests for the user feedback endpoint (``backend/routers/feedback.py``).

Deterministic and offline: the shared user DB connection is a per-test
aiosqlite file and the graph is never touched. Covers persistence, validation,
auth, the 503 guard, and slowapi rate limiting.
"""

import json

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


async def test_submit_feedback_persists_metadata_json(client, feedback_db):
    r = await client.post("/feedback", json={**_VALID, "metadata": {"model": "x", "n": 2}})
    assert r.status_code == 200, r.text
    async with feedback_db.execute(
        "SELECT metadata FROM user_feedback WHERE id = ?", (r.json()["id"],)
    ) as cur:
        (raw,) = await cur.fetchone()
    assert json.loads(raw) == {"model": "x", "n": 2}


async def test_submit_feedback_upserts_duplicate_rating(client, feedback_db):
    """Re-rating the same message updates one row instead of inserting a second."""
    first = await client.post("/feedback", json={**_VALID, "rating": "up"})
    second = await client.post(
        "/feedback", json={**_VALID, "rating": "down", "comment": "changed my mind"}
    )
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    async with feedback_db.execute(
        "SELECT COUNT(*), rating, comment FROM user_feedback "
        "WHERE user_id = ? AND message_id = ?",
        ("tester", "msg-1"),
    ) as cur:
        count, rating, comment = await cur.fetchone()
    assert count == 1
    assert rating == "down"
    assert comment == "changed my mind"


async def test_init_feedback_table_migrates_legacy_schema(tmp_path):
    """An older table without the metadata column gets an additive ALTER + index."""
    db_path = str(tmp_path / "legacy.db")
    conn = await aiosqlite.connect(db_path)
    await conn.execute(
        "CREATE TABLE user_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, "
        "thread_id TEXT, message_id TEXT, rating TEXT, comment TEXT, route TEXT, created_at REAL)"
    )
    await conn.commit()
    await init_feedback_table_async(conn)
    async with conn.execute("PRAGMA table_info(user_feedback)") as cur:
        columns = {row[1] for row in await cur.fetchall()}
    assert "metadata" in columns
    async with conn.execute("PRAGMA index_list(user_feedback)") as cur:
        indexes = [row[1] for row in await cur.fetchall()]
    assert any("user_message" in name for name in indexes)
    await conn.close()


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


async def test_submit_feedback_500_rolls_back(auth_headers, monkeypatch):
    """A store write failure returns 500 and rolls the shared conn back."""

    class _BoomConn:
        def __init__(self):
            self.rolled_back = False

        async def execute(self, *args, **kwargs):
            raise RuntimeError("disk I/O error")

        async def commit(self):  # pragma: no cover - never reached
            pass

        async def rollback(self):
            self.rolled_back = True

    boom = _BoomConn()
    import backend.routers.feedback as feedback_mod

    monkeypatch.setattr(feedback_mod, "_get_user_conn", lambda request: boom)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=auth_headers
    ) as ac:
        r = await ac.post("/feedback", json=_VALID)
    assert r.status_code == 500, r.text
    assert boom.rolled_back is True
