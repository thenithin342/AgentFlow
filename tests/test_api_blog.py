import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


class FakeGraph:
    def __init__(self, values):
        self._values = values

    async def aget_state(self, config):
        class FakeSnap:
            values = self._values
        return FakeSnap()


async def test_get_blog_output_found(monkeypatch, auth_headers):
    fake = FakeGraph({"blog_output": {"title": "Test", "content": "Hello"}})
    monkeypatch.setattr(app.state, "graph", fake, raising=False)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers
    ) as client:
        res = await client.get("/threads/t123/blog")
        assert res.status_code == 200
        data = res.json()
        assert data["thread_id"] == "t123"
        assert data["blog_output"]["title"] == "Test"


async def test_get_blog_output_not_found(monkeypatch, auth_headers):
    fake = FakeGraph({})  # No blog_output
    monkeypatch.setattr(app.state, "graph", fake, raising=False)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers
    ) as client:
        res = await client.get("/threads/t123/blog")
        assert res.status_code == 200
        data = res.json()
        assert data["blog_output"] is None
