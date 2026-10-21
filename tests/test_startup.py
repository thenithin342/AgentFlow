"""Exercise real startup with a fresh local checkpoint database."""

from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import backend.main as main
from backend.routers import threads


def test_swap_node_supports_dataclasses_without_replace_protocol():
    @dataclass(frozen=True)
    class Node:
        runnable: object
        metadata: dict

    original = Node(object(), {"name": "blog_writer"})
    replacement = object()
    updated = main._swap_node_runnable(original, replacement)
    assert isinstance(updated, Node)
    assert updated.runnable is replacement
    assert updated.metadata is original.metadata
    assert original.runnable is not replacement


@pytest.mark.asyncio
async def test_fresh_startup_can_list_threads_before_the_first_chat(auth_headers, monkeypatch):
    monkeypatch.setattr(main.settings, "postgres_conn_string", None)
    monkeypatch.setattr(main, "warm_embeddings", lambda: None)
    # Startup replaces the shared blog node; restore it when this test finishes.
    monkeypatch.setitem(main.builder.nodes, "blog_writer", main.builder.nodes["blog_writer"])
    app = FastAPI()
    app.include_router(threads.router)
    async with main.lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/threads", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"threads": []}
