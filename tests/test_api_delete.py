import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.graph.build_graph import get_default_graph

@pytest.mark.asyncio
async def test_delete_thread():
    # Write a quick dummy state to the checkpointer
    graph = get_default_graph()
    config = {"configurable": {"thread_id": "test-delete-123"}}
    
    # Just need to check the API response
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.delete("/threads/test-delete-123")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "deleted"
        assert data["thread_id"] == "test-delete-123"
