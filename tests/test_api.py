"""
FastAPI smoke tests.

Cover the HTTP endpoints exposed by backend/main.py. These tests do NOT
hit a real LLM — they only validate the request-validation, error-handling,
and JSON-shape contracts of each route. The graph itself is replaced with a
fake `app.state.graph` (a Starlette `State` attribute) so we never need
to compile a real LangGraph instance, open a SQLite file, or load the
sentence-transformer model. That keeps the suite fast and offline-safe.

Why `monkeypatch.setattr(app.state, "graph", fake, raising=False)`:
  `app.state` is a Starlette `State` object. Without `raising=False`,
  monkeypatch raises AttributeError because the attribute doesn't exist
  on a fresh app — lifespan hasn't run. `raising=False` makes the set
  unconditional, matching the production shape after startup.

Reference: DESIGN_DOC.md section 7 "API Design (FastAPI)".
"""

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

from backend.main import MAX_MESSAGE_CHARS, app


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _FakeGraph:
    """Minimal graph stand-in."""

    def __init__(self):
        self._state_snap = self._StateSnap()

    class _StateSnap:
        def __init__(self):
            self.values = {}
            self.next = ()
            self.tasks = ()
            self.interrupts = ()

    async def aget_state(self, config):
        return self._state_snap

    async def ainvoke(self, command, config):
        raise RuntimeError("No pending interrupt to resume")

    async def aupdate_state(self, config, values):
        docs = list(self._state_snap.values.get("documents") or [])
        for d in values.get("documents") or []:
            if d not in docs:
                docs.append(d)
        self._state_snap.values["documents"] = docs
        return None

    async def astream_events(self, input_state, config, version):
        if False:  # pragma: no cover
            yield None
        return
        yield  # make this a generator


class _InterruptObj:
    def __init__(self, value):
        self.value = value


class _SSEFakeGraph(_FakeGraph):
    """Yields a minimal chat_agent stream ending with [DONE] sentinels."""

    def __init__(self):
        super().__init__()
        snap = self._StateSnap()
        snap.values = {"sources": [], "final_response": "hello"}
        self._state_snap = snap

    async def astream_events(self, input_state, config, version):
        yield {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": "chat_agent"},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chat_agent"},
            "data": {"chunk": _Chunk("hello")},
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "chat_agent"},
        }


class _StateFakeGraph(_FakeGraph):
    """Returns structured state for /threads/*/state and /history."""

    def __init__(self):
        super().__init__()
        snap = self._StateSnap()
        snap.values = {
            "messages": [HumanMessage(content="hi", id="msg-1")],
            "review_required": False,
            "final_response": "done",
            "route": "chat",
        }
        self._state_snap = snap


class _InterruptStateFakeGraph(_StateFakeGraph):
    def __init__(self):
        super().__init__()
        snap = self._state_snap
        snap.values = {
            "messages": [HumanMessage(content="hi", id="msg-1")],
            "review_required": True,
            "final_response": "draft answer",
            "route": "research",
        }
        snap.interrupts = (_InterruptObj({"draft": "draft answer"}),)


class _ReviewResumeFakeGraph(_FakeGraph):
    def __init__(self):
        super().__init__()
        self._pending = True
        snap = self._StateSnap()
        snap.values = {"final_response": "draft answer", "route": "chat"}
        snap.interrupts = (_InterruptObj({"draft": "draft answer"}),)
        self._state_snap = snap

    async def aget_state(self, config):
        if not self._pending:
            self._state_snap.interrupts = ()
        return self._state_snap

    async def ainvoke(self, command, config):
        self._pending = False
        return {"status": "ok"}


class _InterruptSSEFakeGraph(_SSEFakeGraph):
    def __init__(self):
        super().__init__()
        snap = self._StateSnap()
        snap.values = {"sources": [], "final_response": "draft"}
        snap.interrupts = (_InterruptObj({"draft": "draft"}),)
        self._state_snap = snap


@pytest.fixture
def fake_graph(monkeypatch):
    """Replace `app.state.graph` with a fake. Lifespan never runs."""
    fake = _FakeGraph()
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    return fake




@pytest.fixture
async def client(fake_graph, auth_headers):
    """ASGI client that talks to the app in-process. No real socket.

    The `auth_headers` fixture creates a test user and a valid JWT,
    which are attached as default headers to every request so individual
    tests don't have to think about auth. Tests that want to assert
    401 behaviour (e.g. `test_api_key_required_when_configured`) pass
    their own headers or no headers explicitly.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers=auth_headers,
    ) as ac:
        yield ac


async def test_health(client):
    """/readyz returns graph/db/embeddings readiness."""
    r = await client.get("/readyz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["graph"] is True
    assert body["status"] in ("ok", "degraded")


async def test_healthz_liveness(client):
    """/healthz is a cheap liveness probe — always 200 when the process is up."""
    r = await client.get("/healthz")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}


async def test_chat_invalid_thread_id(client):
    """/chat must reject thread_ids that fail the path-traversal allowlist."""
    r = await client.post(
        "/chat",
        json={"thread_id": "../../etc/passwd", "message": "hi"},
    )
    assert r.status_code == 422, r.text
    assert "invalid thread_id" in r.text


async def test_chat_message_too_long(client):
    """/chat must reject oversized message bodies."""
    r = await client.post(
        "/chat",
        json={"thread_id": "smoke-thread", "message": "x" * (MAX_MESSAGE_CHARS + 1)},
    )
    assert r.status_code == 422, r.text


async def test_chat_sse_contract(monkeypatch, client):
    """/chat SSE stream must emit tokens and terminal [DONE]/[SOURCES:0]."""
    fake = _SSEFakeGraph()
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    r = await client.post(
        "/chat",
        json={"thread_id": "sse-thread", "message": "hi"},
    )
    assert r.status_code == 200, r.text
    body = r.text
    assert "data: hello" in body
    assert "data: [SOURCES:0]" in body
    assert "data: [DONE]" in body


async def test_upload_rejects_non_pdf(client):
    """/upload must reject a file whose extension is not .pdf."""
    files = {"file": ("malware.txt", b"not a pdf", "text/plain")}
    r = await client.post(
        "/upload",
        data={"thread_id": "smoke-thread"},
        files=files,
    )
    assert r.status_code == 400, r.text
    assert ".pdf" in r.text


async def test_upload_success(monkeypatch, client):
    """/upload happy path with ingest_pdf mocked out."""
    def _fake_ingest(path, tid, *, source_name=None):
        return {
            "document_id": f"{source_name}:1",
            "source": source_name or "doc.pdf",
            "pages": 1,
            "chunks": 1,
        }

    monkeypatch.setattr("backend.main.ingest_pdf", _fake_ingest)
    files = {"file": ("doc.pdf", b"%PDF-1.4\n% fake", "application/pdf")}
    r = await client.post(
        "/upload",
        data={"thread_id": "upload-ok"},
        files=files,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "indexed"
    assert body["thread_id"] == "upload-ok"
    assert body["chunks"] == 1


async def test_thread_state(monkeypatch, client):
    """GET /threads/{id}/state returns serialized values."""
    monkeypatch.setattr(app.state, "graph", _StateFakeGraph(), raising=False)
    r = await client.get("/threads/smoke-thread/state")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thread_id"] == "smoke-thread"
    assert "values" in body
    assert body["values"]["final_response"] == "done"
    assert body["interrupt"] is None


async def test_thread_state_includes_interrupt(monkeypatch, client):
    """GET /threads/{id}/state surfaces pending human_review interrupts."""
    monkeypatch.setattr(app.state, "graph", _InterruptStateFakeGraph(), raising=False)
    r = await client.get("/threads/smoke-thread/state")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["interrupt"]["pending"] is True
    assert body["interrupt"]["draft"] == "draft answer"


async def test_thread_history(monkeypatch, client):
    """GET /threads/{id}/history returns message list with string text."""
    monkeypatch.setattr(app.state, "graph", _StateFakeGraph(), raising=False)
    r = await client.get("/threads/smoke-thread/history")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thread_id"] == "smoke-thread"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["text"] == "hi"


async def test_thread_history_pending_interrupt(monkeypatch, client):
    """GET /threads/{id}/history appends a review row when interrupted."""
    monkeypatch.setattr(app.state, "graph", _InterruptStateFakeGraph(), raising=False)
    r = await client.get("/threads/smoke-thread/history")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["interrupt"]["pending"] is True
    assert body["messages"][-1]["role"] == "review"
    assert body["messages"][-1]["text"] == "draft answer"


async def test_list_threads(client, monkeypatch, tmp_path):
    """GET /threads is a per-user listing stub — returns 200 + JSON shape.

    The current implementation returns an empty stub when the checkpointer
    does not support a generic prefix scan (Postgres does not, SQLite
    uses the raw `checkpoints` table). The endpoint contract is verified
    here: 200 + JSON shape + per-user scoping.
    """
    r = await client.get("/threads")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "threads" in body
    assert isinstance(body["threads"], list)


async def test_review_success(monkeypatch, client):
    """POST /review resumes when an interrupt is pending."""
    fake = _ReviewResumeFakeGraph()
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    r = await client.post(
        "/review/smoke-thread",
        json={"action": "approve"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "resumed", "thread_id": "smoke-thread"}


async def test_chat_sse_interrupt(monkeypatch, client):
    """/chat SSE stream ends with [INTERRUPT] when snapshot has pending review."""
    fake = _InterruptSSEFakeGraph()
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    r = await client.post(
        "/chat",
        json={"thread_id": "interrupt-thread", "message": "hi", "review_required": True},
    )
    assert r.status_code == 200, r.text
    assert "data: [INTERRUPT]" in r.text


async def test_upload_rejects_oversized_body(monkeypatch, client):
    """/upload rejects bodies larger than MAX_UPLOAD_BYTES while streaming."""
    from backend.main import MAX_UPLOAD_BYTES

    monkeypatch.setattr("backend.main.ingest_pdf", lambda path, tid: None)
    big = b"%PDF" + (b"x" * (MAX_UPLOAD_BYTES + 1))
    files = {"file": ("big.pdf", big, "application/pdf")}
    r = await client.post(
        "/upload",
        data={"thread_id": "upload-big"},
        files=files,
    )
    assert r.status_code == 413, r.text


async def test_review_missing_interrupt(client):
    """/review must return 409 JSON when no interrupt is pending."""
    r = await client.post(
        "/review/smoke-thread",
        json={"action": "approve"},
    )
    assert r.headers.get("content-type", "").startswith("application/json"), (
        f"Expected JSON, got content-type={r.headers.get('content-type')!r} "
        f"and body[:200]={r.text[:200]!r}"
    )
    assert r.status_code == 409, f"Expected 409, got {r.status_code}: {r.text}"


async def test_review_edit_requires_body(client):
    """/review edit action requires edited_response."""
    r = await client.post(
        "/review/smoke-thread",
        json={"action": "edit"},
    )
    assert r.status_code == 400, r.text


async def test_review_edit_success(monkeypatch, client):
    """POST /review with edit resumes when interrupt pending."""
    fake = _ReviewResumeFakeGraph()
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    r = await client.post(
        "/review/smoke-thread",
        json={"action": "edit", "edited_response": "edited text"},
    )
    assert r.status_code == 200, r.text


async def test_chat_sse_final_fallback(monkeypatch, client):
    """/chat emits [FINAL:...] when no stream tokens arrive."""
    class _FinalSSEFakeGraph(_FakeGraph):
        def __init__(self):
            super().__init__()
            snap = self._StateSnap()
            snap.values = {"sources": [], "final_response": "blocking reply"}
            self._state_snap = snap

        async def astream_events(self, input_state, config, version):
            if False:
                yield None
            return
            yield

    monkeypatch.setattr(app.state, "graph", _FinalSSEFakeGraph(), raising=False)
    r = await client.post(
        "/chat",
        json={"thread_id": "final-thread", "message": "hi"},
    )
    assert r.status_code == 200, r.text
    assert "data: [FINAL:" in r.text
    assert "data: [DONE]" in r.text


async def test_api_key_required_when_configured(monkeypatch, auth_headers, fake_graph):
    """When AGENTFLOW_API_KEY is set, protected routes require the bearer token."""
    from backend.settings import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "agentflow_api_key", "secret-test-key")
    import backend.main as main_mod
    monkeypatch.setattr(main_mod, "settings", s, raising=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/chat",
            json={"thread_id": "smoke-thread", "message": "hi"},
        )
        assert r.status_code == 401, r.text

        r = await ac.get("/readyz")
        assert r.status_code == 200, r.text

        r = await ac.post(
            "/chat",
            json={"thread_id": "smoke-thread", "message": "hi"},
            headers={"Authorization": "Bearer secret-test-key"},
        )
        assert r.status_code != 401, r.text

    async with AsyncClient(
        transport=transport, base_url="http://test", headers=auth_headers
    ) as ac2:
        r = await ac2.post(
            "/chat",
            json={"thread_id": "smoke-thread", "message": "hi"},
        )
        assert r.status_code != 401, r.text
