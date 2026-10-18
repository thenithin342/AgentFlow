"""Deterministic unit tests for the live black-box API suite.

These tests never open a socket: the suite is driven in-process through
``httpx.ASGITransport(app)`` with a fake graph attached to ``app.state``
(the same technique used by ``tests/test_api.py``). Every scenario and
failure path in ``evals/online/api_blackbox.py`` is exercised offline.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from evals.online import api_blackbox as bb

pytestmark = pytest.mark.tier_a

BASE_URL = "http://test"
RUN_ID = "live-bb-test"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _Interrupt:
    def __init__(self, value):
        self.value = value


class _StateSnap:
    def __init__(self, values=None, interrupts=()):
        self.values = values or {}
        self.next = ()
        self.tasks = ()
        self.interrupts = tuple(interrupts)


class _BaseFakeGraph:
    def __init__(self):
        self._state_snap = _StateSnap()

    async def aget_state(self, config):
        return self._state_snap

    async def ainvoke(self, command, config):
        return {"status": "ok"}

    async def aupdate_state(self, config, values):
        return None


class _SSEFakeGraph(_BaseFakeGraph):
    """Normal chat turn: token + trace markers + FINAL/SOURCES/DONE."""

    def __init__(self):
        super().__init__()
        self._state_snap = _StateSnap({"sources": [], "final_response": "hello"})

    async def astream_events(self, input_state, config, version):
        yield {"event": "on_chain_start", "metadata": {"langgraph_node": "chat_agent"}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chat_agent"},
            "data": {"chunk": _Chunk("hello")},
        }
        yield {"event": "on_chain_end", "metadata": {"langgraph_node": "chat_agent"}}


class _RagFakeGraph(_SSEFakeGraph):
    """Reports >= 1 retrieved source so the RAG round-trip can be asserted."""

    def __init__(self):
        super().__init__()
        self._state_snap = _StateSnap(
            {
                "sources": [{"source": "test.pdf", "chunk": "Northwind facts"}],
                "final_response": "Northwind facts are ...",
            }
        )

    async def astream_events(self, input_state, config, version):
        yield {"event": "on_chain_start", "metadata": {"langgraph_node": "research_agent"}}
        yield {
            "event": "on_tool_start",
            "name": "rag_search",
            "metadata": {"langgraph_node": "research_agent"},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "research_agent"},
            "data": {"chunk": _Chunk("Northwind facts")},
        }
        yield {"event": "on_chain_end", "metadata": {"langgraph_node": "research_agent"}}


class _ErrorFakeGraph(_BaseFakeGraph):
    async def astream_events(self, input_state, config, version):
        raise RuntimeError("boom")
        if False:  # pragma: no cover - makes this an async generator
            yield None


class _InterruptReviewFakeGraph(_SSEFakeGraph):
    """Pauses on an interrupt, then clears it when resumed."""

    def __init__(self):
        super().__init__()
        self._pending = True
        self._state_snap = _StateSnap(
            {"sources": [], "final_response": "draft"},
            interrupts=(_Interrupt({"draft": "draft"}),),
        )

    async def aget_state(self, config):
        if not self._pending:
            self._state_snap.interrupts = ()
        return self._state_snap

    async def ainvoke(self, command, config):
        self._pending = False
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(auth_headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        yield ac


def _suite(client, auth_headers, **kwargs):
    return bb.ApiBlackboxSuite(
        client,
        base_url=BASE_URL,
        auth_headers=auth_headers,
        run_id=RUN_ID,
        timeout=5.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

def test_parse_sse_extracts_data_payloads():
    body = "data: hello\n\ndata: [SOURCES:0]\n\ndata: [DONE]\n\n"
    assert bb.parse_sse(body) == ["hello", "[SOURCES:0]", "[DONE]"]


def test_parse_sse_joins_multiline_frames():
    assert bb.parse_sse("data: line1\ndata: line2\n\n") == ["line1\nline2"]


def test_parse_sse_preserves_leading_space():
    assert bb.parse_sse("data:  indented\n\n") == [" indented"]


def test_classify_route_detects_research():
    assert bb.classify_route(["[SOURCES:0]", "[NODE_START:research_agent|t=1]"]) == "research"
    assert bb.classify_route(["[NODE_START:chat_agent|t=1]"]) == "chat"


def test_classify_route_counts_tool_use_as_research():
    assert bb.classify_route(["[TOOL_START:rag_search]", "[DONE]"]) == "research"


def test_final_text_decodes_marker():
    assert bb._final_text(['[FINAL:"hello world"]', "[DONE]"]) == "hello world"
    assert bb._final_text(['[FINAL:""]']) is None
    assert bb._final_text(["[SOURCES:0]", "[DONE]"]) is None
    assert bb._final_text(['[FINAL:"unterminated]']) is None


# ---------------------------------------------------------------------------
# /chat SSE contract
# ---------------------------------------------------------------------------

async def test_chat_sse_contract_passes(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_chat_sse_contract()
    assert result["passed"], result["failures"]
    assert result["details"]["status"] == 200


async def test_chat_sse_contract_detects_error_marker(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _ErrorFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_chat_sse_contract()
    assert not result["passed"]
    assert any("[ERROR]" in failure for failure in result["failures"])


async def test_chat_sse_contract_requires_done(monkeypatch, client, auth_headers):
    suite = _suite(client, auth_headers)

    async def _truncated(*args, **kwargs):
        return {
            "status": 200,
            "events": ["hello", "[SOURCES:0]", '[FINAL:"x"]'],
            "ttft_s": 0.01,
            "e2e_s": 0.02,
            "error": None,
        }

    monkeypatch.setattr(suite, "_stream_chat", _truncated)
    result = await suite.test_chat_sse_contract()
    assert not result["passed"]
    assert any("DONE" in failure for failure in result["failures"])


async def test_chat_sse_contract_requires_final(monkeypatch, client, auth_headers):
    suite = _suite(client, auth_headers)

    async def _no_final(*args, **kwargs):
        return {
            "status": 200,
            "events": ["hello", "[SOURCES:0]", "[DONE]"],
            "ttft_s": 0.01,
            "e2e_s": 0.02,
            "error": None,
        }

    monkeypatch.setattr(suite, "_stream_chat", _no_final)
    result = await suite.test_chat_sse_contract()
    assert not result["passed"]
    assert any("FINAL" in failure for failure in result["failures"])


# ---------------------------------------------------------------------------
# Latency SLAs
# ---------------------------------------------------------------------------

async def test_chat_latency_sla_passes(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_chat_latency_sla()
    assert result["passed"], result["failures"]
    assert len(result["details"]["measurements"]) == 2


async def test_chat_latency_sla_fails_when_threshold_exceeded(
    monkeypatch, client, auth_headers
):
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    suite = _suite(client, auth_headers, slas={"ttft_seconds": 1e-9, "e2e_chat_seconds": 1e-9})
    result = await suite.test_chat_latency_sla()
    assert not result["passed"]
    assert any("TTFT" in failure for failure in result["failures"])


# ---------------------------------------------------------------------------
# Upload + RAG round-trip
# ---------------------------------------------------------------------------

async def test_upload_and_rag_turn_passes(monkeypatch, client, auth_headers):
    import backend.routers.upload as upload_mod

    def _fake_ingest(path, thread_id, *, source_name=None):
        return {"document_id": "doc-1", "source": source_name, "pages": 1, "chunks": 2}

    monkeypatch.setattr(upload_mod, "ingest_pdf", _fake_ingest)
    monkeypatch.setattr(app.state, "graph", _RagFakeGraph(), raising=False)

    result = await _suite(client, auth_headers).test_upload_and_rag_turn()
    assert result["passed"], result["failures"]
    assert result["details"]["sources"] >= 1
    assert result["details"]["final"]


async def test_upload_and_rag_turn_fails_on_empty_final(monkeypatch, client, auth_headers):
    # The backend suppresses [FINAL] entirely for empty text, so a server that
    # instead frames an empty payload ([FINAL:""]) is exercised via a stub.
    suite = _suite(client, auth_headers)

    class _UploadResp:
        status_code = 200
        text = ""

        def json(self):
            return {"status": "indexed", "chunks": 2}

    async def _fake_post(url, **kwargs):
        return _UploadResp()

    async def _empty_final_stream(*args, **kwargs):
        return {
            "status": 200,
            "events": ["[SOURCES:1]", '[FINAL:""]', "[DONE]"],
            "ttft_s": 0.01,
            "e2e_s": 0.02,
            "error": None,
        }

    monkeypatch.setattr(client, "post", _fake_post)
    monkeypatch.setattr(suite, "_stream_chat", _empty_final_stream)

    result = await suite.test_upload_and_rag_turn()
    assert not result["passed"]
    assert any("empty [FINAL]" in failure for failure in result["failures"])


async def test_upload_scenario_reports_failure(monkeypatch, client, auth_headers):
    monkeypatch.setattr(bb, "MINIMAL_PDF", b"not really a pdf")
    monkeypatch.setattr(app.state, "graph", _RagFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_upload_and_rag_turn()
    assert not result["passed"]
    assert any("/upload expected 200" in failure for failure in result["failures"])


# ---------------------------------------------------------------------------
# Review interrupt / resume
# ---------------------------------------------------------------------------

async def test_review_interrupt_flow_passes(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _InterruptReviewFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_review_interrupt_flow()
    assert result["passed"], result["failures"]
    assert result["details"]["review_status"] == 200


async def test_review_flow_fails_without_interrupt(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_review_interrupt_flow()
    assert not result["passed"]
    assert any("INTERRUPT" in failure for failure in result["failures"])


# ---------------------------------------------------------------------------
# Authentication boundaries
# ---------------------------------------------------------------------------

async def test_auth_enforcement_passes(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    result = await _suite(client, auth_headers).test_auth_enforcement()
    assert result["passed"], result["failures"]


async def test_auth_enforcement_static_api_key(monkeypatch, client, auth_headers):
    from backend.settings import get_settings

    monkeypatch.setattr(get_settings(), "agentflow_api_key", "secret-test-key")
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    suite = _suite(client, {"Authorization": "Bearer secret-test-key"})
    result = await suite.test_auth_enforcement()
    assert result["passed"], result["failures"]


async def test_auth_enforcement_rejects_wrong_static_key(monkeypatch, client, auth_headers):
    from backend.settings import get_settings

    monkeypatch.setattr(get_settings(), "agentflow_api_key", "secret-test-key")
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    suite = _suite(client, {"Authorization": "Bearer wrong-key"})
    result = await suite.test_auth_enforcement()
    assert not result["passed"]
    assert any("valid credentials" in failure for failure in result["failures"])


# ---------------------------------------------------------------------------
# Runner orchestration
# ---------------------------------------------------------------------------

async def test_run_includes_warmup_and_scenarios(monkeypatch, client, auth_headers):
    monkeypatch.setattr(app.state, "graph", _SSEFakeGraph(), raising=False)
    report = await _suite(client, auth_headers, warmup=True).run("chat")
    names = [s["name"] for s in report["scenarios"]]
    assert names == ["warmup", "chat_sse_contract"]
    assert report["summary"]["passed_all"] is True


async def test_run_marks_transport_exception_as_failure(monkeypatch, client, auth_headers):
    suite = _suite(client, auth_headers)

    async def _boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(suite, "_stream_chat", _boom)
    report = await suite.run("chat")
    assert report["summary"]["passed_all"] is False
    assert "exception" in report["scenarios"][0]["failures"][0]


async def test_run_rejects_unknown_suite(client, auth_headers):
    with pytest.raises(ValueError):
        await _suite(client, auth_headers).run("nope")


# ---------------------------------------------------------------------------
# Configuration / CLI
# ---------------------------------------------------------------------------

def test_resolve_base_url_precedence(monkeypatch):
    monkeypatch.delenv("AGENTFLOW_BASE_URL", raising=False)
    assert bb.resolve_base_url() == bb.DEFAULT_BASE_URL
    monkeypatch.setenv("AGENTFLOW_BASE_URL", "http://env-host:9000")
    assert bb.resolve_base_url() == "http://env-host:9000"
    assert bb.resolve_base_url("http://cli-host:1234/") == "http://cli-host:1234"


async def test_resolve_auth_static_key(monkeypatch):
    monkeypatch.setenv("AGENTFLOW_API_KEY", "env-key")
    headers = await bb.resolve_auth(None, BASE_URL)
    assert headers == {"Authorization": "Bearer env-key"}


async def test_resolve_auth_none_without_credentials(monkeypatch):
    for var in ("AGENTFLOW_API_KEY", "AGENTFLOW_USERNAME", "AGENTFLOW_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    assert await bb.resolve_auth(None, BASE_URL) is None


async def test_resolve_auth_login(monkeypatch, client):
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"access_token": "jwt-token"}

    async def _fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(client, "post", _fake_post)
    headers = await bb.resolve_auth(
        client, BASE_URL, username="u", password="p"
    )
    assert headers == {"Authorization": "Bearer jwt-token"}
    assert captured["url"] == f"{BASE_URL}/auth/login"


async def test_resolve_auth_explicit_login_beats_env_key(monkeypatch, client):
    monkeypatch.setenv("AGENTFLOW_API_KEY", "env-key")

    class _Resp:
        status_code = 200

        def json(self):
            return {"access_token": "jwt-token"}

    async def _fake_post(url, **kwargs):
        return _Resp()

    monkeypatch.setattr(client, "post", _fake_post)
    headers = await bb.resolve_auth(client, BASE_URL, username="u", password="p")
    assert headers == {"Authorization": "Bearer jwt-token"}


async def test_resolve_auth_login_rejects_non_json(monkeypatch, client):
    class _Resp:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    async def _fake_post(url, **kwargs):
        return _Resp()

    monkeypatch.setattr(client, "post", _fake_post)
    with pytest.raises(ValueError):
        await bb.resolve_auth(client, BASE_URL, username="u", password="p")


def test_build_parser_defaults():
    args = bb.build_parser().parse_args([])
    assert args.base_url is None
    assert args.suite == "all"
    assert args.warmup is False
    assert args.out is None


def test_load_slas_reads_config():
    slas = bb.load_slas()
    assert slas["ttft_seconds"] == 1.5
    assert slas["e2e_chat_seconds"] == 5.0
    assert slas["e2e_research_seconds"] == 15.0


# ---------------------------------------------------------------------------
# Reporting + main()
# ---------------------------------------------------------------------------

def _sample_report(passed: bool) -> dict:
    return {
        "base_url": BASE_URL,
        "run_id": RUN_ID,
        "suite": "all",
        "warmup": False,
        "timestamp": "2026-10-18T00:00:00+00:00",
        "git_sha": "abc123",
        "slas": {"ttft_seconds": 1.5},
        "scenarios": [
            {
                "name": "chat_sse_contract",
                "passed": passed,
                "duration_s": 0.01,
                "details": {},
                "failures": [] if passed else ["boom"],
            }
        ],
        "summary": {
            "total": 1,
            "passed": 1 if passed else 0,
            "failed": 0 if passed else 1,
            "passed_all": passed,
        },
    }


def test_write_blackbox_report_writes_json_and_md(tmp_path):
    json_path, md_path = bb.write_blackbox_report(_sample_report(True), tmp_path)
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["scenarios"][0]["name"] == "chat_sse_contract"
    assert "chat_sse_contract" in md_path.read_text(encoding="utf-8")


class _DummyClient:
    async def aclose(self):
        return None


def _patch_main(monkeypatch, report, auth_headers):
    monkeypatch.setattr(bb, "build_client", lambda base_url, timeout=30.0: _DummyClient())

    async def _fake_auth(client, base_url, **kwargs):
        return auth_headers

    async def _fake_run(client, **kwargs):
        return report

    monkeypatch.setattr(bb, "resolve_auth", _fake_auth)
    monkeypatch.setattr(bb, "run_blackbox_suite", _fake_run)


def test_main_returns_zero_on_success(monkeypatch, tmp_path):
    _patch_main(monkeypatch, _sample_report(True), {"Authorization": "Bearer x"})
    assert bb.main(["--api-key", "x", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "online_blackbox.json").exists()


def test_main_returns_one_on_failure(monkeypatch, tmp_path):
    _patch_main(monkeypatch, _sample_report(False), {"Authorization": "Bearer x"})
    assert bb.main(["--jwt", "x", "--out", str(tmp_path)]) == 1


def test_main_returns_one_without_credentials(monkeypatch):
    monkeypatch.setattr(bb, "build_client", lambda base_url, timeout=30.0: _DummyClient())

    async def _no_auth(client, base_url, **kwargs):
        return None

    monkeypatch.setattr(bb, "resolve_auth", _no_auth)
    assert bb.main([]) == 1
