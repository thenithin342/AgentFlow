"""Live Black-Box API Test Suite.

Validates a running AgentFlow instance strictly from an external HTTP/SSE
client perspective: the ``/chat`` streaming contract, latency SLAs (TTFT and
end-to-end), an ``/upload`` -> ``/chat`` RAG round-trip, the ``/review``
interrupt/resume flow, and authentication boundaries.

The suite talks to whatever ``httpx.AsyncClient`` it is given, so the same
scenario code runs live against uvicorn *and* in-process against
``httpx.ASGITransport(app)`` in the deterministic test suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentflow.eval.online.api_blackbox")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_RESULTS_DIR = HERE.parent / "results"
DEFAULT_BASE_URL = "http://localhost:8000"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- SSE wire markers (mirrors backend/routers/chat.py) ---------------------
NODE_START_PREFIX = "[NODE_START:"
NODE_END_PREFIX = "[NODE_END:"
TOOL_START_PREFIX = "[TOOL_START:"
SOURCES_PREFIX = "[SOURCES:"
FINAL_PREFIX = "[FINAL:"
ERROR_PREFIX = "[ERROR]"
INTERRUPT_MARKER = "[INTERRUPT]"
DONE_MARKER = "[DONE]"

SUITES = ("chat", "latency", "rag", "review", "auth", "all")

# --- Latency SLAs (PRD §6). Overridden by evals/config.yaml thresholds. -----
DEFAULT_SLAS: dict[str, float] = {
    "ttft_seconds": 1.5,
    "e2e_chat_seconds": 5.0,
    "e2e_research_seconds": 15.0,
}

# A minimal, structurally valid PDF (magic bytes + one text page) so the
# suite can exercise the multipart upload path without shipping a fixture.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    b"/Contents 4 0 R>>endobj\n"
    b"4 0 obj<</Length 47>>stream\n"
    b"BT /F1 12 Tf 72 712 Td (Northwind facts) Tj ET\n"
    b"endstream endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------

def _data_line(line: str) -> str | None:
    """Return the payload of an SSE ``data:`` line, or None for other lines."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):]
    # Per the SSE spec a single optional leading space is framing, not data.
    if payload.startswith(" "):
        payload = payload[1:]
    return payload


def parse_sse(text: str) -> list[str]:
    """Return the ordered list of ``data:`` payloads from an SSE body.

    Frames are terminated by a blank line; a frame may carry multiple
    ``data:`` lines which are concatenated with newlines.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    events: list[str] = []
    for frame in normalized.split("\n\n"):
        if not frame.strip():
            continue
        data_lines = [
            payload
            for payload in (_data_line(ln) for ln in frame.split("\n"))
            if payload is not None
        ]
        if data_lines:
            events.append("\n".join(data_lines))
    return events


def _first_with_prefix(events: list[str], prefix: str) -> str | None:
    for event in events:
        if event.startswith(prefix):
            return event
    return None


def _has(events: list[str], marker: str) -> bool:
    return any(event.startswith(marker) for event in events)


def _count_sources(events: list[str]) -> int | None:
    marker = _first_with_prefix(events, SOURCES_PREFIX)
    if not marker or not marker.endswith("]"):
        return None
    try:
        return int(marker[len(SOURCES_PREFIX):-1])
    except ValueError:
        return None


def _final_text(events: list[str]) -> str | None:
    """Decode the ``[FINAL:<json_str>]`` payload, or None when absent/invalid.

    Returns None for a missing/malformed marker and for an empty string, so
    callers can treat a falsy result as "no usable final answer".
    """
    marker = _first_with_prefix(events, FINAL_PREFIX)
    if not marker or not marker.endswith("]"):
        return None
    raw = marker[len(FINAL_PREFIX):-1]
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(value, str):
        return value or None
    return raw or None


def classify_route(events: list[str]) -> str:
    """Classify a stream as ``research`` or ``chat`` from observed evidence.

    A stream counts as research when a research node starts or any tool is
    invoked (``[TOOL_START:...]``), which is what the PRD §6 research/tool
    latency SLA is meant to cover.
    """
    for event in events:
        if event.startswith(TOOL_START_PREFIX):
            return "research"
        if event.startswith(NODE_START_PREFIX) and "research_agent" in event:
            return "research"
    return "chat"


def _result(
    name: str,
    passed: bool,
    failures: list[str] | None = None,
    details: dict[str, Any] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build a normalised per-scenario result record."""
    return {
        "name": name,
        "passed": bool(passed),
        "duration_s": round(float(duration_s), 4),
        "details": details or {},
        "failures": list(failures or []),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_slas() -> dict[str, float]:
    """Read latency SLAs from ``evals/config.yaml``, falling back to defaults."""
    slas = dict(DEFAULT_SLAS)
    try:
        from evals.config import get_thresholds

        thresholds = get_thresholds() or {}
    except Exception:
        return slas

    ttft = thresholds.get("ttft_seconds")
    if isinstance(ttft, (int, float)):
        slas["ttft_seconds"] = float(ttft)
    e2e = thresholds.get("latency_e2e_p95")
    if isinstance(e2e, dict):
        if isinstance(e2e.get("chat"), (int, float)):
            slas["e2e_chat_seconds"] = float(e2e["chat"])
        if isinstance(e2e.get("research"), (int, float)):
            slas["e2e_research_seconds"] = float(e2e["research"])
    return slas


def resolve_base_url(cli_value: str | None = None) -> str:
    """Resolve the target base URL: ``--base-url`` > env > localhost default."""
    return (cli_value or os.environ.get("AGENTFLOW_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


async def resolve_auth(
    client: Any,
    base_url: str,
    *,
    api_key: str | None = None,
    jwt: str | None = None,
    username: str | None = None,
    password: str | None = None,
    timeout: float = 30.0,
) -> dict[str, str] | None:
    """Resolve request auth headers, or None when no credentials are available.

    Resolution order: explicit ``--api-key``, explicit ``--jwt``, an explicit
    ``--username``/``--password`` login, then environment fallbacks
    (``AGENTFLOW_API_KEY`` before ``AGENTFLOW_USERNAME``/``AGENTFLOW_PASSWORD``).
    Explicit login credentials deliberately take precedence over a static key
    from the environment. Raises ``ValueError`` when a login round-trip fails.
    """
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    if jwt:
        return {"Authorization": f"Bearer {jwt}"}

    explicit_login = bool(username and password)
    user = username or os.environ.get("AGENTFLOW_USERNAME")
    pwd = password or os.environ.get("AGENTFLOW_PASSWORD")

    static_key = os.environ.get("AGENTFLOW_API_KEY")
    if static_key and not explicit_login:
        return {"Authorization": f"Bearer {static_key}"}
    if not user or not pwd:
        return None

    resp = await client.post(
        f"{base_url}/auth/login",
        json={"username": user, "password": pwd},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise ValueError(f"/auth/login failed with status {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    token = (payload or {}).get("access_token")
    if not token:
        raise ValueError("/auth/login returned no access_token")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class ApiBlackboxSuite:
    """Black-box scenarios executed against a live or in-process HTTP client."""

    def __init__(
        self,
        client: Any,
        *,
        base_url: str = DEFAULT_BASE_URL,
        auth_headers: dict[str, str] | None = None,
        run_id: str | None = None,
        slas: dict[str, float] | None = None,
        warmup: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.client = client
        self.base_url = resolve_base_url(base_url)
        self.auth_headers = dict(auth_headers or {})
        self.run_id = run_id or f"live-blackbox-{uuid.uuid4().hex[:12]}"
        self.slas = {**DEFAULT_SLAS, **(slas or {})}
        self.warmup = warmup
        self.timeout = timeout

    # -- helpers ------------------------------------------------------------
    def _headers(self, authed: bool = True) -> dict[str, str]:
        return dict(self.auth_headers) if authed else {}

    def _thread_id(self, tag: str) -> str:
        return f"{self.run_id}-{tag}"

    async def _stream_chat(
        self,
        message: str,
        thread_id: str,
        *,
        review_required: bool = False,
        authed: bool = True,
    ) -> dict[str, Any]:
        """POST /chat and consume the SSE stream, timing TTFT and end-to-end."""
        body = {
            "thread_id": thread_id,
            "message": message,
            "review_required": review_required,
        }
        start = time.perf_counter()
        ttft: float | None = None
        events: list[str] = []
        status = 0
        error: str | None = None

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat",
            json=body,
            headers=self._headers(authed),
            timeout=self.timeout,
        ) as resp:
            status = resp.status_code
            if status != 200:
                await resp.aread()
                error = resp.text[:300]
            else:
                async for line in resp.aiter_lines():
                    payload = _data_line(line)
                    if payload is None:
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    events.append(payload)

        elapsed = time.perf_counter() - start
        return {
            "status": status,
            "events": events,
            "ttft_s": ttft if ttft is not None else elapsed,
            "e2e_s": elapsed,
            "error": error,
        }

    # -- scenarios ----------------------------------------------------------
    async def test_chat_sse_contract(self) -> dict[str, Any]:
        """Tokens + valid trace markers, [SOURCES], [FINAL], [DONE], no [ERROR]."""
        start = time.perf_counter()
        failures: list[str] = []
        res = await self._stream_chat("hello", self._thread_id("contract"))
        events = res["events"]
        details = {
            "status": res["status"],
            "n_events": len(events),
            "final": _final_text(events),
        }

        if res["status"] != 200:
            failures.append(f"expected HTTP 200, got {res['status']}: {res['error']}")

        errors = [e for e in events if e.startswith(ERROR_PREFIX)]
        if errors:
            failures.append(f"unexpected [ERROR] events: {errors[:3]}")
        if not _has(events, DONE_MARKER):
            failures.append("missing terminal [DONE] marker")
        if not _has(events, FINAL_PREFIX):
            failures.append("missing [FINAL:<json>] marker")
        if not _has(events, SOURCES_PREFIX):
            failures.append("missing [SOURCES:<count>] marker")
        if not any(not e.startswith("[") for e in events):
            failures.append("no token text chunks received")

        for event in events:
            if event.startswith(NODE_START_PREFIX) and ("|t=" not in event or not event.endswith("]")):
                failures.append(f"malformed node-start marker: {event!r}")

        return _result(
            "chat_sse_contract", not failures, failures, details, time.perf_counter() - start
        )

    async def test_chat_latency_sla(self) -> dict[str, Any]:
        """TTFT < 1.5s and e2e < 5s (chat) / < 15s (research)."""
        start = time.perf_counter()
        failures: list[str] = []
        measurements: list[dict[str, Any]] = []

        queries = (
            ("chat", "hello"),
            ("research", "research the latest LangGraph release notes"),
        )
        for label, message in queries:
            res = await self._stream_chat(message, self._thread_id(f"latency-{label}"))
            if res["status"] != 200:
                failures.append(f"{label}: expected HTTP 200, got {res['status']}")
                continue
            route = classify_route(res["events"])
            sla_key = (
                "e2e_research_seconds" if route == "research" else "e2e_chat_seconds"
            )
            e2e_limit = self.slas[sla_key]
            ttft = res["ttft_s"]
            e2e = res["e2e_s"]
            if ttft > self.slas["ttft_seconds"]:
                failures.append(
                    f"{label}: TTFT {ttft:.3f}s exceeds {self.slas['ttft_seconds']}s"
                )
            if e2e > e2e_limit:
                failures.append(
                    f"{label}: e2e {e2e:.3f}s exceeds {e2e_limit}s ({route})"
                )
            measurements.append(
                {"label": label, "route": route, "ttft_s": ttft, "e2e_s": e2e, "e2e_limit_s": e2e_limit}
            )

        return _result(
            "chat_latency_sla",
            not failures,
            failures,
            {"measurements": measurements},
            time.perf_counter() - start,
        )

    async def test_upload_and_rag_turn(self) -> dict[str, Any]:
        """Upload a PDF then confirm a same-thread chat retrieves >= 1 source."""
        start = time.perf_counter()
        failures: list[str] = []
        thread_id = self._thread_id("rag")

        upload = await self.client.post(
            f"{self.base_url}/upload",
            data={"thread_id": thread_id},
            files={"file": ("test.pdf", MINIMAL_PDF, "application/pdf")},
            headers=self._headers(),
            timeout=self.timeout,
        )
        details: dict[str, Any] = {"upload_status": upload.status_code}
        if upload.status_code != 200:
            failures.append(
                f"/upload expected 200, got {upload.status_code}: {upload.text[:200]}"
            )
        else:
            upload_body = upload.json()
            details["upload"] = upload_body
            if upload_body.get("status") != "indexed":
                failures.append(f"/upload status != 'indexed': {upload_body.get('status')!r}")

        res = await self._stream_chat("what does the uploaded document say?", thread_id)
        details["chat_status"] = res["status"]
        if res["status"] != 200:
            failures.append(f"follow-up /chat expected 200, got {res['status']}")
        else:
            sources = _count_sources(res["events"])
            details["sources"] = sources
            if sources is None:
                failures.append("no [SOURCES:<count>] marker after upload")
            elif sources < 1:
                failures.append(f"expected >= 1 source after upload, got {sources}")
            final_text = _final_text(res["events"])
            details["final"] = final_text
            if not _has(res["events"], FINAL_PREFIX):
                failures.append("missing [FINAL:<json>] after upload round-trip")
            elif not final_text:
                failures.append("empty [FINAL] answer after upload round-trip")

        return _result(
            "upload_and_rag_turn", not failures, failures, details, time.perf_counter() - start
        )

    async def test_review_interrupt_flow(self) -> dict[str, Any]:
        """A review_required turn pauses on [INTERRUPT] and resumes via /review."""
        start = time.perf_counter()
        failures: list[str] = []
        thread_id = self._thread_id("review")

        res = await self._stream_chat(
            "generate a report", thread_id, review_required=True
        )
        events = res["events"]
        details: dict[str, Any] = {"stream_status": res["status"]}
        if res["status"] != 200:
            failures.append(f"/chat expected 200, got {res['status']}")
        if not _has(events, INTERRUPT_MARKER):
            failures.append("missing [INTERRUPT] marker for a review_required turn")
        if _has(events, DONE_MARKER):
            failures.append("stream should pause on [INTERRUPT] without [DONE]")

        review = await self.client.post(
            f"{self.base_url}/review/{thread_id}",
            json={"action": "approve"},
            headers=self._headers(),
            timeout=self.timeout,
        )
        details["review_status"] = review.status_code
        if review.status_code != 200:
            failures.append(
                f"/review expected 200, got {review.status_code}: {review.text[:200]}"
            )
        else:
            body = review.json()
            details["review"] = body
            if body.get("status") != "resumed":
                failures.append(f"/review status != 'resumed': {body.get('status')!r}")

        return _result(
            "review_interrupt_flow", not failures, failures, details, time.perf_counter() - start
        )

    async def test_auth_enforcement(self) -> dict[str, Any]:
        """Protected routes reject unauthenticated/invalid tokens and accept valid ones."""
        start = time.perf_counter()
        failures: list[str] = []
        details: dict[str, Any] = {}

        unauth_chat = await self.client.post(
            f"{self.base_url}/chat",
            json={"thread_id": self._thread_id("auth"), "message": "hi"},
            headers={},
            timeout=self.timeout,
        )
        details["chat_unauth"] = unauth_chat.status_code
        if unauth_chat.status_code != 401:
            failures.append(f"/chat unauthenticated: expected 401, got {unauth_chat.status_code}")

        unauth_threads = await self.client.get(
            f"{self.base_url}/threads", headers={}, timeout=self.timeout
        )
        details["threads_unauth"] = unauth_threads.status_code
        if unauth_threads.status_code != 401:
            failures.append(
                f"/threads unauthenticated: expected 401, got {unauth_threads.status_code}"
            )

        unauth_upload = await self.client.post(
            f"{self.base_url}/upload",
            data={"thread_id": self._thread_id("authup")},
            files={"file": ("t.pdf", MINIMAL_PDF, "application/pdf")},
            headers={},
            timeout=self.timeout,
        )
        details["upload_unauth"] = unauth_upload.status_code
        if unauth_upload.status_code != 401:
            failures.append(
                f"/upload unauthenticated: expected 401, got {unauth_upload.status_code}"
            )

        bad_token = await self.client.get(
            f"{self.base_url}/threads",
            headers={"Authorization": "Bearer not-a-real-token"},
            timeout=self.timeout,
        )
        details["threads_bad_token"] = bad_token.status_code
        if bad_token.status_code != 401:
            failures.append(f"/threads invalid token: expected 401, got {bad_token.status_code}")

        authed = await self.client.get(
            f"{self.base_url}/threads", headers=self._headers(), timeout=self.timeout
        )
        details["threads_authed"] = authed.status_code
        if authed.status_code != 200:
            failures.append(f"/threads valid credentials: expected 200, got {authed.status_code}")

        return _result(
            "auth_enforcement", not failures, failures, details, time.perf_counter() - start
        )

    async def _warmup(self) -> dict[str, Any]:
        """Issue health/ping requests so cold starts do not skew latency SLAs."""
        start = time.perf_counter()
        failures: list[str] = []
        details: dict[str, Any] = {}
        for path in ("/healthz", "/readyz"):
            try:
                resp = await self.client.get(
                    f"{self.base_url}{path}", timeout=self.timeout
                )
                details[path] = resp.status_code
                if resp.status_code not in (200, 503):
                    failures.append(f"{path}: unexpected status {resp.status_code}")
            except Exception as exc:  # noqa: BLE001 - report any transport error
                failures.append(f"{path}: {exc!r}")
        return _result("warmup", not failures, failures, details, time.perf_counter() - start)

    def _selected(self, suite: str) -> list[tuple[str, Any]]:
        catalogue = {
            "chat": ("chat_sse_contract", self.test_chat_sse_contract),
            "latency": ("chat_latency_sla", self.test_chat_latency_sla),
            "rag": ("upload_and_rag_turn", self.test_upload_and_rag_turn),
            "review": ("review_interrupt_flow", self.test_review_interrupt_flow),
            "auth": ("auth_enforcement", self.test_auth_enforcement),
        }
        if suite == "all":
            return list(catalogue.values())
        if suite not in catalogue:
            raise ValueError(f"unknown suite: {suite}")
        return [catalogue[suite]]

    async def run(self, suite: str = "all") -> dict[str, Any]:
        """Execute the selected scenarios and return a report dict."""
        scenarios: list[dict[str, Any]] = []
        if self.warmup:
            started = time.perf_counter()
            try:
                scenarios.append(await self._warmup())
            except Exception as exc:  # noqa: BLE001 - a failed warmup is a result
                scenarios.append(
                    _result(
                        "warmup",
                        False,
                        [f"exception: {exc!r}"],
                        duration_s=time.perf_counter() - started,
                    )
                )

        for name, method in self._selected(suite):
            started = time.perf_counter()
            try:
                scenarios.append(await method())
            except Exception as exc:  # noqa: BLE001 - transport errors are failures
                scenarios.append(
                    _result(
                        name,
                        False,
                        [f"exception: {exc!r}"],
                        duration_s=time.perf_counter() - started,
                    )
                )

        passed = sum(1 for s in scenarios if s["passed"])
        return {
            "name": "online_blackbox",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "base_url": self.base_url,
            "run_id": self.run_id,
            "suite": suite,
            "warmup": self.warmup,
            "slas": dict(self.slas),
            "scenarios": scenarios,
            "summary": {
                "total": len(scenarios),
                "passed": passed,
                "failed": len(scenarios) - passed,
                "passed_all": passed == len(scenarios),
            },
        }


async def run_blackbox_suite(
    client: Any,
    *,
    base_url: str = DEFAULT_BASE_URL,
    auth_headers: dict[str, str] | None = None,
    suite: str = "all",
    warmup: bool = False,
    slas: dict[str, float] | None = None,
    run_id: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Convenience wrapper: build a suite and run it."""
    blackbox = ApiBlackboxSuite(
        client,
        base_url=base_url,
        auth_headers=auth_headers,
        run_id=run_id,
        slas=slas,
        warmup=warmup,
        timeout=timeout,
    )
    return await blackbox.run(suite)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Online Black-Box API Suite Report",
        "",
        f"- Base URL: `{report['base_url']}`",
        f"- Run ID: `{report['run_id']}`",
        f"- Suite: `{report['suite']}`",
        f"- Warmup: {report['warmup']}",
        f"- Generated: {report['timestamp']}",
        f"- Git SHA: `{report['git_sha']}`",
        "",
        "## Summary",
        "",
        "| Total | Passed | Failed |",
        "|---|---|---|",
        f"| {report['summary']['total']} | {report['summary']['passed']} | "
        f"{report['summary']['failed']} |",
        "",
        "## Scenarios",
        "",
        "| Scenario | Result | Duration (s) |",
        "|---|---|---|",
    ]
    for scenario in report["scenarios"]:
        status = "PASS" if scenario["passed"] else "FAIL"
        lines.append(f"| {scenario['name']} | {status} | {scenario['duration_s']} |")

    slas = report.get("slas", {})
    if slas:
        lines += ["", "## SLAs (seconds)", "", "| SLA | Threshold |", "|---|---|"]
        for key, value in slas.items():
            lines.append(f"| {key} | {value} |")

    failures = [
        (scenario["name"], message)
        for scenario in report["scenarios"]
        for message in scenario["failures"]
    ]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{name}`: {message}" for name, message in failures]

    lines.append("")
    return "\n".join(lines)


def write_blackbox_report(
    report: dict[str, Any], output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Write ``online_blackbox.json`` + ``online_blackbox.md``; return both paths."""
    out = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "online_blackbox.json"
    md_path = out / "online_blackbox.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_render_markdown(report))
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_client(base_url: str, timeout: float = 30.0) -> Any:
    """Construct a live ``httpx.AsyncClient`` pointed at ``base_url``."""
    import httpx

    return httpx.AsyncClient(base_url=base_url, timeout=timeout)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="api_blackbox.py",
        description="AgentFlow live black-box API suite (/chat, /upload, /review, auth)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Target base URL (default: $AGENTFLOW_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument("--api-key", default=None, help="Static API key (Bearer)")
    parser.add_argument("--jwt", default=None, help="Pre-issued JWT bearer token")
    parser.add_argument("--username", default=None, help="Username for /auth/login")
    parser.add_argument("--password", default=None, help="Password for /auth/login")
    parser.add_argument(
        "--suite",
        choices=list(SUITES),
        default="all",
        help="Which scenarios to run (default: all)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Ping /healthz and /readyz before scenarios to avoid cold-start penalties",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report output directory (default: evals/results/)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when every scenario passes, else 1."""
    args = build_parser().parse_args(argv)
    base_url = resolve_base_url(args.base_url)

    async def _run() -> int:
        client = build_client(base_url, timeout=args.timeout)
        try:
            auth_headers = await resolve_auth(
                client,
                base_url,
                api_key=args.api_key,
                jwt=args.jwt,
                username=args.username,
                password=args.password,
                timeout=args.timeout,
            )
            if auth_headers is None:
                print(
                    "No credentials: pass --api-key/--jwt or --username/--password "
                    "(or set AGENTFLOW_API_KEY / AGENTFLOW_USERNAME+AGENTFLOW_PASSWORD).",
                    file=sys.stderr,
                )
                return 1

            report = await run_blackbox_suite(
                client,
                base_url=base_url,
                auth_headers=auth_headers,
                suite=args.suite,
                warmup=args.warmup,
                slas=load_slas(),
                timeout=args.timeout,
            )
            json_path, md_path = write_blackbox_report(report, args.out)
            summary = report["summary"]
            print(
                f"Black-box suite [{args.suite}] vs {base_url}: "
                f"{summary['passed']}/{summary['total']} passed"
            )
            print(f"  JSON: {json_path}")
            print(f"  Markdown: {md_path}")
            for scenario in report["scenarios"]:
                if not scenario["passed"]:
                    for message in scenario["failures"]:
                        print(f"  FAIL {scenario['name']}: {message}", file=sys.stderr)
            return 0 if summary["passed_all"] else 1
        finally:
            await client.aclose()

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface as a failed run, not a traceback
        logger.exception("api_blackbox_run_failed")
        print(f"Black-box suite failed to run: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
