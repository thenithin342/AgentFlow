"""Deterministic unit tests for the operational monitor + safety scanner.

Zero network: the Prometheus parser, KPI aggregation, log processing, safety
detectors, masking, threshold evaluation, and reporting are all pure functions.
The one async path (output sampling) is driven through a stub HTTP client.
Toxicity scoring is injected so no LLM is ever called.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from evals.online import monitor as mon

pytestmark = pytest.mark.tier_a


# ---------------------------------------------------------------------------
# Prometheus parser
# ---------------------------------------------------------------------------

def test_parse_prometheus_types_help_and_labels():
    text = "\n".join(
        [
            "# HELP http_requests_total Total requests",
            "# TYPE http_requests_total counter",
            'http_requests_total{method="POST",handler="/chat",status="200"} 8',
            'http_requests_total{method="POST",handler="/chat",status="500"} 2',
            "",
        ]
    )
    parsed = mon.parse_prometheus(text)
    assert parsed.types["http_requests_total"] == "counter"
    assert parsed.help["http_requests_total"] == "Total requests"
    assert parsed.sum("http_requests_total", status="200") == 8.0
    assert parsed.sum("http_requests_total", handler="/chat") == 10.0


def test_parse_prometheus_skips_malformed_lines():
    text = "garbage line without value\nhttp_requests_total 5\n# not a type\n"
    parsed = mon.parse_prometheus(text)
    assert parsed.sum("http_requests_total") == 5.0


def test_parse_labels_handles_escaped_quotes():
    labels = mon.parse_labels(r'method="GET",path="/a\"b",code="ok"')
    assert labels == {"method": "GET", 'path': '/a"b', "code": "ok"}


def test_parse_prometheus_parses_inf_and_nan():
    parsed = mon.parse_prometheus('m_bucket{le="+Inf"} 3\nm 1')
    assert parsed.series("m_bucket")[0].value == 3.0


# ---------------------------------------------------------------------------
# Histogram quantiles / percentiles
# ---------------------------------------------------------------------------

def _latency_histogram() -> str:
    return "\n".join(
        [
            "# TYPE http_request_duration_seconds histogram",
            'http_request_duration_seconds_bucket{le="0.1"} 0',
            'http_request_duration_seconds_bucket{le="0.5"} 50',
            'http_request_duration_seconds_bucket{le="1"} 100',
            'http_request_duration_seconds_bucket{le="+Inf"} 100',
            "http_request_duration_seconds_count 100",
            "",
        ]
    )


def test_histogram_quantiles_interpolate_buckets():
    parsed = mon.parse_prometheus(_latency_histogram())
    quants = mon.histogram_quantiles(parsed, mon.LATENCY_METRIC, (0.5, 0.95))
    assert quants["p50"] == pytest.approx(0.5)
    assert quants["p95"] == pytest.approx(0.95)


def test_histogram_quantile_without_samples_is_none():
    parsed = mon.parse_prometheus("")
    assert mon.histogram_quantiles(parsed, mon.LATENCY_METRIC, (0.5,))["p50"] is None


def test_percentile_empty_single_and_midpoint():
    assert mon.percentile([], 0.5) is None
    assert mon.percentile([3.0], 0.9) == 3.0
    assert mon.percentile([100.0, 200.0, 300.0, 400.0], 0.5) == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Log processing
# ---------------------------------------------------------------------------

def test_parse_structlog_lines_reads_json_and_plain():
    lines = [
        json.dumps({"event": "chat_stream_timeout", "level": "warning", "timeout_s": 300}),
        "[AgentFlow] router LLM failed; falling back to 'chat'",
        "",
        "not json",
    ]
    events = mon.parse_structlog_lines(lines)
    assert [mon._event_name(e) for e in events] == [
        "chat_stream_timeout",
        "[AgentFlow] router LLM failed; falling back to 'chat'",
        "not json",
    ]


def test_parse_structlog_lines_reads_console_format():
    line = "2026-10-18T03:00:00.123456Z [info     ] request_completed   path=/chat"
    events = mon.parse_structlog_lines([line])
    assert events[0]["event"] == "request_completed"
    assert events[0]["level"] == "info"


def test_filter_window_drops_old_but_keeps_untimestamped():
    now = datetime(2026, 10, 18, tzinfo=timezone.utc)
    old = (now - timedelta(hours=48)).isoformat()
    recent = (now - timedelta(hours=1)).isoformat()
    events = [
        {"event": "old", "timestamp": old},
        {"event": "recent", "timestamp": recent},
        {"event": "no_ts"},
    ]
    kept = mon.filter_window(events, window_hours=24, now=now)
    assert [e["event"] for e in kept] == ["recent", "no_ts"]


def test_is_fallback_event_structured_and_text():
    assert mon.is_fallback_event({"router_fallback": True})
    assert mon.is_fallback_event({"event": "router LLM failed; falling back to 'chat'"})
    assert not mon.is_fallback_event({"event": "request_completed"})


# ---------------------------------------------------------------------------
# KPI aggregation
# ---------------------------------------------------------------------------

def test_compute_kpis_requests_and_latency_from_metrics():
    metrics = mon.parse_prometheus(
        "\n".join(
            [
                'http_requests_total{status="200"} 8',
                'http_requests_total{status="500"} 2',
                _latency_histogram(),
                "",
            ]
        )
    )
    kpis = mon.compute_operational_kpis(metrics, [], window_hours=24)
    assert kpis["requests"] == {"total": 10, "success": 8, "errors": 2, "source": "metrics"}
    assert kpis["success_rate"] == pytest.approx(0.8)
    assert kpis["error_rate"] == pytest.approx(0.2)
    assert kpis["latency_seconds"]["p50"] == pytest.approx(0.5)


def test_compute_kpis_counts_non_numeric_status_as_success():
    metrics = mon.parse_prometheus('http_requests_total{status="2xx"} 4\n')
    kpis = mon.compute_operational_kpis(metrics, [], window_hours=24)
    assert kpis["requests"]["success"] == 4
    assert kpis["requests"]["errors"] == 0


def test_compute_kpis_fallback_and_timeout_rates_from_logs():
    events = [
        {"event": "request_completed", "path": "/chat"},
        {"event": "request_completed", "path": "/chat"},
        {"event": "request_completed", "path": "/chat"},
        {"event": "request_completed", "path": "/chat"},
        {"event": "chat_stream_timeout", "timeout_s": 300},
        {"event": "[AgentFlow] router LLM failed; falling back to 'chat'"},
    ]
    kpis = mon.compute_operational_kpis(None, events, window_hours=24)
    assert kpis["requests"]["source"] == "logs"
    assert kpis["fallback"] == {"count": 1, "chat_requests": 4, "rate": 0.25}
    assert kpis["timeout"]["rate"] == 0.25


def test_compute_kpis_latency_falls_back_to_log_durations():
    events = [
        {"event": "request_completed", "duration_ms": ms} for ms in (100, 200, 300, 400)
    ]
    kpis = mon.compute_operational_kpis(None, events, window_hours=24)
    assert kpis["latency_seconds"]["p50"] == pytest.approx(250.0)


def test_compute_kpis_no_signal_is_none_not_zero():
    kpis = mon.compute_operational_kpis(None, [], window_hours=24)
    assert kpis["success_rate"] is None
    assert kpis["error_rate"] is None
    assert kpis["fallback"]["rate"] is None


def test_compute_kpis_aggregates_usage_cost():
    events = [
        {"event": "turn", "model": "m", "usage_metadata": {"input_tokens": 1000, "output_tokens": 500}}
    ]
    pricing = {"m": {"input_per_1k": 1.0, "output_per_1k": 2.0}, "default": {}}
    kpis = mon.compute_operational_kpis(None, events, window_hours=24, pricing=pricing)
    assert kpis["usage"]["input_tokens"] == 1000
    assert kpis["usage"]["output_tokens"] == 500
    assert kpis["usage"]["cost_per_query"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Cost / token accounting
# ---------------------------------------------------------------------------

def test_estimate_cost_uses_model_then_default():
    pricing = {
        "m": {"input_per_1k": 1.0, "output_per_1k": 3.0},
        "default": {"input_per_1k": 0.5, "output_per_1k": 0.5},
    }
    assert mon.estimate_cost(1000, 1000, "m", pricing=pricing) == pytest.approx(4.0)
    assert mon.estimate_cost(1000, 1000, "unknown", pricing=pricing) == pytest.approx(1.0)


def test_aggregate_usage_sums_and_averages():
    records = [
        {"usage_metadata": {"input_tokens": 10, "output_tokens": 5}},
        {"input_tokens": 20, "output_tokens": 5},
    ]
    usage = mon.aggregate_usage(records, pricing={"default": {"input_per_1k": 0.0}})
    assert usage["n_queries"] == 2
    assert usage["total_tokens"] == 40
    assert usage["cost_per_query"] == 0.0


# ---------------------------------------------------------------------------
# Safety detectors + masking
# ---------------------------------------------------------------------------

def test_detect_pii_positive_and_negative():
    text = "email jane.doe@example.com, phone 555-123-4567, ssn 123-45-6789"
    pii = mon.detect_pii(text)
    assert pii["email"] == ["jane.doe@example.com"]
    assert "123-45-6789" in pii["ssn"]
    assert mon.detect_pii("nothing sensitive here")["email"] == []


def test_detect_secrets_provider_prefixes_and_entropy():
    text = "key gsk_test_abc123 and AIzaSyABCDEFGHIJ1234567890"
    secrets = mon.detect_secrets(text)
    assert secrets["groq_key"] == ["gsk_test_abc123"]
    assert secrets["google_key"] == ["AIzaSyABCDEFGHIJ1234567890"]


def test_mask_variants():
    assert mon.mask("gsk_test_abc123") == "gsk_***"
    assert mon.mask("sk-abcdef123456") == "sk-***"
    assert mon.mask("AIzaSyABCDEFGHIJ") == "AIza***"
    assert mon.mask("jane.doe@example.com") == "jane.doe@***"
    assert mon.mask("Bearer abcdefghij") == "Bearer ***"
    assert mon.mask("GROQ_API_KEY=gsk_test_abc123") == "GROQ_API_KEY=***"


def test_redact_text_masks_and_is_idempotent():
    once = mon.redact_text("contact jane@example.com with gsk_test_abc123")
    assert "jane@example.com" not in once
    assert "gsk_test_abc123" not in once
    assert mon.redact_text(once) == once


def test_redact_report_walks_nested_structures():
    report = {"notes": ["leak jane@example.com"], "n": 1, "nested": {"k": "gsk_test_abc123"}}
    redacted = mon.redact_report(report)
    assert "jane@example.com" not in json.dumps(redacted)
    assert "gsk_test_abc123" not in json.dumps(redacted)


# ---------------------------------------------------------------------------
# scan_live_outputs
# ---------------------------------------------------------------------------

def test_scan_live_outputs_counts_and_masks():
    samples = [
        "reach me at jane.doe@example.com",
        "here is the key gsk_test_abc123",
        "I cannot help with that request",
    ]
    result = mon.scan_live_outputs(samples)
    assert result["n_samples"] == 3
    assert result["pii"]["counts"]["total"] == 1
    assert result["secrets"]["counts"]["total"] == 1
    assert result["refusals"]["count"] == 1
    blob = json.dumps(result)
    assert "jane.doe@example.com" not in blob
    assert "gsk_test_abc123" not in blob
    assert "jane.doe@***" in blob


def test_scan_live_outputs_samples_toxicity_with_injected_fn():
    result = mon.scan_live_outputs(["clean", "toxic"], toxicity_fn=lambda _text: 0.4)
    assert result["toxicity"]["n_scored"] == 2
    assert result["toxicity"]["median"] == pytest.approx(0.4)
    assert result["toxicity"]["max"] == pytest.approx(0.4)


def test_scan_live_outputs_records_toxicity_failures_without_raising():
    def _boom(_text):
        raise RuntimeError("judge down")

    result = mon.scan_live_outputs(["a", "b"], toxicity_fn=_boom)
    assert result["toxicity"]["n_scored"] == 0
    assert result["toxicity"]["failures"] == 2


def test_scan_live_outputs_respects_sample_size():
    result = mon.scan_live_outputs(["a", "b", "c"], sample_size=2)
    assert result["n_samples"] == 2


def test_high_entropy_hits_are_reported_but_not_gated():
    result = mon.scan_live_outputs(["opaque aZ3kP9vQ2wX7mN4bR8tY6uL1sD5fG0hJ"])
    counts = result["secrets"]["counts"]
    assert counts["high_entropy"] == 1
    assert counts["total"] == 1
    assert counts["strong_total"] == 0
    checks = mon.evaluate_thresholds(
        {"kpis": {}, "safety": result}, mon.load_monitor_thresholds()
    )["checks"]
    assert checks["leakage_protected_hits"]["passed"] is True


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------

def _report(**overrides) -> dict:
    base = {
        "kpis": {
            "success_rate": 0.99,
            "error_rate": 0.01,
            "fallback": {"rate": 0.0},
            "timeout": {"rate": 0.0},
            "usage": {"cost_per_query": 0.0},
        },
        "safety": {
            "pii": {"counts": {"total": 0}},
            "secrets": {"counts": {"total": 0}},
            "toxicity": {"median": 0.0, "max": 0.0},
        },
    }
    base.update(overrides)
    return base


def test_evaluate_thresholds_all_pass():
    checks = mon.evaluate_thresholds(_report(), mon.load_monitor_thresholds())["checks"]
    assert checks
    assert all(check["passed"] for check in checks.values())


def test_evaluate_thresholds_flags_failures():
    report = _report(
        kpis={
            "success_rate": 0.5,
            "error_rate": 0.5,
            "fallback": {"rate": 0.9},
            "timeout": {"rate": 0.9},
            "usage": {},
        },
        safety={
            "pii": {"counts": {"total": 3}},
            "secrets": {"counts": {"total": 1}},
            "toxicity": {"median": 0.9, "max": 1.0},
        },
    )
    checks = mon.evaluate_thresholds(report, mon.load_monitor_thresholds())["checks"]
    failed = {name for name, check in checks.items() if not check["passed"]}
    assert {"success_rate", "error_rate", "fallback_rate", "timeout_rate"} <= failed
    assert {"leakage_pii_hits", "leakage_protected_hits", "toxicity_median"} <= failed


def test_evaluate_thresholds_skips_none_values():
    report = _report(
        kpis={
            "success_rate": None,
            "error_rate": None,
            "fallback": {"rate": None},
            "timeout": {"rate": None},
            "usage": {},
        }
    )
    checks = mon.evaluate_thresholds(report, mon.load_monitor_thresholds())["checks"]
    assert "success_rate" not in checks
    assert "fallback_rate" not in checks


def test_load_monitor_thresholds_reads_config():
    thresholds = mon.load_monitor_thresholds()
    assert thresholds["fallback_rate"] == pytest.approx(0.10)
    assert thresholds["timeout_rate"] == pytest.approx(0.02)
    assert thresholds["success_rate"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_build_and_write_monitor_report(tmp_path):
    kpis = mon.compute_operational_kpis(None, [], window_hours=24)
    safety = mon.scan_live_outputs(["key gsk_test_abc123", "reach jane@example.com"])
    report = mon.build_monitor_report(
        kpis=kpis,
        safety=safety,
        thresholds=mon.load_monitor_thresholds(),
        base_url="http://example.test",
        notes=["note"],
    )
    assert report["summary"]["passed_all"] is False
    json_path, md_path = mon.write_monitor_report(report, tmp_path)
    raw_json = json_path.read_text(encoding="utf-8")
    raw_md = md_path.read_text(encoding="utf-8")
    assert "gsk_test_abc123" not in raw_json
    assert "jane@example.com" not in raw_json
    assert "gsk_***" in raw_json
    assert "# Online Operational Monitor Report" in raw_md


# ---------------------------------------------------------------------------
# Output sampling
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _StubClient:
    def __init__(self, responses: dict):
        self._responses = responses
        self.calls: list[str] = []

    async def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        return self._responses[url]


async def test_sample_outputs_from_api_collects_agent_messages():
    now = datetime.now(timezone.utc).isoformat()
    client = _StubClient(
        {
            "http://x/threads": _Resp(
                200,
                {"threads": [
                    {"thread_id": "t1", "last_seen": now},
                    {"thread_id": "t2", "last_seen": now},
                ]},
            ),
            "http://x/threads/t1/history": _Resp(
                200,
                {"messages": [
                    {"role": "user", "text": "hi"},
                    {"role": "agent", "text": "hello Alice"},
                ]},
            ),
            "http://x/threads/t2/history": _Resp(
                200, {"messages": [{"role": "agent", "text": "second"}]}
            ),
        }
    )
    outputs = await mon.sample_outputs_from_api(client, "http://x", sample_size=20)
    assert outputs == ["hello Alice", "second"]


async def test_sample_outputs_from_api_truncates_to_sample_size():
    now = datetime.now(timezone.utc).isoformat()
    client = _StubClient(
        {
            "http://x/threads": _Resp(200, {"threads": [{"thread_id": "t1", "last_seen": now}]}),
            "http://x/threads/t1/history": _Resp(
                200, {"messages": [{"role": "agent", "text": "one"}, {"role": "agent", "text": "two"}]}
            ),
        }
    )
    outputs = await mon.sample_outputs_from_api(client, "http://x", sample_size=1)
    assert outputs == ["one"]


async def test_sample_outputs_from_api_respects_window():
    old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    client = _StubClient({"http://x/threads": _Resp(200, {"threads": [{"thread_id": "t1", "last_seen": old}]})})
    outputs = await mon.sample_outputs_from_api(client, "http://x", window_hours=24)
    assert outputs == []
    assert "http://x/threads/t1/history" not in client.calls


async def test_sample_outputs_from_api_handles_threads_error():
    client = _StubClient({"http://x/threads": _Resp(500, {})})
    assert await mon.sample_outputs_from_api(client, "http://x") == []


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_build_parser_accepts_flags():
    args = mon.build_parser().parse_args(
        ["--metrics-url", "http://m/metrics", "--sample-size", "5", "--no-toxicity"]
    )
    assert args.metrics_url == "http://m/metrics"
    assert args.sample_size == 5
    assert args.no_toxicity is True


def test_resolve_metrics_url_precedence():
    assert mon.resolve_metrics_url("http://explicit", "http://base") == "http://explicit"
    assert mon.resolve_metrics_url(None, "http://base") == "http://base/metrics"


def test_resolve_metrics_url_env_fallback(monkeypatch):
    monkeypatch.setenv("AGENTFLOW_METRICS_URL", "http://env/metrics")
    assert mon.resolve_metrics_url(None, "") == "http://env/metrics"
