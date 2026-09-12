"""Unit tests for dependencies.py (Session 5-B rate limit key + helpers)."""

from unittest.mock import MagicMock

from fastapi import Request

from backend.dependencies import get_rate_limit_key, sse


def test_get_rate_limit_key_user():
    request = MagicMock(spec=Request)
    user = MagicMock()
    user.username = "testuser"
    request.state.user = user

    key = get_rate_limit_key(request)
    assert key == "user:testuser"


def test_get_rate_limit_key_forwarded_for():
    request = MagicMock(spec=Request)
    request.state.user = None
    request.headers = {"X-Forwarded-For": "203.0.113.195, 70.41.3.18"}

    key = get_rate_limit_key(request)
    assert key == "203.0.113.195"


def test_get_rate_limit_key_remote_addr():
    request = MagicMock(spec=Request)
    request.state.user = None
    request.headers = {}
    request.client.host = "192.168.1.50"

    key = get_rate_limit_key(request)
    assert key == "192.168.1.50"


def test_sse_formatting():
    event = sse("hello world")
    assert event == b"data: hello world\n\n"

    multiline = sse("line 1\r\nline 2")
    assert b"data: line 1\n" in multiline
    assert b"data: line 2\n" in multiline
