"""Tokenizer availability must not prevent LLM module use."""

import groq
import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

from backend import llm


def test_token_count_falls_back_to_utf8_bytes_when_encoding_unavailable(monkeypatch):
    calls = []

    def unavailable(_name):
        calls.append(_name)
        raise OSError("encoding cache unavailable")

    monkeypatch.setattr(llm.tiktoken, "get_encoding", unavailable)
    llm._token_encoding.cache_clear()
    try:
        assert llm._count_input_tokens("café") == len("café".encode())
        assert llm._count_input_tokens("another request") == len("another request")
        assert calls == ["cl100k_base"]
    finally:
        llm._token_encoding.cache_clear()


def test_token_count_uses_encoding_when_available(monkeypatch):
    class Encoding:
        def encode(self, text, **_kwargs):
            return [text]

    monkeypatch.setattr(llm.tiktoken, "get_encoding", lambda _name: Encoding())
    llm._token_encoding.cache_clear()
    try:
        assert llm._count_input_tokens("hello world") == 1
    finally:
        llm._token_encoding.cache_clear()


def test_budget_wrapper_accepts_content_blocks_and_prompt_values(monkeypatch):
    monkeypatch.setattr(llm, "_count_input_tokens", lambda text: len(text))
    wrapper = llm.TokenBudgetWrapper(bound=RunnableLambda(lambda value: value), budget=100)
    message = HumanMessage(content=[{"type": "text", "text": "hello"}])
    wrapper._check([message])
    wrapper._check(ChatPromptValue(messages=[message]))
    wrapper._check("hello")


@pytest.mark.parametrize("target", ["application", "judge"])
def test_rate_limits_fail_over_without_sleeping_on_the_exhausted_key(monkeypatch, target):
    from types import SimpleNamespace

    import evals.harness.llm_judge as judge

    calls = []
    monkeypatch.setattr(llm, "_collect_groq_keys", lambda: ["primary", "secondary"])
    monkeypatch.setattr(llm, "_build_fallback", lambda: None)

    def make_client(**kwargs):
        assert kwargs["max_retries"] == 0
        key = kwargs["api_key"].get_secret_value()

        def invoke(_messages):
            calls.append(key)
            if key == "primary":
                response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
                raise groq.RateLimitError("quota exhausted", response=response, body={})
            return SimpleNamespace(content='{"score": 1.0}')

        return RunnableLambda(invoke)

    if target == "application":
        monkeypatch.setattr(llm, "ChatGroq", make_client)
        assert llm._build_groq_pool("test-model").invoke([]).content == '{"score": 1.0}'
    else:
        monkeypatch.setattr(judge, "ChatGroq", make_client)
        monkeypatch.setattr(judge, "_judge_client", None)
        assert judge._call_judge([]) == {"score": 1.0}
    assert calls == ["primary", "secondary"]


@pytest.mark.parametrize(("limit", "hint", "expected_wait"), [
    ("tokens per minute", "9.0675s", 10.0675),
    ("tokens per day", "9s", None),
    ("tokens per minute", "60s", None),
])
def test_judge_retries_only_short_minute_limits(monkeypatch, limit, hint, expected_wait):
    from types import SimpleNamespace

    import evals.harness.llm_judge as judge

    calls = []
    waits = []

    def invoke(_messages):
        calls.append(True)
        if len(calls) == 1:
            response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
            raise groq.RateLimitError(
                f"Rate limit on {limit}. Please try again in {hint}.", response=response, body={}
            )
        return SimpleNamespace(content='{"score": 1.0}')

    monkeypatch.setattr(judge, "_get_judge_client", lambda: SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(judge.time, "sleep", waits.append)
    result = judge._call_judge([])
    if expected_wait is None:
        assert result is None and len(calls) == 1 and waits == []
        assert "RateLimitError" in judge.last_judge_error
    else:
        assert result == {"score": 1.0} and len(calls) == 2
        assert waits == [pytest.approx(expected_wait)]
