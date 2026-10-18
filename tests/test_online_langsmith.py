"""Tier A deterministic unit tests for LangSmith online eval runner.

Zero LLM calls, zero network access.
Verifies dataset mapping, target function wrappers, evaluator conversions,
client initialization / flush hygiene, and CLI parsing.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals.online.langsmith_runner import (
    DATASET_DESCRIPTIONS,
    DATASET_NAME_MAP,
    build_parser,
    close_langsmith_client,
    evaluate_g_eval,
    evaluate_rag_faithfulness,
    evaluate_rag_relevancy,
    evaluate_router_match,
    flush_langsmith_client,
    get_langsmith_client,
    is_valid_langsmith_configured,
    load_dataset_rows,
    main,
    row_to_langsmith_example,
    run_langsmith_evaluation,
    sync_datasets_to_langsmith,
    target_graph_turn,
    target_graph_turn_sync,
    target_router,
)

pytestmark = pytest.mark.tier_a


# ---------------------------------------------------------------------------
# 1. Dataset parsing and payload construction
# ---------------------------------------------------------------------------

def test_row_to_langsmith_example_router():
    row = {"input": "What is Python?", "expected": "research", "kind": "router"}
    inp, outp = row_to_langsmith_example("router", row)
    assert inp == {"input": "What is Python?"}
    assert outp == {"expected": "research"}


def test_row_to_langsmith_example_rag_qa():
    row = {
        "query": "What is the refund policy?",
        "reference": "30 days full refund.",
        "expected_facts": ["30 days"],
        "kind": "rag_qa",
    }
    inp, outp = row_to_langsmith_example("rag_qa", row)
    assert inp == {"query": "What is the refund policy?"}
    assert outp == {"reference": "30 days full refund.", "expected_facts": ["30 days"]}

    # rag-qa with hyphen
    inp2, outp2 = row_to_langsmith_example("rag-qa", row)
    assert inp2 == inp
    assert outp2 == outp


def test_row_to_langsmith_example_generator():
    row = {
        "query": "Summarize findings",
        "context": "Here is finding A and B.",
        "reference": "A and B were found.",
        "kind": "generator",
    }
    inp, outp = row_to_langsmith_example("generator", row)
    assert inp == {"query": "Summarize findings", "context": "Here is finding A and B."}
    assert outp == {"reference": "A and B were found."}


def test_row_to_langsmith_example_application():
    row = {
        "query": "Draft an analysis",
        "route": "analysis",
        "rubric": {"correctness": "Must be factual", "style": "Clear"},
        "kind": "application",
    }
    inp, outp = row_to_langsmith_example("application", row)
    assert inp == {"query": "Draft an analysis", "route": "analysis"}
    assert outp == {"rubric": {"correctness": "Must be factual", "style": "Clear"}}


def test_row_to_langsmith_example_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported dataset key"):
        row_to_langsmith_example("unknown_dataset", {"input": "x"})


def test_load_dataset_rows_real_files():
    for key in ("router", "rag_qa", "generator", "application"):
        rows = load_dataset_rows(key, limit=3)
        assert len(rows) == 3
        # Check that row_to_langsmith_example works on all loaded rows
        for r in rows:
            inp, outp = row_to_langsmith_example(key, r)
            assert isinstance(inp, dict) and len(inp) > 0
            assert isinstance(outp, dict) and len(outp) > 0


def test_load_dataset_rows_unknown_key():
    with pytest.raises(ValueError, match="Unknown dataset key"):
        load_dataset_rows("invalid_key")


# ---------------------------------------------------------------------------
# 2. Target functions with mocked model / router outputs
# ---------------------------------------------------------------------------

def test_target_router_with_input_key():
    with patch("backend.graph.router._route_for_message", return_value="research") as mock_route:
        res = target_router({"input": "Find latest AI papers"})
        assert res == {"route": "research"}
        mock_route.assert_called_once_with("Find latest AI papers")


def test_target_router_with_query_key():
    with patch("backend.graph.router._route_for_message", return_value="analysis") as mock_route:
        res = target_router({"query": "Compare sales Q1 and Q2"})
        assert res == {"route": "analysis"}
        mock_route.assert_called_once_with("Compare sales Q1 and Q2")


@pytest.mark.asyncio
async def test_target_graph_turn_async():
    mock_compiled_graph = MagicMock()
    mock_compiled_graph.ainvoke = AsyncMock(return_value={
        "final_response": "Processed answer",
        "sources": ["https://example.com/source"],
        "documents": ["doc1", "doc2"],
    })

    with patch("backend.graph.build_graph.builder.compile", return_value=mock_compiled_graph) as mock_compile:
        res = await target_graph_turn({"query": "Hello AgentFlow"})
        assert res["output"] == "Processed answer"
        assert res["sources"] == ["https://example.com/source"]
        assert res["documents"] == ["doc1", "doc2"]
        mock_compile.assert_called_once()
        mock_compiled_graph.ainvoke.assert_called_once()


def test_target_graph_turn_sync():
    mock_compiled_graph = MagicMock()
    mock_compiled_graph.invoke.return_value = {
        "final_response": "Sync processed answer",
        "sources": ["https://example.com"],
        "documents": [],
    }

    with patch("backend.graph.build_graph.builder.compile", return_value=mock_compiled_graph) as mock_compile:
        res = target_graph_turn_sync({"query": "Sync query"})
        assert res["output"] == "Sync processed answer"
        assert res["sources"] == ["https://example.com"]
        assert res["documents"] == []
        mock_compile.assert_called_once()
        mock_compiled_graph.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Evaluator wrappers translating LangSmith run/example dicts to judge arguments
# ---------------------------------------------------------------------------

def test_evaluate_router_match():
    # Exact match
    run = {"outputs": {"route": "research"}}
    example = {"outputs": {"expected": "research"}}
    res = evaluate_router_match(run, example)
    assert res == {"key": "router_accuracy", "score": 1.0}

    # Mismatch
    run_bad = {"outputs": {"route": "chat"}}
    res_bad = evaluate_router_match(run_bad, example)
    assert res_bad == {"key": "router_accuracy", "score": 0.0}

    # Object-based (e.g. SimpleNamespace or LangSmith Example/Run classes)
    run_obj = SimpleNamespace(outputs={"route": "blog"})
    ex_obj = SimpleNamespace(outputs={"expected": "blog"})
    assert evaluate_router_match(run_obj, ex_obj) == {"key": "router_accuracy", "score": 1.0}


def test_evaluate_rag_faithfulness():
    with patch("evals.harness.llm_judge.judge_faithfulness", return_value=0.92) as mock_judge:
        run = {
            "outputs": {
                "output": "The capital is Paris.",
                "documents": ["Paris is the capital of France."],
            }
        }
        example = {"inputs": {"query": "What is the capital of France?"}}
        res = evaluate_rag_faithfulness(run, example)
        assert res == {"key": "faithfulness", "score": 0.92}
        mock_judge.assert_called_once_with(
            "What is the capital of France?",
            "Paris is the capital of France.",
            "The capital is Paris.",
        )


def test_evaluate_rag_faithfulness_with_context_in_example():
    with patch("evals.harness.llm_judge.judge_faithfulness", return_value=0.85) as mock_judge:
        run = {"outputs": {"output": "42 is the answer."}}
        example = {
            "inputs": {
                "query": "What is 42?",
                "context": "Context provided in example.",
            }
        }
        res = evaluate_rag_faithfulness(run, example)
        assert res == {"key": "faithfulness", "score": 0.85}
        mock_judge.assert_called_once_with(
            "What is 42?",
            "Context provided in example.",
            "42 is the answer.",
        )


def test_evaluate_rag_relevancy():
    with patch("evals.harness.llm_judge.judge_answer_relevancy", return_value=0.88) as mock_judge:
        run = {"outputs": {"output": "Photosynthesis is the process..."}}
        example = {"inputs": {"query": "Explain photosynthesis"}}
        res = evaluate_rag_relevancy(run, example)
        assert res == {"key": "answer_relevancy", "score": 0.88}
        mock_judge.assert_called_once_with(
            "Explain photosynthesis",
            "Photosynthesis is the process...",
        )


def test_evaluate_g_eval():
    rubric = {"correctness": "Accurate facts", "completeness": "Covers all points"}
    with patch("evals.harness.llm_judge.judge_g_eval", return_value={"correctness": 4.0, "completeness": 5.0}) as mock_judge:
        run = {"outputs": {"output": "Detailed blog post"}}
        example = {
            "inputs": {"query": "Write a blog on tech"},
            "outputs": {"rubric": rubric},
        }
        res = evaluate_g_eval(run, example)
        assert res == {"key": "g_eval_composite", "score": 4.5}
        mock_judge.assert_called_once_with(
            "Write a blog on tech",
            "Detailed blog post",
            rubric,
        )


def test_evaluate_g_eval_empty():
    with patch("evals.harness.llm_judge.judge_g_eval", return_value=None):
        run = {"outputs": {"output": "Some response"}}
        example = {"inputs": {"query": "Query"}, "outputs": {"rubric": {}}}
        res = evaluate_g_eval(run, example)
        assert res == {"key": "g_eval_composite", "score": 0.0}


# ---------------------------------------------------------------------------
# 4. Dataset synchronization (sync_datasets_to_langsmith)
# ---------------------------------------------------------------------------

def test_sync_datasets_to_langsmith_idempotent():
    mock_client = MagicMock()
    mock_ds = SimpleNamespace(id="ds-12345")
    mock_client.has_dataset.return_value = True
    mock_client.read_dataset.return_value = mock_ds

    # Simulate existing examples in LangSmith matching the first row of router.jsonl
    first_row = load_dataset_rows("router", limit=1)[0]
    first_inp, _ = row_to_langsmith_example("router", first_row)
    existing_example = SimpleNamespace(inputs=first_inp, outputs={})
    mock_client.list_examples.return_value = [existing_example]

    synced = sync_datasets_to_langsmith(
        client=mock_client,
        datasets=["router"],
        limit=2,
    )

    assert "router" in synced
    assert synced["router"] == "ds-12345"
    # One example already existed, so only 1 new example should be uploaded
    mock_client.create_examples.assert_called_once()
    call_kwargs = mock_client.create_examples.call_args[1]
    assert len(call_kwargs["inputs"]) == 1
    assert len(call_kwargs["outputs"]) == 1


def test_sync_datasets_creates_new_dataset_if_not_found():
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = False
    mock_client.read_dataset.side_effect = Exception("Not found")
    new_ds = SimpleNamespace(id="new-ds-6789")
    mock_client.create_dataset.return_value = new_ds
    mock_client.list_examples.return_value = []

    synced = sync_datasets_to_langsmith(
        client=mock_client,
        datasets=["router"],
        limit=1,
    )
    assert synced["router"] == "new-ds-6789"
    mock_client.create_dataset.assert_called_once_with(
        dataset_name=DATASET_NAME_MAP["router"],
        description=DATASET_DESCRIPTIONS["router"],
    )
    mock_client.create_examples.assert_called_once()


# ---------------------------------------------------------------------------
# 5. CLI parser options
# ---------------------------------------------------------------------------

def test_build_parser():
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(["--dataset", "rag-qa", "--limit", "10", "--sync-only", "--prefix", "Custom-Prefix"])
    assert args.dataset == "rag-qa"
    assert args.limit == 10
    assert args.sync_only is True
    assert args.prefix == "Custom-Prefix"

    args_default = parser.parse_args([])
    assert args_default.dataset == "router"
    assert args_default.limit is None
    assert args_default.sync_only is False
    assert args_default.prefix == "AgentFlow-Eval"


# ---------------------------------------------------------------------------
# 6. Graceful skip behavior when LANGCHAIN_API_KEY is absent or dummy
# ---------------------------------------------------------------------------

def test_is_valid_langsmith_configured_with_dummy_or_missing(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "dummy_key_for_ci")
    assert not is_valid_langsmith_configured()
    assert get_langsmith_client() is None

    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    assert not is_valid_langsmith_configured()

    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert not is_valid_langsmith_configured()


def test_sync_datasets_skips_when_client_none(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    result = sync_datasets_to_langsmith(client=None, datasets=["router"])
    assert result == {}


def test_run_langsmith_evaluation_skips_when_client_none(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    result = run_langsmith_evaluation(dataset="router", client=None)
    assert result is None


def test_main_clean_exit_when_unconfigured(monkeypatch, capsys):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "dummy_key_for_ci")
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "LangSmith is not configured" in captured.err


# ---------------------------------------------------------------------------
# 7. Thread hygiene and client flush
# ---------------------------------------------------------------------------

def test_flush_langsmith_client_only_flushes():
    mock_client = MagicMock()
    flush_langsmith_client(mock_client)
    mock_client.flush.assert_called_once()
    # Flush must NOT close: a shared client has to survive the whole run.
    mock_client.close.assert_not_called()

    # None client does not raise
    flush_langsmith_client(None)

    # Client with exceptions does not raise
    faulty_client = MagicMock()
    faulty_client.flush.side_effect = RuntimeError("Flush error")
    flush_langsmith_client(faulty_client)


def test_close_langsmith_client_flushes_then_closes():
    mock_client = MagicMock()
    close_langsmith_client(mock_client)
    mock_client.flush.assert_called_once()
    mock_client.close.assert_called_once()

    # Idempotent / None-safe
    close_langsmith_client(None)
    close_langsmith_client(mock_client)
    assert mock_client.close.call_count == 2

    # A raising close must not propagate
    faulty_client = MagicMock()
    faulty_client.close.side_effect = RuntimeError("Close error")
    close_langsmith_client(faulty_client)


# ---------------------------------------------------------------------------
# 8. run_langsmith_evaluation flow with mock evaluate
# ---------------------------------------------------------------------------

def test_run_langsmith_evaluation_mocked():
    mock_client = MagicMock()
    mock_results = [{"evaluation_results": {"results": [SimpleNamespace(key="router_accuracy", score=1.0)]}}]

    with (
        patch("evals.online.langsmith_runner.sync_datasets_to_langsmith") as mock_sync,
        patch("langsmith.evaluation.evaluate", return_value=mock_results) as mock_eval,
    ):
        res = run_langsmith_evaluation(
            dataset="router",
            limit=5,
            prefix="Test-Prefix",
            client=mock_client,
        )
        assert res == mock_results
        mock_sync.assert_called_once_with(mock_client, datasets=["router"], limit=5)
        mock_eval.assert_called_once()
        # A passed-in client is flushed but never closed (the caller owns it).
        mock_client.flush.assert_called_once()
        mock_client.close.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Client ownership / shared-client reuse (regression: --dataset all)
# ---------------------------------------------------------------------------

class _RecordingClient:
    """Minimal flush/close accounting for the multi-dataset reuse test."""

    def __init__(self):
        self.flush_calls = 0
        self.close_calls = 0
        self.closed = False

    def flush(self):
        self.flush_calls += 1

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_main_all_datasets_keeps_client_open_until_end():
    """Regression: `--dataset all` must not evaluate on a closed client."""
    client = _RecordingClient()
    open_at_eval = []

    def fake_evaluate(target, data=None, evaluators=None, experiment_prefix=None, client=None, **kwargs):
        open_at_eval.append(not client.closed)
        return []

    with (
        patch("evals.online.langsmith_runner.is_valid_langsmith_configured", return_value=True),
        patch("evals.online.langsmith_runner.get_langsmith_client", return_value=client),
        patch("evals.online.langsmith_runner.sync_datasets_to_langsmith", return_value={}),
        patch("langsmith.evaluation.evaluate", side_effect=fake_evaluate),
    ):
        rc = main(["--dataset", "all"])

    assert rc == 0
    assert open_at_eval == [True, True, True, True]
    assert client.closed is True
    assert client.close_calls == 1


def test_run_langsmith_evaluation_closes_owned_client():
    """A client created internally (client=None) is closed by the runner."""
    client = MagicMock()
    with (
        patch("evals.online.langsmith_runner.get_langsmith_client", return_value=client),
        patch("evals.online.langsmith_runner.sync_datasets_to_langsmith", return_value={}),
        patch("langsmith.evaluation.evaluate", return_value=[]),
    ):
        run_langsmith_evaluation(dataset="router", client=None)
    client.flush.assert_called_once()
    client.close.assert_called_once()


# ---------------------------------------------------------------------------
# 10. --limit bounds the evaluated rows, not just the upload
# ---------------------------------------------------------------------------

def test_run_langsmith_evaluation_applies_limit():
    client = MagicMock()
    captured = {}

    def fake_evaluate(target, data=None, evaluators=None, experiment_prefix=None, client=None, **kwargs):
        captured["data"] = data
        return []

    with (
        patch("evals.online.langsmith_runner.sync_datasets_to_langsmith", return_value={}),
        patch("langsmith.evaluation.evaluate", side_effect=fake_evaluate),
    ):
        run_langsmith_evaluation(dataset="router", limit=3, client=client)

    assert isinstance(captured["data"], list)
    assert len(captured["data"]) == 3
    assert all("inputs" in ex and "outputs" in ex for ex in captured["data"])


def test_run_langsmith_evaluation_without_limit_uses_dataset_name():
    client = MagicMock()
    captured = {}

    def fake_evaluate(target, data=None, evaluators=None, experiment_prefix=None, client=None, **kwargs):
        captured["data"] = data
        return []

    with (
        patch("evals.online.langsmith_runner.sync_datasets_to_langsmith", return_value={}),
        patch("langsmith.evaluation.evaluate", side_effect=fake_evaluate),
    ):
        run_langsmith_evaluation(dataset="router", client=client)

    assert captured["data"] == DATASET_NAME_MAP["router"]


# ---------------------------------------------------------------------------
# 11. Sync refreshes changed outputs and refuses risky uploads
# ---------------------------------------------------------------------------

def test_sync_datasets_updates_changed_outputs():
    mock_client = MagicMock()
    mock_ds = SimpleNamespace(id="ds-upd")
    mock_client.has_dataset.return_value = True
    mock_client.read_dataset.return_value = mock_ds

    first_row = load_dataset_rows("router", limit=1)[0]
    first_inp, first_out = row_to_langsmith_example("router", first_row)
    existing = SimpleNamespace(id="ex-1", inputs=first_inp, outputs={"expected": "stale"})
    mock_client.list_examples.return_value = [existing]

    synced = sync_datasets_to_langsmith(client=mock_client, datasets=["router"], limit=1)

    assert synced["router"] == "ds-upd"
    mock_client.create_examples.assert_not_called()
    mock_client.update_examples.assert_called_once()
    assert mock_client.update_examples.call_args[1]["updates"] == [
        {"id": "ex-1", "outputs": first_out}
    ]


def test_sync_datasets_no_update_when_outputs_match():
    mock_client = MagicMock()
    mock_ds = SimpleNamespace(id="ds-ok")
    mock_client.has_dataset.return_value = True
    mock_client.read_dataset.return_value = mock_ds

    first_row = load_dataset_rows("router", limit=1)[0]
    first_inp, first_out = row_to_langsmith_example("router", first_row)
    existing = SimpleNamespace(id="ex-1", inputs=first_inp, outputs=first_out)
    mock_client.list_examples.return_value = [existing]

    sync_datasets_to_langsmith(client=mock_client, datasets=["router"], limit=1)

    mock_client.create_examples.assert_not_called()
    mock_client.update_examples.assert_not_called()


def test_sync_datasets_skips_upload_when_listing_fails():
    mock_client = MagicMock()
    mock_ds = SimpleNamespace(id="ds-x")
    mock_client.has_dataset.return_value = True
    mock_client.read_dataset.return_value = mock_ds
    mock_client.list_examples.side_effect = RuntimeError("pagination boom")

    synced = sync_datasets_to_langsmith(client=mock_client, datasets=["router"], limit=2)

    assert synced["router"] == "ds-x"
    mock_client.create_examples.assert_not_called()
    mock_client.update_examples.assert_not_called()


# ---------------------------------------------------------------------------
# 12. Graceful behaviour when the langsmith package is unimportable
# ---------------------------------------------------------------------------

def test_is_valid_langsmith_configured_when_langsmith_missing(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_pt_real_looking_key")
    with patch.dict("sys.modules", {"langsmith": None}):
        assert not is_valid_langsmith_configured()
        assert get_langsmith_client() is None
