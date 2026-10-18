"""LangSmith Online Evaluation Runner & Dataset Sync.

Wires real golden datasets (router, rag_qa, generator, application) into
LangSmith experiments against the compiled graph and router targets, and
evaluates them with LLM-as-judge metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentflow.eval.online.langsmith")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATASETS_DIR = HERE.parent / "datasets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME_MAP = {
    "router": "AgentFlow-Router-Golden",
    "rag-qa": "AgentFlow-RAG-QA-Golden",
    "rag_qa": "AgentFlow-RAG-QA-Golden",
    "generator": "AgentFlow-Generator-Golden",
    "application": "AgentFlow-Application-Golden",
}

DATASET_FILE_MAP = {
    "router": "router.jsonl",
    "rag-qa": "rag_qa.jsonl",
    "rag_qa": "rag_qa.jsonl",
    "generator": "generator.jsonl",
    "application": "application.jsonl",
}

DATASET_DESCRIPTIONS = {
    "router": "Golden pairs for testing AgentFlow routing accuracy",
    "rag-qa": "Golden pairs for testing AgentFlow end-to-end RAG QA",
    "rag_qa": "Golden pairs for testing AgentFlow end-to-end RAG QA",
    "generator": "Golden pairs for testing AgentFlow generator faithfulness and relevancy",
    "application": "Golden queries and G-Eval rubrics per skill",
}

DUMMY_API_KEYS = frozenset({"dummy_key_for_ci", "dummy", "test", ""})


def is_valid_langsmith_configured() -> bool:
    """Return True if LangSmith is available and LANGCHAIN_API_KEY is non-dummy."""
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    if not api_key or api_key in DUMMY_API_KEYS:
        return False
    try:
        import langsmith  # noqa: F401

        return True
    except ImportError:
        return False


def get_langsmith_client() -> Any | None:
    """Return a configured LangSmith Client or None if unavailable/unconfigured."""
    if not is_valid_langsmith_configured():
        return None
    try:
        from langsmith import Client

        return Client()
    except Exception as exc:
        logger.warning("Could not initialize LangSmith Client: %s", exc)
        return None


def flush_langsmith_client(client: Any | None = None) -> None:
    """Drain pending LangSmith traces without closing the client.

    ``flush`` is safe to call repeatedly and keeps a client reusable across
    datasets. Closing is deliberately *not* done here: a client shared by the
    caller (e.g. ``main --dataset all``) must survive the whole loop. Callers
    that own a client should close it exactly once via
    :func:`close_langsmith_client`.
    """
    if client is not None:
        try:
            if hasattr(client, "flush"):
                client.flush()
        except Exception as exc:
            logger.debug("Error flushing LangSmith client: %s", exc)


def close_langsmith_client(client: Any | None = None) -> None:
    """Flush then close a LangSmith client so background threads do not hang exit.

    Idempotent and ``None``-safe: closing the same client twice, or a client
    that was never opened, is a no-op.
    """
    if client is None:
        return
    try:
        flush_langsmith_client(client)
    except Exception as exc:  # pragma: no cover - flush already swallows
        logger.debug("Error flushing LangSmith client before close: %s", exc)
    try:
        if hasattr(client, "close"):
            client.close()
    except Exception as exc:
        logger.debug("Error closing LangSmith client: %s", exc)


def row_to_langsmith_example(dataset_key: str, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a raw golden dataset row into (inputs, outputs) for LangSmith."""
    norm_key = dataset_key.lower().replace("-", "_")
    if norm_key == "router":
        return {"input": row["input"]}, {"expected": row["expected"]}
    elif norm_key == "rag_qa":
        return {"query": row["query"]}, {
            "reference": row["reference"],
            "expected_facts": row.get("expected_facts", []),
        }
    elif norm_key == "generator":
        return {
            "query": row["query"],
            "context": row["context"],
        }, {
            "reference": row["reference"],
        }
    elif norm_key == "application":
        return {
            "query": row["query"],
            "route": row["route"],
        }, {
            "rubric": row.get("rubric", {}),
        }
    else:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")


def load_dataset_rows(dataset_key: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load rows from the specified golden dataset JSONL file."""
    norm_key = dataset_key.lower().replace("-", "_")
    filename = DATASET_FILE_MAP.get(norm_key)
    if not filename:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    filepath = DATASETS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    rows: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _fingerprint(payload: Any) -> str:
    """Stable JSON fingerprint for dedupe/comparison of example inputs/outputs."""
    try:
        return json.dumps(payload if payload is not None else {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(payload)


def _update_examples(
    client: Any,
    dataset_id: Any,
    updates: list[dict[str, Any]],
    dataset_name: str,
) -> None:
    """Refresh changed outputs on existing examples (batched when supported)."""
    try:
        if hasattr(client, "update_examples"):
            client.update_examples(dataset_id=dataset_id, updates=updates)
        elif hasattr(client, "update_example"):
            for upd in updates:
                client.update_example(example_id=upd["id"], outputs=upd["outputs"])
        else:
            logger.warning(
                "Client cannot update examples; %d stale rows left in %s",
                len(updates),
                dataset_name,
            )
            return
        logger.info("Updated %d existing examples in dataset %s", len(updates), dataset_name)
    except Exception as exc:
        logger.error("Failed to update examples in %s: %s", dataset_name, exc)


def sync_datasets_to_langsmith(
    client: Any | None = None,
    datasets: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, str]:
    """Upload golden rows from evals/datasets/*.jsonl to LangSmith.

    Idempotent: checks if dataset exists; creates or updates examples without
    duplicating rows.
    """
    if client is None:
        client = get_langsmith_client()
    if client is None:
        logger.warning("LangSmith client unavailable or API key missing/dummy; skipping dataset sync.")
        return {}

    target_keys = datasets or ["router", "rag_qa", "generator", "application"]
    synced: dict[str, str] = {}

    for raw_key in target_keys:
        key = raw_key.lower().replace("-", "_")
        dataset_name = DATASET_NAME_MAP.get(key)
        if not dataset_name:
            logger.warning("Skipping unrecognized dataset key: %s", raw_key)
            continue

        try:
            rows = load_dataset_rows(key, limit=limit)
        except Exception as exc:
            logger.error("Failed to load dataset rows for %s: %s", key, exc)
            continue

        examples: list[tuple[dict[str, Any], dict[str, Any]]] = [
            row_to_langsmith_example(key, row) for row in rows
        ]

        # Retrieve or create dataset
        ds = None
        if hasattr(client, "has_dataset"):
            try:
                if client.has_dataset(dataset_name=dataset_name):
                    ds = client.read_dataset(dataset_name=dataset_name)
            except Exception:
                ds = None

        if ds is None:
            try:
                ds = client.read_dataset(dataset_name=dataset_name)
            except Exception:
                try:
                    ds = client.create_dataset(
                        dataset_name=dataset_name,
                        description=DATASET_DESCRIPTIONS.get(key, f"AgentFlow dataset {dataset_name}"),
                    )
                except Exception as exc:
                    logger.error("Failed to create dataset %s: %s", dataset_name, exc)
                    continue

        ds_id = getattr(ds, "id", None) or dataset_name

        # Map existing examples by input fingerprint so we can dedupe (create)
        # and refresh changed outputs (update) without re-uploading rows.
        listing_supported = hasattr(client, "list_examples")
        listing_ok = not listing_supported
        existing: dict[str, tuple[Any, Any]] = {}
        if listing_supported:
            try:
                existing_examples = list(
                    client.list_examples(
                        dataset_id=getattr(ds, "id", None),
                        dataset_name=dataset_name,
                    )
                )
                for ex in existing_examples:
                    inp = getattr(ex, "inputs", None) or (ex.get("inputs") if isinstance(ex, dict) else {}) or {}
                    ex_id = getattr(ex, "id", None) or (ex.get("id") if isinstance(ex, dict) else None)
                    outp = getattr(ex, "outputs", None) or (ex.get("outputs") if isinstance(ex, dict) else {}) or {}
                    existing[_fingerprint(inp)] = (ex_id, outp)
                listing_ok = True
            except Exception as exc:
                logger.debug("Could not list existing examples for %s: %s", dataset_name, exc)

        new_inputs: list[dict[str, Any]] = []
        new_outputs: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        for inp, outp in examples:
            fp = _fingerprint(inp)
            if fp not in existing:
                new_inputs.append(inp)
                new_outputs.append(outp)
                existing[fp] = (None, outp)
                continue
            ex_id, ex_out = existing[fp]
            if ex_id is not None and _fingerprint(ex_out) != _fingerprint(outp):
                updates.append({"id": ex_id, "outputs": outp})

        if new_inputs and hasattr(client, "create_examples"):
            if listing_supported and not listing_ok:
                # Could not enumerate existing rows: uploading would risk
                # duplicating them, so refuse rather than corrupt the dataset.
                logger.warning(
                    "Skipping upload for %s: could not list existing examples; "
                    "refusing to risk duplicates.",
                    dataset_name,
                )
            else:
                try:
                    client.create_examples(
                        inputs=new_inputs,
                        outputs=new_outputs,
                        dataset_id=getattr(ds, "id", None),
                    )
                    logger.info("Uploaded %d new examples to dataset %s", len(new_inputs), dataset_name)
                except Exception as exc:
                    logger.error("Failed to upload examples to %s: %s", dataset_name, exc)

        if updates:
            _update_examples(client, getattr(ds, "id", None), updates, dataset_name)

        synced[key] = str(ds_id)

    return synced


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------

def target_router(inputs: dict[str, Any]) -> dict[str, str]:
    """Extract user query, call router classification, and return route."""
    user_text = inputs.get("input") or inputs.get("query", "")
    from backend.graph.router import _route_for_message

    route_name = _route_for_message(str(user_text))
    return {"route": route_name}


async def target_graph_turn(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run an async turn through the compiled graph with an InMemorySaver checkpointer."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from backend.graph.build_graph import builder

    graph = builder.compile(checkpointer=InMemorySaver())
    query = inputs.get("query") or inputs.get("input", "")
    thread_id = f"eval-ls-{uuid.uuid4().hex[:8]}"
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=str(query))]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "output": result.get("final_response") or "",
        "sources": result.get("sources") or [],
        "documents": result.get("documents") or [],
    }


def target_graph_turn_sync(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run a synchronous turn through the compiled graph with an InMemorySaver checkpointer."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from backend.graph.build_graph import builder

    graph = builder.compile(checkpointer=InMemorySaver())
    query = inputs.get("query") or inputs.get("input", "")
    thread_id = f"eval-ls-{uuid.uuid4().hex[:8]}"
    result = graph.invoke(
        {"messages": [HumanMessage(content=str(query))]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "output": result.get("final_response") or "",
        "sources": result.get("sources") or [],
        "documents": result.get("documents") or [],
    }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _extract_inputs_outputs(run: Any, example: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Safely extract (run_inputs, run_outputs, example_inputs, example_outputs)."""
    run_inputs = getattr(run, "inputs", None) or (run.get("inputs") if isinstance(run, dict) else {}) or {}
    run_outputs = getattr(run, "outputs", None) or (run.get("outputs") if isinstance(run, dict) else {}) or {}
    example_inputs = getattr(example, "inputs", None) or (example.get("inputs") if isinstance(example, dict) else {}) or {}
    example_outputs = getattr(example, "outputs", None) or (example.get("outputs") if isinstance(example, dict) else {}) or {}
    return run_inputs, run_outputs, example_inputs, example_outputs


def evaluate_router_match(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate router classification match between example and run outputs."""
    _, run_outputs, _, example_outputs = _extract_inputs_outputs(run, example)
    expected = example_outputs.get("expected")
    actual = run_outputs.get("route")
    match = expected is not None and expected == actual
    return {"key": "router_accuracy", "score": 1.0 if match else 0.0}


def evaluate_rag_faithfulness(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate faithfulness of generated answer against retrieved context using judge."""
    from evals.harness.llm_judge import judge_faithfulness

    _, run_outputs, example_inputs, _ = _extract_inputs_outputs(run, example)
    query = str(example_inputs.get("query") or example_inputs.get("input") or "")

    context = example_inputs.get("context")
    if not context:
        docs = run_outputs.get("documents") or []
        if isinstance(docs, list):
            context = "\n\n".join(
                getattr(d, "page_content", str(d)) for d in docs
            )
        else:
            context = str(docs)
    context = str(context or "")

    answer = str(run_outputs.get("output") or run_outputs.get("final_response") or "")
    score = judge_faithfulness(query, context, answer)
    return {"key": "faithfulness", "score": float(score) if score is not None else 0.0}


def evaluate_rag_relevancy(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate answer relevancy to the query using judge."""
    from evals.harness.llm_judge import judge_answer_relevancy

    _, run_outputs, example_inputs, _ = _extract_inputs_outputs(run, example)
    query = str(example_inputs.get("query") or example_inputs.get("input") or "")
    answer = str(run_outputs.get("output") or run_outputs.get("final_response") or "")
    score = judge_answer_relevancy(query, answer)
    return {"key": "answer_relevancy", "score": float(score) if score is not None else 0.0}


def evaluate_g_eval(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate answer against G-Eval rubric dimensions and return composite mean."""
    from evals.harness.llm_judge import judge_g_eval

    _, run_outputs, example_inputs, example_outputs = _extract_inputs_outputs(run, example)
    query = str(example_inputs.get("query") or example_inputs.get("input") or "")
    answer = str(run_outputs.get("output") or run_outputs.get("final_response") or "")
    rubric = example_outputs.get("rubric") or example_inputs.get("rubric") or {}

    dim_scores = judge_g_eval(query, answer, rubric)
    if dim_scores:
        mean_score = sum(dim_scores.values()) / len(dim_scores)
    else:
        mean_score = 0.0
    return {"key": "g_eval_composite", "score": float(mean_score)}


# ---------------------------------------------------------------------------
# Runner Execution
# ---------------------------------------------------------------------------

def run_langsmith_evaluation(
    dataset: str = "router",
    target: Any | None = None,
    evaluators: list[Any] | None = None,
    limit: int | None = None,
    prefix: str = "AgentFlow-Eval",
    client: Any | None = None,
) -> Any | None:
    """Execute evaluation for a specified dataset in LangSmith.

    Returns the ExperimentResults or None if LangSmith is unconfigured.
    """
    owns_client = client is None
    if client is None:
        client = get_langsmith_client()
    if client is None:
        logger.warning("LangSmith client unavailable or API key missing/dummy; skipping evaluation.")
        return None

    norm = dataset.lower().replace("-", "_")
    dataset_name = DATASET_NAME_MAP.get(norm, dataset)

    # Sync dataset to LangSmith if needed
    sync_datasets_to_langsmith(client, datasets=[norm], limit=limit)

    if target is None:
        target = target_router if norm == "router" else target_graph_turn

    if evaluators is None:
        if norm == "router":
            evaluators = [evaluate_router_match]
        elif norm in ("rag_qa", "generator"):
            evaluators = [evaluate_rag_faithfulness, evaluate_rag_relevancy]
        elif norm == "application":
            evaluators = [evaluate_g_eval]
        else:
            evaluators = [evaluate_router_match]

    # `evaluate(data=<name>)` always scores the whole remote dataset, so honour
    # `--limit` by evaluating an explicit inline slice of the golden rows
    # instead (langsmith does not accept a row limit for named datasets).
    data: Any = dataset_name
    if limit is not None:
        try:
            rows = load_dataset_rows(norm, limit=limit)
            data = [
                {"inputs": inp, "outputs": outp}
                for inp, outp in (row_to_langsmith_example(norm, r) for r in rows)
            ]
        except Exception as exc:
            logger.warning("Could not build limited eval slice for %s: %s", norm, exc)
            data = dataset_name

    from langsmith.evaluation import evaluate

    try:
        results = evaluate(
            target,
            data=data,
            evaluators=evaluators,
            experiment_prefix=prefix,
            client=client,
        )
        return results
    finally:
        # Never close a client we did not create: `main` reuses one client
        # across every dataset in a run and closes it once at the end.
        if owns_client:
            close_langsmith_client(client)
        else:
            flush_langsmith_client(client)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LangSmith Online Evaluation Runner & Dataset Sync"
    )
    parser.add_argument(
        "--dataset",
        choices=["router", "rag-qa", "rag_qa", "generator", "application", "all"],
        default="router",
        help="Dataset to evaluate or sync (default: router)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to sync or evaluate",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only synchronize datasets to LangSmith without running evaluation",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="AgentFlow-Eval",
        help="Experiment prefix for LangSmith evaluation runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for LangSmith online eval runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not is_valid_langsmith_configured():
        print(
            "LangSmith is not configured (LANGCHAIN_API_KEY missing or dummy). Exiting cleanly.",
            file=sys.stderr,
        )
        return 0

    client = get_langsmith_client()
    if client is None:
        print("Failed to initialize LangSmith client. Exiting cleanly.", file=sys.stderr)
        return 0

    try:
        datasets_to_run = (
            ["router", "rag_qa", "generator", "application"]
            if args.dataset == "all"
            else [args.dataset]
        )

        if args.sync_only:
            print(f"Synchronizing datasets to LangSmith: {datasets_to_run}")
            synced = sync_datasets_to_langsmith(client, datasets=datasets_to_run, limit=args.limit)
            print(f"Successfully synced: {synced}")
            return 0

        for ds_key in datasets_to_run:
            print(f"Running LangSmith evaluation for {ds_key} (prefix={args.prefix})...")
            run_langsmith_evaluation(
                dataset=ds_key,
                limit=args.limit,
                prefix=args.prefix,
                client=client,
            )
        return 0
    finally:
        # The client is owned by `main`; drain and close exactly once here.
        close_langsmith_client(client)


if __name__ == "__main__":
    sys.exit(main())
