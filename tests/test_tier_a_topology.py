"""Tier A graph topology gates (Phase 1 — Offline Tier A).

Pure structural assertions over the uncompiled `builder` StateGraph —
no LLM, no graph invocation, no DB side effects: the exact node set and
the router conditional route-map keys cannot silently drift.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.tier_a

EXPECTED_NODES = {
    "router",
    "research_agent",
    "analysis_agent",
    "chat_agent",
    "blog_writer",
    "synthesizer",
    "human_review",
    "memory_reader",
    "memory_writer",
    "stm_compressor",
}

EXPECTED_ROUTES = {"research", "analysis", "chat", "blog"}


def test_node_set():
    from backend.graph.build_graph import builder

    assert set(builder.nodes.keys()) == EXPECTED_NODES


def test_route_map_keys():
    """The path map passed to add_conditional_edges("router", ...) exposes
    exactly the four route keys — and this test must FAIL, not silently skip,
    if the branch shape changes (langgraph version drift is a regression too).
    Normalize the version-specific branch shape, then require the 'ends'
    mapping unconditionally. Always: the route_query behavioral pins below
    keep the contract version-proof."""
    from backend.graph.build_graph import builder
    from backend.graph.router import route_query

    branches = getattr(builder, "branches", {}) or {}
    router_branch = branches.get("router")
    assert router_branch is not None, "builder exposes no 'router' branch"

    # langgraph stores the router branch as a dict `{edge_fn_name: BranchSpec}`
    # in current versions; `BranchSpec.ends` is the path map. Older versions
    # store the path map directly or an object with an `.ends` attribute.
    # Normalize, then REQUIRE an ends mapping — never skip: version-shape
    # drift must surface as a failure, not a silent pass.
    ends = None
    if isinstance(router_branch, dict):
        branch_spec = next(iter(router_branch.values()), None)
        ends = getattr(branch_spec, "ends", branch_spec)
    else:
        ends = getattr(router_branch, "ends", None)
    assert isinstance(ends, dict), (
        "router branch does not expose an 'ends' mapping "
        f"(got {ends!r}) — update the normalization for this langgraph version"
    )
    assert set(ends.keys()) == EXPECTED_ROUTES

    for label in EXPECTED_ROUTES:
        assert route_query({"route": label}) == label
    assert route_query({}) == "chat"
    assert route_query({"route": None}) == "chat"
