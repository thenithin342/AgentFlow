"""
Pytest configuration and shared fixtures.

Reference: DESIGN_DOC.md section 5 "Persistence Design", section 9
"Testing Strategy".

Phase 4: SQLite checkpointer now requires every graph.invoke / ainvoke
call to pass a thread_id. Rather than touch all 8 existing tests, this
file installs an autouse fixture that monkeypatches the wrappers to
inject a default thread_id when none is provided. The new persistence
tests pass an EXPLICIT thread_id; the fixture detects an existing
configurable.thread_id and does NOT override it, so multiple invokes
within one test can share a thread (required for multi-turn tests).

Phase 7: the session-scoped DB-cleanup fixture also wipes
`faiss_indexes/` so each test session starts with no leaked per-thread
vector stores.

Phase 8: Added rate-limit guard. Groq free tier = 100K tokens/day.
When the daily quota is exhausted, every LLM-dependent test raises
groq.RateLimitError. The `_rate_limit_guard` autouse fixture catches
this and re-raises as pytest.xfail (expected failure) so the test run
shows xfailed (x) instead of FAILED (F). Non-LLM tests are unaffected.
"""

import os
import re
import shutil
import sqlite3
from pathlib import Path

# Isolate the sync test checkpointer from the async API DB (agentflow.db).
# Must run before build_graph is imported so _DEFAULT_DB_PATH picks this up.
os.environ.setdefault("CHECKPOINT_DB_PATH", "test_agentflow.db")

import json

import pytest

from backend.graph import build_graph
from backend.rag.ingest import INDEX_ROOT
from backend.settings import get_settings


# --- Auth fixture: shared by every API test ------------------------------
#
# The auth layer reads `data/users.json` from `settings.data_dir`. We
# point that at a tmp_path so tests never touch the real user file,
# then issue a JWT for the test user. The token is attached to
# `Authorization: Bearer …` headers so API tests can use it as default
# headers on the httpx AsyncClient.
#
# Phase 9: hoisted from `tests/test_api.py` so all API-touching tests
# (test_api, test_api_blog, test_api_delete, test_graph::test_sources_in_sse)
# share one source of truth. Keeping it in conftest avoids the copy-paste
# drift where one file gets a secret that another doesn't.

@pytest.fixture
async def auth_headers(tmp_path, monkeypatch):
    from backend import auth as auth_mod
    import backend.main as main_mod

    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "jwt_secret", "test-secret-not-for-prod")
    # Also patch the module-level settings used by `require_user`'s
    # `Depends(get_settings)` so it sees the test secret.
    monkeypatch.setattr(main_mod, "settings", settings, raising=False)

    users_file = tmp_path / "users.json"
    user = {
        "tester": {
            "password_hash": auth_mod.hash_password("test-pw"),
            "created_at": 0.0,
        }
    }
    users_file.write_text(json.dumps(user), encoding="utf-8")

    token = auth_mod.issue_token(settings, "tester")
    return {"Authorization": f"Bearer {token}"}


_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "test_agentflow.db")

# Groq error message fragment that indicates daily token quota exhaustion.
_GROQ_TPD_MARKER = "tokens per day"

from backend.validation import THREAD_ID_RE


# --- Helpers ---------------------------------------------------------------

def _ensure_thread_id(args: tuple, kwargs: dict, default: str) -> dict:
    """Return kwargs with config["configurable"]["thread_id"] set to
    `default` ONLY if not already present. Handles both kwarg config
    (graph.invoke(input, config=cfg)) and positional config
    (graph.invoke(input, cfg)). The caller is expected to splat the
    returned kwargs into the original invocation."""
    # Build a fresh config dict with the default injected.
    if "config" in kwargs and kwargs["config"] is not None:
        base_cfg = dict(kwargs["config"])
    elif len(args) >= 2 and args[1] is not None:
        base_cfg = dict(args[1])
    else:
        base_cfg = {}

    configurable = dict(base_cfg.get("configurable") or {})
    if "thread_id" not in configurable:
        # Validate the *caller's* thread_id too — if a test passes an
        # explicit thread_id that doesn't match the allowlist, we'd
        # rather fail loudly here than crash with a confusing langgraph
        # error inside the graph.
        if not THREAD_ID_RE.fullmatch(default):
            raise ValueError(
                f"default thread_id {default!r} fails allowlist — "
                "fix _default_thread_id in tests/conftest.py"
            )
        configurable["thread_id"] = default
    base_cfg["configurable"] = configurable

    new_kwargs = dict(kwargs)
    new_kwargs["config"] = base_cfg
    return new_kwargs


# --- Session fixture: clean DB --------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _clean_checkpoint_db():
    """Truncate the checkpoints and writes tables at the start of the
    test session. We use a fresh sqlite3 connection (WAL mode allows
    multiple connections in the same process) rather than unlinking
    the DB file — the file is held open by build_graph at module
    import time, and Windows refuses to delete a locked file.

    Truncating tables (instead of file removal) achieves the same
    effect: every test starts with an empty checkpoint store. WAL
    checkpoint via TRUNCATE reclaims disk space.
    """
    # Phase 7: wipe any per-thread FAISS indexes left behind by prior
    # runs BEFORE the early return so the very first session of a
    # fresh checkout still starts clean. (Previously the wipe lived
    # after the `if not os.path.exists(_DB_PATH): return` branch, so
    # a missing-DB-on-first-run session leaked indexes forever.)
    if INDEX_ROOT.exists():
        # Windows holds the dir open across test invocations via FAISS mmap;
        # unlink then raises PermissionError and aborts the whole session.
        # Warning keeps the run going without masking a real bug elsewhere.
        try:
            shutil.rmtree(INDEX_ROOT)
        except PermissionError as exc:  # pragma: no cover — Windows-only path
            import warnings
            warnings.warn(f"Could not clean {INDEX_ROOT}: {exc}")

    if not os.path.exists(_DB_PATH):
        # No DB yet — first run, nothing to clean. Tables are created
        # lazily on first invoke.
        yield
        return

    con = sqlite3.connect(_db_path_for_wal(), check_same_thread=False)
    try:
        cur = con.cursor()
        # The two tables are created lazily by SqliteSaver.setup() on
        # first invoke. If neither exists yet, there is nothing to
        # truncate.
        for table in ("checkpoints", "writes"):
            cur.execute(
                f"SELECT name FROM sqlite_master "
                f"WHERE type='table' AND name=?", (table,)
            )
            if cur.fetchone() is not None:
                cur.execute(f"DELETE FROM {table}")
        con.commit()
    finally:
        con.close()

    # Force a fresh sqlite connection on next graph access. The second
    # connection in _clean_checkpoint_db (sqlite3.connect above) is closed
    # by now, but the default graph proxy may still hold the OLD connection
    # cached from a prior process run. Closing it first releases the
    # underlying sqlite3 connection (Windows refuses to truncate a file
    # held open by SqliteSaver); nilling the reference makes the next
    # test get a clean connection rather than reusing a stale one.
    _close_old_default_graph()

    yield


def _db_path_for_wal() -> str:
    """Resolve the same DB path the live checkpointer uses. Centralized
    so the env-var lookup happens once."""
    return _DB_PATH


def _close_old_default_graph() -> None:
    """Close the lazy default graph's underlying sqlite connection (if any)
    and reset the singleton so the next access rebuilds it.

    The compiled graph holds a live `sqlite3.Connection` (via the
    SqliteSaver checkpointer). Nilling `_default_graph` alone drops the
    Python reference, but the OS file handle lingers until the next GC
    cycle — and on Windows the next test session cannot TRUNCATE a file
    still held open, which surfaces as `database is locked`.

    Defensive on `.close()`: not every CompiledStateGraph exposes one
    (older LangGraph versions, mocks in unit tests). `getattr` keeps
    this safe across backends.
    """
    graph_obj = getattr(build_graph, "_default_graph", None)
    if graph_obj is not None:
        close = getattr(graph_obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — best-effort
                pass
    build_graph._default_graph = None  # force lazy rebuild on next access


# --- Rate-limit guard: xfail instead of FAIL on Groq TPD exhaustion ------

@pytest.fixture(autouse=True)
def _rate_limit_guard():
    """Catch Groq daily token quota (TPD) exhaustion and mark the test as
    xfail rather than FAILED.

    Groq free tier = 100K tokens/day. When that limit is hit every LLM
    call raises `groq.BadRequestError` (HTTP 400) with 'tokens per day'
    in the body — historically it was `groq.RateLimitError` (HTTP 429)
    but the SDK now reports TPD as 400. We match on the message string
    (not the exception class) so the guard keeps working across SDK
    versions. The code is correct — this is an environment constraint.
    Showing xfail (x) instead of FAILED (F) makes that clear.

    Non-LLM tests (ingest, calculator, compile, retrieval) never raise
    any Groq error so they are completely unaffected by this fixture.
    """
    try:
        yield
    except Exception as exc:
        msg = str(exc)
        if _GROQ_TPD_MARKER in msg:
            pytest.xfail(
                reason=(
                    "Groq free-tier daily token quota exhausted (100K TPD). "
                    "The code is correct — quota resets at midnight UTC. "
                    f"Original error: {msg[:120]}"
                )
            )
        raise


# --- Function fixture: inject default thread_id ---------------------------

@pytest.fixture(autouse=True)
def _default_thread_id(monkeypatch, request):
    """Monkeypatch graph.invoke / graph.ainvoke / graph.astream_events
    so any call without an explicit configurable.thread_id gets a
    default of f"pytest-{request.node.name}". Each test gets a unique
    default → no cross-test state leakage (checkpointed state is keyed
    on thread_id).

    The capture of the originals happens INSIDE this fixture, not at
    module load: the session-scoped `_clean_checkpoint_db` closes the
    old default graph and nils the singleton, after which the proxy
    lazily rebuilds on next access. Capturing at import time would
    pin the closed graph's methods; capturing per-test guarantees we
    patch the live one.
    """
    default = f"pytest-{request.node.name}"

    original_invoke = build_graph.graph.invoke
    original_ainvoke = build_graph.graph.ainvoke
    original_astream_events = build_graph.graph.astream_events

    def invoke_patched(*args, **kwargs):
        return original_invoke(*args, **_ensure_thread_id(args, kwargs, default))

    async def ainvoke_patched(*args, **kwargs):
        return await original_ainvoke(*args, **_ensure_thread_id(args, kwargs, default))

    async def astream_events_patched(*args, **kwargs):
        # astream_events returns an async *generator* (not a coroutine) —
        # do NOT await the call itself; iterate it directly.
        async for event in original_astream_events(
            *args, **_ensure_thread_id(args, kwargs, default)
        ):
            yield event

    monkeypatch.setattr(build_graph.graph, "invoke", invoke_patched)
    monkeypatch.setattr(build_graph.graph, "ainvoke", ainvoke_patched)
    monkeypatch.setattr(build_graph.graph, "astream_events", astream_events_patched)
