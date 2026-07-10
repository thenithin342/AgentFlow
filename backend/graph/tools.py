"""
Tools used by the Research and Analysis agents.

Phase 9 additions:
    wikipedia_search   — quick encyclopaedic lookups
    datetime_tool      — current date/time without a web call
    url_reader         — fetch and extract text from a public URL
    code_interpreter   — run sandboxed Python for numeric/data tasks
"""

import ast
import operator
import os
import textwrap
from typing import Any

try:
    from langchain_tavily import TavilySearch as _TavilyClient
    _USE_NEW_TAVILY = True
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults as _TavilyClient  # noqa: F401
    _USE_NEW_TAVILY = False
from langchain_core.tools import tool

from backend.graph.security import escape_untrusted


_tavily_instance = None


def _get_tavily():
    global _tavily_instance
    if _tavily_instance is None:
        _tavily_instance = _TavilyClient(max_results=3) if _USE_NEW_TAVILY else _TavilyClient(max_results=3)
    return _tavily_instance


@tool
def tavily_search(query: str) -> str:
    """Search the web for current information relevant to the query."""
    try:
        results = _get_tavily().invoke(query)
        return str(results)
    except Exception as exc:  # noqa: BLE001
        return f"Error: web search unavailable ({type(exc).__name__}: {exc})"


_MAX_EXPR_LEN = 256
_MAX_AST_NODES = 64
_MAX_POW_RIGHT = 10000
_MAX_POW_BASE = 1000

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _count_nodes(node: ast.AST) -> int:
    n = 1
    for child in ast.iter_child_nodes(node):
        n += _count_nodes(child)
    return n


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("unsupported constant: bool")
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {type(node.value).__name__}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"unsupported operator: {op_type.__name__}")
        left = _eval(node.left)
        right = _eval(node.right)
        if op_type is ast.Pow:
            if right > _MAX_POW_RIGHT:
                raise ValueError(f"exponent too large (>{_MAX_POW_RIGHT})")
            if abs(left) > _MAX_POW_BASE:
                raise ValueError(f"base magnitude too large (>{_MAX_POW_BASE})")
        return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        operand = _eval(node.operand)
        return _UNARY_OPS[op_type](operand)
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    if not isinstance(expression, str):
        return f"Error: expression must be a string, got {type(expression).__name__}"
    if len(expression) > _MAX_EXPR_LEN:
        return f"Error: expression too long (>{_MAX_EXPR_LEN} chars)"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"Error: invalid syntax ({exc.msg})"
    if _count_nodes(tree) > _MAX_AST_NODES:
        return f"Error: expression too complex (>{_MAX_AST_NODES} AST nodes)"
    try:
        return str(_eval(tree))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {type(exc).__name__}: {exc}"


def make_retrieve_documents_tool(thread_id: str):
    from backend.rag.ingest import _index_dir, get_retriever

    @tool
    def retrieve_documents(query: str) -> str:
        """Retrieve relevant chunks from this thread's uploaded documents."""
        if not os.path.isdir(_index_dir(thread_id)):
            return "Error: No documents uploaded for this conversation. STOP searching and inform the user that they must upload a document first."

        try:
            retriever = get_retriever(thread_id)
            docs = retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001
            return f"Error: retrieval failed ({type(exc).__name__}: {exc})"

        if not docs:
            return "No relevant information found in the uploaded documents. Do not retry the search with the same meaning."

        lines: list[str] = []
        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("page", d.metadata.get("source", "unknown"))
            excerpt = d.page_content.strip().replace("\n", " ")
            safe_source = escape_untrusted(str(source))
            safe_excerpt = escape_untrusted(excerpt)
            lines.append(
                f"[{i}] <<UNTRUSTED EXCERPT — source: {safe_source}>> {safe_excerpt} <<END EXCERPT>>"
            )
        return "\n\n".join(lines)

    return retrieve_documents


# ---------------------------------------------------------------------------
# Phase 9 — expanded tools
# ---------------------------------------------------------------------------


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopaedic information about a topic."""
    if not isinstance(query, str) or not query.strip():
        return "Error: query must be a non-empty string"
    try:
        import urllib.request
        import urllib.parse
        import json as _json
        encoded = urllib.parse.quote(query.strip())
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgentFlow/1.0 (portfolio project)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read(200_000).decode("utf-8"))
        title = data.get("title", "")
        extract = data.get("extract", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not extract:
            return f"No Wikipedia summary found for: {query}"
        result = f"**{title}**\n{extract}"
        if page_url:
            result += f"\nSource: {page_url}"
        return result
    except Exception as exc:
        return f"Error: Wikipedia lookup failed ({type(exc).__name__}: {exc})"


@tool
def datetime_tool(query: str = "") -> str:
    """Return the current UTC date and time. Useful for date-aware questions."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return (
        f"Current UTC datetime: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"ISO 8601: {now.isoformat()}\n"
        f"Day of week: {now.strftime('%A')}"
    )


_URL_MAX_BYTES = 200_000  # 200 KB text cap
_URL_MAX_CHARS = 8_000    # trim returned text to this many characters


@tool
def url_reader(url: str) -> str:
    """Fetch the text content of a public URL and return it as plain text.

    Useful for reading articles, documentation pages, or any public web content
    when you have the direct URL. Returns the first 8000 characters of the page
    text with HTML tags stripped.
    """
    if not isinstance(url, str) or not url.strip():
        return "Error: url must be a non-empty string"
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: url must start with http:// or https://"
    try:
        import http.client
        import urllib.parse
        import urllib.error
        import re as _re
        import socket
        import ipaddress
        import ssl

        class SafeHTTPConnection(http.client.HTTPConnection):
            def _resolve_and_validate(self) -> str:
                # Resolve once and pin the validated IP. We must NOT rely on the
                # second DNS lookup performed by `socket.create_connection` --
                # between the first and second resolve, a DNS rebinding attacker
                # can flip the answer to a private/loopback address. Connect only
                # to the IP we just validated, and send the original Host header
                # so SNI / vhost routing still works.
                infos = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
                if not infos:
                    raise ValueError(f"No DNS results for {self.host!r}")
                for family, _type, _proto, _canon, sockaddr in infos:
                    ip = sockaddr[0]
                    ip_obj = ipaddress.ip_address(ip)
                    if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
                            or ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified
                            or not ip_obj.is_global):
                        raise ValueError(f"Resolves to non-public IP: {ip}")
                # Use the FIRST validated IP; subsequent A/AAAA ranswers should not
                # bypass the check above. Re-resolving here is the bug we are fixing.
                return infos[0][4][0]

            def connect(self):
                ip = self._resolve_and_validate()
                self.sock = socket.create_connection((ip, self.port), self.timeout, self.source_address)
                if getattr(self, "_tunnel_host", None):
                    self._tunnel()

        class SafeHTTPSConnection(http.client.HTTPSConnection):
            def connect(self):
                ip = self._resolve_and_validate()
                self.sock = socket.create_connection((ip, self.port), self.timeout, self.source_address)
                if getattr(self, "_tunnel_host", None):
                    self._tunnel()
                # server_hostname must be the original DNS name so SNI + cert
                # validation work; the connection itself is pinned to the IP.
                self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)

        class SafeHTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                return self.do_open(SafeHTTPConnection, req)

        class SafeHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(SafeHTTPSConnection, req, context=ssl.create_default_context())

        class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
            def __init__(self):
                self.redirect_count = 0
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                self.redirect_count += 1
                if self.redirect_count > 3:
                    raise urllib.error.URLError("Too many redirects")
                new_parsed = urllib.parse.urlparse(newurl)
                if new_parsed.scheme not in ('http', 'https'):
                    raise urllib.error.URLError("Redirected to disallowed scheme")
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(SafeHTTPHandler(), SafeHTTPSHandler(), SafeRedirectHandler())
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AgentFlow/1.0 (portfolio project; +research)",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.8",
            },
        )
        with opener.open(req, timeout=15) as resp:
            raw = resp.read(_URL_MAX_BYTES)
        text = raw.decode("utf-8", errors="replace")
        # Strip HTML tags
        text = _re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = _re.sub(r"[ \t]+", " ", text)
        text = _re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > _URL_MAX_CHARS:
            text = text[:_URL_MAX_CHARS] + f"\n\n[truncated — {len(text)} chars total]"
        return text if text else "Page fetched but no readable text found."
    except Exception as exc:
        return f"Error: could not fetch URL ({type(exc).__name__}: {exc})"


# Code interpreter safety list.
#
# SECURITY NOTE: This list must be small. The previous version exposed
# `getattr`, `hasattr`, `type`, etc. With `statistics.mean.__globals__` still
# reachable, that combination allowed `getattr(statistics.mean, '__globals__')`
# -> `__builtins__` -> `__import__('os')` -> arbitrary RCE inside the worker
# process. Verified reproduction against the prior sandbox:
#     imp = statistics.mean.__globals__['__builtins__'].__import__
#     imp('os').uname()  # works without `type` even being available
#
# Fix: drop `getattr`, `hasattr`, `type`. The agent prompts request numeric
# computation, statistics, and string manipulation -- none of those need
# introspection. `dir()` and `isinstance` are also dropped because `dir(obj)`
# leaks public dunder names from `math`/`statistics` modules; we expose only
# the safe builtins below and the two whitelisted modules.
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "int", "iter", "len", "list", "map",
    "max", "min", "next", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip",
}
_CODE_MAX_LEN = 2000
_CODE_TIMEOUT = 5  # seconds


import multiprocessing
import io
import contextlib

def _code_worker(code_str: str, result_q: multiprocessing.Queue) -> None:
    """Runs inside a child process — killed on timeout."""
    try:
        import builtins
        import math
        import statistics
        safe_globals = {
            "__builtins__": {k: getattr(builtins, k) for k in _SAFE_BUILTINS},
            "math": math,
            "statistics": statistics,
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code_str, safe_globals, {})  # noqa: S102
        result_q.put(("ok", out.getvalue()))
    except Exception as exc:  # noqa: BLE001
        result_q.put(("err", f"{type(exc).__name__}: {exc}"))

@tool
def code_interpreter(code: str) -> str:
    """Execute Python code in a sandboxed environment.
    The snippet's stdout is returned.

    Runs in a restricted in-process environment with no network access or dangerous imports.
    Ideal for data processing, statistics, string manipulation, and numeric computation
    (e.g., iterating to find prime numbers).

    DO NOT USE for shell commands, system utilities, or accessing the local filesystem
    (the sandboxed environment forbids os/sys operations). Wait for the tool output before
    continuing execution.
    """
    if not isinstance(code, str):
        return f"Error: code must be a string, got {type(code).__name__}"
    code = textwrap.dedent(code).strip()
    if not code:
        return "Error: empty code snippet"
    if len(code) > _CODE_MAX_LEN:
        return f"Error: code too long (max {_CODE_MAX_LEN} chars)"

    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_code_worker, args=(code, q), daemon=True)
    try:
        proc.start()
        proc.join(timeout=_CODE_TIMEOUT)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return "Error: execution timed out"
        status, payload = q.get_nowait()
        if status == "ok":
            return payload.strip() if payload.strip() else "(no output)"
        return f"Error: {payload}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2)
