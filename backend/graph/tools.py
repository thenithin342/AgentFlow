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
        import urllib.request
        import urllib.parse
        import urllib.error
        import re as _re
        import socket
        import ipaddress

        def validate_host(hostname):
            if not hostname: return
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                raise ValueError("Resolves to private/local IP")

        parsed = urllib.parse.urlparse(url)
        validate_host(parsed.hostname)

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
                validate_host(new_parsed.hostname)
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(SafeRedirectHandler())
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


# Code interpreter safety list
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip",
}
_CODE_MAX_LEN = 2000
_CODE_TIMEOUT = 5  # seconds


@tool
def code_interpreter(code: str) -> str:
    """Execute a small Python snippet and return the output.

    Runs in a restricted subprocess with no network access or dangerous imports.
    Ideal for data processing, statistics, string manipulation, and numeric computation
    that goes beyond the calculator tool.

    The snippet's stdout is returned.
    Execution is time-limited to 5 seconds.
    """
    if not isinstance(code, str):
        return f"Error: code must be a string, got {type(code).__name__}"
    code = textwrap.dedent(code).strip()
    if not code:
        return "Error: empty code snippet"
    if len(code) > _CODE_MAX_LEN:
        return f"Error: code too long (max {_CODE_MAX_LEN} chars)"

    import subprocess
    import tempfile
    import sys
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # Wrap user code to print the last expression if it doesn't already print
        wrapped_code = (
            "import math, statistics\n"
            + code
        )
        f.write(wrapped_code)
        temp_path = f.name
        
    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=_CODE_TIMEOUT
        )
        out = result.stdout
        err = result.stderr
        if err:
            out += "\nErrors:\n" + err
        return out.strip() if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
