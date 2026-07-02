"""
Tools used by the Research and Analysis agents.
"""

import ast
import operator
import os
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
        _tavily_instance = _TavilyClient(max_results=5) if _USE_NEW_TAVILY else _TavilyClient(max_results=5)
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
            return "No documents uploaded for this conversation."

        try:
            retriever = get_retriever(thread_id)
            docs = retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001
            return f"Error: retrieval failed ({type(exc).__name__}: {exc})"
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
