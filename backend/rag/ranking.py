"""Lightweight lexical reranking of vector-search candidates."""

from __future__ import annotations

import math
import re
from typing import Any, TypeVar

_Doc = TypeVar("_Doc")
_STOP = frozenset({
    "a", "and", "are", "by", "did", "do", "does", "for", "from", "how",
    "in", "is", "me", "my", "of", "the", "this", "that", "to", "was",
    "were", "what", "when", "which", "who", "with",
})


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    terms = set()
    for word in words:
        if word in _STOP or len(word) <= 2:
            continue
        # shortcut: English suffixes only; use a language-aware reranker for multilingual corpora.
        for suffix in ("ing", "ed", "ly", "al", "s", "e"):
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                word = word[:-len(suffix)]
        terms.add(word)
    return terms


def rerank_documents(query: str, documents: list[_Doc]) -> list[_Doc]:
    """Move candidates with matching, distinctive query terms earlier.

    Fuse vector and lexical ranks so incidental exact matches cannot discard
    semantic relevance. No provider call or corpus-wide index is needed.
    """
    query_terms = _terms(query)
    if not query_terms or len(documents) < 2:
        return list(documents)
    terms_by_doc = [_terms(str(getattr(doc, "page_content", ""))) for doc in documents]
    count = len(documents)
    weights = {
        term: 1 + math.log((count + 1) / (1 + sum(term in terms for terms in terms_by_doc)))
        for term in query_terms
    }
    lexical_order = sorted(
        range(count),
        key=lambda index: -sum(
            weights[term] for term in query_terms if term in terms_by_doc[index]
        ),
    )
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order)}
    # Reciprocal rank fusion gives both retrieval signals equal influence.
    order = sorted(
        range(count),
        key=lambda index: -(1 / (60 + index) + 1 / (60 + lexical_rank[index])),
    )
    return [documents[index] for index in order]


def retrieve_ranked_documents(query: str, retriever: Any, k: int = 4) -> list[Any]:
    """Rerank a wider vector candidate pool without mutating a cached retriever."""
    candidates = retriever.invoke(query, k=3 * k)
    return rerank_documents(query, candidates)[:k]
