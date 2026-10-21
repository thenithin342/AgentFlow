"""Query-aware ordering of vector retrieval candidates."""

from langchain_core.documents import Document

from backend.rag.ranking import _terms, rerank_documents, retrieve_ranked_documents


def test_distinctive_query_terms_move_relevant_chunk_first():
    documents = [
        Document(page_content="Company overview and operating history"),
        Document(page_content="The South region led revenue in fiscal 2024"),
        Document(page_content="Revenue data for individual products"),
    ]
    ranked = rerank_documents("Which region led revenue?", documents)
    assert ranked[0] is documents[1]
    assert set(map(id, ranked)) == set(map(id, documents))


def test_no_lexical_evidence_keeps_vector_order():
    documents = [Document(page_content="alpha"), Document(page_content="beta")]
    assert rerank_documents("unknown topic", documents) == documents


def test_inflected_query_terms_match_relevant_document():
    assert _terms("dynamic routing") == _terms("route dynamically")


def test_retrieval_reranks_beyond_the_final_top_k_without_changing_shared_settings():
    documents = [
        Document(page_content="General company information."),
        Document(page_content="Other general information."),
        Document(page_content="Revenue by region."),
    ]

    class Retriever:
        search_kwargs = {"k": 2}

        def invoke(self, query, **kwargs):
            assert kwargs["k"] == 6
            return documents

    retriever = Retriever()
    result = retrieve_ranked_documents("Revenue by region?", retriever, k=2)
    assert len(result) == 2 and documents[2] in result
    assert retriever.search_kwargs == {"k": 2}


def test_semantic_match_survives_incidental_exact_matches():
    documents = [
        Document(page_content="Routing classifies intent as research, analysis, chat, or blog."),
        Document(page_content="Company support handles questions directly."),
        Document(page_content="Our chat agent handles questions without tools."),
        Document(page_content="Our research agent handles questions using search."),
    ]
    ranked = rerank_documents("Which agent handles my question?", documents)
    assert documents[0] in ranked[:2]
