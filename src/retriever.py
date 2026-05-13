"""Retrieve relevant document chunks for a user query."""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import config


def retrieve_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = None,
    max_distance: float = None,
) -> List[Tuple[Document, float]]:
    """Retrieve chunks with relevance scores, filtered by distance threshold.

    Args:
        vector_store: A Chroma vector store instance.
        query: The user's query string.
        k: Number of chunks to retrieve (default from config).
        max_distance: Maximum L2 distance threshold (default from config).

    Returns:
        List of (Document, score) tuples ordered by relevance.
    """
    if k is None:
        k = config.k_retrieval
    if max_distance is None:
        max_distance = config.relevance_threshold

    if not query.strip():
        return []

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    results = [(doc, score) for doc, score in results if score <= max_distance]
    return results


def retrieve_similar(
    vector_store: Chroma,
    query: str,
    k: int = None,
    max_distance: float = None,
) -> List[Document]:
    """Retrieve the k most similar document chunks for a query.

    Args:
        vector_store: A Chroma vector store instance.
        query: The user's query string.
        k: Number of chunks to retrieve (default from config).
        max_distance: Maximum L2 distance threshold (default from config).

    Returns:
        List of Document objects ordered by relevance (most relevant first).
    """
    scored = retrieve_with_scores(vector_store, query, k=k, max_distance=max_distance)
    return [doc for doc, _score in scored]


def format_retrieval_diagnostics(
    scored_documents: List[Tuple[Document, float]],
    preview_chars: int = 220,
) -> List[dict]:
    """Format scored retrieval results for benchmark diagnostics.

    Args:
        scored_documents: List of (Document, score) tuples.
        preview_chars: Maximum characters to include from each chunk.

    Returns:
        List of diagnostic dictionaries with rank, score, metadata, and preview.
    """
    diagnostics = []
    for rank, (doc, score) in enumerate(scored_documents, start=1):
        preview = " ".join(doc.page_content.split())
        if len(preview) > preview_chars:
            preview = f"{preview[:preview_chars].rstrip()}..."

        diagnostics.append(
            {
                "rank": rank,
                "score": score,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "preview": preview,
            }
        )

    return diagnostics


def format_retrieved_context(documents: List[Document]) -> str:
    """Format retrieved documents into a single context string for the LLM.

    Args:
        documents: List of retrieved Document objects.

    Returns:
        Formatted context string with source annotations.
    """
    if not documents:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(
            f"[Source {i+1}: {source}, page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)
