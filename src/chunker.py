"""Split documents into overlapping chunks for embedding and retrieval."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter with configured chunk size and overlap.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split a list of documents into smaller overlapping chunks.

    Args:
        documents: List of Document objects from the loader.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    if not documents:
        return []

    splitter = create_splitter()
    chunks = splitter.split_documents(documents)

    # Preserve original source metadata in each chunk
    for i, chunk in enumerate(chunks):
        if "chunk_index" not in chunk.metadata:
            chunk.metadata["chunk_index"] = i
        if "chunk_total" not in chunk.metadata:
            chunk.metadata["chunk_total"] = len(chunks)

    return chunks


def get_chunk_stats(documents: List[Document], chunks: List[Document]) -> dict:
    """Return statistics about the chunking process.

    Args:
        documents: Original document list.
        chunks: Resulting chunk list.

    Returns:
        Dictionary with stats (doc_count, chunk_count, avg_chunk_size).
    """
    if not chunks:
        return {"doc_count": len(documents), "chunk_count": 0, "avg_chunk_size": 0}

    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    return {
        "doc_count": len(documents),
        "chunk_count": len(chunks),
        "avg_chunk_size": total_chars // len(chunks),
    }
