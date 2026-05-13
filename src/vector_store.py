"""ChromaDB vector store for storing and querying document embeddings."""

import shutil
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import config
from src.embedder import create_embeddings


def get_chroma_settings() -> Settings:
    """Create ChromaDB settings for local persistent storage."""
    return Settings(anonymized_telemetry=False)


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create a persistent ChromaDB client.

    Returns:
        A PersistentClient connected to the configured directory.
    """
    return chromadb.PersistentClient(
        path=config.chroma_persist_dir,
        settings=get_chroma_settings(),
    )


def reset_vector_store() -> bool:
    """Delete the local ChromaDB directory.

    Use this when changing embedding models or when an old Chroma schema is
    incompatible with the installed ChromaDB version.
    """
    persist_path = Path(config.chroma_persist_dir)
    if not persist_path.exists():
        return False

    try:
        shutil.rmtree(persist_path)
        return True
    except OSError:
        return False


def create_vector_store(
    collection_name: str = "documents",
) -> Chroma:
    """Create a Chroma vector store with the configured embeddings.

    Args:
        collection_name: Name for the ChromaDB collection.

    Returns:
        A Chroma vector store instance.
    """
    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_dir,
        client_settings=get_chroma_settings(),
    )

    return vector_store


def add_documents_to_store(
    vector_store: Chroma,
    chunks: List[Document],
    embeddings=None,
) -> int:
    """Add document chunks to the vector store with their embeddings.

    Embeds each chunk individually to handle Gemini API content policy
    rejections gracefully (skips chunks that fail embedding).

    Args:
        vector_store: A Chroma vector store instance.
        chunks: List of chunked Document objects.
        embeddings: Embedding model instance. If None, creates one via config.

    Returns:
        Number of documents added.
    """
    if not chunks:
        return 0

    if embeddings is None:
        embeddings = create_embeddings()

    collection = vector_store._collection

    added = 0
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        if not text or not text.strip():
            continue

        chunk_id = (
            f"chunk_{chunk.metadata.get('source', 'unknown')}_"
            f"{chunk.metadata.get('page', 'nopage')}_"
            f"{chunk.metadata.get('chunk_index', i)}"
        )

        try:
            embedding = embeddings.embed_documents([text])[0]
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[chunk.metadata],
            )
            added += 1
        except Exception:
            continue

    return added


def delete_collection(collection_name: str = "documents") -> bool:
    """Delete a ChromaDB collection.

    Args:
        collection_name: Name of the collection to delete.

    Returns:
        True if deleted successfully.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        return True
    except (ValueError, Exception):
        # Collection doesn't exist or other ChromaDB error
        return False


def get_collection_stats(
    vector_store: Chroma = None,
    collection_name: str = "documents",
) -> dict:
    """Get statistics about a vector store collection.

    Args:
        vector_store: Optional Chroma instance (used to extract collection name).
        collection_name: Fallback collection name if vector_store not provided.

    Returns:
        Dictionary with collection stats.
    """
    if vector_store is not None:
        collection_name = vector_store._collection.name

    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
    except Exception:
        count = 0

    return {
        "total_chunks": count,
        "collection_name": collection_name,
    }
