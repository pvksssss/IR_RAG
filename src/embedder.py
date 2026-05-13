"""Generate embeddings for document chunks using the configured provider."""

from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from src.config import config


def create_embeddings():
    """Create an embeddings model for the configured provider.

    Returns:
        A provider-specific embeddings instance.
    """
    if config.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=config.local_embedding_model,
            base_url=config.local_base_url,
        )

    return GoogleGenerativeAIEmbeddings(
        model=config.gemini_embedding_model,
        google_api_key=config.google_api_key,
    )


def embed_chunks(
    chunks: List[Document],
    embeddings=None,
) -> List[List[float]]:
    """Generate embedding vectors for a list of document chunks.

    Args:
        chunks: List of chunked Document objects.
        embeddings: Optional pre-configured embeddings model.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not chunks:
        return []

    if embeddings is None:
        embeddings = create_embeddings()

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    return vectors


def embed_query(query: str, embeddings=None) -> List[float]:
    """Generate an embedding vector for a user query.

    Args:
        query: The user's question string.
        embeddings: Optional pre-configured embeddings model.

    Returns:
        Embedding vector for the query.
    """
    if embeddings is None:
        embeddings = create_embeddings()

    return embeddings.embed_query(query)
