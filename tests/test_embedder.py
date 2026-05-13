"""Tests for embedding module."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.embedder import create_embeddings, embed_chunks, embed_query
from src.config import config


class TestCreateEmbeddings:
    """Tests for embedding model creation."""

    @patch("src.embedder.OllamaEmbeddings")
    def test_create_ollama_embeddings(self, mock_cls, monkeypatch):
        monkeypatch.setattr(config, "embedding_provider", "ollama")
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        emb = create_embeddings()
        mock_cls.assert_called_once_with(
            model=config.local_embedding_model,
            base_url=config.local_base_url,
        )
        assert emb == mock_instance

    @patch("src.embedder.GoogleGenerativeAIEmbeddings")
    def test_create_gemini_embeddings(self, mock_cls, monkeypatch):
        monkeypatch.setattr(config, "embedding_provider", "gemini")
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        emb = create_embeddings()
        mock_cls.assert_called_once_with(
            model=config.gemini_embedding_model,
            google_api_key=config.google_api_key,
        )
        assert emb == mock_instance


class TestEmbedChunks:
    """Tests for chunk embedding."""

    def test_embed_empty_list(self):
        result = embed_chunks([])
        assert result == []

    @patch("src.embedder.create_embeddings")
    def test_embed_chunks_returns_vectors(self, mock_create_emb):
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_create_emb.return_value = mock_emb

        chunks = [
            Document(page_content="First chunk", metadata={"source": "test.txt"}),
            Document(page_content="Second chunk", metadata={"source": "test.txt"}),
        ]

        result = embed_chunks(chunks)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


class TestEmbedQuery:
    """Tests for query embedding."""

    @patch("src.embedder.create_embeddings")
    def test_embed_query_returns_vector(self, mock_create_emb):
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_create_emb.return_value = mock_emb

        result = embed_query("What is RAG?")
        assert result == [0.1, 0.2, 0.3]
        mock_emb.embed_query.assert_called_once_with("What is RAG?")
