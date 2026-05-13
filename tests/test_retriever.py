"""Tests for retriever module."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.retriever import (
    retrieve_similar,
    retrieve_with_scores,
    format_retrieved_context,
    format_retrieval_diagnostics,
)
from src.config import config


class TestRetrieveSimilar:
    """Tests for similarity-based retrieval."""

    def test_empty_query_returns_empty(self):
        mock_store = MagicMock()
        result = retrieve_similar(mock_store, "")
        assert result == []

    def test_retrieve_returns_documents(self):
        mock_store = MagicMock()
        doc = Document(
            page_content="Relevant content about RAG.",
            metadata={"source": "notes.pdf", "page": 3},
        )
        mock_store.similarity_search_with_score.return_value = [(doc, 0.5)]

        result = retrieve_similar(mock_store, "What is RAG?")
        assert len(result) == 1
        assert "Relevant content" in result[0].page_content

    def test_retrieve_with_custom_k(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []

        retrieve_similar(mock_store, "query", k=10)
        mock_store.similarity_search_with_score.assert_called_once_with(
            query="query", k=10,
        )

    def test_retrieve_filters_by_threshold(self):
        mock_store = MagicMock()
        doc_good = Document(page_content="Good", metadata={})
        doc_bad = Document(page_content="Bad", metadata={})
        mock_store.similarity_search_with_score.return_value = [
            (doc_good, 0.3),
            (doc_bad, 2.0),
        ]

        result = retrieve_similar(mock_store, "query", max_distance=1.0)
        assert len(result) == 1
        assert result[0].page_content == "Good"


class TestRetrieveWithScores:
    """Tests for retrieval with relevance scores."""

    def test_empty_query_returns_empty(self):
        mock_store = MagicMock()
        result = retrieve_with_scores(mock_store, "")
        assert result == []

    def test_retrieve_with_scores(self):
        mock_store = MagicMock()
        doc = Document(page_content="Test", metadata={"source": "test.txt"})
        mock_store.similarity_search_with_score.return_value = [(doc, 0.95)]

        result = retrieve_with_scores(mock_store, "test query")
        assert len(result) == 1
        assert result[0][1] == 0.95

    def test_threshold_filters_distant_results(self):
        mock_store = MagicMock()
        doc1 = Document(page_content="Close", metadata={})
        doc2 = Document(page_content="Far", metadata={})
        mock_store.similarity_search_with_score.return_value = [
            (doc1, 0.8),
            (doc2, 2.5),
        ]

        result = retrieve_with_scores(mock_store, "query", max_distance=1.5)
        assert len(result) == 1
        assert result[0][0].page_content == "Close"

    def test_default_threshold_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "relevance_threshold", 0.5)
        mock_store = MagicMock()
        doc = Document(page_content="Over threshold", metadata={})
        mock_store.similarity_search_with_score.return_value = [(doc, 0.8)]

        result = retrieve_with_scores(mock_store, "query")
        assert len(result) == 0


class TestFormatRetrievalDiagnostics:
    """Tests for retrieval diagnostic formatting."""

    def test_empty_scored_documents(self):
        result = format_retrieval_diagnostics([])
        assert result == []

    def test_formats_scored_documents(self):
        doc = Document(
            page_content="First chunk content.\nWith a new line.",
            metadata={"source": "notes.pdf", "page": 3, "chunk_index": 7},
        )

        result = format_retrieval_diagnostics([(doc, 0.42)])

        assert result == [
            {
                "rank": 1,
                "score": 0.42,
                "source": "notes.pdf",
                "page": 3,
                "chunk_index": 7,
                "preview": "First chunk content. With a new line.",
            }
        ]

    def test_missing_metadata_uses_defaults(self):
        doc = Document(page_content="No metadata chunk.", metadata={})

        result = format_retrieval_diagnostics([(doc, 1.5)])

        assert result[0]["source"] == "unknown"
        assert result[0]["page"] is None
        assert result[0]["chunk_index"] is None

    def test_preview_is_truncated(self):
        doc = Document(page_content="A" * 300, metadata={"source": "long.txt"})

        result = format_retrieval_diagnostics([(doc, 0.1)], preview_chars=20)

        assert result[0]["preview"] == "AAAAAAAAAAAAAAAAAAAA..."


class TestFormatRetrievedContext:
    """Tests for context formatting."""

    def test_format_empty_list(self):
        result = format_retrieved_context([])
        assert "No relevant documents" in result

    def test_format_with_documents(self):
        docs = [
            Document(
                page_content="First chunk content.",
                metadata={"source": "notes.pdf", "page": 1},
            ),
            Document(
                page_content="Second chunk content.",
                metadata={"source": "notes.pdf", "page": 2},
            ),
        ]

        result = format_retrieved_context(docs)
        assert "[Source 1:" in result
        assert "[Source 2:" in result
        assert "First chunk content" in result
        assert "Second chunk content" in result
        assert "notes.pdf" in result
        assert "page 1" in result
        assert "page 2" in result
        assert "---" in result

    def test_format_missing_metadata(self):
        docs = [
            Document(page_content="No metadata chunk.", metadata={}),
        ]

        result = format_retrieved_context(docs)
        assert "Source 1" in result
        assert "No metadata chunk" in result
