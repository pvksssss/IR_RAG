"""Tests for document chunker module."""

from langchain_core.documents import Document

from src.chunker import chunk_documents, get_chunk_stats, create_splitter
from src.config import config


class TestCreateSplitter:
    """Tests for splitter creation."""

    def test_splitter_uses_config_values(self):
        splitter = create_splitter()
        assert splitter._chunk_size == config.chunk_size
        assert splitter._chunk_overlap == config.chunk_overlap


class TestChunkDocuments:
    """Tests for document chunking."""

    def test_single_short_document(self):
        doc = Document(
            page_content="This is a short document.",
            metadata={"source": "test.txt"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1
        assert "short document" in chunks[0].page_content

    def test_long_document_gets_chunked(self):
        # Create a document longer than chunk_size
        long_text = "Lorem ipsum dolor sit amet. " * 200
        doc = Document(
            page_content=long_text,
            metadata={"source": "test.txt"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        # Each chunk should be roughly <= chunk_size
        for chunk in chunks:
            assert len(chunk.page_content) <= config.chunk_size + config.chunk_overlap + 100

    def test_empty_list_returns_empty(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunks_preserve_metadata(self):
        doc = Document(
            page_content="Test " * 50,
            metadata={"source": "myfile.pdf"},
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "myfile.pdf"
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata


class TestGetChunkStats:
    """Tests for chunk statistics."""

    def test_stats_with_documents(self):
        docs = [
            Document(page_content="A" * 500, metadata={"source": "test.txt"}),
        ]
        chunks = chunk_documents(docs)
        stats = get_chunk_stats(docs, chunks)
        assert stats["doc_count"] == 1
        assert stats["chunk_count"] > 0
        assert stats["avg_chunk_size"] > 0

    def test_stats_empty(self):
        stats = get_chunk_stats([], [])
        assert stats["doc_count"] == 0
        assert stats["chunk_count"] == 0
        assert stats["avg_chunk_size"] == 0
