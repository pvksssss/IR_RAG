"""Tests for ChromaDB vector store module."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.vector_store import (
    get_chroma_client,
    get_chroma_settings,
    create_vector_store,
    add_documents_to_store,
    delete_collection,
    reset_vector_store,
    get_collection_stats,
)
from src.config import config


class TestChromaClient:
    """Tests for ChromaDB client creation."""

    @patch("src.vector_store.chromadb.PersistentClient")
    @patch("src.vector_store.get_chroma_settings")
    def test_get_chroma_client(self, mock_settings, mock_client_cls):
        mock_client = MagicMock()
        mock_settings.return_value = "settings"
        mock_client_cls.return_value = mock_client

        client = get_chroma_client()
        mock_client_cls.assert_called_once_with(
            path=config.chroma_persist_dir,
            settings="settings",
        )
        assert client == mock_client

    def test_get_chroma_settings_disables_telemetry(self):
        settings = get_chroma_settings()
        assert settings.anonymized_telemetry is False


class TestCreateVectorStore:
    """Tests for vector store creation."""

    @patch("src.vector_store.create_embeddings")
    @patch("src.vector_store.Chroma")
    @patch("src.vector_store.get_chroma_settings")
    def test_create_vector_store(
        self, mock_settings, mock_chroma_cls, mock_create_emb
    ):
        mock_emb = MagicMock()
        mock_settings.return_value = "settings"
        mock_create_emb.return_value = mock_emb
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store

        store = create_vector_store(collection_name="test_collection")
        mock_chroma_cls.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_emb,
            persist_directory=config.chroma_persist_dir,
            client_settings="settings",
        )
        assert store == mock_store


class TestResetVectorStore:
    """Tests for resetting local ChromaDB files."""

    def test_reset_missing_store(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing_chroma"
        monkeypatch.setattr(config, "chroma_persist_dir", str(missing))

        assert reset_vector_store() is False

    def test_reset_existing_store(self, tmp_path, monkeypatch):
        store = tmp_path / "chroma_db"
        store.mkdir()
        (store / "chroma.sqlite3").write_text("test")
        monkeypatch.setattr(config, "chroma_persist_dir", str(store))

        assert reset_vector_store() is True
        assert not store.exists()


class TestAddDocumentsToStore:
    """Tests for adding documents to vector store."""

    def test_add_empty_chunks(self):
        mock_store = MagicMock()
        mock_embeddings = MagicMock()
        result = add_documents_to_store(mock_store, [], embeddings=mock_embeddings)
        assert result == 0

    def test_add_chunks(self):
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_store._collection = mock_collection
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        chunks = [
            Document(
                page_content="Test chunk 1",
                metadata={"source": "test.txt", "chunk_index": 0}
            ),
            Document(
                page_content="Test chunk 2",
                metadata={"source": "test.txt", "chunk_index": 1}
            ),
        ]

        result = add_documents_to_store(mock_store, chunks, embeddings=mock_embeddings)
        assert result == 2
        assert mock_embeddings.embed_documents.call_count == 2
        assert mock_collection.add.call_count == 2
        assert mock_embeddings.embed_documents.call_args_list[0].args == (["Test chunk 1"],)

        first_call = mock_collection.add.call_args_list[0].kwargs
        assert first_call["documents"] == ["Test chunk 1"]
        assert first_call["ids"] == ["chunk_test.txt_nopage_0"]

    def test_skip_chunk_when_embedding_fails(self):
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_store._collection = mock_collection
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.side_effect = [
            Exception("policy blocked"),
            [[0.4, 0.5, 0.6]],
        ]
        chunks = [
            Document(
                page_content="Blocked chunk",
                metadata={"source": "test.txt", "chunk_index": 0}
            ),
            Document(
                page_content="Allowed chunk",
                metadata={"source": "test.txt", "chunk_index": 1}
            ),
        ]

        result = add_documents_to_store(mock_store, chunks, embeddings=mock_embeddings)
        assert result == 1
        assert mock_collection.add.call_count == 1


class TestDeleteCollection:
    """Tests for collection deletion."""

    @patch("src.vector_store.get_chroma_client")
    def test_delete_existing_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = delete_collection("test_collection")
        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")

    @patch("src.vector_store.get_chroma_client")
    def test_delete_nonexistent_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = ValueError("not found")
        mock_get_client.return_value = mock_client

        result = delete_collection("nonexistent")
        assert result is False


class TestGetCollectionStats:
    """Tests for collection statistics."""

    @patch("src.vector_store.get_chroma_client")
    def test_get_stats(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_collection.name = "test_docs"
        mock_get_client.return_value.get_collection.return_value = mock_collection

        mock_store = MagicMock()
        mock_store._collection.name = "test_docs"

        stats = get_collection_stats(mock_store)
        assert stats["total_chunks"] == 42
        assert stats["collection_name"] == "test_docs"

    @patch("src.vector_store.get_chroma_client")
    def test_get_stats_on_error(self, mock_get_client):
        mock_get_client.return_value.get_collection.side_effect = Exception("db error")

        stats = get_collection_stats(collection_name="missing")
        assert stats["total_chunks"] == 0

    @patch("src.vector_store.get_chroma_client")
    def test_get_stats_without_vector_store(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_get_client.return_value.get_collection.return_value = mock_collection

        stats = get_collection_stats(collection_name="my_collection")
        assert stats["total_chunks"] == 10
        assert stats["collection_name"] == "my_collection"
