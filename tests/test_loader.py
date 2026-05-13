"""Tests for document loader module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.loader import (
    validate_file,
    load_pdf,
    load_txt,
    load_markdown,
    load_document,
)
from src.config import config


class TestValidateFile:
    """Tests for file validation."""

    def test_valid_txt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_path = f.name

        try:
            result = validate_file(tmp_path)
            assert isinstance(result, Path)
            assert result.suffix == ".txt"
        finally:
            Path(tmp_path).unlink()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            validate_file("/nonexistent/file.pdf")

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                validate_file(tmp_path)
        finally:
            Path(tmp_path).unlink()


class TestLoadTxt:
    """Tests for TXT file loading."""

    def test_load_simple_txt(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello world.\nThis is a test document.")
            tmp_path = f.name

        try:
            docs = load_txt(tmp_path)
            assert len(docs) == 1
            assert "Hello world" in docs[0].page_content
            assert docs[0].metadata["source"].endswith(".txt")
        finally:
            Path(tmp_path).unlink()

    def test_load_empty_txt(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("   ")
            tmp_path = f.name

        try:
            docs = load_txt(tmp_path)
            assert len(docs) == 0
        finally:
            Path(tmp_path).unlink()


class TestLoadMarkdown:
    """Tests for Markdown file loading."""

    def test_load_md_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Title\n\nSome **bold** text.")
            tmp_path = f.name

        try:
            docs = load_markdown(tmp_path)
            assert len(docs) == 1
            assert "# Title" in docs[0].page_content
        finally:
            Path(tmp_path).unlink()


class TestLoadDocument:
    """Tests for generic document loader."""

    def test_load_document_dispatches_correctly(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("test content for dispatch")
            tmp_path = f.name

        try:
            docs = load_document(tmp_path)
            assert len(docs) == 1
            assert "test content for dispatch" in docs[0].page_content
        finally:
            Path(tmp_path).unlink()
