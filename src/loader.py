"""Load and extract text from PDF, TXT, and Markdown files."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config import config


def validate_file(file_path: str) -> Path:
    """Validate that a file exists and has an allowed extension.

    Args:
        file_path: Path to the file to validate.

    Returns:
        Path object for the validated file.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: File extension not allowed or file too large.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in config.allowed_extensions:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Allowed: {config.allowed_extensions}"
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > config.max_upload_size_mb:
        raise ValueError(
            f"File too large: {file_size_mb:.1f}MB. "
            f"Max: {config.max_upload_size_mb}MB"
        )

    return path


def load_pdf(file_path: str) -> List[Document]:
    """Load and extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page.
    """
    from PyPDF2 import PdfReader

    path = validate_file(file_path)
    reader = PdfReader(str(path))

    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": path.name,
                        "page": i + 1,
                        "file_path": str(path.absolute()),
                    },
                )
            )

    return documents


def load_txt(file_path: str) -> List[Document]:
    """Load text from a plain text file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        List with a single Document object.
    """
    path = validate_file(file_path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return []

    return [
        Document(
            page_content=text.strip(),
            metadata={
                "source": path.name,
                "file_path": str(path.absolute()),
            },
        )
    ]


def load_markdown(file_path: str) -> List[Document]:
    """Load text from a Markdown file (treated as plain text).

    Args:
        file_path: Path to the .md file.

    Returns:
        List with a single Document object.
    """
    # Markdown is loaded as plain text, preserving the raw markdown.
    return load_txt(file_path)


LOADER_MAP = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_markdown,
}


def load_document(file_path: str) -> List[Document]:
    """Load a document of any supported type.

    Args:
        file_path: Path to the document file.

    Returns:
        List of Document objects with page content and metadata.
    """
    path = validate_file(file_path)
    ext = path.suffix.lower()

    loader = LOADER_MAP.get(ext)
    if loader is None:
        raise ValueError(f"No loader for extension: {ext}")

    return loader(file_path)
