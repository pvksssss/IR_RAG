from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
    )

    # Gemini API (LLM + Embeddings)
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/embedding-001"

    # Ollama (LLM + Embeddings)
    ollama_llm_model: str = "qwen2.5:3b"
    ollama_llm_base_url: str = "http://localhost:11434"
    llm_provider: str = "ollama"  # "ollama" or "gemini"
    embedding_provider: str = "ollama"  # "ollama" or "gemini"

    # Ollama (local Embeddings)
    local_embedding_model: str = "bge-m3"
    local_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    k_retrieval: int = 6
    relevance_threshold: float = 1.5  # max L2 distance (lower = more similar)

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"

    # Upload limits
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".txt", ".md"]

    # Temperature for LLM responses
    temperature: float = 0.3


config = Config()
