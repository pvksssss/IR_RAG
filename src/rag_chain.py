"""RAG pipeline: orchestrate document ingestion and question answering."""

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import config
from src.loader import load_document
from src.chunker import chunk_documents, get_chunk_stats
from src.embedder import create_embeddings
from src.vector_store import (
    create_vector_store,
    add_documents_to_store,
    delete_collection,
    reset_vector_store,
    get_collection_stats,
)
from src.retriever import (
    retrieve_similar,
    retrieve_with_scores,
    format_retrieved_context,
    format_retrieval_diagnostics,
)

# Prompt template for RAG
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý trả lời câu hỏi dựa trên context tài liệu được cung cấp.

Quy tắc:
- Trả lời dựa trên thông tin trong context. Nếu context có thông tin liên quan dù chỉ một phần, hãy trả lời phần đó.
- Trả lời ngắn, trực tiếp, đủ ý; ưu tiên 1-3 câu nếu câu hỏi không yêu cầu liệt kê.
- Giữ nguyên tên riêng, model, dataset, metric, công thức và số liệu chính xác như trong context.
- Chỉ nói "Tôi không tìm thấy thông tin này trong tài liệu." khi context hoàn toàn không liên quan đến câu hỏi.
- Không thêm thông tin ngoài context, không suy đoán.

Context:
{context}"""),
    ("human", "{question}"),
])


def create_llm():
    """Create an LLM instance based on config.llm_provider.

    Returns:
        A ChatOllama or ChatGoogleGenerativeAI instance.
    """
    if config.llm_provider == "ollama":
        return ChatOllama(
            model=config.ollama_llm_model,
            base_url=config.ollama_llm_base_url,
            temperature=config.temperature,
        )
    else:
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.google_api_key,
            temperature=config.temperature,
        )


def ingest_document(
    file_path: str,
    collection_name: str = "documents",
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """Full document ingestion pipeline: load → chunk → embed → store.

    Args:
        file_path: Path to the document file.
        collection_name: ChromaDB collection name.
        clear_existing: If True, delete existing collection before ingesting.

    Returns:
        Dictionary with ingestion stats.
    """
    # Step 1: Load document
    documents = load_document(file_path)
    if not documents:
        return {"status": "error", "message": "No content extracted from file."}

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    if not chunks:
        return {"status": "error", "message": "No chunks created from document."}

    chunk_stats = get_chunk_stats(documents, chunks)

    # Step 3: Clear existing collection if requested
    if clear_existing:
        if not reset_vector_store():
            delete_collection(collection_name)

    # Step 4: Create vector store and add chunks
    vector_store = create_vector_store(collection_name)
    num_added = add_documents_to_store(vector_store, chunks)

    return {
        "status": "success",
        "file_name": documents[0].metadata.get("source", "unknown"),
        "num_pages": len(documents),
        "num_chunks": num_added,
        "avg_chunk_size": chunk_stats.get("avg_chunk_size", 0),
    }


def ask_question(
    query: str,
    collection_name: str = "documents",
    vector_store=None,
    llm=None,
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Answer a question using the RAG pipeline: retrieve → augment → generate.

    Args:
        query: The user's question.
        collection_name: ChromaDB collection name.
        vector_store: Optional pre-loaded vector store.
        llm: Optional pre-configured LLM instance.
        include_diagnostics: Whether to include scored retrieval diagnostics.

    Returns:
        Dictionary with answer text and source documents.
    """
    if not query.strip():
        result = {"answer": "Vui lòng nhập câu hỏi.", "sources": []}
        if include_diagnostics:
            result["retrieval_diagnostics"] = []
        return result

    # Get or create vector store
    if vector_store is None:
        vector_store = create_vector_store(collection_name)

    # Check if store has documents
    stats = get_collection_stats(vector_store)
    if stats["total_chunks"] == 0:
        result = {
            "answer": "Chưa có tài liệu nào được tải lên. Vui lòng upload tài liệu trước khi đặt câu hỏi.",
            "sources": [],
        }
        if include_diagnostics:
            result["retrieval_diagnostics"] = []
        return result

    # Step 1: Retrieve relevant chunks
    retrieval_diagnostics = []
    if include_diagnostics:
        scored_docs = retrieve_with_scores(vector_store, query)
        retrieved_docs = [doc for doc, _score in scored_docs]
        retrieval_diagnostics = format_retrieval_diagnostics(scored_docs)
    else:
        retrieved_docs = retrieve_similar(vector_store, query)

    if not retrieved_docs:
        result = {
            "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
            "sources": [],
        }
        if include_diagnostics:
            result["retrieval_diagnostics"] = []
        return result

    # Get or create LLM (only when we have documents to process)
    if llm is None:
        llm = create_llm()

    # Step 2: Format context from retrieved docs
    context = format_retrieved_context(retrieved_docs)

    # Step 3: Generate answer with LLM
    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({"context": context, "question": query})

    # Step 4: Extract sources for citation
    sources = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        source_str = f"{source}" + (f" (trang {page})" if page else "")
        if source_str not in sources:
            sources.append(source_str)

    result = {
        "answer": answer,
        "sources": sources,
        "num_chunks_retrieved": len(retrieved_docs),
    }
    if include_diagnostics:
        result["retrieval_diagnostics"] = retrieval_diagnostics

    return result


def get_session_info(collection_name: str = "documents") -> Dict[str, Any]:
    """Get information about the current RAG session.

    Args:
        collection_name: ChromaDB collection name.

    Returns:
        Dictionary with session info.
    """
    vector_store = create_vector_store(collection_name)
    stats = get_collection_stats(vector_store)

    model = (
        config.ollama_llm_model
        if config.llm_provider == "ollama"
        else config.gemini_model
    )

    embedding_model = (
        config.local_embedding_model
        if config.embedding_provider == "ollama"
        else config.gemini_embedding_model
    )

    return {
        "has_documents": stats["total_chunks"] > 0,
        "total_chunks": stats["total_chunks"],
        "collection_name": stats["collection_name"],
        "model": model,
        "embedding_model": embedding_model,
        "provider": config.llm_provider,
    }
