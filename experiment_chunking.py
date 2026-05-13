"""Chunking Strategy Experiment.

Compares different chunking configurations on the benchmark dataset:
1. Recursive (small): chunk_size=500, overlap=100
2. Recursive (large): chunk_size=1000, overlap=200 (current default)
3. Recursive (xlarge): chunk_size=1500, overlap=300
4. Semantic chunking: breakpoint_threshold_type="percentile"

For each config, we ingest all PDFs, run 20 benchmark questions, and measure:
- Number of chunks created
- Average chunk size (characters)
- Retrieval accuracy (% of questions with correct/partial answers)

Usage:
    python experiment_chunking.py
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.loader import load_document
from src.embedder import create_embeddings
from src.vector_store import (
    create_vector_store,
    add_documents_to_store,
    delete_collection,
    reset_vector_store,
    get_collection_stats,
)
from src.retriever import retrieve_similar, format_retrieved_context
from src.rag_chain import create_llm, RAG_PROMPT

PDF_DIR = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\extracted_data\Data\pdfs"
CSV_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\extracted_data\Data\[Data]-Benchmark-Rag.csv"
OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\notebooklm-rag\chunking_experiment_results.json"

MAX_QUESTIONS = 20

CHUNKING_CONFIGS = [
    {
        "name": "recursive_small",
        "description": "Recursive: 500 chars, 100 overlap",
        "type": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    {
        "name": "recursive_default",
        "description": "Recursive: 1000 chars, 200 overlap (current default)",
        "type": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    {
        "name": "recursive_large",
        "description": "Recursive: 1500 chars, 300 overlap",
        "type": "recursive",
        "chunk_size": 1500,
        "chunk_overlap": 300,
    },
    {
        "name": "semantic",
        "description": "Semantic chunking (percentile breakpoint)",
        "type": "semantic",
        "breakpoint_threshold_type": "percentile",
    },
]


def load_all_documents() -> list[Document]:
    """Load all benchmark PDFs into a flat document list."""
    documents = []
    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        docs = load_document(path)
        documents.extend(docs)
    return documents


def chunk_with_config(documents: list[Document], cfg: dict) -> list[Document]:
    """Chunk documents using the specified configuration."""
    if cfg["type"] == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
    elif cfg["type"] == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker

        embeddings = create_embeddings()
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=cfg.get("breakpoint_threshold_type", "percentile"),
        )
        # Filter out pages with empty/problematic content that cause NaN embeddings
        valid_docs = [d for d in documents if d.page_content.strip() and len(d.page_content.strip()) > 10]
        chunks = splitter.split_documents(valid_docs)
    else:
        raise ValueError(f"Unknown chunking type: {cfg['type']}")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    return chunks


def load_questions() -> list[list[str]]:
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader if len(row) >= 3]


def evaluate_chunking_config(
    cfg: dict,
    documents: list[Document],
    questions: list[list[str]],
    llm,
) -> dict[str, Any]:
    """Run full pipeline for one chunking config and measure accuracy."""
    collection_name = f"experiment_{cfg['name']}"

    print(f"\n{'='*60}")
    print(f"Config: {cfg['description']}")
    print(f"{'='*60}")

    # Step 1: Chunk
    print("  Chunking documents...")
    chunks = chunk_with_config(documents, cfg)
    num_chunks = len(chunks)
    avg_size = sum(len(c.page_content) for c in chunks) // max(num_chunks, 1)
    print(f"  -> {num_chunks} chunks, avg {avg_size} chars")

    # Step 2: Ingest into vector store
    print("  Ingesting into vector store...")
    delete_collection(collection_name)
    vector_store = create_vector_store(collection_name)
    num_added = add_documents_to_store(vector_store, chunks)
    print(f"  -> {num_added} chunks stored")

    # Step 3: Run questions
    print(f"  Running {len(questions)} questions...")
    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    from benchmark_rag import evaluate_answer

    correct = 0
    partial = 0
    incorrect = 0
    not_found = 0
    results_detail = []

    for row in questions:
        num, question, ground_truth = row[0], row[1], row[2]

        retrieved_docs = retrieve_similar(vector_store, question)
        if not retrieved_docs:
            not_found += 1
            results_detail.append({"num": num, "status": "not_found", "score": 0.0})
            continue

        context = format_retrieved_context(retrieved_docs)
        answer = None
        for attempt in range(3):
            try:
                answer = chain.invoke({"context": context, "question": question})
                break
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait = 20 * (attempt + 1)
                    print(f"    Q{num} rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Q{num} error: {e}")
                    break

        if answer is None:
            not_found += 1
            results_detail.append({"num": num, "status": "error", "score": 0.0})
            continue

        time.sleep(5)
        evaluation = evaluate_answer(answer, ground_truth)

        status = evaluation["status"]
        if status == "correct":
            correct += 1
        elif status == "partial":
            partial += 1
        elif status == "not_found":
            not_found += 1
        else:
            incorrect += 1

        results_detail.append({
            "num": num,
            "status": status,
            "score": evaluation["score"],
        })

    total = len(questions)
    accuracy = round(correct / total, 3) if total > 0 else 0.0
    weighted_accuracy = round((correct + 0.5 * partial) / total, 3) if total > 0 else 0.0

    print(f"  Results: correct={correct}, partial={partial}, incorrect={incorrect}, not_found={not_found}")
    print(f"  Accuracy: {accuracy*100:.1f}%, Weighted: {weighted_accuracy*100:.1f}%")

    # Cleanup
    delete_collection(collection_name)

    return {
        "config": cfg,
        "num_chunks": num_chunks,
        "num_stored": num_added,
        "avg_chunk_size": avg_size,
        "total_questions": total,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "not_found": not_found,
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "per_question": results_detail,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("CHUNKING STRATEGY EXPERIMENT")
    print("=" * 60)
    print(f"LLM: {config.llm_provider} / {config.gemini_model if config.llm_provider == 'gemini' else config.ollama_llm_model}")
    print(f"Embeddings: {config.embedding_provider} / {config.local_embedding_model if config.embedding_provider == 'ollama' else config.gemini_embedding_model}")
    print(f"K retrieval: {config.k_retrieval}")
    print(f"Max questions per config: {MAX_QUESTIONS}")

    # Load documents once
    print("\nLoading all PDFs...")
    documents = load_all_documents()
    print(f"Loaded {len(documents)} pages from PDFs")

    # Load questions
    questions = load_questions()[:MAX_QUESTIONS]
    print(f"Using {len(questions)} benchmark questions")

    # Create LLM once (shared across configs)
    llm = create_llm()

    # Run each config
    all_results = []
    for cfg in CHUNKING_CONFIGS:
        try:
            result = evaluate_chunking_config(cfg, documents, questions, llm)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR in config '{cfg['name']}': {e}")
            print("  Skipping this config...")
            all_results.append({
                "config": cfg,
                "num_chunks": 0,
                "num_stored": 0,
                "avg_chunk_size": 0,
                "total_questions": len(questions),
                "correct": 0,
                "partial": 0,
                "incorrect": 0,
                "not_found": len(questions),
                "accuracy": 0.0,
                "weighted_accuracy": 0.0,
                "per_question": [],
                "error": str(e),
            })

    # Summary table
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<30} {'Chunks':<8} {'Avg Size':<10} {'Accuracy':<10} {'Weighted':<10}")
    print("-" * 68)
    for r in all_results:
        print(
            f"{r['config']['description'][:30]:<30} "
            f"{r['num_chunks']:<8} "
            f"{r['avg_chunk_size']:<10} "
            f"{r['accuracy']*100:.1f}%{'':4} "
            f"{r['weighted_accuracy']*100:.1f}%"
        )

    # Find best config
    best = max(all_results, key=lambda x: x["weighted_accuracy"])
    print(f"\nBest config: {best['config']['description']}")
    print(f"  Weighted accuracy: {best['weighted_accuracy']*100:.1f}%")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "llm_provider": config.llm_provider,
            "llm_model": config.gemini_model if config.llm_provider == "gemini" else config.ollama_llm_model,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.local_embedding_model if config.embedding_provider == "ollama" else config.gemini_embedding_model,
            "k_retrieval": config.k_retrieval,
            "relevance_threshold": config.relevance_threshold,
        },
        "num_questions": len(questions),
        "configs_tested": len(all_results),
        "best_config": best["config"]["name"],
        "results": [{k: v for k, v in r.items() if k != "per_question"} for r in all_results],
        "detailed_results": all_results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
