"""Evaluate RAG pipeline quality using Ragas metrics.

Metrics:
- Context Precision: Are the retrieved chunks relevant to the question?
- Context Recall: Do the retrieved chunks cover the ground truth?
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question asked?

Usage:
    python evaluate_ragas.py
"""

import csv
import json
import sys
import time
from datetime import datetime
from typing import Any

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections.context_precision import ContextPrecisionWithReference
from ragas.metrics.collections.context_recall import ContextRecall
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy

from src.rag_chain import create_llm, create_vector_store, get_session_info, ingest_document
from src.retriever import retrieve_similar, format_retrieved_context
from src.embedder import create_embeddings
from src.config import config

PDF_DIR = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\extracted_data\Data\pdfs"
CSV_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\extracted_data\Data\[Data]-Benchmark-Rag.csv"
OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\notebooklm-rag\ragas_evaluation_results.json"

COLLECTION_NAME = "ragas_eval"


def load_questions() -> list[list[str]]:
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader if len(row) >= 3]


def build_evaluation_samples(
    questions: list[list[str]],
    vector_store,
    llm,
    max_questions: int = 20,
) -> list[SingleTurnSample]:
    """Run RAG pipeline and collect samples for Ragas evaluation."""
    from langchain_core.output_parsers import StrOutputParser
    from src.rag_chain import RAG_PROMPT

    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    samples = []
    for row in questions[:max_questions]:
        num, question, ground_truth = row[0], row[1], row[2]
        print(f"  Q{num}: {question[:60]}...")

        retrieved_docs = retrieve_similar(vector_store, question)
        if not retrieved_docs:
            print(f"    -> No docs retrieved, skipping")
            continue

        context = format_retrieved_context(retrieved_docs)
        contexts = [doc.page_content for doc in retrieved_docs]

        for attempt in range(3):
            try:
                answer = chain.invoke({"context": context, "question": question})
                break
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait = 20 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Error: {e}")
                    answer = None
                    break
        else:
            print(f"    Failed after 3 retries, skipping")
            continue

        if answer is None:
            continue

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        samples.append(sample)
        time.sleep(5)

    return samples


def run_ragas_evaluation(samples: list[SingleTurnSample]) -> dict[str, Any]:
    """Run Ragas evaluation on collected samples."""
    llm = create_llm()
    embeddings = create_embeddings()

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        ContextPrecisionWithReference(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    print("\nRunning Ragas evaluation (this may take a few minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    # Step 1: Ensure documents are ingested
    info = get_session_info(COLLECTION_NAME)
    if info["total_chunks"] == 0:
        print("\nIngesting PDFs for evaluation...")
        import os
        pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
        for i, pdf_file in enumerate(pdf_files):
            path = os.path.join(PDF_DIR, pdf_file)
            print(f"  Ingesting: {pdf_file}")
            ingest_document(path, collection_name=COLLECTION_NAME, clear_existing=(i == 0))

    info = get_session_info(COLLECTION_NAME)
    print(f"\nVector store: {info['total_chunks']} chunks")
    print(f"LLM: {info['model']} ({info['provider']})")
    print(f"Embeddings: {info['embedding_model']}")

    # Step 2: Build evaluation samples
    print("\n" + "-" * 60)
    print("Building evaluation samples...")
    questions = load_questions()
    vector_store = create_vector_store(COLLECTION_NAME)
    llm = create_llm()

    samples = build_evaluation_samples(questions, vector_store, llm, max_questions=10)
    print(f"\nCollected {len(samples)} samples for evaluation")

    if not samples:
        print("ERROR: No samples collected. Ensure documents are ingested.")
        sys.exit(1)

    # Step 3: Run Ragas evaluation
    print("\n" + "-" * 60)
    results = run_ragas_evaluation(samples)

    # Step 4: Output results
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    scores = results.scores.mean()
    print(f"\n{'Metric':<25} {'Score':<10}")
    print("-" * 35)
    for metric_name, score in scores.items():
        print(f"{metric_name:<25} {score:.4f}")

    # Save full results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_provider": config.llm_provider,
            "llm_model": info["model"],
            "embedding_provider": config.embedding_provider,
            "embedding_model": info["embedding_model"],
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "k_retrieval": config.k_retrieval,
            "relevance_threshold": config.relevance_threshold,
        },
        "num_samples": len(samples),
        "aggregate_scores": {k: round(v, 4) for k, v in scores.items()},
        "per_sample_scores": results.scores.to_list(),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
