"""Benchmark RAG pipeline with CSV questions and ground truth."""

import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Any

from src.rag_chain import ask_question, get_session_info, ingest_document

PDF_DIR = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\extracted_data\Data\pdfs"
CSV_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\extracted_data\Data\[Data]-Benchmark-Rag.csv"
OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\test_prj\notebooklm-rag\benchmark_results.json"

EVALUATION_STATUSES = ("correct", "partial", "incorrect", "not_found", "error")
STOPWORDS = {
    "các", "của", "cho", "trong", "được", "dùng", "dụng", "với", "một",
    "này", "những", "thông", "tin", "hình", "mô", "để", "khi", "là",
    "và", "hoặc", "the", "and", "for", "from", "that", "this", "used",
    "model", "metric", "dataset", "mô", "hình",
}
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+(?:[/_\-.][\wÀ-ỹ]+)*", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text for deterministic benchmark matching."""
    text = text.lower()
    text = re.sub(r"[^\wÀ-ỹ/_\-.]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _is_specific_term(term: str) -> bool:
    return any(char in term for char in "/_-") or any(char.isdigit() for char in term)


def extract_key_terms(ground_truth: str) -> list[str]:
    """Extract deterministic key terms from a ground-truth answer."""
    terms: list[str] = []
    seen: set[str] = set()

    for raw_term in TOKEN_RE.findall(ground_truth):
        term = normalize_text(raw_term)
        if not term or term in STOPWORDS or len(term) < 3:
            continue
        if not _is_specific_term(term) and len(term) < 5:
            continue
        if term not in seen:
            terms.append(term)
            seen.add(term)

    return terms[:16]


def extract_key_phrases(ground_truth: str) -> list[str]:
    """Extract meaningful bigram phrases from ground truth for Vietnamese."""
    normalized = normalize_text(ground_truth)
    words = normalized.split()
    phrases: list[str] = []
    seen: set[str] = set()

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 in STOPWORDS and w2 in STOPWORDS:
            continue
        bigram = f"{w1} {w2}"
        if len(bigram) >= 6 and bigram not in seen:
            phrases.append(bigram)
            seen.add(bigram)

    return phrases[:16]


def evaluate_answer(answer: str, ground_truth: str) -> dict[str, Any]:
    """Evaluate an answer against ground truth with deterministic heuristics."""
    normalized_answer = normalize_text(answer)
    normalized_ground_truth = normalize_text(ground_truth)

    if normalized_answer.startswith("error"):
        return {
            "status": "error",
            "score": 0.0,
            "matched_terms": [],
            "missing_terms": [],
            "reason": "Answer is an error response.",
        }

    not_found_markers = ("không tìm thấy", "khong tim thay", "không có thông tin", "not found")
    has_not_found = any(marker in answer.lower() for marker in not_found_markers)

    if has_not_found:
        key_terms_check = extract_key_terms(ground_truth)
        if key_terms_check:
            matched_check = [t for t in key_terms_check if t in normalized_answer]
            if len(matched_check) >= 2:
                has_not_found = False
        if has_not_found:
            phrases_check = extract_key_phrases(ground_truth)
            if phrases_check:
                matched_phrases_check = [p for p in phrases_check if p in normalized_answer]
                if len(matched_phrases_check) >= 2:
                    has_not_found = False

    if has_not_found:
        return {
            "status": "not_found",
            "score": 0.0,
            "matched_terms": [],
            "missing_terms": [],
            "reason": "Answer reports that the information was not found.",
        }

    key_terms = extract_key_terms(ground_truth)
    if not key_terms:
        phrases = extract_key_phrases(ground_truth)
        if phrases:
            matched_phrases = [p for p in phrases if p in normalized_answer]
            missing_phrases = [p for p in phrases if p not in normalized_answer]
            score = round(len(matched_phrases) / len(phrases), 3)
            if score >= 0.5:
                status = "correct"
                reason = "Phrase matching: answer covers key bigram phrases."
            elif score >= 0.25:
                status = "partial"
                reason = "Phrase matching: partial overlap with ground truth."
            else:
                status = "incorrect"
                reason = "Phrase matching: low overlap with ground truth."
            return {
                "status": status,
                "score": score,
                "matched_terms": matched_phrases,
                "missing_terms": missing_phrases,
                "reason": reason,
            }
        score = 1.0 if normalized_ground_truth in normalized_answer else 0.0
        status = "correct" if score == 1.0 else "incorrect"
        return {
            "status": status,
            "score": score,
            "matched_terms": [],
            "missing_terms": [],
            "reason": "No key terms or phrases extracted; used full containment.",
        }

    matched_terms = [term for term in key_terms if term in normalized_answer]
    missing_terms = [term for term in key_terms if term not in normalized_answer]
    important_terms = [term for term in key_terms if _is_specific_term(term)]
    missing_important = [term for term in important_terms if term not in matched_terms]
    score = round(len(matched_terms) / len(key_terms), 3)

    if missing_important:
        status = "partial" if matched_terms else "incorrect"
        reason = "Missing important ground-truth term(s)."
    elif score >= 0.65:
        status = "correct"
        reason = "Answer covers the key ground-truth terms."
    elif score >= 0.3:
        status = "partial"
        reason = "Answer overlaps with the ground truth but misses key terms."
    else:
        status = "incorrect"
        reason = "Answer has low overlap with the ground truth."

    return {
        "status": status,
        "score": score,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "reason": reason,
    }


def summarize_evaluations(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize benchmark evaluation labels."""
    total = len(results)
    counts = {status: 0 for status in EVALUATION_STATUSES}

    for result in results:
        status = result.get("evaluation", {}).get("status", "incorrect")
        counts[status if status in counts else "incorrect"] += 1

    if total == 0:
        return {"status_counts": counts, "accuracy": 0.0, "weighted_accuracy": 0.0}

    accuracy = round(counts["correct"] / total, 3)
    weighted_accuracy = round((counts["correct"] + 0.5 * counts["partial"]) / total, 3)
    return {
        "status_counts": counts,
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
    }


def ingest_all_pdfs(collection_name: str = "benchmark") -> dict[str, Any]:
    """Ingest all benchmark PDFs into a Chroma collection."""
    print("=" * 60)
    print("INGESTING ALL PDFs")
    print("=" * 60)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    total_pages = 0

    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        print(f"\nIngesting: {pdf_file}")
        result = ingest_document(
            path,
            collection_name=collection_name,
            clear_existing=(pdf_file == pdf_files[0]),
        )
        if result["status"] == "success":
            print(f"  Pages: {result['num_pages']}, Chunks: {result['num_chunks']}")
            total_pages += result["num_pages"]
        else:
            print(f"  ERROR: {result.get('message', 'unknown')}")

    info = get_session_info(collection_name)
    print(f"\nTotal: {total_pages} pages, {info['total_chunks']} chunks in vector store")
    print(f"Model: {info['model']}, Embedding: {info['embedding_model']}")
    return info


def load_questions() -> list[list[str]]:
    """Load benchmark questions from CSV."""
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader if len(row) >= 3]


def run_questions(collection_name: str = "benchmark") -> list[dict[str, Any]]:
    """Run benchmark questions against the RAG pipeline."""
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK - 50 QUESTIONS")
    print("=" * 60)

    results = []
    for row in load_questions():
        num, question, ground_truth = row[0], row[1], row[2]
        print(f"\nQ{num}: {question[:80]}...")
        try:
            answer = ask_question(
                question,
                collection_name=collection_name,
                include_diagnostics=True,
            )
            ans_text = answer["answer"]
            sources = answer.get("sources", [])
            retrieved = answer.get("num_chunks_retrieved", 0)
            retrieval_diagnostics = answer.get("retrieval_diagnostics", [])
            evaluation = evaluate_answer(ans_text, ground_truth)

            results.append({
                "num": num,
                "question": question,
                "ground_truth": ground_truth,
                "answer": ans_text,
                "sources": sources,
                "chunks_retrieved": retrieved,
                "found_answer": evaluation["status"] in {"correct", "partial", "incorrect"},
                "evaluation": evaluation,
                "retrieval_diagnostics": retrieval_diagnostics,
            })

            print(f"  Chunks: {retrieved}, Sources: {len(sources)}")
            print(f"  Eval: {evaluation['status']}, Score: {evaluation['score']}")
            if evaluation["missing_terms"]:
                print(f"  Missing: {', '.join(evaluation['missing_terms'][:5])}")
            print(f"  Answer: {ans_text[:150]}...")
        except Exception as e:
            ans_text = f"ERROR: {str(e)}"
            evaluation = evaluate_answer(ans_text, ground_truth)
            print(f"  ERROR: {e}")
            results.append({
                "num": num,
                "question": question,
                "ground_truth": ground_truth,
                "answer": ans_text,
                "sources": [],
                "chunks_retrieved": 0,
                "found_answer": False,
                "evaluation": evaluation,
                "retrieval_diagnostics": [],
            })

    return results


def save_results(info: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Save benchmark results to JSON and return the output payload."""
    total = len(results)
    summary = summarize_evaluations(results)
    found = sum(1 for result in results if result["found_answer"])
    not_found = total - found

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": info["model"],
        "embedding_model": info["embedding_model"],
        "total_chunks": info["total_chunks"],
        "total_questions": total,
        "found_count": found,
        "not_found_count": not_found,
        **summary,
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def print_summary(output: dict[str, Any]) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total questions: {output['total_questions']}")
    for status, count in output["status_counts"].items():
        print(f"{status}: {count}")
    print(f"Accuracy: {output['accuracy'] * 100:.1f}%")
    print(f"Weighted accuracy: {output['weighted_accuracy'] * 100:.1f}%")
    print(f"\nResults saved to: {OUTPUT_PATH}")


def main() -> None:
    """Run the full benchmark."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    info = ingest_all_pdfs("benchmark")
    results = run_questions("benchmark")
    output = save_results(info, results)
    print_summary(output)


if __name__ == "__main__":
    main()
