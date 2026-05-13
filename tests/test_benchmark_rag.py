"""Tests for benchmark evaluation helpers."""

from benchmark_rag import evaluate_answer, extract_key_phrases, summarize_evaluations


class TestEvaluateAnswer:
    """Tests for deterministic benchmark answer evaluation."""

    def test_correct_when_key_terms_match(self):
        result = evaluate_answer(
            "Metric squad_v2 từ thư viện evaluate được dùng để đánh giá mô hình.",
            "Metric squad_v2 từ thư viện evaluate được dùng để đánh giá mô hình.",
        )
        assert result["status"] == "correct"
        assert "squad_v2" in result["matched_terms"]

    def test_partial_when_model_id_is_missing(self):
        result = evaluate_answer(
            "Mô hình E5 được sử dụng để tạo vector đại diện ngữ nghĩa.",
            "Mô hình intfloat/multilingual-e5-base từ Hugging Face được sử dụng để tạo vector đại diện ngữ nghĩa.",
        )
        assert result["status"] == "partial"
        assert "intfloat/multilingual-e5-base" in result["missing_terms"]

    def test_missing_squad_metric_is_not_correct(self):
        result = evaluate_answer(
            "Hàm compute_metrics được sử dụng để đánh giá performance của mô hình QA.",
            "Metric squad_v2 từ thư viện evaluate được dùng để đánh giá mô hình.",
        )
        assert result["status"] in {"partial", "incorrect"}
        assert "squad_v2" in result["missing_terms"]

    def test_not_found_answer(self):
        result = evaluate_answer(
            "Tôi không tìm thấy thông tin này trong tài liệu.",
            "Đáp án đúng có trong ground truth.",
        )
        assert result["status"] == "not_found"
        assert result["score"] == 0.0

    def test_error_answer(self):
        result = evaluate_answer(
            "ERROR: quota exceeded",
            "Đáp án đúng có trong ground truth.",
        )
        assert result["status"] == "error"
        assert result["score"] == 0.0

    def test_bigram_fallback_for_vietnamese_compound_words(self):
        ground_truth = (
            "SFT còn dạy mô hình \"trả lời như thế nào\", "
            "bao gồm việc trả lời đúng vai trò, đúng cấu trúc "
            "và kiểu nội dung bài toán yêu cầu."
        )
        answer = (
            "SFT giúp mô hình học được trả lời như thế nào, "
            "bao gồm vai trò, cấu trúc và kiểu nội dung bài toán yêu cầu."
        )
        result = evaluate_answer(answer, ground_truth)
        assert result["status"] in {"correct", "partial"}
        assert result["score"] > 0.0

    def test_bigram_extraction(self):
        phrases = extract_key_phrases("trả lời đúng vai trò và cấu trúc")
        assert len(phrases) > 0
        assert any("vai" in p for p in phrases)


class TestSummarizeEvaluations:
    """Tests for benchmark evaluation summary aggregation."""

    def test_summary_counts_and_scores(self):
        results = [
            {"evaluation": {"status": "correct"}},
            {"evaluation": {"status": "partial"}},
            {"evaluation": {"status": "incorrect"}},
            {"evaluation": {"status": "not_found"}},
            {"evaluation": {"status": "error"}},
        ]

        summary = summarize_evaluations(results)

        assert summary["status_counts"] == {
            "correct": 1,
            "partial": 1,
            "incorrect": 1,
            "not_found": 1,
            "error": 1,
        }
        assert summary["accuracy"] == 0.2
        assert summary["weighted_accuracy"] == 0.3
