"""
Final validation tests to confirm the complete pipeline works with all configurations.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.evaluator import run_evaluation


class TestFinalValidation:
    """Final validation tests for the complete pipeline."""

    @pytest.fixture
    def comprehensive_qa_dataset(self):
        """Comprehensive QA dataset for final testing."""
        return [
            {
                "chunk_id": "final_test_1",
                "human_readable_id": "AAPL_2021_1A_0",
                "question": "What are Apple's main revenue sources?",
                "answer": "Apple's main revenue sources include iPhone sales, services, and other products.",
                "ticker": "AAPL",
                "year": 2021,
                "section": "1A",
            },
            {
                "chunk_id": "final_test_2",
                "human_readable_id": "TSLA_2022_1A_1",
                "question": "What are Tesla's major risks?",
                "answer": "Tesla faces risks from competition, regulation, and supply chain issues.",
                "ticker": "TSLA",
                "year": 2022,
                "section": "1A",
            },
            {
                "chunk_id": "final_test_3",
                "human_readable_id": "NVDA_2023_1A_2",
                "question": "How does NVIDIA generate revenue?",
                "answer": "NVIDIA generates revenue from data center, gaming, and automotive segments.",
                "ticker": "NVDA",
                "year": 2023,
                "section": "1A",
            },
        ]

    def test_all_basic_methods_work(self, comprehensive_qa_dataset):
        """Test that all basic methods work together in one evaluation."""
        with patch("src.vector_store.document_store.DocumentStore"):
            # Test multiple methods together
            run_evaluation(
                qa_dataset=comprehensive_qa_dataset,
                num_questions=3,
                methods=["baseline", "web_search", "unfiltered_text"],
                k_values=[1, 3, 5],
                run_id="final_validation_all_methods",
                use_docker=False,
                docker_port=6333,
            )

            # Check results were created
            results_dir = Path("data/results/final_validation_all_methods")
            assert results_dir.exists()

            individual_file = results_dir / "individual_results.json"
            assert individual_file.exists()

            summary_file = results_dir / "summary.json"
            assert summary_file.exists()

            csv_file = results_dir / "summary.csv"
            assert csv_file.exists()

    def test_different_k_values(self, comprehensive_qa_dataset):
        """Test that different k values work correctly."""
        with patch("src.vector_store.document_store.DocumentStore"):
            # Test with different k values
            run_evaluation(
                qa_dataset=comprehensive_qa_dataset,
                num_questions=2,
                methods=["baseline"],
                k_values=[1, 5, 10, 20],
                run_id="final_validation_k_values",
                use_docker=False,
                docker_port=6333,
            )

            # Check results were created
            results_dir = Path("data/results/final_validation_k_values")
            assert results_dir.exists()

    def test_docker_mode_fallback(self, comprehensive_qa_dataset):
        """Test that Docker mode gracefully falls back to in-memory."""
        with patch("src.vector_store.document_store.DocumentStore"):
            # This should work even if Docker isn't available
            run_evaluation(
                qa_dataset=comprehensive_qa_dataset,
                num_questions=1,
                methods=["baseline"],
                k_values=[1, 3, 5],
                run_id="final_validation_docker",
                use_docker=True,  # Try Docker first
                docker_port=6333,
            )

            # Check results were created (should fallback to in-memory if Docker fails)
            results_dir = Path("data/results/final_validation_docker")
            assert results_dir.exists()

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with minimal dataset
        minimal_qa = [
            {
                "chunk_id": "edge_case_1",
                "question": "Simple question?",
                "answer": "Simple answer",
                "ticker": "TEST",
                "year": 2024,
                "section": "1A",
            }
        ]

        with patch("src.vector_store.document_store.DocumentStore"):
            # Test with zero questions requested
            run_evaluation(
                qa_dataset=minimal_qa,
                num_questions=0,
                methods=["baseline"],
                k_values=[1],
                run_id="final_validation_zero_questions",
                use_docker=False,
                docker_port=6333,
            )

            # Should still create directory structure
            results_dir = Path("data/results/final_validation_zero_questions")
            assert results_dir.exists()

    def test_mixed_qa_formats(self):
        """Test with mixed QA dataset formats."""
        mixed_qa = [
            # Format 1: Full fields
            {
                "chunk_id": "mixed_1",
                "human_readable_id": "MIXED_2024_1A_0",
                "question": "Mixed format question 1?",
                "answer": "Mixed answer 1",
                "ticker": "MIXED",
                "year": 2024,
                "section": "1A",
            },
            # Format 2: Missing human_readable_id
            {
                "chunk_id": "mixed_2",
                "question": "Mixed format question 2?",
                "answer": "Mixed answer 2",
                "ticker": "MIXED2",
                "year": 2024,
                "section": "1B",
            },
            # Format 3: With id field
            {
                "id": "mixed_3",
                "chunk_id": "mixed_3",
                "question": "Mixed format question 3?",
                "answer": "Mixed answer 3",
                "ticker": "MIXED3",
                "year": 2024,
                "section": "1C",
            },
        ]

        with patch("src.vector_store.document_store.DocumentStore"):
            run_evaluation(
                qa_dataset=mixed_qa,
                num_questions=3,
                methods=["baseline"],
                k_values=[1, 3],
                run_id="final_validation_mixed_formats",
                use_docker=False,
                docker_port=6333,
            )

            # Should handle all formats without errors
            results_dir = Path("data/results/final_validation_mixed_formats")
            assert results_dir.exists()

    def test_comprehensive_integration(self, comprehensive_qa_dataset):
        """Comprehensive test that exercises most of the pipeline functionality."""
        with patch("src.vector_store.document_store.DocumentStore"):
            # Test comprehensive configuration
            run_evaluation(
                qa_dataset=comprehensive_qa_dataset,
                num_questions=3,
                methods=["baseline", "web_search"],  # Multiple methods
                k_values=[1, 3, 5, 10],  # Multiple k values
                run_id="final_comprehensive_test",
                use_docker=False,
                docker_port=6333,
            )

            # Verify comprehensive results
            results_dir = Path("data/results/final_comprehensive_test")
            assert results_dir.exists()

            # Check all expected files exist
            individual_file = results_dir / "individual_results.json"
            summary_file = results_dir / "summary.json"
            csv_file = results_dir / "summary.csv"

            assert individual_file.exists()
            assert summary_file.exists()
            assert csv_file.exists()

            # Verify file sizes (should contain actual data)
            assert individual_file.stat().st_size > 100  # Has real content
            assert summary_file.stat().st_size > 50  # Has real content
            assert csv_file.stat().st_size > 50  # Has real content
