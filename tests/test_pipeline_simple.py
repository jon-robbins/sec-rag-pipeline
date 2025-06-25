"""
Simple tests that validate the pipeline is working correctly.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evaluator import run_evaluation


class TestPipelineSimple:
    """Simple tests for pipeline functionality."""

    @pytest.fixture
    def sample_qa_dataset(self):
        """Sample QA dataset for testing."""
        return [
            {
                "chunk_id": "test_chunk_1",
                "human_readable_id": "AAPL_2021_1A_0",
                "question": "What is the revenue?",
                "answer": "Revenue was $100B",
                "ticker": "AAPL",
                "year": 2021,
                "section": "1A",
            }
        ]

    def test_baseline_method_works(self, sample_qa_dataset):
        """Test that baseline method executes without errors."""
        # Mock OpenAI to avoid API calls
        with patch("src.methods.baseline.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Test baseline answer"
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_client.chat.completions.create.return_value = mock_response

            # Mock DocumentStore to avoid needing data file
            with patch("src.vector_store.document_store.DocumentStore"):
                # This should not raise any exceptions
                run_evaluation(
                    qa_dataset=sample_qa_dataset,
                    num_questions=1,
                    methods=["baseline"],
                    k_values=[1, 3, 5],
                    run_id="test_simple_baseline",
                    use_docker=False,
                    docker_port=6333,
                )

                # If we get here, the pipeline executed successfully
                assert True

    def test_web_search_method_works(self, sample_qa_dataset):
        """Test that web search method executes without errors."""
        with patch("src.methods.web_search.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Test web search answer"
            mock_response.usage.prompt_tokens = 200
            mock_response.usage.completion_tokens = 75
            mock_response.usage.total_tokens = 275
            mock_client.chat.completions.create.return_value = mock_response

            with patch("src.vector_store.document_store.DocumentStore"):
                run_evaluation(
                    qa_dataset=sample_qa_dataset,
                    num_questions=1,
                    methods=["web_search"],
                    k_values=[1, 3, 5],
                    run_id="test_simple_web_search",
                    use_docker=False,
                    docker_port=6333,
                )

                assert True

    def test_multiple_methods_work_together(self, sample_qa_dataset):
        """Test that multiple methods can run together."""
        with patch("src.methods.baseline.OpenAI") as mock_baseline_openai, patch(
            "src.methods.web_search.OpenAI"
        ) as mock_web_openai:

            # Setup baseline mock
            mock_baseline_client = MagicMock()
            mock_baseline_openai.return_value = mock_baseline_client
            baseline_response = MagicMock()
            baseline_response.choices[0].message.content = "Baseline answer"
            baseline_response.usage.prompt_tokens = 100
            baseline_response.usage.completion_tokens = 50
            baseline_response.usage.total_tokens = 150
            mock_baseline_client.chat.completions.create.return_value = (
                baseline_response
            )

            # Setup web search mock
            mock_web_client = MagicMock()
            mock_web_openai.return_value = mock_web_client
            web_response = MagicMock()
            web_response.choices[0].message.content = "Web search answer"
            web_response.usage.prompt_tokens = 200
            web_response.usage.completion_tokens = 75
            web_response.usage.total_tokens = 275
            mock_web_client.chat.completions.create.return_value = web_response

            with patch("src.vector_store.document_store.DocumentStore"):
                run_evaluation(
                    qa_dataset=sample_qa_dataset,
                    num_questions=1,
                    methods=["baseline", "web_search"],
                    k_values=[1, 3, 5],
                    run_id="test_simple_multi_method",
                    use_docker=False,
                    docker_port=6333,
                )

                assert True

    def test_invalid_method_doesnt_crash(self, sample_qa_dataset):
        """Test that invalid methods don't crash the pipeline."""
        with patch("src.vector_store.document_store.DocumentStore"):
            # This should not raise any exceptions
            run_evaluation(
                qa_dataset=sample_qa_dataset,
                num_questions=1,
                methods=["invalid_method_name"],
                k_values=[1, 3, 5],
                run_id="test_simple_invalid",
                use_docker=False,
                docker_port=6333,
            )

            assert True

    def test_qa_dataset_field_handling(self):
        """Test that different QA dataset field formats are handled."""
        qa_variations = [
            # Standard format
            {
                "chunk_id": "test_chunk",
                "human_readable_id": "AAPL_2021_1A_0",
                "question": "Test question?",
                "answer": "Test answer",
                "ticker": "AAPL",
                "year": 2021,
                "section": "1A",
            },
            # Missing human_readable_id
            {
                "chunk_id": "test_chunk_2",
                "question": "Test question 2?",
                "answer": "Test answer 2",
                "ticker": "TSLA",
                "year": 2022,
                "section": "1B",
            },
            # Has 'id' field instead
            {
                "id": "test_chunk_3",
                "chunk_id": "test_chunk_3",
                "question": "Test question 3?",
                "answer": "Test answer 3",
                "ticker": "NVDA",
                "year": 2023,
                "section": "1C",
            },
        ]

        with patch("src.methods.baseline.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Test answer"
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 25
            mock_response.usage.total_tokens = 75
            mock_client.chat.completions.create.return_value = mock_response

            with patch("src.vector_store.document_store.DocumentStore"):
                # Should handle all QA format variations without crashing
                run_evaluation(
                    qa_dataset=qa_variations,
                    num_questions=3,
                    methods=["baseline"],
                    k_values=[1, 3, 5],
                    run_id="test_simple_qa_formats",
                    use_docker=False,
                    docker_port=6333,
                )

                assert True

    def test_results_are_created(self):
        """Test that results files are actually created."""
        sample_data = [
            {
                "chunk_id": "test_results_chunk",
                "human_readable_id": "TEST_2024_1A_0",
                "question": "What is the revenue for this company?",
                "answer": "Test results answer",
                "ticker": "TEST",
                "year": 2024,
                "section": "1A",
            }
        ]

        # Mock DocumentStore to avoid needing data file
        with patch("src.vector_store.document_store.DocumentStore"):
            run_evaluation(
                qa_dataset=sample_data,
                num_questions=1,
                methods=["baseline"],
                k_values=[1, 3, 5],
                run_id="test_results_creation_final",
                use_docker=False,
                docker_port=6333,
            )

            # Check that results directory was created
            results_dir = Path("data/results/test_results_creation_final")
            assert results_dir.exists()

            # Check that individual results file exists
            individual_file = results_dir / "individual_results.json"
            assert individual_file.exists()

            # Check that summary files exist
            summary_file = results_dir / "summary.json"
            assert summary_file.exists()

            csv_file = results_dir / "summary.csv"
            assert csv_file.exists()

            # Verify individual results structure (check actual format)
            with open(individual_file, "r") as f:
                individual_data = json.load(f)

            # Handle both possible result formats
            if "individual" in individual_data:
                # New format with wrapper
                results = individual_data["individual"]
                assert len(results) == 1
                result = results[0]
            else:
                # Old format - individual_data is directly a list
                results = individual_data
                assert len(results) == 1
                result = results[0]

            # Verify required fields exist
            assert "question_id" in result
            assert "question" in result
            assert "baseline" in result

            baseline_result = result["baseline"]
            assert "answer" in baseline_result
            assert "token_usage" in baseline_result
            assert "retrieval_metrics" in baseline_result

            # Verify content is reasonable (using real API)
            assert isinstance(baseline_result["answer"], str)
            assert len(baseline_result["answer"]) > 0
            assert baseline_result["token_usage"]["total_tokens"] > 0
            assert baseline_result["token_usage"]["prompt_tokens"] > 0
            assert baseline_result["token_usage"]["completion_tokens"] > 0
