"""
Integration tests for the run_evaluation.py script.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_evaluation import EvaluationConfig, EvaluationRunner
from sec_insights.rag.chunkers import Chunk

# A dummy chunk object for use in tests
DUMMY_CHUNK = Chunk(
    id="chunk1",
    text="This is a test chunk.",
    metadata={"ticker": "TEST", "fiscal_year": 2023, "section": "1A"},
)

# A dummy QA pair for the dataset
DUMMY_QA_PAIR = {
    "chunk_id": "chunk1",
    "question": "What is in the test chunk?",
    "answer": "The test chunk contains a test.",
    "ticker": "TEST",
    "year": 2023,
}


@pytest.fixture
def mock_dependencies():
    """Mocks all external dependencies for the evaluation run."""
    with patch("run_evaluation.RAGPipeline") as mock_rag_pipeline, patch(
        "run_evaluation.ComprehensiveEvaluator"
    ) as mock_evaluator, patch(
        "sec_insights.evaluation.evaluator.QADatasetManager"
    ) as mock_qa_manager, patch(
        "sec_insights.evaluation.reporting.EvaluationReporter"
    ) as mock_reporter:

        # Mock RAGPipeline instance and its methods
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.get_chunks.return_value = [DUMMY_CHUNK]
        mock_rag_pipeline.return_value = mock_pipeline_instance

        # Mock ComprehensiveEvaluator instance
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        # Simulate evaluate_all_scenarios returning some results and a temp path
        mock_evaluator_instance.evaluate_all_scenarios.return_value = (
            {"summary": {}, "individual": []},
            Path("/tmp/dummy_run"),
        )

        # Mock QADatasetManager instance
        mock_qa_manager_instance = MagicMock()
        mock_qa_manager_instance.load_or_generate.return_value = [DUMMY_QA_PAIR]
        mock_qa_manager.return_value = mock_qa_manager_instance

        # Mock EvaluationReporter instance and its methods
        mock_reporter_instance = MagicMock()
        mock_reporter_instance.aggregate_results.return_value = {
            "summary": {},
            "individual": [],
        }
        mock_reporter.return_value = mock_reporter_instance

        yield {
            "RAGPipeline": mock_rag_pipeline,
            "ComprehensiveEvaluator": mock_evaluator,
            "QADatasetManager": mock_qa_manager,
            "EvaluationReporter": mock_reporter,
        }


def test_basic_in_memory_run(mock_dependencies, tmp_path):
    """
    Tests a basic evaluation run with use_docker=False.
    """
    # Point RESULTS_DIR to a temporary directory for this test
    with patch("sec_insights.evaluation.reporting.RESULTS_DIR", tmp_path):
        config = EvaluationConfig(
            num_questions=1,
            use_docker=False,
            quiet=True,  # Run in quiet mode to avoid printing to console during tests
            resume=False,  # Start a fresh run
        )

        runner = EvaluationRunner(config)
        runner.run()

        # Assert that the core components were initialized
        mock_dependencies["RAGPipeline"].assert_called_once()
        mock_dependencies["ComprehensiveEvaluator"].assert_called_once()

        # Assert that the evaluator was called to run the scenarios
        evaluator_instance = mock_dependencies["ComprehensiveEvaluator"].return_value
        evaluator_instance.evaluate_all_scenarios.assert_called_once()

        # Assert that the reporter was used
        reporter_instance = mock_dependencies["EvaluationReporter"].return_value
        reporter_instance.results_to_dataframe.assert_called_once()


def test_docker_fallback_run(mock_dependencies, tmp_path):
    """
    Tests that a Docker connection failure gracefully falls back to in-memory.
    """
    with patch("sec_insights.evaluation.reporting.RESULTS_DIR", tmp_path):
        # Simulate a connection failure on the first call, but succeed on the second
        mock_pipeline_instance = MagicMock()
        mock_dependencies["RAGPipeline"].side_effect = [
            Exception("Docker connection failed"),
            mock_pipeline_instance,  # Success on the second call
        ]

        config = EvaluationConfig(
            num_questions=1,
            use_docker=True,  # Attempt to use Docker
            quiet=True,
            resume=False,
        )

        runner = EvaluationRunner(config)
        runner.run()

        # We expect RAGPipeline to be called twice (failed docker, successful in-memory)
        assert mock_dependencies["RAGPipeline"].call_count == 2
        mock_dependencies["ComprehensiveEvaluator"].assert_called_once()
        mock_dependencies[
            "ComprehensiveEvaluator"
        ].return_value.evaluate_all_scenarios.assert_called_once()


def test_qa_generation_is_triggered(mock_dependencies, tmp_path):
    """
    Tests that the QA generation process is triggered when not enough questions exist.
    """
    with patch("sec_insights.evaluation.reporting.RESULTS_DIR", tmp_path):
        # The evaluator will call the QA manager, so we check the mock on that class
        evaluator_instance = mock_dependencies["ComprehensiveEvaluator"].return_value

        config = EvaluationConfig(
            num_questions=10,  # Request more questions than available
            quiet=True,
            resume=False,
        )

        runner = EvaluationRunner(config)
        runner.run()

        # Check that the evaluator was called, which in turn would call the qa_manager
        evaluator_instance.evaluate_all_scenarios.assert_called_once()
        # The arguments to evaluate_all_scenarios now contain num_questions
        _, call_kwargs = evaluator_instance.evaluate_all_scenarios.call_args
        assert call_kwargs["num_questions"] == 10
