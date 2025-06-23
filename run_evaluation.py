#!/usr/bin/env python3
"""
Runs the comprehensive evaluation of the RAG pipeline using a class-based runner.
"""
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import sec_insights.evaluation.reporting as evaluation_reporting
from sec_insights.evaluation.evaluator import ComprehensiveEvaluator
from sec_insights.rag.config import RESULTS_DIR
from sec_insights.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run.

    Phase parameters (phase_1_k, phase_2_k) only apply to reranked systems:
    - reranked_rag: Uses BGE reranker
    - ensemble_rerank_rag: Uses query expansion + dual rerankers (BGE + Jina)

    These parameters standardize the evaluation to ensure fair comparison between
    reranked systems by using the same retrieval scope and generation context size.
    """

    num_questions: int = 50
    log_level: str = "WARNING"
    methods: Optional[List[str]] = field(default_factory=list)
    k_values: Optional[List[int]] = field(default_factory=list)
    # Standardized phase parameters for reranked systems only
    phase_1_k: int = 30
    phase_2_k: int = 10
    use_rrf: bool = False
    reprocess_path: Optional[str] = None
    resume: bool = True
    run_id: Optional[str] = None
    target_tokens: Optional[int] = None
    overlap_tokens: Optional[int] = None
    hard_ceiling: Optional[int] = None
    use_docker: bool = False
    docker_port: int = 6333

    def get_rag_params(self) -> dict:
        """Get standardized parameters for reranked RAG methods.

        Returns:
            dict: Dictionary containing phase_1_k, phase_2_k, and use_rrf parameters
                 for consistent evaluation of reranked systems.
        """
        return {
            "phase_1_k": self.phase_1_k,
            "phase_2_k": self.phase_2_k,
            "use_rrf": self.use_rrf,
        }


class EvaluationRunner:
    """Orchestrates the RAG evaluation process based on a configuration."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.pipeline: Optional[RAGPipeline] = None
        self.evaluator: Optional[ComprehensiveEvaluator] = None
        self.temp_dir_to_delete: Optional[Path] = None

    def _setup(self):
        """Initializes logging, pipeline, and evaluator."""
        logging.basicConfig(
            level=self.config.log_level.upper(),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logging.info("Initializing RAG pipeline for evaluation...")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        project_root = Path(__file__).parent

        pipeline_kwargs = {
            "root_dir": project_root,
            "target_tokens": self.config.target_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "hard_ceiling": self.config.hard_ceiling,
            "use_docker": self.config.use_docker,
            "docker_port": self.config.docker_port,
        }
        # Filter out None values to use defaults in RAGPipeline
        pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v is not None}

        self.pipeline = None
        try:
            if not self.config.reprocess_path:
                self.pipeline = RAGPipeline(**pipeline_kwargs)
        except Exception as e:
            logger.warning(f"RAGPipeline initialization failed: {e}")
            if self.config.use_docker:
                logger.warning("Attempting fallback to in-memory mode.")
                pipeline_kwargs["use_docker"] = False
                self.pipeline = RAGPipeline(**pipeline_kwargs)
            else:
                raise

        self.evaluator = ComprehensiveEvaluator(
            root_dir=project_root, pipeline=self.pipeline
        )

    def _reprocess(self) -> dict:
        """Reprocesses results from a file."""
        logging.info("Reprocessing raw results from: %s", self.config.reprocess_path)

        # Ensure evaluator is initialized for reprocessing
        if self.evaluator is None:
            project_root = Path(__file__).parent
            self.evaluator = ComprehensiveEvaluator(
                root_dir=project_root, pipeline=None
            )

        if self.config.reprocess_path is None:
            raise ValueError("reprocess_path cannot be None for reprocessing.")

        with open(self.config.reprocess_path, "r") as f:
            data = json.load(f)

        if self.evaluator is None:
            raise RuntimeError("Evaluator not initialized during reprocessing.")

        if isinstance(data, dict) and "individual" in data:
            logging.info("Aggregating from 'individual' results in the file.")
            return self.evaluator.aggregate_results(data["individual"])
        elif isinstance(data, list):
            logging.info("Aggregating from raw individual results list.")
            return self.evaluator.aggregate_results(data)
        elif isinstance(data, dict) and "completed_results" in data:
            logging.info("Aggregating from checkpoint file.")
            return self.evaluator.aggregate_results(data["completed_results"])
        else:
            raise ValueError(f"Unexpected data format in {self.config.reprocess_path}")

    def _run_new_evaluation(self) -> dict:
        """Runs a new, full evaluation."""
        logging.info(
            "Starting evaluation with %d questions...", self.config.num_questions
        )

        assert (
            self.evaluator is not None
        ), "Evaluator must be initialized before running evaluation"
        results, self.temp_dir_to_delete = self.evaluator.evaluate_all_scenarios(
            num_questions=self.config.num_questions,
            methods=self.config.methods,
            k_values=self.config.k_values,
            phase_1_k=self.config.phase_1_k,
            phase_2_k=self.config.phase_2_k,
            use_rrf=self.config.use_rrf,
            resume=self.config.resume,
            run_id=self.config.run_id,
        )
        return results

    def _save_and_cleanup(self, results: dict, reporter: Any):
        """Saves results to files and cleans up temporary directories."""
        reporter.print_results(results)
        reporter.save_results(results)

        if self.temp_dir_to_delete:
            if not self.temp_dir_to_delete.exists():
                self.temp_dir_to_delete.mkdir(parents=True, exist_ok=True)
            logging.info("Cleaning up temporary directory: %s", self.temp_dir_to_delete)
            shutil.rmtree(self.temp_dir_to_delete)

        logging.info("Evaluation complete.")

    def run(self):
        """Executes the entire evaluation process."""
        self._setup()
        reporter = evaluation_reporting.EvaluationReporter(self.config.run_id)

        if self.config.reprocess_path:
            results = self._reprocess()
        else:
            assert self.evaluator is not None
            results, self.temp_dir_to_delete = self.evaluator.evaluate_all_scenarios(
                num_questions=self.config.num_questions,
                methods=self.config.methods,
                k_values=self.config.k_values,
                phase_1_k=self.config.phase_1_k,
                phase_2_k=self.config.phase_2_k,
                use_rrf=self.config.use_rrf,
                resume=self.config.resume,
                run_id=self.config.run_id,
            )

        # Return dataframe in CI environment for testing purposes
        if os.getenv("CI"):
            return reporter.results_to_dataframe(results)

        self._save_and_cleanup(results, reporter)
        return None


if __name__ == "__main__":
    # Configuration for rerankers-only evaluation with RRF implementation
    from datetime import datetime

    config = EvaluationConfig(
        num_questions=300,
        log_level="WARNING",
        methods=[
            "reranked_rag",
            "ensemble_rerank_rag",
        ],
        k_values=[1, 3, 5, 7, 10],
        target_tokens=150,
        overlap_tokens=50,
        hard_ceiling=500,
        phase_1_k=30,
        phase_2_k=10,
        use_rrf=False,
        use_docker=True,
        docker_port=6333,
        resume=True,
        run_id=f"rerankers_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    # Initialize and run the evaluation.
    runner = EvaluationRunner(config)
    runner.run()
