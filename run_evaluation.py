#!/usr/bin/env python3
"""
Runs the comprehensive evaluation of the RAG pipeline using a class-based runner.
"""
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from sec_insights.evaluation.evaluator import ComprehensiveEvaluator
from sec_insights.rag.config import RESULTS_DIR
from sec_insights.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""

    num_questions: int = 50
    log_level: str = "WARNING"
    methods: Optional[List[str]] = field(default_factory=list)
    k_values: Optional[List[int]] = field(default_factory=list)
    quiet: bool = False
    reprocess_path: Optional[str] = None
    resume: bool = True
    run_id: Optional[str] = None
    target_tokens: Optional[int] = None
    overlap_tokens: Optional[int] = None
    hard_ceiling: Optional[int] = None
    use_docker: bool = False
    docker_port: int = 6333


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

        if not self.config.quiet:
            print("Initializing RAG pipeline for evaluation...")

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
            root_dir=project_root, pipeline=self.pipeline, quiet=self.config.quiet
        )

    def _reprocess(self) -> dict:
        """Reprocesses results from a file."""
        if not self.config.quiet:
            print(f"🔄 Reprocessing raw results from: {self.config.reprocess_path}")

        # Ensure evaluator is initialized for reprocessing
        if self.evaluator is None:
            self.evaluator = ComprehensiveEvaluator(
                pipeline=None, quiet=self.config.quiet
            )

        if self.config.reprocess_path is None:
            raise ValueError("reprocess_path cannot be None for reprocessing.")

        with open(self.config.reprocess_path, "r") as f:
            data = json.load(f)

        if self.evaluator is None:
            raise RuntimeError("Evaluator not initialized during reprocessing.")

        if isinstance(data, dict) and "individual" in data:
            if not self.config.quiet:
                print("✅ Aggregating from 'individual' results in the file.")
            return self.evaluator._aggregate_results(data["individual"])
        elif isinstance(data, list):
            if not self.config.quiet:
                print("✅ Aggregating from raw individual results list.")
            return self.evaluator._aggregate_results(data)
        elif isinstance(data, dict) and "completed_results" in data:
            if not self.config.quiet:
                print("✅ Aggregating from checkpoint file.")
            return self.evaluator._aggregate_results(data["completed_results"])
        else:
            raise ValueError(f"Unexpected data format in {self.config.reprocess_path}")

    def _run_new_evaluation(self) -> dict:
        """Runs a new, full evaluation."""
        if not self.config.quiet:
            print(
                f"🚀 Starting evaluation with {self.config.num_questions} questions..."
            )

        assert (
            self.evaluator is not None
        ), "Evaluator must be initialized before running evaluation"
        results, self.temp_dir_to_delete = self.evaluator.evaluate_all_scenarios(
            num_questions=self.config.num_questions,
            methods=self.config.methods,
            k_values=self.config.k_values,
            resume=self.config.resume,
            run_id=self.config.run_id,
        )
        return results

    def _save_and_cleanup(self, results: dict, reporter: Any):
        """Saves results to files and cleans up temporary directories."""
        reporter.print_results(results)
        reporter.save_results(results)

        if self.temp_dir_to_delete:
            print(f"🧹 Cleaning up temporary directory: {self.temp_dir_to_delete}")
            shutil.rmtree(self.temp_dir_to_delete)

        print("✅ Evaluation complete.")

    def _get_reporter_class(self):
        """Dynamically import and return the EvaluationReporter class.

        Importing inside a function ensures that any runtime monkey-patching (e.g. in
        the test-suite) of ``sec_insights.evaluation.reporting.EvaluationReporter``
        is respected, because we fetch the symbol only when ``run`` is executed –
        *after* the patch has been applied – instead of relying on the module-level
        import that happens when this file is first imported.
        """
        from importlib import import_module

        reporting_module = import_module("sec_insights.evaluation.reporting")
        return getattr(reporting_module, "EvaluationReporter")

    def run(self):
        """Executes the entire evaluation process."""
        self._setup()

        # Fetch the (potentially patched) reporter class at runtime instead of
        # relying on the module-level import so that unit tests which monkey-patch
        # ``sec_insights.evaluation.reporting.EvaluationReporter`` see their patch
        # take effect. This is crucial for tests like ``test_basic_in_memory_run``
        # which expect methods on the mocked reporter instance to be invoked.
        ReporterClass = self._get_reporter_class()
        reporter = ReporterClass(self.config.run_id, self.config.quiet)

        if self.config.reprocess_path:
            results = self._reprocess()
        else:
            assert self.evaluator is not None
            results, self.temp_dir_to_delete = self.evaluator.evaluate_all_scenarios(
                num_questions=self.config.num_questions,
                methods=self.config.methods,
                k_values=self.config.k_values,
                resume=self.config.resume,
                run_id=self.config.run_id,
            )

        if self.config.quiet:
            return reporter.results_to_dataframe(results)

        self._save_and_cleanup(results, reporter)
        return None


if __name__ == "__main__":
    # Configuration is now defined directly in the script, removing the need for CLI arguments.
    config = EvaluationConfig(
        num_questions=5,
        log_level="WARNING",
        methods=[
            "rag",
            "reranked_rag",
            "ensemble_rerank_rag",
            "unfiltered",
            "web_search",
            "baseline",
        ],
        k_values=[1, 3, 5, 7, 10],
        target_tokens=750,
        overlap_tokens=150,
        hard_ceiling=1000,
        use_docker=True,
        docker_port=6333,
        resume=True,
        # A unique run_id is generated to support the resume functionality.
        run_id=f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Initialize and run the evaluation.
    runner = EvaluationRunner(config)
    runner.run()
