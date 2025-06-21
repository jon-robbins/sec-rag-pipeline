#!/usr/bin/env python3
"""
Runs the comprehensive evaluation of the RAG pipeline using a class-based runner.
"""
import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import logging
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import RAGPipeline
from evaluation.evaluator import ComprehensiveEvaluator
from rag.config import RESULTS_DIR

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
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        if not self.config.quiet:
            print("Initializing RAG pipeline for evaluation...")
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        pipeline_kwargs = {
            "target_tokens": self.config.target_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "hard_ceiling": self.config.hard_ceiling,
            "use_docker": self.config.use_docker,
            "docker_port": self.config.docker_port,
        }
        # Filter out None values to use defaults in RAGPipeline
        pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v is not None}

        # Don't initialize a full pipeline if we're just reprocessing
        if not (self.config.quiet and self.config.reprocess_path):
             self.pipeline = RAGPipeline(**pipeline_kwargs)
        
        self.evaluator = ComprehensiveEvaluator(self.pipeline, quiet=self.config.quiet)

    def _reprocess(self) -> dict:
        """Reprocesses results from a file."""
        if not self.config.quiet:
            print(f"🔄 Reprocessing raw results from: {self.config.reprocess_path}")
        
        with open(self.config.reprocess_path, 'r') as f:
            data = json.load(f)

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
            print(f"🚀 Starting evaluation with {self.config.num_questions} questions...")
        
        results, self.temp_dir_to_delete = self.evaluator.evaluate_all_scenarios(
            num_questions=self.config.num_questions,
            methods=self.config.methods,
            k_values=self.config.k_values,
            resume=self.config.resume,
            run_id=self.config.run_id
        )
        return results

    def _save_and_cleanup(self, results: dict):
        """Saves results to files and cleans up temporary directories."""
        if self.config.quiet:
            if self.temp_dir_to_delete:
                shutil.rmtree(self.temp_dir_to_delete)
            return

        self.evaluator.print_results(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_results_filename = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
        final_csv_filename = RESULTS_DIR / f"evaluation_results_{timestamp}.csv"
        
        print(f"💾 Saving final aggregated results to {final_results_filename}...")
        with open(final_results_filename, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"📊 Exporting results to CSV: {final_csv_filename}...")
        self.evaluator.export_to_csv(results, final_csv_filename)
        
        if self.temp_dir_to_delete:
            print(f"🧹 Cleaning up temporary directory: {self.temp_dir_to_delete}")
            shutil.rmtree(self.temp_dir_to_delete)
            
        print("✅ Evaluation complete.")

    def run(self):
        """Executes the entire evaluation process."""
        self._setup()
        
        if self.config.reprocess_path:
            results = self._reprocess()
        else:
            results = self._run_new_evaluation()
        
        if self.config.quiet:
            # In quiet mode, return the DataFrame directly
            return self.evaluator.results_to_dataframe(results)
        else:
            self._save_and_cleanup(results)
            return None

if __name__ == "__main__":
    # Configuration is now defined directly in the script, removing the need for CLI arguments.
    config = EvaluationConfig(
        num_questions=300,
        log_level="WARNING",
        methods=["rag", "reranked_rag", "ensemble_rerank_rag", "unfiltered", "web_search", "baseline"],
        k_values=[1, 3, 5, 7, 10],
        target_tokens=150,
        overlap_tokens=50,
        hard_ceiling=500,
        use_docker=True,
        docker_port=6333,
        resume=True,
        # A unique run_id is generated to support the resume functionality.
        run_id=f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Initialize and run the evaluation.
    runner = EvaluationRunner(config)
    runner.run()