#!/usr/bin/env python3
"""
Runs the comprehensive evaluation of the RAG pipeline.

This script initializes the RAG system and then runs a comparative
evaluation across three different scenarios:
1. RAG Pipeline: Full retrieval-augmented generation.
2. Unfiltered Context: Generation with the full SEC filing as context.
3. Web Search: Generation using a standard model without specific context.
"""
import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import logging
import shutil

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import RAGPipeline
from evaluation.evaluator import ComprehensiveEvaluator
from rag.config import RESULTS_DIR

def run_evaluation(
    num_questions: int, 
    log_level: str = "WARNING",
    methods: list = None,
    k_values: list = None,
    quiet: bool = False,
    reprocess_path: str = None,
    resume: bool = True,
    run_id: str = None
):
    """
    Initialize the RAG pipeline and run the comprehensive evaluation.
    
    Args:
        num_questions: The number of questions to evaluate.
        log_level: The logging level to use.
        methods: A list of evaluation methods to run (e.g., ['rag', 'reranked_rag']). 
                 If None, all methods are run.
        k_values: A list of integers for retrieval evaluation (e.g., [1, 5, 10]).
                  If None, default values are used.
        quiet: If True, suppresses all print statements and returns a DataFrame.
        reprocess_path: Path to a raw results JSON file to reprocess.
        resume: If True, resumes the evaluation from the last saved state.
        run_id: Identifier for the current run.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not quiet:
        print("Initializing RAG pipeline for evaluation...")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir_to_delete = None
    
    # In quiet mode, we don't want to initialize a full pipeline if we're just reprocessing
    pipeline = RAGPipeline() if not (quiet and reprocess_path) else None
    evaluator = ComprehensiveEvaluator(pipeline, quiet=quiet)
    
    if reprocess_path:
        if not quiet:
            print(f"🔄 Reprocessing raw results from: {reprocess_path}")
        with open(reprocess_path, 'r') as f:
            data = json.load(f)
            
        # Check if this is already aggregated results or raw results
        if isinstance(data, dict) and "individual" in data:
            # Already aggregated - use as is
            results = data
            if not quiet:
                print("✅ File contains aggregated results, using directly.")
        elif isinstance(data, list):
            # Raw individual results - need to aggregate
            results = evaluator._aggregate_results(data)
            if not quiet:
                print("✅ Aggregated raw individual results.")
        elif isinstance(data, dict) and "completed_results" in data:
            # Checkpoint file format - extract completed_results and aggregate
            raw_results = data["completed_results"]
            results = evaluator._aggregate_results(raw_results)
            if not quiet:
                print("✅ Loaded checkpoint file and aggregated results.")
        else:
            raise ValueError(f"Unexpected data format in {reprocess_path}")
    
    else:
        # Run a new evaluation
        if not quiet:
            print(f"🚀 Starting evaluation with {num_questions} questions...")
        
        results, temp_dir_to_delete = evaluator.evaluate_all_scenarios(
            num_questions=num_questions,
            methods=methods,
            k_values=k_values,
            resume=resume,
            run_id=run_id
        )
    
    # If in quiet mode, return the DataFrame and skip file saving/cleanup
    if quiet:
        if temp_dir_to_delete:
            shutil.rmtree(temp_dir_to_delete)
        return evaluator.results_to_dataframe(results)

    # --- Standard Mode: Print and Save Results ---
    evaluator.print_results(results)

    # Save the detailed results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_filename = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
    final_csv_filename = RESULTS_DIR / f"evaluation_results_{timestamp}.csv"
    
    print(f"💾 Saving final aggregated results to {final_results_filename}...")
    with open(final_results_filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"📊 Exporting results to CSV: {final_csv_filename}...")
    evaluator.export_to_csv(results, final_csv_filename)
    
    # --- Cleanup ---
    if temp_dir_to_delete:
        print(f"🧹 Cleaning up temporary directory: {temp_dir_to_delete}")
        shutil.rmtree(temp_dir_to_delete)
        
    print("✅ Evaluation complete.")
    return None # Explicitly return None in non-quiet mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation.")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions to use for the evaluation."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the script."
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        default=None,
        choices=["rag", "reranked_rag", "ensemble_rerank_rag", "unfiltered", "web_search", "baseline"],
        help="A list of evaluation methods to run (e.g., rag ensemble_rerank_rag)."
    )
    parser.add_argument(
        "--k-values",
        nargs='+',
        type=int,
        default=None,
        help="A list of k-values for retrieval metrics (e.g., 1 5 10)."
    )
    parser.add_argument(
        "--reprocess",
        type=str,
        default=None,
        help="Path to a raw results JSON file to reprocess, skipping the main evaluation."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume functionality and start fresh evaluation."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specify a run ID for checkpoint management. If not provided, a timestamp will be generated."
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        num_questions=args.num_questions, 
        log_level=args.log_level,
        methods=args.methods,
        k_values=args.k_values,
        reprocess_path=args.reprocess,
        resume=not args.no_resume,
        run_id=args.run_id
    )