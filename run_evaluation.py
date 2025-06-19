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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import RAGPipeline
from evaluation.evaluator import ComprehensiveEvaluator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

def run_evaluation(
    num_questions: int, 
    log_level: str = "WARNING",
    methods: list = None,
    k_values: list = None,
    quiet: bool = False
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
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not quiet:
        print("Initializing RAG pipeline for evaluation...")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize the RAG pipeline
    pipeline = RAGPipeline()
    
    # Initialize the evaluator
    evaluator = ComprehensiveEvaluator(pipeline, quiet=quiet)
    
    # Run the evaluation
    if not quiet:
        print(f"🚀 Starting evaluation with {num_questions} questions...")
    
    results = evaluator.evaluate_all_scenarios(
        num_questions=num_questions,
        methods=methods,
        k_values=k_values
    )
    
    # If in quiet mode, return the DataFrame
    if quiet:
        return evaluator.results_to_dataframe(results)

    # --- Standard Mode: Print and Save Results ---
    evaluator.print_results(results)

    # Save the detailed results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
    csv_filename = RESULTS_DIR / f"evaluation_results_{timestamp}.csv"
    
    print(f"💾 Saving detailed results to {results_filename}...")
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"📊 Exporting results to CSV: {csv_filename}...")
    evaluator.export_to_csv(results, csv_filename)
    
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
        help="A list of evaluation methods to run (e.g., rag reranked_rag)."
    )
    parser.add_argument(
        "--k-values",
        nargs='+',
        type=int,
        default=None,
        help="A list of k-values for retrieval metrics (e.g., 1 5 10)."
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        num_questions=args.num_questions, 
        log_level=args.log_level,
        methods=args.methods,
        k_values=args.k_values
    )