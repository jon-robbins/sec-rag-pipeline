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

from rag.pipeline import RAGPipeline
from evaluation.evaluator import ComprehensiveEvaluator

def run_pipeline_evaluation(num_questions: int = 20):
    """
    Initializes the RAG pipeline, runs the comprehensive evaluation,
    and prints the results.
    
    Args:
        num_questions: The number of questions to evaluate.
    """
    # Ensure the main project directory is in the Python path
    sys.path.insert(0, str(Path(__file__).parent.parent))


    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set.")
        return 1
    
    try:
        # 1. Initialize the entire RAG system.
        print("="*80)
        print("STEP 1: Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            target_tokens=750,
            overlap_tokens=150
        )
        print("="*80)

        # 2. Run the evaluation using the pipeline
        print("STEP 2: Running RAG Pipeline Evaluation...")
        evaluator = ComprehensiveEvaluator(pipeline)
        results = evaluator.evaluate_all_scenarios(num_questions=num_questions)
        print("="*80)
        
        # 3. Print the final comparison
        print("STEP 3: Final Results")
        evaluator.print_results(results)
        
        # 4. Save detailed results
        output_file = "evaluation_results.json"
        with open(output_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to {output_file}")

    except Exception as e:
        print(f"\n❌ An error occurred during the evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline evaluation.")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="The number of questions to run the evaluation on."
    )
    args = parser.parse_args()
    
    sys.exit(run_pipeline_evaluation(num_questions=args.num_questions))