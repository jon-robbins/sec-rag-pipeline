#!/usr/bin/env python3
"""
Main entry point for running the RAG evaluation pipeline.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.evaluation.evaluator import run_evaluation

logger = logging.getLogger(__name__)


def load_existing_qa_dataset() -> list:
    """Load an existing QA dataset for testing."""
    qa_file = Path("data/processed/qa_dataset_50.jsonl")
    if not qa_file.exists():
        raise FileNotFoundError(f"QA dataset not found: {qa_file}")

    qa_data = []
    with open(qa_file, "r") as f:
        for line in f:
            qa_data.append(json.loads(line))

    logger.info("Loaded %d QA pairs from %s", len(qa_data), qa_file)
    return qa_data


def main():
    """
    Parses command-line arguments and runs the evaluation.
    """
    parser = argparse.ArgumentParser(description="Run the RAG evaluation pipeline.")
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "baseline",
            "web_search",
            "unfiltered_text",
            "vanilla_rag",
            "rerank_rag",
            "ensemble_rerank_rag",
        ],
        help="List of methods to evaluate.",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="List of k values for retrieval metrics.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Use Docker for the vector store.",
    )
    parser.add_argument(
        "--docker-port",
        type=int,
        default=6333,
        help="Port for the Dockerized vector store.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Identifier for the evaluation run.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting evaluation run: %s", args.run_id)

    # Load existing QA dataset
    qa_dataset = load_existing_qa_dataset()

    # Run the evaluation using the legacy run_evaluation function
    run_evaluation(
        qa_dataset=qa_dataset,
        num_questions=args.num_questions,
        methods=args.methods,
        k_values=args.k_values,
        run_id=args.run_id,
        use_docker=args.use_docker,
        docker_port=args.docker_port,
    )

    logger.info("Evaluation run %s finished.", args.run_id)


if __name__ == "__main__":
    main()
