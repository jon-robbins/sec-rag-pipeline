#!/usr/bin/env python3
"""
This module contains the core logic for running the evaluation pipeline.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.evaluation.reporting import EvaluationReporter
from src.methods import (
    baseline,
    ensemble_rerank_rag,
    rerank_rag,
    unfiltered_text,
    vanilla_rag,
    web_search,
)
from src.metrics.metrics import (
    calculate_bleu_score,
    calculate_retrieval_metrics,
    calculate_rouge_scores,
)
from src.openai_functions.answer_question import AnswerGenerator
from src.preprocessing.dataset import QADatasetManager
from src.rag.pipeline import RAGPipeline

# --- Legacy imports removed during refactor ---
# from .scenarios import (
#     format_question_with_context,
#     run_baseline_scenario,
#     run_rag_scenario,
#     run_reranked_rag_scenario,
#     run_unfiltered_context_scenario,
#     run_web_search_scenario,
# )
# from .scenarios_financerag import ensemble_rerank_rag
from src.rag.reranker import BGEReranker
from src.utils.client_utils import get_openai_client
from src.utils.config import RESULTS_DIR
from src.vector_store.document_store import DocumentStore
from src.vector_store.embedding import EmbeddingManager
from src.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Orchestrates the comprehensive evaluation of different RAG scenarios.
    """

    def __init__(self, root_dir: Path, pipeline: Optional[RAGPipeline]):
        self.pipeline = pipeline
        self.openai_client = get_openai_client()
        self._initialize_components(root_dir, pipeline)

    def _initialize_components(self, root_dir: Path, pipeline: Optional[RAGPipeline]):
        """Initialize required components for the evaluator."""
        # The evaluator needs its own DocumentStore for the 'unfiltered' scenario
        raw_data_path = root_dir / "data" / "raw" / "df_filings_full.parquet"
        doc_store_tickers = (
            pipeline.document_store.tickers
            if pipeline and hasattr(pipeline, "document_store")
            else None
        )
        self.doc_store = DocumentStore(
            raw_data_path=raw_data_path, tickers_of_interest=doc_store_tickers
        )

        logging.info("Initializing BGE Reranker...")
        self.reranker = BGEReranker()

        # Create config-specific QA dataset path using default chunking configuration
        # Note: RAGPipeline doesn't store chunking config, so use defaults
        chunking_config = {
            "target_tokens": 750,
            "overlap_tokens": 150,
            "hard_ceiling": 1000,
        }
        self.qa_manager = QADatasetManager(root_dir, chunking_config)

    def evaluate_all_scenarios(
        self,
        num_questions: int = 50,
        methods: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        phase_1_k: int = 30,
        phase_2_k: int = 10,
        use_rrf: bool = False,
        resume: bool = True,
        run_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        """
        Runs the full evaluation across all scenarios.

        Args:
            num_questions: Number of questions to evaluate
            methods: List of evaluation methods to run
            k_values: List of k values for retrieval metrics
            phase_1_k: Initial retrieval count (only for reranked systems)
            phase_2_k: Final selection count for generation (only for reranked systems)
            use_rrf: Whether to use Reciprocal Rank Fusion for ensemble reranking
            resume: Whether to resume from checkpoint
            run_id: Unique identifier for this evaluation run
        """
        methods = methods or [
            "rag",
            "reranked_rag",
            "unfiltered",
            "web_search",
            "baseline",
            "ensemble_rerank_rag",
        ]
        k_values = k_values or [1, 3, 5, 7, 10]
        run_id = run_id or time.strftime("%Y%m%d_%H%M%S")

        reporter = EvaluationReporter(run_id)

        logging.info("Pre-loading all necessary data for evaluation...")
        self.doc_store.get_all_sentences()

        all_chunks = []
        if self.pipeline and hasattr(self.pipeline, "get_chunks"):
            all_chunks = self.pipeline.get_chunks()
        else:
            # For baseline test without full pipeline, use empty chunks list
            all_chunks = []

        qa_dataset = self.qa_manager.load_or_generate(num_questions, all_chunks)

        all_results: List[Dict[str, Any]] = []
        start_index = 0
        if resume:
            checkpoint = self._load_checkpoint(run_id)
            if (
                checkpoint
                and checkpoint.get("methods") == methods
                and len(checkpoint.get("qa_dataset", [])) == len(qa_dataset)
            ):
                # Load checkpoint data - mypy has trouble with JSON deserialization types
                try:
                    completed_results = checkpoint["completed_results"]
                    current_index = checkpoint["current_index"]
                    if isinstance(completed_results, list) and isinstance(
                        current_index, int
                    ):
                        all_results.extend(completed_results)
                        start_index = current_index
                except (KeyError, TypeError):
                    # If checkpoint format is unexpected, start fresh
                    pass
                logging.info(
                    "Resuming from question %d/%d", start_index + 1, len(qa_dataset)
                )
            elif checkpoint:
                logging.warning("Checkpoint incompatible. Starting fresh.")

        progress_bar = tqdm(
            qa_dataset[start_index:],
            desc="Evaluating scenarios",
            initial=start_index,
            total=len(qa_dataset),
        )

        try:
            for i, qa_item in enumerate(progress_bar, start=start_index):
                single_result = self.evaluate_single_question(
                    qa_item, methods, k_values, phase_1_k, phase_2_k, use_rrf
                )
                single_result["question"] = qa_item["question"]
                single_result["ground_truth_answer"] = qa_item["answer"]
                single_result["ground_truth_chunk_id"] = qa_item["chunk_id"]
                single_result["ticker"] = qa_item.get("ticker")
                single_result["year"] = qa_item.get("year")
                single_result["section"] = qa_item.get("section")
                all_results.append(single_result)

                if (i + 1) % 10 == 0:
                    self._save_checkpoint(
                        all_results, qa_dataset, i + 1, methods, k_values, run_id
                    )
                time.sleep(1)  # Rate limiting

            self._cleanup_checkpoint(run_id)
        except (KeyboardInterrupt, Exception) as e:
            logging.error("Process interrupted: %s. Progress saved.", e)
            self._save_checkpoint(
                all_results, qa_dataset, len(all_results), methods, k_values, run_id
            )
            raise

        temp_results_dir = RESULTS_DIR / f"run_{run_id}"
        temp_results_dir.mkdir(parents=True, exist_ok=True)

        # Save results and let reporter handle aggregation
        results_data = {"individual": all_results}

        # Add chunking configuration metadata to results
        chunking_config = {
            "target_tokens": 750,
            "overlap_tokens": 150,
            "hard_ceiling": 1000,
        }
        results_data["chunking_config"] = chunking_config
        results_data["qa_dataset_path"] = str(self.qa_manager.qa_dataset_path)

        reporter.save_results(results_data)

        return results_data, temp_results_dir

    def evaluate_single_question(
        self,
        qa_item: Dict[str, Any],
        methods: List[str],
        k_values: List[int],
        phase_1_k: int = 30,
        phase_2_k: int = 10,
        use_rrf: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a single QA item across the selected retrieval-generation methods."""
        results: Dict[str, Any] = {}
        ground_truth_answer = qa_item["answer"]
        ground_truth_chunk_id = qa_item["chunk_id"]

        if self.pipeline is None:
            raise RuntimeError(
                "Cannot evaluate questions without an active RAG pipeline."
            )

        # --- Vanilla RAG ---
        if "rag" in methods:
            answer, retrieved_ids, tokens = vanilla_rag.run_rag_scenario(
                self.pipeline, qa_item
            )
            results["rag"] = {
                "answer": answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_chunk_ids=retrieved_ids,
                    ground_truth_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacency_map=getattr(self.pipeline, "adjacent_map", None),
                    adjacency_bonus=0.5,
                ),
                "rouge": calculate_rouge_scores(answer, ground_truth_answer),
                "bleu": calculate_bleu_score(answer, ground_truth_answer),
                "tokens": tokens,
            }

        # --- Reranked RAG ---
        if "reranked_rag" in methods:
            answer, retrieved_ids, tokens = rerank_rag.run_reranked_rag_scenario(
                pipeline=self.pipeline,
                qa_item=qa_item,
                reranker=self.reranker,
                phase_1_k=phase_1_k,
                phase_2_k=phase_2_k,
            )
            results["reranked_rag"] = {
                "answer": answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_chunk_ids=retrieved_ids,
                    ground_truth_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacency_map=getattr(self.pipeline, "adjacent_map", None),
                    adjacency_bonus=0.5,
                ),
                "rouge": calculate_rouge_scores(answer, ground_truth_answer),
                "bleu": calculate_bleu_score(answer, ground_truth_answer),
                "tokens": tokens,
            }

        # --- Ensemble Rerank RAG ---
        if "ensemble_rerank_rag" in methods:
            augmented_question = vanilla_rag.format_question_with_context(
                qa_item["question"], qa_item["ticker"], qa_item["year"]
            )
            answer, retrieved_ids, tokens = ensemble_rerank_rag.run_ensemble_rerank_rag(
                rag_pipeline=self.pipeline,
                question=augmented_question,
                phase_1_k=phase_1_k,
                phase_2_k=phase_2_k,
                use_rrf=use_rrf,
            )
            results["ensemble_rerank_rag"] = {
                "answer": answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_chunk_ids=retrieved_ids,
                    ground_truth_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacency_map=getattr(self.pipeline, "adjacent_map", None),
                    adjacency_bonus=0.5,
                ),
                "rouge": calculate_rouge_scores(answer, ground_truth_answer),
                "bleu": calculate_bleu_score(answer, ground_truth_answer),
                "tokens": tokens,
            }

        # --- Unfiltered (full-context) ---
        if "unfiltered" in methods:
            answer, tokens = unfiltered_text.run_unfiltered_context_scenario(
                self.doc_store, self.openai_client, qa_item
            )
            results["unfiltered"] = {
                "answer": answer,
                "rouge": calculate_rouge_scores(answer, ground_truth_answer),
                "bleu": calculate_bleu_score(answer, ground_truth_answer),
                "tokens": tokens,
            }

        # --- Web Search ---
        if "web_search" in methods:
            answer, tokens = web_search.run_web_search_scenario(
                self.openai_client, qa_item
            )
            results["web_search"] = {
                "answer": answer,
                "rouge": calculate_rouge_scores(answer, ground_truth_answer),
                "bleu": calculate_bleu_score(answer, ground_truth_answer),
                "tokens": tokens,
            }

        # --- Baseline LLM ---
        if "baseline" in methods:
            answer, tokens = baseline.run_baseline_scenario(self.openai_client, qa_item)
            results["baseline"] = {
                "answer": answer,
                "rouge": calculate_rouge_scores(answer, ground_truth_answer),
                "bleu": calculate_bleu_score(answer, ground_truth_answer),
                "tokens": tokens,
            }

        return results

    def _get_checkpoint_path(self, run_id: str) -> Path:
        return RESULTS_DIR / f"checkpoint_{run_id}.json"

    def _save_checkpoint(
        self,
        results: List[Dict],
        qa_dataset: List[Dict],
        current_index: int,
        methods: List[str],
        k_values: List[int],
        run_id: str,
    ):
        checkpoint_path = self._get_checkpoint_path(run_id)
        checkpoint_data = {
            "current_index": current_index,
            "completed_results": results,
            "qa_dataset": qa_dataset,
            "methods": methods,
            "k_values": k_values,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

    def _load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        checkpoint_path = self._get_checkpoint_path(run_id)
        if checkpoint_path.exists():
            logging.info("Found checkpoint: %s", checkpoint_path)
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        return None

    def _cleanup_checkpoint(self, run_id: str):
        checkpoint_path = self._get_checkpoint_path(run_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logging.info("Checkpoint cleaned up: %s", checkpoint_path)


def run_evaluation(
    qa_dataset: List[Dict[str, Any]],
    num_questions: int,
    methods: List[str],
    k_values: List[int],
    run_id: str,
    use_docker: bool = False,
    docker_port: int = 6333,
):
    """
    Runs the evaluation across specified methods and saves the results.
    """
    openai_client = get_openai_client()
    raw_data_path = Path.cwd() / "data" / "raw" / "df_filings_full.parquet"
    doc_store = DocumentStore(raw_data_path=raw_data_path)
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(
        embedding_manager=embedding_manager, use_docker=use_docker, port=docker_port
    )
    answer_generator = AnswerGenerator(openai_client)
    rag_pipeline = RAGPipeline(vector_store, answer_generator)
    reporter = EvaluationReporter(run_id)
    reranker = BGEReranker()

    results = []

    for item in tqdm(qa_dataset[:num_questions], desc="Evaluating Questions"):
        item_id = (
            item.get("id") or item.get("human_readable_id") or item.get("chunk_id")
        )
        question_results = {"question_id": item_id, "question": item["question"]}

        for method in methods:
            logger.info("Running method '%s' for question '%s'", method, item_id)
            answer: Optional[str] = None
            retrieved_ids: List[str] = []
            tokens: Dict[str, Any] = {}

            if method == "baseline":
                answer, tokens = baseline.run_baseline_scenario(openai_client, item)
            elif method == "web_search":
                answer, tokens = web_search.run_web_search_scenario(openai_client, item)
            elif method == "unfiltered_text":
                answer, tokens = unfiltered_text.run_unfiltered_context_scenario(
                    doc_store, openai_client, item
                )
            elif method == "vanilla_rag":
                answer, retrieved_ids, tokens = vanilla_rag.run_rag_scenario(
                    rag_pipeline, item
                )
            elif method == "rerank_rag":
                answer, retrieved_ids, tokens = rerank_rag.run_reranked_rag_scenario(
                    pipeline=rag_pipeline, qa_item=item, reranker=reranker
                )
            elif method == "ensemble_rerank_rag":
                answer, retrieved_ids, tokens = (
                    ensemble_rerank_rag.run_ensemble_rerank_rag(
                        rag_pipeline=rag_pipeline, question=item["question"]
                    )
                )

            retrieval_metrics = {}
            if retrieved_ids:
                # Handle both chunk_id (singular) and chunk_ids (plural) from dataset
                ground_truth_chunks = item.get("chunk_ids") or [item.get("chunk_id")]
                if (
                    ground_truth_chunks and ground_truth_chunks[0]
                ):  # Make sure we have valid chunk IDs
                    retrieval_metrics = calculate_retrieval_metrics(
                        retrieved_chunk_ids=retrieved_ids,
                        ground_truth_chunk_id=ground_truth_chunks[
                            0
                        ],  # Use first/only chunk ID
                        k_values=k_values,
                    )

            question_results[method] = {
                "answer": answer,
                "retrieval_metrics": retrieval_metrics,
                "token_usage": tokens,
            }
        results.append(question_results)

    reporter.save_results({"individual": results})
    logger.info("Evaluation complete. Results for run %s saved.", run_id)
