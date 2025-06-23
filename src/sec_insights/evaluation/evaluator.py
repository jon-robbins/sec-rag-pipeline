#!/usr/bin/env python3
"""
Defines the ComprehensiveEvaluator for running and evaluating RAG scenarios.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from ..rag.client_utils import get_openai_client
from ..rag.config import RESULTS_DIR
from ..rag.document_store import DocumentStore
from ..rag.pipeline import RAGPipeline
from ..rag.reranker import BGEReranker
from .dataset import QADatasetManager
from .metrics import (
    calculate_bleu_score,
    calculate_retrieval_metrics,
    calculate_rouge_scores,
)
from .reporting import EvaluationReporter
from .scenarios import (
    format_question_with_context,
    run_baseline_scenario,
    run_rag_scenario,
    run_reranked_rag_scenario,
    run_unfiltered_context_scenario,
    run_web_search_scenario,
)
from .scenarios_financerag import ensemble_rerank_rag

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

        # Create config-specific QA dataset path using pipeline chunking configuration
        if pipeline:
            chunking_config = {
                "target_tokens": pipeline.target_tokens,
                "overlap_tokens": pipeline.overlap_tokens,
                "hard_ceiling": pipeline.hard_ceiling,
            }
        else:
            # Default configuration if no pipeline provided
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
        if self.pipeline:
            all_chunks = self.pipeline.get_chunks()

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
                all_results = checkpoint["completed_results"]
                start_index = checkpoint["current_index"]
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

        aggregated_results = reporter.aggregate_results(all_results)

        # Add chunking configuration metadata to results
        if self.pipeline:
            aggregated_results["chunking_config"] = {
                "target_tokens": self.pipeline.target_tokens,
                "overlap_tokens": self.pipeline.overlap_tokens,
                "hard_ceiling": self.pipeline.hard_ceiling,
            }
            aggregated_results["qa_dataset_path"] = str(self.qa_manager.qa_dataset_path)

        reporter.save_results(aggregated_results)

        return aggregated_results, temp_results_dir

    def evaluate_single_question(
        self,
        qa_item: Dict[str, Any],
        methods: List[str],
        k_values: List[int],
        phase_1_k: int = 30,
        phase_2_k: int = 10,
        use_rrf: bool = False,
    ) -> Dict[str, Any]:
        results = {}
        ground_truth_answer = qa_item["answer"]
        ground_truth_chunk_id = qa_item["chunk_id"]

        if self.pipeline is None:
            raise RuntimeError(
                "Cannot evaluate new questions without an active RAG pipeline."
            )

        if "rag" in methods:
            rag_answer, retrieved_ids, rag_tokens = run_rag_scenario(
                self.pipeline, qa_item
            )
            results["rag"] = {
                "answer": rag_answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_chunk_ids=retrieved_ids,
                    ground_truth_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacency_map=self.pipeline.adjacent_map,
                    adjacency_bonus=0.5,
                ),
                "rouge": calculate_rouge_scores(rag_answer, ground_truth_answer),
                "bleu": calculate_bleu_score(rag_answer, ground_truth_answer),
                "tokens": rag_tokens,
            }

        if "reranked_rag" in methods:
            reranked_answer, reranked_ids, reranked_tokens = run_reranked_rag_scenario(
                self.pipeline,
                qa_item,
                self.reranker,
                phase_1_k=phase_1_k,  # Use configurable Phase 1: Initial retrieval
                phase_2_k=phase_2_k,  # Use configurable Phase 2: Final selection for generation
            )
            results["reranked_rag"] = {
                "answer": reranked_answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_chunk_ids=reranked_ids,
                    ground_truth_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacency_map=self.pipeline.adjacent_map,
                    adjacency_bonus=0.5,
                ),
                "rouge": calculate_rouge_scores(reranked_answer, ground_truth_answer),
                "bleu": calculate_bleu_score(reranked_answer, ground_truth_answer),
                "tokens": reranked_tokens,
            }

        if "ensemble_rerank_rag" in methods:
            augmented_question = format_question_with_context(
                qa_item["question"], qa_item["ticker"], qa_item["year"]
            )
            scenario_output = ensemble_rerank_rag(
                self.pipeline,
                augmented_question,
                [ground_truth_chunk_id],
                k_values,
                phase_1_k=phase_1_k,  # Use configurable Phase 1: Initial retrieval
                phase_2_k=phase_2_k,  # Use configurable Phase 2: Final selection for generation
                use_rrf=use_rrf,  # Use configurable RRF for ensemble reranking
            )
            results["ensemble_rerank_rag"] = {
                "answer": scenario_output["answer"],
                "retrieval": scenario_output["retrieval"],
                "rouge": calculate_rouge_scores(
                    scenario_output["answer"], ground_truth_answer
                ),
                "bleu": calculate_bleu_score(
                    scenario_output["answer"], ground_truth_answer
                ),
                "tokens": scenario_output.get("tokens", {}),
                "contexts": scenario_output.get("contexts", []),
            }

        if "unfiltered" in methods:
            unfiltered_answer, unfiltered_tokens = run_unfiltered_context_scenario(
                self.doc_store, self.openai_client, qa_item
            )
            results["unfiltered"] = {
                "answer": unfiltered_answer,
                "rouge": calculate_rouge_scores(unfiltered_answer, ground_truth_answer),
                "bleu": calculate_bleu_score(unfiltered_answer, ground_truth_answer),
                "tokens": unfiltered_tokens,
            }

        if "web_search" in methods:
            web_search_answer, web_search_tokens = run_web_search_scenario(
                self.openai_client, qa_item
            )
            results["web_search"] = {
                "answer": web_search_answer,
                "rouge": calculate_rouge_scores(web_search_answer, ground_truth_answer),
                "bleu": calculate_bleu_score(web_search_answer, ground_truth_answer),
                "tokens": web_search_tokens,
            }

        if "baseline" in methods:
            baseline_answer, baseline_tokens = run_baseline_scenario(
                self.openai_client, qa_item
            )
            results["baseline"] = {
                "answer": baseline_answer,
                "rouge": calculate_rouge_scores(baseline_answer, ground_truth_answer),
                "bleu": calculate_bleu_score(baseline_answer, ground_truth_answer),
                "tokens": baseline_tokens,
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
