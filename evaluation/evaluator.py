#!/usr/bin/env python3
"""
Defines the ComprehensiveEvaluator for running and evaluating RAG scenarios.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import time

from openai import OpenAI
from tqdm.auto import tqdm

# Ensure the project root is in the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from evaluation.scenarios import (
    run_rag_scenario,
    run_unfiltered_context_scenario,
    run_web_search_scenario,
)
from evaluation.metrics import calculate_retrieval_metrics, calculate_rouge_scores
from evaluation.generate_qa_dataset import generate_qa_pairs, BalancedChunkSampler
from rag.config import QA_DATASET_PATH
from rag.document_store import DocumentStore


class ComprehensiveEvaluator:
    """
    Orchestrates the comprehensive evaluation of different RAG scenarios.
    """
    def __init__(self, pipeline: RAGPipeline):
        """
        Initializes the evaluator with a RAG pipeline instance.

        Args:
            pipeline: An initialized RAGPipeline object.
        """
        self.pipeline = pipeline
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qa_dataset_path = QA_DATASET_PATH
        self.qa_dataset = None
        self.doc_store = DocumentStore(tickers_of_interest=pipeline.document_store.tickers)

    def _get_chunks_from_pipeline(self):
        """
        Get chunks directly from the RAG pipeline (always fresh).
        """
        if not hasattr(self.pipeline, 'get_chunks') or not self.pipeline.get_chunks():
            raise RuntimeError("Pipeline chunks not available. Ensure the RAG pipeline is properly initialized.")
        return self.pipeline.get_chunks()

    def _load_or_generate_qa_dataset(self, num_questions: int) -> List[Dict[str, Any]]:
        """
        Loads the QA dataset, generates more questions if needed, and samples
        the requested number of questions for the evaluation.
        """
        qa_data = []
        if self.qa_dataset_path.exists():
            with open(self.qa_dataset_path, "r") as f:
                for line in f:
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line in {self.qa_dataset_path}")
        
        print(f"✅ Found {len(qa_data)} existing QA pairs in {self.qa_dataset_path}.")

        # --- Generate more questions if needed ---
        if len(qa_data) < num_questions:
            num_to_generate = num_questions - len(qa_data)
            print(f" Generating {num_to_generate} missing questions")

            # Get chunks from the pipeline
            chunks = self._get_chunks_from_pipeline()

            # Create a balanced sample of chunks for new QA generation
            sampler = BalancedChunkSampler()
            grouped_chunks = sampler.group_chunks_by_keys(chunks)
            balanced_chunks = sampler.stratified_sample(grouped_chunks)
            
            # We need to sample enough chunks to generate the desired number of questions.
            # The generator creates ~2 questions per chunk.
            num_chunks_to_sample = (num_to_generate + 1) // 2
            if len(balanced_chunks) < num_chunks_to_sample:
                print(f"⚠️ Warning: Not enough balanced chunks ({len(balanced_chunks)}) to generate {num_to_generate} new questions. Using all available.")
                chunks_for_qa = balanced_chunks
            else:
                chunks_for_qa = random.sample(balanced_chunks, num_chunks_to_sample)
            
            print(f"Selected {len(chunks_for_qa)} chunks to generate ~{num_to_generate} new questions.")
            
            # Generate new QA pairs and append to the existing file
            generate_qa_pairs(chunks_for_qa, str(self.qa_dataset_path), append=True)
            
            # --- Reload the full dataset ---
            qa_data = []
            with open(self.qa_dataset_path, "r") as f:
                for line in f:
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line in {self.qa_dataset_path}")
            print(f"✅ Reloaded dataset with a total of {len(qa_data)} QA pairs.")

        # --- Final sampling ---
        if len(qa_data) >= num_questions:
            return random.sample(qa_data, num_questions)
        else:
            print(f"⚠️ Warning: Final dataset size ({len(qa_data)}) is less than requested ({num_questions}). Using all available questions.")
            return qa_data

    def evaluate_single_question(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single question across all three scenarios.
        """
        results = {}
        ground_truth_answer = qa_item["answer"]
        ground_truth_chunk_id = qa_item["chunk_id"]

        # 1. RAG Scenario
        rag_answer, retrieved_ids = run_rag_scenario(self.pipeline, qa_item)
        results["rag"] = {
            "answer": rag_answer,
            "retrieval": calculate_retrieval_metrics(
                retrieved_chunk_ids=retrieved_ids,
                true_chunk_id=ground_truth_chunk_id,
                k_values=[1, 3, 5, 10]
            ),
            "rouge": calculate_rouge_scores(rag_answer, ground_truth_answer)
        }

        # 2. Unfiltered Context Scenario
        unfiltered_answer = run_unfiltered_context_scenario(
            doc_store=self.doc_store,
            openai_client=self.openai_client,
            qa_item=qa_item
        )
        results["unfiltered"] = {
            "answer": unfiltered_answer,
            "rouge": calculate_rouge_scores(unfiltered_answer, ground_truth_answer)
        }

        # 3. Web Search Scenario
        web_search_answer = run_web_search_scenario(self.openai_client, qa_item)
        results["web_search"] = {
            "answer": web_search_answer,
            "rouge": calculate_rouge_scores(web_search_answer, ground_truth_answer)
        }
        
        return results

    def evaluate_all_scenarios(self, num_questions: int = 50) -> Dict[str, Any]:
        """
        Runs the full evaluation across all scenarios and aggregates the results.
        """
        # Pre-load all necessary data before starting the evaluation loop
        print("Pre-loading all necessary data for evaluation...")
        self.doc_store.get_all_sentences()  # This will load the document store data
        
        self.qa_dataset = self._load_or_generate_qa_dataset(num_questions)
        
        all_results = []
        progress_bar = tqdm(self.qa_dataset, desc="🔬 Evaluating scenarios", unit="question")

        for qa_item in progress_bar:
            single_result = self.evaluate_single_question(qa_item)
            single_result["question"] = qa_item["question"]
            single_result["ground_truth_answer"] = qa_item["answer"]
            all_results.append(single_result)
            time.sleep(2)  # Add a delay to avoid rate limiting
        
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates metrics from all evaluation runs.
        """
        summary = {
            "rag": {"retrieval": defaultdict(list), "rouge": defaultdict(list)},
            "unfiltered": {"rouge": defaultdict(list)},
            "web_search": {"rouge": defaultdict(list)}
        }
        
        for res in results:
            # RAG
            for metric, value in res["rag"]["retrieval"].items():
                summary["rag"]["retrieval"][metric].append(value)
            for metric, scores in res["rag"]["rouge"].items():
                summary["rag"]["rouge"][metric].append(scores)
            
            # Unfiltered
            for metric, scores in res["unfiltered"]["rouge"].items():
                summary["unfiltered"]["rouge"][metric].append(scores)
            
            # Web Search
            for metric, scores in res["web_search"]["rouge"].items():
                summary["web_search"]["rouge"][metric].append(scores)

        # Calculate averages, filtering out None values
        final_summary = {"total_questions": len(results)}
        for scenario, metrics in summary.items():
            final_summary[scenario] = {}
            if "retrieval" in metrics:
                final_summary[scenario]["retrieval"] = {}
                for m, values in metrics["retrieval"].items():
                    # Filter out None values before calculating average
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        final_summary[scenario]["retrieval"][m] = sum(valid_values) / len(valid_values)
                    else:
                        final_summary[scenario]["retrieval"][m] = 0.0
            if "rouge" in metrics:
                final_summary[scenario]["rouge"] = {
                    m: {
                        "precision": sum(s['precision'] for s in scores) / len(scores),
                        "recall": sum(s['recall'] for s in scores) / len(scores),
                        "fmeasure": sum(s['fmeasure'] for s in scores) / len(scores),
                    }
                    for m, scores in metrics["rouge"].items()
                }
        
        return {"summary": final_summary, "detailed": results}

    def print_results(self, results: Dict[str, Any]):
        """
        Prints a formatted summary of the evaluation results.
        """
        summary = results.get("summary", {})
        if not summary:
            print("No summary found in results.")
            return

        print("\n" + "="*80)
        print(f"📊 Evaluation Summary (Total Questions: {summary.get('total_questions', 0)})")
        print("="*80)
        
        for scenario, metrics in summary.items():
            if scenario == "total_questions": continue
            
            print(f"\n--- Scenario: {scenario.upper()} ---")
            
            if "retrieval" in metrics:
                print("  [Retrieval Metrics]")
                ret = metrics["retrieval"]
                print(f"    - Recall@1:  {ret.get('recall_at_1', 0):.3f}")
                print(f"    - Recall@3:  {ret.get('recall_at_3', 0):.3f}")
                print(f"    - Recall@5:  {ret.get('recall_at_5', 0):.3f}")
                print(f"    - Recall@10: {ret.get('recall_at_10', 0):.3f}")
                print(f"    - MRR:       {ret.get('mrr', 0):.3f}")
            
            if "rouge" in metrics:
                print("\n  [Generation Metrics (F1-Score)]")
                rouge = metrics["rouge"]
                print(f"    - ROUGE-1: {rouge.get('rouge1', {}).get('fmeasure', 0):.3f}")
                print(f"    - ROUGE-2: {rouge.get('rouge2', {}).get('fmeasure', 0):.3f}")
                print(f"    - ROUGE-L: {rouge.get('rougeL', {}).get('fmeasure', 0):.3f}")
        
        print("\n" + "="*80) 