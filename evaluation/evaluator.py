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
import csv
import numpy as np
import pandas as pd

from openai import OpenAI
from tqdm.auto import tqdm

# Ensure the project root is in the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from rag.reranker import BGEReranker
from evaluation.scenarios import (
    run_rag_scenario,
    run_reranked_rag_scenario,
    run_unfiltered_context_scenario,
    run_web_search_scenario,
    run_baseline_scenario,
    format_question_with_context,
)
from evaluation.scenarios_financerag import ensemble_rerank_rag
from evaluation.metrics import calculate_retrieval_metrics, calculate_rouge_scores
from evaluation.generate_qa_dataset import generate_qa_pairs, BalancedChunkSampler
from rag.config import QA_DATASET_PATH, RESULTS_DIR
from rag.document_store import DocumentStore


class ComprehensiveEvaluator:
    """
    Orchestrates the comprehensive evaluation of different RAG scenarios.
    """
    def __init__(self, pipeline: RAGPipeline, quiet: bool = False):
        """
        Initializes the evaluator with a RAG pipeline instance.

        Args:
            pipeline: An initialized RAGPipeline object.
            quiet: If True, suppresses print statements and progress bars.
        """
        self.pipeline = pipeline
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qa_dataset_path = QA_DATASET_PATH
        self.qa_dataset = None
        self.doc_store = DocumentStore(tickers_of_interest=pipeline.document_store.tickers)
        if not quiet:
            print("Initializing BGE Reranker...")
        self.reranker = BGEReranker()
        self.quiet = quiet

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
                        if not self.quiet:
                            print(f"Warning: Skipping malformed line in {self.qa_dataset_path}")
        
        if not self.quiet:
            print(f"✅ Found {len(qa_data)} existing QA pairs in {self.qa_dataset_path}.")

        # --- Generate more questions if needed ---
        if len(qa_data) < num_questions:
            num_to_generate = num_questions - len(qa_data)
            if not self.quiet:
                print(f"🧬 Generating {num_to_generate} missing questions...")

            # Get chunks from the pipeline
            chunks = self._get_chunks_from_pipeline()

            # Create a balanced sample of chunks for new QA generation
            # Scale max_per_group conservatively based on questions needed
            max_per_group = max(3, min(10, (num_to_generate + 19) // 20))  # 3-10 range, +1 per 20 questions
            sampler = BalancedChunkSampler(max_per_group=max_per_group)
            grouped_chunks = sampler.group_chunks_by_keys(chunks)
            balanced_chunks = sampler.stratified_sample(grouped_chunks)
            
            if not self.quiet:
                print(f"📊 Using max_per_group={max_per_group} for {num_to_generate} questions (found {len(balanced_chunks)} balanced chunks)")
            
            # We need to sample enough chunks to generate the desired number of questions.
            # The generator creates ~2 questions per chunk.
            num_chunks_to_sample = (num_to_generate + 1) // 2
            if len(balanced_chunks) < num_chunks_to_sample:
                if not self.quiet:
                    print(f"⚠️ Warning: Not enough balanced chunks ({len(balanced_chunks)}) to generate {num_to_generate} new questions. Using all available.")
                chunks_for_qa = balanced_chunks
            else:
                chunks_for_qa = random.sample(balanced_chunks, num_chunks_to_sample)
            
            if not self.quiet:
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
                        if not self.quiet:
                            print(f"Warning: Skipping malformed line in {self.qa_dataset_path}")
            if not self.quiet:
                print(f"✅ Reloaded dataset with a total of {len(qa_data)} QA pairs.")

        # --- Final sampling ---
        if len(qa_data) >= num_questions:
            return random.sample(qa_data, num_questions)
        else:
            if not self.quiet:
                print(f"⚠️ Warning: Final dataset size ({len(qa_data)}) is less than requested ({num_questions}). Using all available questions.")
            return qa_data

    def evaluate_single_question(self, qa_item: Dict[str, Any], methods: List[str], k_values: List[int]) -> Dict[str, Any]:
        """
        Evaluates a single question across specified scenarios.
        """
        results = {}
        ground_truth_answer = qa_item["answer"]
        ground_truth_chunk_id = qa_item["chunk_id"]

        # 1. RAG Scenario
        if "rag" in methods:
            rag_answer, retrieved_ids, rag_tokens = run_rag_scenario(self.pipeline, qa_item)
            results["rag"] = {
                "answer": rag_answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_ids=retrieved_ids,
                    true_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacent_map=self.pipeline.adjacent_map,
                    adjacent_credit=0.5
                ),
                "rouge": calculate_rouge_scores(rag_answer, ground_truth_answer),
                "tokens": rag_tokens
            }
        
        # 2. Reranked RAG Scenario
        if "reranked_rag" in methods:
            reranked_answer, reranked_ids, reranked_tokens = run_reranked_rag_scenario(
                pipeline=self.pipeline,
                qa_item=qa_item,
                reranker=self.reranker
            )
            results["reranked_rag"] = {
                "answer": reranked_answer,
                "retrieval": calculate_retrieval_metrics(
                    retrieved_ids=reranked_ids,
                    true_chunk_id=ground_truth_chunk_id,
                    k_values=k_values,
                    adjacent_map=self.pipeline.adjacent_map,
                    adjacent_credit=0.5
                ),
                "rouge": calculate_rouge_scores(reranked_answer, ground_truth_answer),
                "tokens": reranked_tokens
            }

        # 3. Ensemble Reranked RAG Scenario
        if "ensemble_rerank_rag" in methods:
            augmented_question = format_question_with_context(
                qa_item["question"], qa_item["ticker"], qa_item["year"]
            )

            # The scenario now returns a dictionary with answer, contexts, retrieval, and tokens
            scenario_output = ensemble_rerank_rag(
                rag_pipeline=self.pipeline,
                question=augmented_question,
                ground_truth_chunks=[ground_truth_chunk_id],
                k_values=k_values,
            )
            
            # We need to manually calculate ROUGE scores here
            rouge_scores = calculate_rouge_scores(scenario_output["answer"], ground_truth_answer)
            
            results["ensemble_rerank_rag"] = {
                "answer": scenario_output["answer"],
                "retrieval": scenario_output["retrieval"], # The metrics are pre-calculated
                "rouge": rouge_scores,
                "tokens": scenario_output.get("tokens", {}), # Use .get for safety
                "contexts": scenario_output.get("contexts", [])
            }

        # 4. Unfiltered Context Scenario
        if "unfiltered" in methods:
            unfiltered_answer, unfiltered_tokens = run_unfiltered_context_scenario(
                doc_store=self.doc_store,
                openai_client=self.openai_client,
                qa_item=qa_item
            )
            results["unfiltered"] = {
                "answer": unfiltered_answer,
                "rouge": calculate_rouge_scores(unfiltered_answer, ground_truth_answer),
                "tokens": unfiltered_tokens
            }

        # 5. Web Search Scenario
        if "web_search" in methods:
            web_search_answer, web_search_tokens = run_web_search_scenario(self.openai_client, qa_item)
            results["web_search"] = {
                "answer": web_search_answer,
                "rouge": calculate_rouge_scores(web_search_answer, ground_truth_answer),
                "tokens": web_search_tokens
            }
        
        # 6. Baseline Scenario
        if "baseline" in methods:
            baseline_answer, baseline_tokens = run_baseline_scenario(self.openai_client, qa_item)
            results["baseline"] = {
                "answer": baseline_answer,
                "rouge": calculate_rouge_scores(baseline_answer, ground_truth_answer),
                "tokens": baseline_tokens
            }
        
        return results

    def evaluate_all_scenarios(
        self, 
        num_questions: int = 50,
        methods: List[str] = None,
        k_values: List[int] = None,
        resume: bool = True,
        run_id: str = None
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation across all scenarios and aggregates the results.
        """
        if methods is None:
            methods = ["rag", "reranked_rag", "unfiltered", "web_search", "baseline", "ensemble_rerank_rag"]
        if k_values is None:
            k_values = [1, 3, 5, 7, 10]
            
        # Generate run ID if not provided
        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            
        # Pre-load all necessary data before starting the evaluation loop
        if not self.quiet:
            print("Pre-loading all necessary data for evaluation...")
        self.doc_store.get_all_sentences()  # This will load the document store data
        
        self.qa_dataset = self._load_or_generate_qa_dataset(num_questions)
        
        # Try to load existing checkpoint
        all_results = []
        start_index = 0

        if resume:
            checkpoint = self._load_checkpoint(run_id)
            if checkpoint:
                # Validate checkpoint compatibility
                if (checkpoint.get("methods") == methods and 
                    checkpoint.get("k_values") == k_values and
                    len(checkpoint.get("qa_dataset", [])) == len(self.qa_dataset)):
                    
                    all_results = checkpoint["completed_results"]
                    start_index = checkpoint["current_index"]
                    
                    if not self.quiet:
                        print(f"🔄 Resuming from question {start_index + 1}/{len(self.qa_dataset)}")
                else:
                    if not self.quiet:
                        print("⚠️ Checkpoint incompatible with current run parameters. Starting fresh.")

        # Evaluation loop with checkpointing
        progress_bar = tqdm(
            self.qa_dataset[start_index:], 
            desc="🔬 Evaluating scenarios", 
            unit="question",
            disable=self.quiet,
            initial=start_index,
            total=len(self.qa_dataset)
        )

        try:
            for i, qa_item in enumerate(progress_bar, start=start_index):
                single_result = self.evaluate_single_question(qa_item, methods, k_values)
                single_result["question"] = qa_item["question"]
                single_result["ground_truth_answer"] = qa_item["answer"]
                single_result["section"] = qa_item.get("section", "N/A")
                single_result["chunk_text"] = qa_item.get("chunk_text", "N/A")
                all_results.append(single_result)

                # --- Print QA Results ---
                if not self.quiet:
                    augmented_question = format_question_with_context(
                        qa_item["question"], qa_item["ticker"], qa_item["year"]
                    )
                    progress_bar.write("\n" + "="*80)
                    progress_bar.write(f"❓ Question (Original): {single_result['question']}")
                    progress_bar.write(f"❓ Question (Augmented): {augmented_question}")
                    progress_bar.write(f"📄 Ground Truth Section: {qa_item.get('section', 'N/A')}")
                    progress_bar.write(f"💬 Ground Truth Context: {qa_item.get('chunk_text', 'N/A')}")
                    progress_bar.write(f"✅ Ground Truth Answer: {single_result['ground_truth_answer']}")
                    progress_bar.write("-"*80)
                    
                    for scenario in methods:
                        if scenario in single_result:
                            answer = single_result[scenario].get('answer', '[No answer generated]')
                            tokens = single_result[scenario].get('tokens', {})
                            total_tokens = tokens.get('total_tokens', 0)
                            progress_bar.write(f"🤖 {scenario.upper()}: {answer} [Tokens: {total_tokens}]")
                
                    progress_bar.write("="*80)

                # Save checkpoint every 10 questions
                if (i + 1) % 10 == 0:
                    self._save_checkpoint(all_results, self.qa_dataset, i + 1, methods, k_values, run_id)

                time.sleep(2)  # Rate limiting

            # Clean up checkpoint on successful completion
            self._cleanup_checkpoint(run_id)

        except KeyboardInterrupt:
            if not self.quiet:
                print(f"\n⚠️ Evaluation interrupted. Progress saved to checkpoint.")
            self._save_checkpoint(all_results, self.qa_dataset, len(all_results), methods, k_values, run_id)
            raise
        except Exception as e:
            if not self.quiet:
                print(f"\n❌ Evaluation failed: {e}. Progress saved to checkpoint.")
            self._save_checkpoint(all_results, self.qa_dataset, len(all_results), methods, k_values, run_id)
            raise

        # --- Save raw results before aggregation for recovery ---
        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_results_dir = RESULTS_DIR / f"run_{run_timestamp}"
        temp_results_dir.mkdir(parents=True, exist_ok=True)
        raw_results_path = temp_results_dir / "raw_results.json"
        
        if not self.quiet:
            print(f"💾 Saving raw, unprocessed results to {raw_results_path}")
        with open(raw_results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        aggregated_results = self._aggregate_results(all_results)
        return aggregated_results, temp_results_dir

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates results from all evaluation runs.
        """
        aggregated = defaultdict(lambda: defaultdict(list))
        scenario_token_metrics = defaultdict(lambda: defaultdict(list))

        for item_result in results:
            for scenario, metrics in item_result.items():
                # Skip non-scenario keys like 'question', 'ground_truth_answer', etc.
                if scenario in ["question", "ground_truth_answer", "section", "chunk_text"]:
                    continue
                    
                # ROUGE scores
                if "rouge" in metrics:
                    for rouge_type, scores in metrics["rouge"].items():
                        # The value from rouge-score can be a dict of f, p, r
                        # We are interested in the F-score
                        if isinstance(scores, dict) and 'fmeasure' in scores:
                            aggregated[scenario][f"{rouge_type}_f"].append(scores['fmeasure'])
                        elif isinstance(scores, dict) and 'f' in scores: # for backwards compatibility
                            aggregated[scenario][f"{rouge_type}_f"].append(scores['f'])
                        else:
                             # Handle cases where it might just be a float (older format)
                            aggregated[scenario][f"{rouge_type}_f"].append(scores)

                # Retrieval metrics
                if "retrieval" in metrics and metrics["retrieval"]:
                    for metric, value in metrics["retrieval"].items():
                        aggregated[scenario][metric].append(value)
                
                # Token counts - track per scenario
                if "tokens" in metrics and metrics["tokens"]:
                    for token_type, value in metrics["tokens"].items():
                        scenario_token_metrics[scenario][token_type].append(value)
        
        num_questions = len(results)
        final_summary = {}
        for scenario, metrics in aggregated.items():
            # Separate ROUGE and retrieval metrics
            rouge_metrics = {m: np.mean(v) for m, v in metrics.items() if "_f" in m}
            retrieval_metrics = {m: np.mean(v) for m, v in metrics.items() if m in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_7", "recall_at_10", "mrr", "ndcg_at_10", "adj_recall_at_1", "adj_recall_at_3", "adj_recall_at_5", "adj_recall_at_7", "adj_recall_at_10", "adj_mrr"]}
            
            # Get scenario-specific token metrics
            scenario_tokens = scenario_token_metrics.get(scenario, {})
            token_summary = {t: np.mean(v) for t, v in scenario_tokens.items()}
            
            final_summary[scenario] = {
                "rouge": rouge_metrics,
                "retrieval": retrieval_metrics,
                "tokens": token_summary,
                "total_cost": self._calculate_cost(scenario_tokens, scenario, num_questions)
            }
        
        return {
            "summary": final_summary, 
            "individual": results,
            "per_question_metrics": self._extract_per_question_metrics(results)
        }

    def _extract_per_question_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract per-question metrics for bootstrap analysis.
        
        Args:
            results: List of per-question evaluation results
            
        Returns:
            Dictionary mapping method names to metric arrays
        """
        per_question_metrics = defaultdict(lambda: defaultdict(list))
        
        for item_result in results:
            for scenario, metrics in item_result.items():
                # Skip non-scenario keys
                if scenario in ["question", "ground_truth_answer", "section", "chunk_text"]:
                    continue
                
                # Extract ROUGE F-scores
                if "rouge" in metrics and metrics["rouge"]:
                    for rouge_type, scores in metrics["rouge"].items():
                        if isinstance(scores, dict) and 'fmeasure' in scores:
                            per_question_metrics[scenario][f"{rouge_type}_f"].append(scores['fmeasure'])
                        elif isinstance(scores, dict) and 'f' in scores:
                            per_question_metrics[scenario][f"{rouge_type}_f"].append(scores['f'])
                        else:
                            # Handle cases where it might just be a float
                            per_question_metrics[scenario][f"{rouge_type}_f"].append(scores)
                
                # Extract retrieval metrics
                if "retrieval" in metrics and metrics["retrieval"]:
                    for metric, value in metrics["retrieval"].items():
                        per_question_metrics[scenario][metric].append(value)
        
        # Convert to regular dict for JSON serialization
        return {scenario: dict(metrics) for scenario, metrics in per_question_metrics.items()}

    def _calculate_cost(self, token_metrics: Dict[str, List[int]], scenario: str = "", num_questions: int = 1) -> float:
        """
        Calculate the total cost based on token metrics and pricing.
        
        Args:
            token_metrics: Dictionary of token counts (prompt_tokens, completion_tokens, etc.)
            scenario: The scenario name (for special pricing like web_search)
            num_questions: Number of questions evaluated
            
        Returns:
            Total cost in USD
        """
        total_cost = 0.0
        
        # GPT-4o-mini pricing (used for generation and query expansion)
        gpt4_mini_input_price = 0.15 / 1_000_000  # $0.15 per 1M input tokens
        gpt4_mini_output_price = 0.60 / 1_000_000  # $0.60 per 1M output tokens
        
        # Text embedding pricing (used for retrieval)
        embedding_price = 0.02 / 1_000_000  # $0.02 per 1M input tokens
        
        # Web search per-call pricing
        web_search_price = 27.5 / 1_000  # $27.5 per 1K calls
        
        # Calculate GPT-4o-mini costs (generation + query expansion)
        if "prompt_tokens" in token_metrics and token_metrics["prompt_tokens"]:
            avg_prompt_tokens = np.mean(token_metrics["prompt_tokens"])
            total_prompt_tokens = avg_prompt_tokens * num_questions
            total_cost += total_prompt_tokens * gpt4_mini_input_price
            
        if "completion_tokens" in token_metrics and token_metrics["completion_tokens"]:
            avg_completion_tokens = np.mean(token_metrics["completion_tokens"])
            total_completion_tokens = avg_completion_tokens * num_questions
            total_cost += total_completion_tokens * gpt4_mini_output_price
        
        # Add embedding costs (retrieval queries)
        # Each question involves 1 embedding call for the query
        if scenario in ["rag", "reranked_rag", "ensemble_rerank_rag"]:
            # Estimate ~50-100 tokens per query for embedding
            estimated_embedding_tokens = 75 * num_questions
            total_cost += estimated_embedding_tokens * embedding_price
        
        # Add web search per-call costs
        if scenario == "web_search":
            total_cost += num_questions * web_search_price
            
        return total_cost

    def print_results(self, results: Dict[str, Any]):
        """
        Prints the aggregated evaluation results in a formatted way.
        """
        if self.quiet:
            return
            
        print("\n" + "="*50)
        print("          🎉 Comprehensive Evaluation Results 🎉")
        print("="*50 + "\n")

        summary = results.get("summary", {})

        for scenario, scenario_data in sorted(summary.items()):
            print(f"--- Scenario: {scenario.replace('_', ' ').title()} ---")
            
            # Print Retrieval Metrics
            retrieval_metrics = scenario_data.get("retrieval", {})
            rouge_data = scenario_data.get("rouge", {})
            
            # Check if retrieval metrics are in the wrong place (legacy format)
            retrieval_from_rouge = {}
            for k in [1, 3, 5, 7, 10]:
                recall_key = f"recall_at_{k}"
                if recall_key in rouge_data:
                    retrieval_from_rouge[recall_key] = rouge_data[recall_key]
            if "mrr" in rouge_data:
                retrieval_from_rouge["mrr"] = rouge_data["mrr"]
            if "ndcg_at_10" in rouge_data:
                retrieval_from_rouge["ndcg_at_10"] = rouge_data["ndcg_at_10"]
            
            # Use retrieval metrics from the proper place or legacy location
            # Check if retrieval_metrics actually contains retrieval data (not just ROUGE data)
            has_real_retrieval = any(k in retrieval_metrics for k in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_7", "recall_at_10", "mrr", "ndcg_at_10", "adj_recall_at_1", "adj_recall_at_3", "adj_recall_at_5", "adj_recall_at_7", "adj_recall_at_10", "adj_mrr"])
            final_retrieval_metrics = retrieval_metrics if has_real_retrieval else retrieval_from_rouge
            
            if final_retrieval_metrics:
                print("  Retrieval Metrics:")
                
                # Print recall metrics in order
                for k in [1, 3, 5, 7, 10]:
                    recall_key = f"recall_at_{k}"
                    if recall_key in final_retrieval_metrics:
                        print(f"    - Recall@{k}: {final_retrieval_metrics[recall_key]:.2%}")
                
                # Print MRR
                if "mrr" in final_retrieval_metrics:
                    print(f"    - MRR: {final_retrieval_metrics['mrr']:.4f}")
                
                # Print NDCG@10
                if "ndcg_at_10" in final_retrieval_metrics:
                    print(f"    - NDCG@10: {final_retrieval_metrics['ndcg_at_10']:.4f}")
            
            # Print ROUGE Scores
            rouge_metrics = scenario_data.get("rouge", {})
            if rouge_metrics:
                print("\n  Generation Quality (ROUGE-F):")
                for rouge_type in ["rouge1_f", "rouge2_f", "rougeL_f"]:
                    if rouge_type in rouge_metrics:
                        display_name = rouge_type.replace("_f", "").upper()
                        print(f"    - {display_name}: {rouge_metrics[rouge_type]:.4f}")
            
            # Print Token Usage and Cost
            token_data = scenario_data.get("tokens", {})
            total_cost = scenario_data.get("total_cost", 0)
            
            if token_data or total_cost > 0:
                print("\n  Token Usage & Cost:")
                if "prompt_tokens" in token_data:
                    print(f"    - Avg Prompt Tokens: {token_data['prompt_tokens']:.0f}")
                if "completion_tokens" in token_data:
                    print(f"    - Avg Completion Tokens: {token_data['completion_tokens']:.0f}")
                if "total_tokens" in token_data:
                    print(f"    - Avg Total Tokens: {token_data['total_tokens']:.0f}")
                
                if total_cost > 0:
                    print(f"    - Total Cost: ${total_cost:.6f}")
                    # Show cost per question
                    num_questions = len(results.get("individual", []))
                    if num_questions > 0:
                        cost_per_question = total_cost / num_questions
                        print(f"    - Cost per Question: ${cost_per_question:.6f}")
                    
                    # Show cost breakdown if it's ensemble (has query expansion)
                    if scenario == "ensemble_rerank_rag":
                        print(f"    - Note: Includes query expansion + generation + embedding costs")
            
            print("-"*(30 + len(scenario)) + "\n")

        print("="*50 + "\n")

    def results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts the final results dictionary to a pandas DataFrame.
        """
        summary = results.get("summary", {})
        if not summary:
            return pd.DataFrame()

        records = []
        for scenario, metrics in summary.items():
            record = {"scenario": scenario}
            
            # Get base data
            rouge_data = metrics.get("rouge", {})
            retrieval_data = metrics.get("retrieval", {})
            
            # Check for retrieval metrics in rouge section (legacy format)
            legacy_retrieval = {}
            for k in [1, 3, 5, 7, 10]:
                recall_key = f"recall_at_{k}"
                if recall_key in rouge_data:
                    legacy_retrieval[recall_key] = rouge_data[recall_key]
            if "mrr" in rouge_data:
                legacy_retrieval["mrr"] = rouge_data["mrr"]
            if "ndcg_at_10" in rouge_data:
                legacy_retrieval["ndcg_at_10"] = rouge_data["ndcg_at_10"]
            
            # Use retrieval metrics from proper location or legacy location
            # Check if retrieval_data actually contains retrieval metrics (not just ROUGE data)
            has_real_retrieval = any(k in retrieval_data for k in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_7", "recall_at_10", "mrr", "ndcg_at_10", "adj_recall_at_1", "adj_recall_at_3", "adj_recall_at_5", "adj_recall_at_7", "adj_recall_at_10", "adj_mrr"])
            final_retrieval_data = retrieval_data if has_real_retrieval else legacy_retrieval
            
            # Filter out retrieval metrics from ROUGE data for clean separation
            clean_rouge_data = {k: v for k, v in rouge_data.items() if k not in legacy_retrieval}
            
            # Add clean ROUGE metrics
            record.update(clean_rouge_data)
            
            # Add retrieval metrics
            record.update(final_retrieval_data)
            
            # Add token and cost info
            tokens = metrics.get("tokens", {})
            record["avg_prompt_tokens"] = tokens.get("prompt_tokens")
            record["avg_completion_tokens"] = tokens.get("completion_tokens")
            record["avg_total_tokens"] = tokens.get("total_tokens")
            record["total_cost"] = metrics.get("total_cost")
            
            # Calculate cost per question if we have the data
            num_questions = len(results.get("individual", []))
            if record["total_cost"] and num_questions > 0:
                record["cost_per_question"] = record["total_cost"] / num_questions
            
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # Reorder columns for better readability
        cols = ["scenario"]
        
        # ROUGE columns
        rouge_cols = ["rouge1_f", "rouge2_f", "rougeL_f"]
        cols.extend([col for col in rouge_cols if col in df.columns])
        
        # Retrieval columns in logical order
        retrieval_cols = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_7", "recall_at_10", "mrr", "ndcg_at_10", "adj_recall_at_1", "adj_recall_at_3", "adj_recall_at_5", "adj_recall_at_7", "adj_recall_at_10", "adj_mrr"]
        cols.extend([col for col in retrieval_cols if col in df.columns])
        
        # Token and cost columns
        token_cols = ["avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens", "total_cost", "cost_per_question"]
        cols.extend([col for col in token_cols if col in df.columns])
        
        # Ensure all expected columns exist and reorder
        final_cols = [col for col in cols if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in final_cols]
        final_cols.extend(remaining_cols)

        return df[final_cols]

    def export_to_csv(self, results: Dict[str, Any], filename: str):
        """
        Exports the aggregated evaluation results to a CSV file.
        """
        summary = results.get("summary", {})
        if not summary:
            print("No summary results found to export.")
            return

        # Use the results_to_dataframe method to get the data
        df = self.results_to_dataframe(results)
        
        if df.empty:
            print("No data to export to CSV.")
            return

        df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")

    # ============================================================================
    # Checkpoint Methods for Resume Functionality
    # ============================================================================
    
    def _get_checkpoint_path(self, run_id: str = None) -> Path:
        """Get the path for checkpoint file."""
        if run_id is None:
            run_id = "current"
        in_process_dir = RESULTS_DIR / "in_process"
        in_process_dir.mkdir(parents=True, exist_ok=True)
        return in_process_dir / f"checkpoint_{run_id}.json"

    def _save_checkpoint(self, results: List[Dict], qa_dataset: List[Dict], 
                        current_index: int, methods: List[str], k_values: List[int], 
                        run_id: str) -> None:
        """Save evaluation progress to checkpoint file."""
        checkpoint_data = {
            "completed_results": results,
            "qa_dataset": qa_dataset,
            "current_index": current_index,
            "methods": methods,
            "k_values": k_values,
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        checkpoint_path = self._get_checkpoint_path(run_id)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        if not self.quiet:
            print(f"💾 Checkpoint saved: {len(results)}/{len(qa_dataset)} questions completed")

    def _load_checkpoint(self, run_id: str = None) -> Dict[str, Any]:
        """Load evaluation progress from checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(run_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
            
            if not self.quiet:
                completed = len(checkpoint_data.get("completed_results", []))
                total = len(checkpoint_data.get("qa_dataset", []))
                print(f"📂 Found checkpoint: {completed}/{total} questions completed")
            
            return checkpoint_data
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ Error loading checkpoint: {e}")
            return None

    def _cleanup_checkpoint(self, run_id: str) -> None:
        """Remove checkpoint file after successful completion."""
        checkpoint_path = self._get_checkpoint_path(run_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            if not self.quiet:
                print(f"🗑️ Checkpoint cleaned up: {checkpoint_path}") 