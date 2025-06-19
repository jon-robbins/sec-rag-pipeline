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
from evaluation.metrics import calculate_retrieval_metrics, calculate_rouge_scores
from evaluation.generate_qa_dataset import generate_qa_pairs, BalancedChunkSampler
from rag.config import QA_DATASET_PATH
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
            sampler = BalancedChunkSampler()
            grouped_chunks = sampler.group_chunks_by_keys(chunks)
            balanced_chunks = sampler.stratified_sample(grouped_chunks)
            
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
                    retrieved_chunk_ids=retrieved_ids,
                    true_chunk_id=ground_truth_chunk_id,
                    k_values=k_values
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
                    retrieved_chunk_ids=reranked_ids,
                    true_chunk_id=ground_truth_chunk_id,
                    k_values=k_values
                ),
                "rouge": calculate_rouge_scores(reranked_answer, ground_truth_answer),
                "tokens": reranked_tokens
            }

        # 3. Unfiltered Context Scenario
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

        # 4. Web Search Scenario
        if "web_search" in methods:
            web_search_answer, web_search_tokens = run_web_search_scenario(self.openai_client, qa_item)
            results["web_search"] = {
                "answer": web_search_answer,
                "rouge": calculate_rouge_scores(web_search_answer, ground_truth_answer),
                "tokens": web_search_tokens
            }
        
        # 5. Baseline Scenario
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
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation across all scenarios and aggregates the results.
        """
        if methods is None:
            methods = ["rag", "reranked_rag", "unfiltered", "web_search", "baseline"]
        if k_values is None:
            k_values = [1, 3, 5, 10]
            
        # Pre-load all necessary data before starting the evaluation loop
        if not self.quiet:
            print("Pre-loading all necessary data for evaluation...")
        self.doc_store.get_all_sentences()  # This will load the document store data
        
        self.qa_dataset = self._load_or_generate_qa_dataset(num_questions)
        
        all_results = []
        progress_bar = tqdm(
            self.qa_dataset, 
            desc="🔬 Evaluating scenarios", 
            unit="question",
            disable=self.quiet
        )

        for qa_item in progress_bar:
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

            time.sleep(2)  # Add a delay to avoid rate limiting
        
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates metrics from all evaluation runs.
        """
        # Determine scenarios from the first result item
        scenarios = list(results[0].keys())
        scenarios.remove("question")
        scenarios.remove("ground_truth_answer")
        scenarios.remove("section")
        scenarios.remove("chunk_text")

        summary = {
            s: {"retrieval": defaultdict(list), "rouge": defaultdict(list), "tokens": defaultdict(list)}
            for s in scenarios
        }

        for res in results:
            for scenario in scenarios:
                if scenario not in res:
                    continue
                    
                # Rouge scores
                for metric, values in res[scenario].get("rouge", {}).items():
                    summary[scenario]["rouge"][metric].append(values["f"]) # f-measure

                # Retrieval metrics
                if "retrieval" in res[scenario]:
                    for metric, value in res[scenario]["retrieval"].items():
                        summary[scenario]["retrieval"][metric].append(value)
                        
                # Token counts
                for token_type, value in res[scenario].get("tokens", {}).items():
                    summary[scenario]["tokens"][token_type].append(value)
        
        final_summary = {}
        for scenario, metrics in summary.items():
            final_summary[scenario] = {
                "rouge": {m: np.mean(v) for m, v in metrics["rouge"].items()},
                "retrieval": {m: np.mean(v) for m, v in metrics["retrieval"].items()},
                "tokens": {t: np.mean(v) for t, v in metrics["tokens"].items()},
                "total_cost": self._calculate_cost(metrics["tokens"])
            }
        
        return {"summary": final_summary, "individual": results}

    def _calculate_cost(self, token_metrics: Dict[str, List[int]]) -> float:
        # This method needs to be implemented to calculate the total cost
        # based on the token metrics.
        # For now, we'll return a placeholder value
        return 0.0

    def print_results(self, results: Dict[str, Any]):
        """
        Prints the aggregated evaluation results in a formatted table.
        """
        summary = results.get("summary", {})
        if not summary:
            print("No summary found in results.")
            return

        print("\n" + "="*20 + " 📊 Evaluation Summary " + "="*20)
        
        # Determine k values from the first RAG result if available
        k_values = []
        if "rag" in summary and "retrieval" in summary["rag"]:
            k_values = sorted([int(k.split('@')[1]) for k in summary["rag"]["retrieval"].keys() if k.startswith('recall@')])

        header = ["Scenario", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        if k_values:
            header.extend([f"Recall@{k}" for k in k_values])
            header.append("MRR")
        header.extend(["Avg Tokens", "Total Cost ($)"])
        
        print(f"{header[0]:<15} | {header[1]:<10} | {header[2]:<10} | {header[3]:<10} | " +
              " | ".join([f"{h:<8}" for h in header[4:-2]]) + 
              (" | " if k_values else "") +
              f"{header[-2]:<12} | {header[-1]:<15}")
        print("-" * (len(header) * 14))

        for scenario, metrics in summary.items():
            rouge = metrics.get("rouge", {})
            retrieval = metrics.get("retrieval", {})
            tokens = metrics.get("tokens", {})
            
            row = [
                scenario,
                f"{rouge.get('rouge1', 0):.4f}",
                f"{rouge.get('rouge2', 0):.4f}",
                f"{rouge.get('rougeL', 0):.4f}"
            ]
            
            if k_values:
                for k in k_values:
                    row.append(f"{retrieval.get(f'recall@{k}', 0):.4f}")
                row.append(f"{retrieval.get('mrr', 0):.4f}")

            row.append(f"{tokens.get('total_tokens', 0):.2f}")
            row.append(f"{metrics.get('total_cost', 0):.4f}")
            
            print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10} | " +
                  " | ".join([f"{str(v):<8}" for v in row[4:-2]]) +
                  (" | " if k_values else "") +
                  f"{row[-2]:<12} | {row[-1]:<15}")
                  
        print("="*55)
    
    def results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts the summary results into a pandas DataFrame.
        """
        summary = results.get("summary", {})
        if not summary:
            return pd.DataFrame()

        records = []
        for scenario, metrics in summary.items():
            record = {"scenario": scenario}
            record.update(metrics.get("rouge", {}))
            record.update(metrics.get("retrieval", {}))
            
            # Add token and cost info
            tokens = metrics.get("tokens", {})
            record["avg_prompt_tokens"] = tokens.get("prompt_tokens")
            record["avg_completion_tokens"] = tokens.get("completion_tokens")
            record["avg_total_tokens"] = tokens.get("total_tokens")
            record["total_cost"] = metrics.get("total_cost")
            
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # Reorder columns for better readability
        cols = ["scenario", "rouge1", "rouge2", "rougeL"]
        retrieval_cols = sorted([col for col in df.columns if col.startswith("recall@") or col == "mrr"])
        cols.extend(retrieval_cols)
        token_cols = ["avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens", "total_cost"]
        cols.extend(token_cols)
        
        # Ensure all expected columns exist
        final_cols = [col for col in cols if col in df.columns]

        return df[final_cols]

    def export_to_csv(self, results: Dict[str, Any], filename: str):
        """
        Exports the individual evaluation results to a detailed CSV file.
        """
        if not results.get("individual"):
            print("No individual results found in the provided results.")
            return

        detailed_results = results["individual"]
        with open(filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header
            header = [
                "Question", "Ground Truth Answer", "Ground Truth Section", "Ground Truth Context",
                "RAG Answer", "RAG Prompt Tokens", "RAG Completion Tokens", "RAG Total Tokens",
                "RAG ROUGE-1 F1", "RAG ROUGE-2 F1", "RAG ROUGE-L F1", "RAG Recall@1", "RAG Recall@3", "RAG MRR",
                "Reranked RAG Answer", "Reranked RAG Prompt Tokens", "Reranked RAG Completion Tokens", "Reranked RAG Total Tokens",
                "Reranked RAG ROUGE-1 F1", "Reranked RAG ROUGE-2 F1", "Reranked RAG ROUGE-L F1", "Reranked RAG Recall@1", "Reranked RAG Recall@3", "Reranked RAG MRR",
                "Unfiltered Answer", "Unfiltered Prompt Tokens", "Unfiltered Completion Tokens", "Unfiltered Total Tokens",
                "Unfiltered ROUGE-1 F1", "Unfiltered ROUGE-2 F1", "Unfiltered ROUGE-L F1",
                "Web Search Answer", "Web Search Prompt Tokens", "Web Search Completion Tokens", "Web Search Total Tokens",
                "Web Search ROUGE-1 F1", "Web Search ROUGE-2 F1", "Web Search ROUGE-L F1",
                "Baseline Answer", "Baseline Prompt Tokens", "Baseline Completion Tokens", "Baseline Total Tokens",
                "Baseline ROUGE-1 F1", "Baseline ROUGE-2 F1", "Baseline ROUGE-L F1"
            ]
            csvwriter.writerow(header)
            
            # Write data rows
            for res in detailed_results:
                row = [
                    res["question"],
                    res["ground_truth_answer"],
                    res.get("section", "N/A"),
                    res.get("chunk_text", "N/A"),
                    
                    # RAG metrics
                    res["rag"]["answer"],
                    res["rag"]["tokens"]["prompt_tokens"],
                    res["rag"]["tokens"]["completion_tokens"],
                    res["rag"]["tokens"]["total_tokens"],
                    res["rag"]["rouge"]["rouge1"]["fmeasure"],
                    res["rag"]["rouge"]["rouge2"]["fmeasure"],
                    res["rag"]["rouge"]["rougeL"]["fmeasure"],
                    res["rag"]["retrieval"]["recall_at_1"],
                    res["rag"]["retrieval"]["recall_at_3"],
                    res["rag"]["retrieval"]["mrr"],
                    
                    # Reranked RAG metrics
                    res["reranked_rag"]["answer"],
                    res["reranked_rag"]["tokens"]["prompt_tokens"],
                    res["reranked_rag"]["tokens"]["completion_tokens"],
                    res["reranked_rag"]["tokens"]["total_tokens"],
                    res["reranked_rag"]["rouge"]["rouge1"]["fmeasure"],
                    res["reranked_rag"]["rouge"]["rouge2"]["fmeasure"],
                    res["reranked_rag"]["rouge"]["rougeL"]["fmeasure"],
                    res["reranked_rag"]["retrieval"]["recall_at_1"],
                    res["reranked_rag"]["retrieval"]["recall_at_3"],
                    res["reranked_rag"]["retrieval"]["mrr"],
                    
                    # Unfiltered metrics
                    res["unfiltered"]["answer"],
                    res["unfiltered"]["tokens"]["prompt_tokens"],
                    res["unfiltered"]["tokens"]["completion_tokens"],
                    res["unfiltered"]["tokens"]["total_tokens"],
                    res["unfiltered"]["rouge"]["rouge1"]["fmeasure"],
                    res["unfiltered"]["rouge"]["rouge2"]["fmeasure"],
                    res["unfiltered"]["rouge"]["rougeL"]["fmeasure"],
                    
                    # Web Search metrics
                    res["web_search"]["answer"],
                    res["web_search"]["tokens"]["prompt_tokens"],
                    res["web_search"]["tokens"]["completion_tokens"],
                    res["web_search"]["tokens"]["total_tokens"],
                    res["web_search"]["rouge"]["rouge1"]["fmeasure"],
                    res["web_search"]["rouge"]["rouge2"]["fmeasure"],
                    res["web_search"]["rouge"]["rougeL"]["fmeasure"],
                    
                    # Baseline metrics
                    res["baseline"]["answer"],
                    res["baseline"]["tokens"]["prompt_tokens"],
                    res["baseline"]["tokens"]["completion_tokens"],
                    res["baseline"]["tokens"]["total_tokens"],
                    res["baseline"]["rouge"]["rouge1"]["fmeasure"],
                    res["baseline"]["rouge"]["rouge2"]["fmeasure"],
                    res["baseline"]["rouge"]["rougeL"]["fmeasure"],
                ]
                csvwriter.writerow(row)

        print(f"✅ Results exported to {filename}") 