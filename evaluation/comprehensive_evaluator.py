#!/usr/bin/env python3
"""
Comprehensive Evaluation System for SEC RAG Pipeline

Compares three scenarios:
1. GPT-4.0 UnfilteredContext - Uses entire SEC filing as context
2. GPT-4.0 WebSearch - Uses no additional context (web knowledge only)
3. RAGPipeline - Uses semantic search with relevant chunks

Metrics: Recall@K, Mean Reciprocal Rank (MRR), ROUGE-1/2/L
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import statistics

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import create_vector_store
from rag.openai_helpers import UsageCostCalculator
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge-score...")
    os.system("pip install rouge-score")
    from rouge_score import rouge_scorer


class ComprehensiveEvaluator:
    """Evaluates three different QA approaches with comprehensive metrics."""
    
    def __init__(self, evaluation_file: str = "data/qa_dataset.jsonl"):
        self.evaluation_file = evaluation_file
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.cost_calculator = UsageCostCalculator()
        
        # Initialize RAG pipeline
        print("🔧 Initializing RAG pipeline...")
        self.rag_vs = create_vector_store(use_docker=False, verbose=False)
        
        # ROUGE scorer for answer quality
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load evaluation dataset
        self.eval_data = self._load_evaluation_data()
        print(f"📊 Loaded {len(self.eval_data)} evaluation questions")
        
    def _load_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load the QA evaluation dataset."""
        data = []
        with open(self.evaluation_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def gpt4_unfiltered_context(self, question: str, source_text: str) -> str:
        """Scenario 1: GPT-4 with entire SEC filing as context."""
        system_prompt = """You are a financial analyst. Answer the question based ONLY on the provided SEC filing context. Be accurate and concise."""
        
        user_prompt = f"""Context: {source_text}

Question: {question}

Answer:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def gpt4_web_search(self, question: str) -> str:
        """Scenario 2: GPT-4 with no additional context (web knowledge only)."""
        system_prompt = """You are a financial analyst. Answer the question using your knowledge of publicly available financial information. Be accurate and concise."""
        
        user_prompt = f"""Question: {question}

Answer:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def rag_pipeline(self, question: str, ticker: str = None, year: int = None, section: str = None) -> str:
        """Scenario 3: RAG pipeline with semantic search."""
        # Search for relevant chunks
        chunks = self.rag_vs.search(
            query=question,
            ticker=ticker,
            fiscal_year=year,
            sections=[section] if section else None,
            top_k=10
        )
        
        # Generate answer using retrieved chunks
        result = self.rag_vs.answer(
            question=question,
            ticker=ticker,
            fiscal_year=year,
            sections=[section] if section else None
        )
        
        return result["answer"]
    
    def compute_retrieval_metrics(self, question: str, correct_chunk_id: str, ticker: str = None, year: int = None, section: str = None) -> Dict[str, float]:
        """Compute Recall@K and MRR for retrieval quality."""
        # Get retrieved chunks
        chunks = self.rag_vs.search(
            query=question,
            ticker=ticker,
            fiscal_year=year,
            sections=[section] if section else None,
            top_k=10
        )
        
        retrieved_ids = [chunk.get('id', '') for chunk in chunks]
        
        # Find position of correct chunk
        correct_position = None
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id == correct_chunk_id:
                correct_position = i + 1  # 1-indexed
                break
        
        # Compute metrics
        recall_at_1 = 1.0 if correct_position == 1 else 0.0
        recall_at_3 = 1.0 if correct_position and correct_position <= 3 else 0.0
        recall_at_5 = 1.0 if correct_position and correct_position <= 5 else 0.0
        recall_at_10 = 1.0 if correct_position and correct_position <= 10 else 0.0
        
        mrr = 1.0 / correct_position if correct_position else 0.0
        
        return {
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "mrr": mrr,
            "retrieved_position": correct_position
        }
    
    def compute_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores for answer quality."""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    
    def evaluate_single_question(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question across all three scenarios."""
        question = qa_item["question"]
        reference_answer = qa_item["answer"]
        source_text = qa_item["source_text"]
        chunk_id = qa_item["chunk_id"]
        ticker = qa_item.get("ticker")
        year = qa_item.get("year")
        section = qa_item.get("section")
        
        results = {"question": question, "reference": reference_answer}
        
        try:
            # Scenario 1: Unfiltered Context
            unfiltered_answer = self.gpt4_unfiltered_context(question, source_text)
            unfiltered_rouge = self.compute_rouge_scores(unfiltered_answer, reference_answer)
            results["unfiltered"] = {
                "answer": unfiltered_answer,
                "rouge": unfiltered_rouge
            }
            
            # Scenario 2: Web Search (No Context)
            web_answer = self.gpt4_web_search(question)
            web_rouge = self.compute_rouge_scores(web_answer, reference_answer)
            results["web_search"] = {
                "answer": web_answer,
                "rouge": web_rouge
            }
            
            # Scenario 3: RAG Pipeline
            rag_answer = self.rag_pipeline(question, ticker, year, section)
            rag_rouge = self.compute_rouge_scores(rag_answer, reference_answer)
            rag_retrieval = self.compute_retrieval_metrics(question, chunk_id, ticker, year, section)
            results["rag"] = {
                "answer": rag_answer,
                "rouge": rag_rouge,
                "retrieval": rag_retrieval
            }
            
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Error evaluating question: {e}")
        
        return results
    
    def evaluate_all_scenarios(self, max_questions: int = 50) -> Dict[str, Any]:
        """Evaluate all scenarios and compute aggregate metrics."""
        print(f"🚀 Starting comprehensive evaluation (max {max_questions} questions)...")
        
        # Select subset for evaluation (balanced across companies)
        eval_subset = self.eval_data[:max_questions]
        
        all_results = []
        unfiltered_rouge = {"rouge1": [], "rouge2": [], "rougeL": []}
        web_rouge = {"rouge1": [], "rouge2": [], "rougeL": []}
        rag_rouge = {"rouge1": [], "rouge2": [], "rougeL": []}
        rag_retrieval = {"recall_at_1": [], "recall_at_3": [], "recall_at_5": [], "recall_at_10": [], "mrr": []}
        
        for i, qa_item in enumerate(eval_subset):
            print(f"📝 Evaluating question {i+1}/{len(eval_subset)}: {qa_item['question'][:60]}...")
            
            result = self.evaluate_single_question(qa_item)
            all_results.append(result)
            
            # Aggregate metrics
            if "error" not in result:
                # ROUGE scores
                for metric in ["rouge1", "rouge2", "rougeL"]:
                    unfiltered_rouge[metric].append(result["unfiltered"]["rouge"][metric])
                    web_rouge[metric].append(result["web_search"]["rouge"][metric])
                    rag_rouge[metric].append(result["rag"]["rouge"][metric])
                
                # Retrieval metrics (only for RAG)
                for metric in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"]:
                    rag_retrieval[metric].append(result["rag"]["retrieval"][metric])
        
        # Compute aggregate scores
        summary = {
            "total_questions": len(eval_subset),
            "successful_evaluations": len([r for r in all_results if "error" not in r]),
            "scenarios": {
                "unfiltered_context": {
                    "rouge1": statistics.mean(unfiltered_rouge["rouge1"]),
                    "rouge2": statistics.mean(unfiltered_rouge["rouge2"]),
                    "rougeL": statistics.mean(unfiltered_rouge["rougeL"])
                },
                "web_search": {
                    "rouge1": statistics.mean(web_rouge["rouge1"]),
                    "rouge2": statistics.mean(web_rouge["rouge2"]),
                    "rougeL": statistics.mean(web_rouge["rougeL"])
                },
                "rag_pipeline": {
                    "rouge1": statistics.mean(rag_rouge["rouge1"]),
                    "rouge2": statistics.mean(rag_rouge["rouge2"]),
                    "rougeL": statistics.mean(rag_rouge["rougeL"]),
                    "recall_at_1": statistics.mean(rag_retrieval["recall_at_1"]),
                    "recall_at_3": statistics.mean(rag_retrieval["recall_at_3"]),
                    "recall_at_5": statistics.mean(rag_retrieval["recall_at_5"]),
                    "recall_at_10": statistics.mean(rag_retrieval["recall_at_10"]),
                    "mrr": statistics.mean(rag_retrieval["mrr"])
                }
            }
        }
        
        return {"summary": summary, "detailed_results": all_results}
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a clean format."""
        summary = results["summary"]
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        print(f"📝 Questions Evaluated: {summary['successful_evaluations']}/{summary['total_questions']}")
        
        print("\n🎯 SCENARIO COMPARISON:")
        print("-" * 50)
        
        scenarios = summary["scenarios"]
        
        # ROUGE Comparison Table
        print(f"{'Scenario':<20} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
        print("-" * 50)
        print(f"{'Unfiltered Context':<20} {scenarios['unfiltered_context']['rouge1']:<10.3f} {scenarios['unfiltered_context']['rouge2']:<10.3f} {scenarios['unfiltered_context']['rougeL']:<10.3f}")
        print(f"{'Web Search':<20} {scenarios['web_search']['rouge1']:<10.3f} {scenarios['web_search']['rouge2']:<10.3f} {scenarios['web_search']['rougeL']:<10.3f}")
        print(f"{'RAG Pipeline':<20} {scenarios['rag_pipeline']['rouge1']:<10.3f} {scenarios['rag_pipeline']['rouge2']:<10.3f} {scenarios['rag_pipeline']['rougeL']:<10.3f}")
        
        # RAG Retrieval Metrics
        print(f"\n🔍 RAG RETRIEVAL METRICS:")
        print("-" * 50)
        rag = scenarios['rag_pipeline']
        print(f"Recall@1:  {rag['recall_at_1']:.3f}")
        print(f"Recall@3:  {rag['recall_at_3']:.3f}")
        print(f"Recall@5:  {rag['recall_at_5']:.3f}")
        print(f"Recall@10: {rag['recall_at_10']:.3f}")
        print(f"MRR:       {rag['mrr']:.3f}")
        
        # Winner Analysis
        print(f"\n🏆 BEST PERFORMING:")
        print("-" * 50)
        rouge1_winner = max(scenarios.keys(), key=lambda x: scenarios[x]['rouge1'])
        rouge2_winner = max(scenarios.keys(), key=lambda x: scenarios[x]['rouge2'])
        rougeL_winner = max(scenarios.keys(), key=lambda x: scenarios[x]['rougeL'])
        
        print(f"ROUGE-1: {rouge1_winner.replace('_', ' ').title()} ({scenarios[rouge1_winner]['rouge1']:.3f})")
        print(f"ROUGE-2: {rouge2_winner.replace('_', ' ').title()} ({scenarios[rouge2_winner]['rouge2']:.3f})")
        print(f"ROUGE-L: {rougeL_winner.replace('_', ' ').title()} ({scenarios[rougeL_winner]['rougeL']:.3f})")


def main():
    """Run comprehensive evaluation."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return 1
    
    evaluator = ComprehensiveEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_all_scenarios(max_questions=20)  # Start with 20 for testing
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results
    output_file = "evaluation/comprehensive_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 