#!/usr/bin/env python3
"""
Generate and evaluate multiple chunking configurations with their corresponding QA datasets.

This script creates 4 different chunking configurations:
1. 150 tokens, 50 overlap, 500 cutoff
2. 350 tokens, 100 overlap, 800 cutoff  
3. 500 tokens, 100 overlap, 800 cutoff
4. 750 tokens, 150 overlap, 1000 cutoff

For each configuration:
- Generates chunks and embeddings
- Creates a 50-question QA dataset
- Evaluates the configuration on the dataset
- Saves questions and results to CSV files
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import shutil

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.pipeline import RAGPipeline
from evaluation.evaluator import ComprehensiveEvaluator
from rag.config import RESULTS_DIR

def generate_and_evaluate_config(config: Dict, config_num: int, total_configs: int):
    """
    Generate chunks, QA dataset, and evaluate a single configuration.
    
    Args:
        config: Dictionary with target_tokens, overlap_tokens, hard_ceiling, name
        config_num: Current configuration number (1-based)
        total_configs: Total number of configurations
    """
    
    print(f"\n{'='*80}")
    print(f"📊 Configuration {config_num}/{total_configs}: {config['name']}")
    print(f"   Target: {config['target_tokens']} tokens")
    print(f"   Overlap: {config['overlap_tokens']} tokens") 
    print(f"   Cutoff: {config['hard_ceiling']} tokens")
    print(f"{'='*80}")
    
    try:
        # 1. Initialize pipeline with custom chunking
        print("🔧 Initializing RAG pipeline with custom chunking...")
        pipeline = RAGPipeline(
            target_tokens=config['target_tokens'], 
            overlap_tokens=config['overlap_tokens'],
            hard_ceiling=config['hard_ceiling']
        )
        
        print(f"✅ Pipeline initialized with {len(pipeline.chunks)} chunks")
        
        # 2. Generate QA dataset (50 questions)
        print("📝 Generating 50-question QA dataset...")
        evaluator = ComprehensiveEvaluator(pipeline, quiet=False)
        qa_dataset = evaluator._load_or_generate_qa_dataset(num_questions=50)
        
        # 3. Save QA dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        qa_filename = RESULTS_DIR / f"qa_dataset_{config['name']}_{timestamp}.json"
        with open(qa_filename, 'w') as f:
            json.dump(qa_dataset, f, indent=2)
        print(f"💾 Saved QA dataset: {qa_filename}")
        
        # 4. Run evaluation
        print("🔬 Running evaluation...")
        eval_results, temp_dir = evaluator.evaluate_all_scenarios(
            num_questions=50,
            methods=['rag', 'reranked_rag', 'ensemble_rerank_rag']
        )
        
        # Clean up temp directory
        if temp_dir:
            shutil.rmtree(temp_dir)
        
        # 5. Extract and structure results
        results_row = {
            'configuration': config['name'],
            'target_tokens': config['target_tokens'],
            'overlap_tokens': config['overlap_tokens'],
            'hard_ceiling': config['hard_ceiling'],
            'total_chunks': len(pipeline.chunks),
            'timestamp': timestamp
        }
        
        # Extract metrics for each method
        summary = eval_results.get('summary', {})
        for method in ['rag', 'reranked_rag', 'ensemble_rerank_rag']:
            if method in summary:
                method_data = summary[method]
                
                # Retrieval metrics
                retrieval = method_data.get('retrieval', {})
                results_row[f'{method}_recall_at_1'] = retrieval.get('recall_at_1', 0)
                results_row[f'{method}_recall_at_3'] = retrieval.get('recall_at_3', 0)
                results_row[f'{method}_recall_at_5'] = retrieval.get('recall_at_5', 0)
                results_row[f'{method}_recall_at_10'] = retrieval.get('recall_at_10', 0)
                results_row[f'{method}_mrr'] = retrieval.get('mrr', 0)
                results_row[f'{method}_ndcg_at_10'] = retrieval.get('ndcg_at_10', 0)
                
                # Adjacency-aware metrics
                results_row[f'{method}_adj_recall_at_1'] = retrieval.get('adj_recall_at_1', 0)
                results_row[f'{method}_adj_recall_at_3'] = retrieval.get('adj_recall_at_3', 0)
                results_row[f'{method}_adj_recall_at_5'] = retrieval.get('adj_recall_at_5', 0)
                results_row[f'{method}_adj_recall_at_10'] = retrieval.get('adj_recall_at_10', 0)
                results_row[f'{method}_adj_mrr'] = retrieval.get('adj_mrr', 0)
                
                # ROUGE metrics
                rouge = method_data.get('rouge', {})
                results_row[f'{method}_rouge1_f'] = rouge.get('rouge1_f', 0)
                results_row[f'{method}_rouge2_f'] = rouge.get('rouge2_f', 0)
                results_row[f'{method}_rougeL_f'] = rouge.get('rougeL_f', 0)
                
                # Token and cost metrics
                tokens = method_data.get('tokens', {})
                results_row[f'{method}_avg_prompt_tokens'] = tokens.get('prompt_tokens', 0)
                results_row[f'{method}_avg_completion_tokens'] = tokens.get('completion_tokens', 0)
                results_row[f'{method}_avg_total_tokens'] = tokens.get('total_tokens', 0)
                results_row[f'{method}_total_cost'] = method_data.get('total_cost', 0)
        
        # 6. Save evaluation results
        eval_csv_filename = RESULTS_DIR / f"evaluation_{config['name']}_{timestamp}.csv"
        results_df = pd.DataFrame([results_row])
        results_df.to_csv(eval_csv_filename, index=False)
        print(f"📊 Saved evaluation results: {eval_csv_filename}")
        
        # 7. Print summary
        print(f"\n📈 Configuration Summary:")
        print(f"   Chunks: {results_row['total_chunks']}")
        if 'rag_mrr' in results_row:
            print(f"   RAG MRR: {results_row['rag_mrr']:.4f}")
        if 'ensemble_rerank_rag_mrr' in results_row:
            print(f"   Ensemble MRR: {results_row['ensemble_rerank_rag_mrr']:.4f}")
        
        return results_row
        
    except Exception as e:
        print(f"❌ Error processing configuration {config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run all configurations."""
    
    # Define the 4 configurations as requested
    configs = [
        # {
        #     "target_tokens": 150, 
        #     "overlap_tokens": 50, 
        #     "hard_ceiling": 500, 
        #     "name": "Small_150_50_500"
        # },
        {
            "target_tokens": 350, 
            "overlap_tokens": 100, 
            "hard_ceiling": 800, 
            "name": "Medium_350_100_800"
        },
        {
            "target_tokens": 500, 
            "overlap_tokens": 100, 
            "hard_ceiling": 800, 
            "name": "Large_500_100_800"
        },
        {
            "target_tokens": 750, 
            "overlap_tokens": 150, 
            "hard_ceiling": 1000, 
            "name": "XLarge_750_150_1000"
        }
    ]
    
    print("🚀 Starting chunking configuration comparison")
    print(f"📋 Will process {len(configs)} configurations with 50 questions each")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Process each configuration
    for i, config in enumerate(configs, 1):
        result = generate_and_evaluate_config(config, i, len(configs))
        if result:
            all_results.append(result)
    
    # Create combined results file
    if all_results:
        print(f"\n🎯 Creating combined results file...")
        combined_df = pd.DataFrame(all_results)
        
        # Sort by best ensemble MRR performance
        if 'ensemble_rerank_rag_mrr' in combined_df.columns:
            combined_df = combined_df.sort_values('ensemble_rerank_rag_mrr', ascending=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = RESULTS_DIR / f"chunking_comparison_all_configs_{timestamp}.csv"
        combined_df.to_csv(combined_filename, index=False)
        
        print(f"📊 Combined results saved: {combined_filename}")
        
        # Print final summary
        print(f"\n🏆 FINAL RESULTS SUMMARY")
        print("="*80)
        summary_cols = ['configuration', 'total_chunks', 'rag_mrr', 'ensemble_rerank_rag_mrr']
        available_cols = [col for col in summary_cols if col in combined_df.columns]
        print(combined_df[available_cols].round(4).to_string(index=False))
        
        if 'ensemble_rerank_rag_mrr' in combined_df.columns:
            best_config = combined_df.iloc[0]
            print(f"\n🥇 Best configuration: {best_config['configuration']}")
            print(f"📈 Best Ensemble MRR: {best_config['ensemble_rerank_rag_mrr']:.4f}")
            print(f"📦 Total chunks: {best_config['total_chunks']}")
    
    print(f"\n✅ All configurations completed!")

if __name__ == "__main__":
    main() 