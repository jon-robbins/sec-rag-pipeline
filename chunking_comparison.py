import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys
from rag.config import RESULTS_DIR
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from evaluation.evaluator import ComprehensiveEvaluator
    
def compare_chunking_configs(num_questions=20, 
                             configs: List[Dict] = [
        {"target_tokens": 150, "overlap_tokens": 25, "name": "Small_150_25"},
        {"target_tokens": 300, "overlap_tokens": 50, "name": "Medium_300_50"},
        {"target_tokens": 500, "overlap_tokens": 100, "name": "Large_500_100"},
        {"target_tokens": 750, "overlap_tokens": 150, "name": "XLarge_750_150"}
        ]
    ):
    """
    Compare chunking configurations using existing pipeline classes.
    """
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n📊 Configuration {i+1}/{len(configs)}: {config['name']}")
        print(f"   Testing: {config['target_tokens']} tokens, {config['overlap_tokens']} overlap")
        
        try:
            # Initialize pipeline with custom chunking
            pipeline = RAGPipeline(
                target_tokens=config['target_tokens'], 
                overlap_tokens=config['overlap_tokens']
            )
            
            # Run evaluation
            evaluator = ComprehensiveEvaluator(pipeline)
            eval_results, temp_dir = evaluator.evaluate_all_scenarios(num_questions=num_questions)
            
            # Clean up temp directory if it exists
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir)
            
            # Extract metrics
            if 'rag' in eval_results.get('summary', {}):
                rag_metrics = eval_results['summary']['rag']
                result = {
                    'configuration': config['name'],
                    'target_tokens': config['target_tokens'],
                    'overlap_tokens': config['overlap_tokens'],
                    'total_chunks': len(pipeline.get_chunks()),
                    'recall_at_1': rag_metrics.get('retrieval', {}).get('recall_at_1', 0),
                    'recall_at_3': rag_metrics.get('retrieval', {}).get('recall_at_3', 0),
                    'recall_at_5': rag_metrics.get('retrieval', {}).get('recall_at_5', 0),
                    'recall_at_10': rag_metrics.get('retrieval', {}).get('recall_at_10', 0),
                    'mrr': rag_metrics.get('retrieval', {}).get('mrr', 0),
                    'rouge1_f': rag_metrics.get('rouge', {}).get('rouge1', {}).get('fmeasure', 0),
                    'rouge2_f': rag_metrics.get('rouge', {}).get('rouge2', {}).get('fmeasure', 0),
                    'rougeL_f': rag_metrics.get('rouge', {}).get('rougeL', {}).get('fmeasure', 0)
                }
                results.append(result)
                
                print(f"   ✅ MRR: {result['mrr']:.3f}, Chunks: {result['total_chunks']}")
            else:
                print("   ⚠️ RAG metrics not found in summary.")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        df_results = pd.DataFrame(results).sort_values('mrr', ascending=False)
        print(f"\n🎯 Chunking Comparison Results:")
        print("="*80)
        print(df_results[['configuration', 'target_tokens', 'overlap_tokens', 'total_chunks',
                         'recall_at_1', 'recall_at_5', 'mrr', 'rouge1_f']].round(4))
        
        best_config = df_results.iloc[0]
        print(f"\n🏆 Best configuration: {best_config['configuration']}")
        print(f"📈 Best MRR: {best_config['mrr']:.4f}")
        
        return df_results
    else:
        print("❌ No valid results generated")
        return pd.DataFrame()

if __name__ == "__main__":
    # Run with 50 questions as requested
    results = compare_chunking_configs(
        num_questions=100,
        configs = [{"target_tokens": 750, "overlap_tokens": 150, "name": "XLarge_750_150"}])
    if results is not None and not results.empty:
        print(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(RESULTS_DIR / f'chunking_comparison_results_{timestamp}.csv', index=False)