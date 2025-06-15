#!/usr/bin/env python3
"""
Test script for the comprehensive evaluator with a single question.
"""

import sys
import os
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator


def test_single_question():
    """Test the evaluator with a single question."""
    
    # Create a sample QA item (matching the structure from normalize_qa_sample.py)
    sample_qa = {
        "chunk_id": "test_chunk_1",
        "ticker": "AAPL",
        "year": 2015,
        "section": "1A",
        "question": "What were Apple's main risk factors in 2015?",
        "answer": "Apple's main risk factors included competition, supply chain dependencies, and regulatory changes.",
        "source_text": "Apple Inc. faces significant risks including intense competition in the technology industry, dependence on third-party suppliers, and potential regulatory changes that could impact operations."
    }
    
    print("🧪 Testing Comprehensive Evaluator with single question...")
    print(f"Question: {sample_qa['question']}")
    print(f"Company: {sample_qa['ticker']}, Year: {sample_qa['year']}")
    
    try:
        evaluator = ComprehensiveEvaluator()
        
        # Test single question evaluation
        result = evaluator.evaluate_single_question(sample_qa)
        
        print("\n✅ Single question evaluation successful!")
        print(f"Scenarios tested: {list(result.keys())}")
        
        # Print sample results
        for scenario in ["unfiltered", "web_search", "rag"]:
            if scenario in result:
                answer = result[scenario]["answer"][:100] + "..." if len(result[scenario]["answer"]) > 100 else result[scenario]["answer"]
                print(f"\n{scenario.upper()} Answer: {answer}")
                if "rouge" in result[scenario]:
                    rouge = result[scenario]["rouge"]
                    print(f"ROUGE-1: {rouge['rouge1']:.3f}, ROUGE-2: {rouge['rouge2']:.3f}, ROUGE-L: {rouge['rougeL']:.3f}")
                if "retrieval" in result[scenario]:
                    retrieval = result[scenario]["retrieval"]
                    print(f"Recall@1: {retrieval['recall_at_1']:.3f}, MRR: {retrieval['mrr']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    success = test_single_question()
    sys.exit(0 if success else 1) 