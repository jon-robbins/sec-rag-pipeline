#!/usr/bin/env python3
"""
Quick test script for the comprehensive evaluator.
Tests with just 2-3 questions to verify everything works.
"""

import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator


def test_comprehensive_evaluator():
    """Test the evaluator with a small sample."""
    print("🧪 Testing Comprehensive Evaluator...")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping test")
        return
    
    # Check if QA dataset exists
    qa_file = "data/qa_dataset.jsonl"
    if not os.path.exists(qa_file):
        print(f"❌ {qa_file} not found, run evaluation/generate_qa_dataset.py first")
        return
    
    try:
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator()
        
        # Test with just 2 questions
        print("📝 Running evaluation with 2 questions...")
        results = evaluator.evaluate_all_scenarios(max_questions=2)
        
        # Print results
        evaluator.print_results(results)
        
        print("✅ Comprehensive evaluator test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_comprehensive_evaluator() 