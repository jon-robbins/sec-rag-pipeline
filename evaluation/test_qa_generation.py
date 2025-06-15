#!/usr/bin/env python3
"""
Quick test script to verify QA generation works correctly.
"""

import os
import sys
import pickle
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.normalize_qa_sample import generate_qa_pairs


def test_qa_generation():
    """Test QA generation with a small sample."""
    print("🧪 Testing QA generation...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping test")
        return
    
    # Load chunks
    chunks_path = Path("data/chunks.pkl")
    if not chunks_path.exists():
        print(f"❌ {chunks_path} not found")
        return
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    # Test with just 2 chunks
    test_chunks = chunks[:2]
    print(f"🎯 Testing with {len(test_chunks)} chunks")
    
    for i, chunk in enumerate(test_chunks):
        print(f"   {i+1}. {chunk.id} ({chunk.metadata['ticker']}) - {len(chunk.text)} chars")
    
    # Generate QA pairs
    output_path = "test_qa_output.jsonl"
    try:
        generate_qa_pairs(test_chunks, output_path)
        
        # Check results
        if Path(output_path).exists():
            with open(output_path, "r") as f:
                lines = f.readlines()
            
            print(f"✅ Generated {len(lines)} QA pairs")
            
            # Show first QA pair
            if lines:
                qa = json.loads(lines[0])
                print(f"\n📖 Sample QA:")
                print(f"   Chunk: {qa['chunk_id']}")
                print(f"   Question: {qa['question']}")
                print(f"   Answer: {qa['answer'][:200]}...")
            
            # Clean up test file
            Path(output_path).unlink()
            print("🧹 Cleaned up test file")
        else:
            print("❌ No output file generated")
            
    except Exception as e:
        print(f"❌ QA generation failed: {e}")


if __name__ == "__main__":
    test_qa_generation() 