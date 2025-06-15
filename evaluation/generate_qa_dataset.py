#!/usr/bin/env python3
"""
Generate a balanced QA dataset from SEC filing chunks.

This script:
1. Loads chunks from data/chunks.pkl
2. Creates a stratified sample balanced across companies, years, sections, and chunk sizes
3. Generates QA pairs using LangChain
4. Saves the results to data/qa_dataset.jsonl

Usage:
    python evaluation/generate_qa_dataset.py
"""

import os
import sys
import pickle
from pathlib import Path
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.normalize_qa_sample import BalancedChunkSampler, generate_qa_pairs, classify_chunk_by_tokens


def main():
    """Main function to generate QA dataset."""
    print("🚀 Starting QA dataset generation...")
    
    # Check if required packages are installed
    try:
        import langchain
        import langchain_openai
        print("✅ LangChain packages found")
    except ImportError:
        print("❌ LangChain not installed. Please install with:")
        print("   pip install langchain langchain-openai")
        return 1
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to generate QA pairs.")
        return 1
    
    # Load chunks
    chunks_path = Path("data/chunks.pkl")
    if not chunks_path.exists():
        print(f"❌ {chunks_path} not found. Please make sure the file exists.")
        return 1
    
    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"📊 Loaded {len(chunks)} chunks from {chunks_path}")
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return 1
    
    # Create balanced sample with company balancing enabled
    print("🎯 Creating balanced sample with company balancing...")
    sampler = BalancedChunkSampler(max_per_group=3, balance_companies=True)  # Smaller sample for QA generation
    grouped = sampler.group_chunks_by_keys(chunks)
    balanced_chunks = sampler.stratified_sample(grouped)
    
    print(f"📊 Balanced sample stats:")
    print(f"   Total chunks: {len(balanced_chunks)}")
    print(f"   Groups represented: {len(grouped)}")
    
    # Show sample distribution
    company_dist = Counter(chunk.metadata["ticker"] for chunk in balanced_chunks)
    chunk_class_dist = Counter()
    
    for chunk in balanced_chunks:
        chunk_class = classify_chunk_by_tokens(chunk.text)
        chunk_class_dist[chunk_class] += 1
    
    print(f"📈 Final distribution after balancing:")
    print(f"   By company: {dict(company_dist)}")
    print(f"   By chunk class: {dict(chunk_class_dist)}")
    
    # Verify balance
    company_counts = list(company_dist.values())
    if len(set(company_counts)) == 1:
        print("✅ Perfect company balance achieved!")
    else:
        min_count, max_count = min(company_counts), max(company_counts)
        print(f"⚖️ Company balance: {min_count}-{max_count} chunks per company")
    
    # Save balanced chunks for reference
    balanced_chunks_path = Path("data/balanced_chunks_for_eval.jsonl")
    sampler.save_chunks_to_jsonl(balanced_chunks, balanced_chunks_path)
    print(f"💾 Saved balanced chunks to {balanced_chunks_path}")
    
    # Generate QA pairs
    print("🤖 Generating QA pairs with improved ChatOpenAI approach...")
    qa_output_path = "data/qa_dataset.jsonl"
    
    try:
        generate_qa_pairs(balanced_chunks, qa_output_path)
    except Exception as e:
        print(f"❌ QA generation failed: {e}")
        return 1
    
    # Check results
    qa_path = Path(qa_output_path)
    if qa_path.exists():
        with open(qa_path, "r") as f:
            lines = f.readlines()
        print(f"✅ Generated {len(lines)} QA pairs")
        print(f"📄 QA dataset saved to: {qa_path}")
        
        # Show sample QA pair
        if lines:
            import json
            sample_qa = json.loads(lines[0])
            print(f"\n📖 Sample QA pair:")
            print(f"   Company: {sample_qa['ticker']}")
            print(f"   Question: {sample_qa['question'][:100]}...")
            print(f"   Answer: {sample_qa['answer'][:100]}...")
    else:
        print("❌ QA dataset generation may have failed")
        return 1
    
    print("🎉 QA dataset generation complete!")
    return 0


if __name__ == "__main__":
    exit(main()) 