#!/usr/bin/env python3
"""
Quick test script to verify company balancing works correctly.
"""

import sys
import pickle
from pathlib import Path
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.normalize_qa_sample import BalancedChunkSampler, classify_chunk_by_tokens


def test_balancing():
    """Test the company balancing functionality."""
    print("🧪 Testing company balancing...")
    
    # Load chunks
    chunks_path = Path("data/chunks.pkl")
    if not chunks_path.exists():
        print(f"❌ {chunks_path} not found")
        return
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"📊 Original dataset: {len(chunks)} chunks")
    
    # Show original distribution
    original_dist = Counter(chunk.metadata["ticker"] for chunk in chunks)
    print(f"📈 Original by company: {dict(original_dist)}")
    
    # Test WITHOUT company balancing
    print("\n🔍 Testing WITHOUT company balancing:")
    sampler_unbalanced = BalancedChunkSampler(max_per_group=3, balance_companies=False)
    grouped = sampler_unbalanced.group_chunks_by_keys(chunks)
    unbalanced_chunks = sampler_unbalanced.stratified_sample(grouped)
    
    unbalanced_dist = Counter(chunk.metadata["ticker"] for chunk in unbalanced_chunks)
    print(f"   Result: {dict(unbalanced_dist)} (total: {len(unbalanced_chunks)})")
    
    # Test WITH company balancing
    print("\n✅ Testing WITH company balancing:")
    sampler_balanced = BalancedChunkSampler(max_per_group=3, balance_companies=True)
    grouped = sampler_balanced.group_chunks_by_keys(chunks)
    balanced_chunks = sampler_balanced.stratified_sample(grouped)
    
    balanced_dist = Counter(chunk.metadata["ticker"] for chunk in balanced_chunks)
    print(f"   Result: {dict(balanced_dist)} (total: {len(balanced_chunks)})")
    
    # Verify perfect balance
    company_counts = list(balanced_dist.values())
    if len(set(company_counts)) == 1:
        print("🎉 Perfect balance achieved!")
    else:
        print(f"⚠️ Balance not perfect: range {min(company_counts)}-{max(company_counts)}")
    
    # Show chunk class distribution
    chunk_class_dist = Counter(classify_chunk_by_tokens(chunk.text) for chunk in balanced_chunks)
    print(f"📊 Chunk classes: {dict(chunk_class_dist)}")


if __name__ == "__main__":
    test_balancing() 