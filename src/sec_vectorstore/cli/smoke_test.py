#!/usr/bin/env python3
"""
Smoke test for the SEC Vector Store.
Tests basic functionality with real SEC filings data.
"""

import os
import sys

# Add parent directories to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
from src.chunkers import SmartChunker
from src.filing_exploder import FilingExploder
from src.embedding import embed_texts
from src.sec_vectorstore import VectorStore


def main(use_docker: bool = False):
    """Run the smoke test."""
    # ✨ SWITCH BETWEEN MODES HERE ✨
    
    print("🚀 Starting vector store smoke test...")
    print(f"📊 Mode: {'Docker' if use_docker else 'In-Memory'}")
    
    # Load and process data
    print("📁 Loading SEC filings data...")
    try:
        df = pd.read_csv("data/df_filings.csv")
        items = FilingExploder().explode(df)
        chunks = SmartChunker().run(items)
        print(f"📝 Generated {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("💡 Make sure you're running from the project root with data/df_filings.csv")
        return 1

    # Get embeddings
    print("🧮 Computing embeddings...")
    try:
        ids, vecs, metas = embed_texts(chunks, refresh=False)
        print(f"✨ Got {len(vecs)} embeddings")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return 1

    # Initialize vector store
    print("🔧 Initializing vector store...")
    try:
        vs = VectorStore(
            collection_name="sec_filings_test",
            dim=1536,
            model="text-embedding-3-small",
            openai_key=os.getenv("OPENAI_API_KEY"),
            use_docker=use_docker,
        )
        
        # Initialize collection and upsert data
        vs.init_collection()
        vs.upsert(vectors=vecs, metas=metas, texts=[c.text for c in chunks])
    except Exception as e:
        print(f"❌ Vector store setup failed: {e}")
        return 1

    # Test search
    print("\n🔍 Testing search...")
    test_query = "Who are some of the primary competitors of Tesla in 2019?"
    print(f"Query: {test_query}")
    
    try:
        answers = vs.search(test_query, top_k=10)
        print(f"\n📋 Found {len(answers)} results:")
        
        for i, hit in enumerate(answers, 1):
            item_desc = hit.get('item_desc', 'Unknown')
            print(f"{i:2d}. {hit['score']:.3f} | {hit['ticker']} {hit['fiscal_year']} | "
                  f"Section {hit.get('item', '?')} ({item_desc}) | {hit['text'][:80]}...")
        
        if not answers:
            print("⚠️  No results found - this might indicate an issue")
            return 1
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 1

    # Show status
    print(f"\n📊 Vector Store Status:")
    status = vs.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print(f"\n✅ Smoke test completed successfully using {'Docker' if use_docker else 'in-memory'} mode!")
    
    if not use_docker:
        print("💡 To test Docker mode:")
        print("   1. Start Docker Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("   2. Set use_docker = True in this file")
        print("   3. Re-run the test")
    
    return 0


if __name__ == "__main__":
    sys.exit(main(use_docker=True)) 