#!/usr/bin/env python3
"""
Test script for the RAG answer functionality.
Demonstrates how to ask questions and get AI-generated answers based on SEC filings.
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


def setup_vector_store(use_docker: bool = False) -> VectorStore:
    """Set up the vector store with SEC filings data."""
    print("🔧 Setting up vector store with SEC filings data...")
    
    # Load and process data
    df = pd.read_csv("data/df_filings.csv")
    items = FilingExploder().explode(df)
    chunks = SmartChunker().run(items)
    print(f"📝 Generated {len(chunks)} chunks")

    # Get embeddings
    ids, vecs, metas = embed_texts(chunks, refresh=False)
    print(f"✨ Got {len(vecs)} embeddings")

    # Initialize vector store
    vs = VectorStore(
        collection_name="sec_answer_test",
        use_docker=use_docker,
        openai_key=os.getenv("OPENAI_API_KEY")
    )
    
    vs.init_collection()
    vs.upsert(vectors=vecs, metas=metas, texts=[c.text for c in chunks])
    
    return vs


def test_questions():
    """List of test questions to demonstrate the system."""
    return [
        "Who are Tesla's main competitors in the electric vehicle market?",
        "What were Meta's main revenue risks in 2020?",
        "How did Tesla's automotive revenue change in 2019?",
        "What are the key risk factors for Tesla's business?",
        "Describe Apple's main business segments in 2021",
    ]


def display_answer(question: str, result: dict):
    """Display a formatted answer result."""
    print(f"\n{'='*80}")
    print(f"❓ Question: {question}")
    print(f"{'='*80}")
    
    print(f"\n🤖 Answer:")
    print(f"{result['answer']}")
    
    print(f"\n📊 Metadata:")
    print(f"  • Confidence: {result['confidence']}")
    print(f"  • Chunks used: {result['chunks_used']}")
    print(f"  • Search results: {result.get('search_results', 0)}")
    print(f"  • Top score: {result.get('top_score', 0.0):.3f}")
    
    if result['sources']:
        print(f"\n📚 Sources:")
        for i, source in enumerate(result['sources'][:5], 1):  # Show top 5 sources
            print(f"  {i}. {source}")
        if len(result['sources']) > 5:
            print(f"  ... and {len(result['sources']) - 5} more sources")


def test_summary(vs: VectorStore):
    """Test the summary functionality."""
    print(f"\n{'='*80}")
    print(f"📋 Testing Summary Functionality")
    print(f"{'='*80}")
    
    topic = "Tesla's business risks and challenges"
    print(f"\n🔍 Generating summary for: {topic}")
    
    result = vs.summarize(topic, ticker="TSLA", fiscal_year=2019, top_k=10)
    
    print(f"\n📝 Summary:")
    print(f"{result['summary']}")
    
    print(f"\n📊 Summary Metadata:")
    print(f"  • Chunks used: {result['chunks_used']}")
    print(f"  • Sources: {len(result['sources'])}")


def main():
    """Run the answer test demonstration."""
    print("🚀 Starting RAG Answer System Test...")
    print("📊 This will demonstrate question answering with SEC filings")
    
    try:
        # Set up vector store
        vs = setup_vector_store(use_docker=False)
        
        # Test questions
        questions = test_questions()
        
        print(f"\n🔍 Testing {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            try:
                print(f"\n⏳ Processing question {i}/{len(questions)}...")
                result = vs.answer(question, top_k=8)
                display_answer(question, result)
                
            except Exception as e:
                print(f"❌ Error with question {i}: {e}")
        
        # Test summary functionality
        test_summary(vs)
        
        print(f"\n✅ RAG Answer System test completed successfully!")
        print(f"💡 The system can now answer questions and generate summaries based on SEC filings!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 