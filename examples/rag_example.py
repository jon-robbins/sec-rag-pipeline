#!/usr/bin/env python3
"""
Simple example of how to use the RAG (Retrieval-Augmented Generation) functionality.

This demonstrates the complete pipeline:
1. Search for relevant SEC filing chunks
2. Generate AI answers based on the retrieved context
3. Get summaries of topics from multiple sources
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.sec_vectorstore import VectorStore


def main():
    print("🚀 RAG Example - Question Answering with SEC Filings")
    print("=" * 60)
    
    # Initialize vector store (assuming data is already loaded)
    print("🔧 Setting up vector store...")
    vs = VectorStore(
        collection_name="sec_filing_test",
        use_docker=True,  # Use in-memory for faster testing
        openai_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Example 1: Ask a specific question
    print("\n📋 Example 1: Direct Question")
    print("-" * 40)
    
    question = "Who are Tesla's main competitors?"
    print(f"❓ Question: {question}")
    
    # Get AI-generated answer
    result = vs.answer(question, ticker="TSLA", fiscal_year=2019)
    
    print(f"\n🤖 Answer:")
    print(f"{result['answer']}")
    
    print(f"\n📊 Details:")
    print(f"  • Confidence: {result['confidence']}")
    print(f"  • Sources used: {result['chunks_used']}")
    print(f"  • Top similarity score: {result['top_score']:.3f}")
    
    # Example 2: Generate a summary
    print("\n\n📋 Example 2: Topic Summary")
    print("-" * 40)
    
    topic = "Revenue and financial performance"
    print(f"📈 Topic: {topic}")
    
    summary_result = vs.summarize(
        topic, 
        ticker="META", 
        fiscal_year=2020,
        top_k=10
    )
    
    print(f"\n📝 Summary:")
    print(f"{summary_result['summary']}")
    
    print(f"\n📚 Based on {summary_result['chunks_used']} sources from SEC filings")
    
    # Example 3: Compare search vs answer
    print("\n\n📋 Example 3: Search vs Answer Comparison")
    print("-" * 40)
    
    query = "What are Tesla's business risks?"
    
    # Just search (returns raw chunks)
    search_results = vs.search(query, ticker="TSLA", top_k=3)
    print(f"🔍 Search results ({len(search_results)} chunks):")
    for i, chunk in enumerate(search_results[:2], 1):
        print(f"  {i}. {chunk['text'][:100]}... (Score: {chunk['score']:.3f})")
    
    # Full answer (uses GPT to synthesize)
    answer_result = vs.answer(query, ticker="TSLA", top_k=3)
    print(f"\n🤖 AI Answer:")
    print(f"{answer_result['answer']}")
    
    print("\n✅ RAG example completed!")
    print("\n💡 Usage patterns:")
    print("  • vs.search() - Get raw relevant chunks")
    print("  • vs.answer() - Get AI-generated answers") 
    print("  • vs.summarize() - Get topic summaries")


if __name__ == "__main__":
    main() 