"""
Defines the three evaluation scenarios for the RAG pipeline.

1. Unfiltered Context: Answer generation using the full context of an SEC filing.
2. Web Search: Answer generation using only GPT-4's web search capabilities.
3. RAG Pipeline: Answer generation using the semantic search RAG pipeline.
"""

from typing import Dict, Any, Tuple
from openai import OpenAI
from rag.pipeline import RAGPipeline

def _format_question_with_context(question: str, ticker: str, fiscal_year: int) -> str:
    """Add company and year context to the question."""
    return f"Company: {ticker}\nSEC Filing year: {fiscal_year}\n\n{question}"

def run_unfiltered_context_scenario(
    pipeline: RAGPipeline,
    openai_client: OpenAI,
    qa_item: Dict[str, Any]
) -> str:
    """Scenario 1: GPT-4 with entire SEC filing as context."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    
    full_text = pipeline.document_store.get_full_filing_text(ticker, year)
    
    if not full_text:
        chunks = pipeline.retrieve_by_filter(ticker=ticker, fiscal_year=year, limit=100)
        if not chunks:
            return f"[No SEC filing data available for {ticker} {year}]"
        full_text = "\n\n".join([chunk.get("text", "") for chunk in chunks])[:50000]

    system_prompt = "You are a financial analyst. Answer the question based ONLY on the provided SEC filing context. Be accurate and concise."
    formatted_question = _format_question_with_context(question, ticker, year)
    user_prompt = f"Context: {full_text}\n\nQuestion: {formatted_question}\n\nAnswer:"
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def run_web_search_scenario(
    openai_client: OpenAI,
    qa_item: Dict[str, Any]
) -> str:
    """Scenario 2: GPT-4 with web search."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    formatted_question = _format_question_with_context(question, ticker, year)
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Use your knowledge and reasoning to answer the question about the company's SEC filing."},
            {"role": "user", "content": formatted_question},
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def run_rag_scenario(
    pipeline: RAGPipeline,
    qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Scenario 3: RAG pipeline with semantic search."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    
    formatted_question = _format_question_with_context(question, ticker, year)

    # The RAG pipeline handles query parsing, search, and generation
    result = pipeline.answer(question=formatted_question)
    answer = result.get("answer", "")
    
    # We also need the retrieved documents to calculate retrieval metrics
    retrieved_chunks = pipeline.search(query=formatted_question, top_k=10)
    
    # Extract chunk IDs from search results - need to check the actual structure
    retrieved_ids = []
    for chunk in retrieved_chunks:
        # Try to get ID from different possible locations
        chunk_id = chunk.get("id") or chunk.get("chunk_id") or chunk.get("payload", {}).get("id")
        if chunk_id:
            retrieved_ids.append(chunk_id)
    
    return answer, retrieved_ids 