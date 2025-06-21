"""
Defines the three evaluation scenarios for the RAG pipeline.

1. Unfiltered Context: Answer generation using the full context of an SEC filing.
2. Web Search: Answer generation using only GPT-4's web search capabilities.
3. RAG Pipeline: Answer generation using the semantic search RAG pipeline.
"""

import logging
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from sec_insights.rag.document_store import DocumentStore
from sec_insights.rag.pipeline import RAGPipeline
from sec_insights.rag.reranker import BGEReranker

logger = logging.getLogger(__name__)


def format_question_with_context(question: str, ticker: str, fiscal_year: int) -> str:
    """Add company and year context to the question for better parsing."""
    # Format with explicit ticker and fiscal year for optimal parsing
    formatted = (
        f"Question about ticker {ticker} for fiscal year {fiscal_year}: {question}"
    )
    logger.debug(f"🔍 Formatted question: {formatted}")
    return formatted


def run_unfiltered_context_scenario(
    doc_store: "DocumentStore", openai_client: OpenAI, qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """Scenario 1: GPT-4 with entire SEC filing as context."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]

    full_text = doc_store.get_full_filing_text(ticker, year)

    if not full_text:
        return f"[No SEC filing data available for {ticker} {year}]", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    system_prompt = """You are a financial analyst assistant. Your job is to answer questions about SEC filings based ONLY on the complete SEC filing context provided.

IMPORTANT GUIDELINES:
1. Answer based ONLY on the information in the complete SEC filing provided
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. You have access to the ENTIRE SEC filing, so search thoroughly for relevant information
4. Include specific details like numbers, dates, or company names when available
5. Do not make assumptions or add information not in the filing
6. If multiple sections provide conflicting information, mention this and cite the sections
7. For financial data, be precise with numbers and units

RESPONSE FORMAT:
- Start directly with the answer
- Don't say "Based on the context" or similar phrases
- Be professional but conversational
- If you're unsure after reviewing the entire filing, express appropriate uncertainty"""

    formatted_question = format_question_with_context(question, ticker, year)
    user_prompt = f"Context: {full_text}\n\nQuestion: {formatted_question}\n\nAnswer:"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=60,
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content.strip(), token_usage


def run_web_search_scenario(
    openai_client: OpenAI, qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """Scenario 2: GPT-4 with web search."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    formatted_question = format_question_with_context(question, ticker, year)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini-search-preview",
        web_search_options={},
        messages=[
            {
                "role": "system",
                "content": """
You are a financial analyst assistant with web search capabilities. Your job is to answer questions about SEC filings using real-time web search.

IMPORTANT GUIDELINES:
1. Use web search to find current and accurate information about the company's SEC filings
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. Search for official SEC filings, company reports, and reliable financial sources
4. Include specific details like numbers, dates, or company names when available
5. Cite or reference the sources you find when possible
6. Focus on factual, verifiable information from authoritative sources
7. For financial data, be precise with numbers and units

RESPONSE FORMAT:
- Start directly with the answer
- Be professional but conversational
- If search results are insufficient, mention what you were unable to find
- Prioritize official SEC documents and filings in your search""",
            },
            {"role": "user", "content": formatted_question},
        ],
        max_tokens=60,
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content.strip(), token_usage


def run_baseline_scenario(
    openai_client: OpenAI, qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """Scenario 5: Direct GPT-4o-mini without any additional context."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    formatted_question = format_question_with_context(question, ticker, year)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a financial analyst assistant. Your job is to answer questions about SEC filings using only your training knowledge.

IMPORTANT GUIDELINES:
1. Answer based ONLY on your existing knowledge from training data
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. You do NOT have access to current documents, web search, or real-time data
4. Include specific details like numbers, dates, or company names only if you're confident from training
5. Be honest about limitations - if information might be outdated or unavailable, say so
6. Do not make up specific financial figures or dates you're uncertain about
7. Focus on general knowledge about companies and typical SEC filing practices

RESPONSE FORMAT:
- Start directly with the answer
- Be professional but conversational
- If your training data is insufficient for specifics, explain this limitation
- Distinguish between general knowledge and specific facts you cannot verify""",
            },
            {"role": "user", "content": formatted_question},
        ],
        temperature=0,
        max_tokens=60,
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content.strip(), token_usage


def run_rag_scenario(
    pipeline: RAGPipeline, qa_item: Dict[str, Any]
) -> Tuple[str, List[str], Dict[str, int]]:
    """Scenario 3: RAG pipeline with semantic search."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]

    formatted_question = format_question_with_context(question, ticker, year)

    # Perform a single search to get the chunks
    retrieved_chunks = pipeline.search(query=formatted_question, top_k=10)

    # Extract chunk IDs from search results for metrics
    retrieved_ids = []
    for chunk in retrieved_chunks:
        chunk_id = (
            chunk.get("id")
            or chunk.get("chunk_id")
            or chunk.get("payload", {}).get("id")
        )
        if chunk_id:
            retrieved_ids.append(chunk_id)

    # Generate the answer using the retrieved chunks and get the response for token usage
    result_with_response = pipeline.answer_generator.generate_answer_with_response(
        question=formatted_question, chunks=retrieved_chunks
    )
    answer = result_with_response[0].get("answer", "")

    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if result_with_response[1] and hasattr(result_with_response[1], "usage"):
        usage = result_with_response[1].usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    return answer, retrieved_ids, token_usage


def run_reranked_rag_scenario(
    pipeline: RAGPipeline,
    qa_item: Dict[str, Any],
    reranker: BGEReranker,
    initial_k: int = 50,
    final_k: int = 10,
) -> Tuple[str, List[str], Dict[str, int]]:
    """Scenario 4: RAG pipeline with BGE reranker."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]

    formatted_question = format_question_with_context(question, ticker, year)

    # 1. Over-retrieve documents
    initial_chunks = pipeline.search(query=formatted_question, top_k=initial_k)
    if not initial_chunks:
        return (
            "[No documents found to rerank]",
            [],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # 2. Rerank the passages
    passages = [chunk.get("payload", {}).get("text", "") for chunk in initial_chunks]
    reranked_results = reranker.rerank(formatted_question, passages, top_k=final_k)

    # 3. Get the reranked chunks in the new order
    reranked_chunks = [initial_chunks[i] for i, score in reranked_results]

    # 4. Generate answer using reranked chunks with token tracking
    result, response = pipeline.answer_generator.generate_answer_with_response(
        question=formatted_question, chunks=reranked_chunks
    )
    answer = result.get("answer", "")

    # 5. Extract chunk IDs for metrics
    reranked_ids = []
    for chunk in reranked_chunks:
        chunk_id = (
            chunk.get("id")
            or chunk.get("chunk_id")
            or chunk.get("payload", {}).get("id")
        )
        if chunk_id:
            reranked_ids.append(chunk_id)

    # 6. Get token usage
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if response and hasattr(response, "usage"):
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    return answer, reranked_ids, token_usage
