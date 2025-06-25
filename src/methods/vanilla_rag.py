import logging
from typing import Any, Dict, List, Tuple

from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def format_question_with_context(question: str, ticker: str, fiscal_year: int) -> str:
    """Add company and year context to the question for better parsing."""
    # Format with explicit ticker and fiscal year for optimal parsing
    formatted = (
        f"Question about ticker {ticker} for fiscal year {fiscal_year}: {question}"
    )
    logger.debug(f"🔍 Formatted question: {formatted}")
    return formatted


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
