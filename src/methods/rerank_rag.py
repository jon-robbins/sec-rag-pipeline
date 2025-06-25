import logging
from typing import Any, Dict, List, Tuple

from src.rag.pipeline import RAGPipeline
from src.rag.reranker import BGEReranker

logger = logging.getLogger(__name__)


def format_question_with_context(question: str, ticker: str, fiscal_year: int) -> str:
    """Add company and year context to the question for better parsing."""
    # Format with explicit ticker and fiscal year for optimal parsing
    formatted = (
        f"Question about ticker {ticker} for fiscal year {fiscal_year}: {question}"
    )
    logger.debug(f"🔍 Formatted question: {formatted}")
    return formatted


def run_reranked_rag_scenario(
    pipeline: RAGPipeline,
    qa_item: Dict[str, Any],
    reranker: BGEReranker,
    phase_1_k: int = 30,  # Phase 1: Initial retrieval (only for reranked systems)
    phase_2_k: int = 10,  # Phase 2: Final selection for generation (only for reranked systems)
) -> Tuple[str, List[str], Dict[str, int]]:
    """Scenario 4: RAG pipeline with BGE reranker.

    This scenario uses a two-phase approach:
    - Phase 1: Retrieve phase_1_k documents via vector search
    - Phase 2: Rerank and select top phase_2_k for generation

    For fair evaluation, NDCG@10 is calculated on the full phase_1_k reranked list.
    Note: Phase parameters only apply to reranked systems.
    """
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]

    formatted_question = format_question_with_context(question, ticker, year)

    # Phase 1: Over-retrieve documents
    phase_1_chunks = pipeline.search(query=formatted_question, top_k=phase_1_k)
    if not phase_1_chunks:
        return (
            "[No documents found to rerank]",
            [],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # Phase 2: Rerank and select final chunks
    passages = [chunk.get("payload", {}).get("text", "") for chunk in phase_1_chunks]

    # Get full reranked order for evaluation (all phase_1_k documents)
    all_reranked_results = reranker.rerank(
        formatted_question, passages, top_k=phase_1_k
    )

    # Select top phase_2_k for generation
    phase_2_chunks = [
        phase_1_chunks[i] for i, score in all_reranked_results[:phase_2_k]
    ]

    # Generate answer using phase_2_k chunks with token tracking
    result, response = pipeline.answer_generator.generate_answer_with_response(
        question=formatted_question, chunks=phase_2_chunks
    )
    answer = result.get("answer", "")

    # Extract chunk IDs for metrics - FULL RERANKED LIST for fair evaluation
    all_reranked_ids = []
    for i, score in all_reranked_results:
        chunk = phase_1_chunks[i]
        chunk_id = (
            chunk.get("id")
            or chunk.get("chunk_id")
            or chunk.get("payload", {}).get("id")
        )
        if chunk_id:
            all_reranked_ids.append(chunk_id)

    # Get token usage
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if response and hasattr(response, "usage"):
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    return answer, all_reranked_ids, token_usage
