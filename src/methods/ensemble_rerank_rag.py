import logging
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from src.openai_functions.query_expansion import expand_query
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def get_rerankers():
    """Initializes and returns the reranker models."""
    logger.info("Initializing reranker models...")
    return {
        "jina": CrossEncoder(
            "jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True
        ),
        "bge": CrossEncoder("BAAI/bge-reranker-base"),
    }


def run_ensemble_rerank_rag(
    rag_pipeline: RAGPipeline,
    question: str,
    phase_1_k: int = 30,
    phase_2_k: int = 10,
    use_rrf: bool = False,
    rrf_k: int = 60,
) -> Tuple[str, List[str], Dict[str, int]]:
    """
    Runs the ensemble reranking RAG scenario.
    """
    rerankers = get_rerankers()

    # Phase 1: Initial Retrieval
    phase_1_chunks = rag_pipeline.search(query=question, top_k=phase_1_k)

    if not phase_1_chunks:
        return (
            "Could not retrieve any documents.",
            [],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # Query Expansion
    (
        expanded_query,
        expansion_prompt_tokens,
        expansion_completion_tokens,
    ) = expand_query(question)

    # Phase 2: Ensemble Reranking
    rerank_candidates = [
        (expanded_query, chunk.get("payload", {}).get("text", ""))
        for chunk in phase_1_chunks
    ]

    jina_scores = rerankers["jina"].predict(rerank_candidates, convert_to_numpy=True)
    bge_scores = rerankers["bge"].predict(rerank_candidates, convert_to_numpy=True)

    if use_rrf:
        jina_ranks = np.argsort(-jina_scores)
        bge_ranks = np.argsort(-bge_scores)
        rrf_scores = np.zeros(len(phase_1_chunks))
        for doc_idx in range(len(phase_1_chunks)):
            jina_rank_pos = np.where(jina_ranks == doc_idx)[0][0]
            bge_rank_pos = np.where(bge_ranks == doc_idx)[0][0]
            rrf_scores[doc_idx] = 1.0 / (rrf_k + jina_rank_pos + 1) + 1.0 / (
                rrf_k + bge_rank_pos + 1
            )
        full_reranked_indices = np.argsort(-rrf_scores)
    else:
        jina_scores_norm = (jina_scores - jina_scores.min()) / (
            jina_scores.max() - jina_scores.min() + 1e-6
        )
        bge_scores_norm = (bge_scores - bge_scores.min()) / (
            bge_scores.max() - bge_scores.min() + 1e-6
        )
        fused_scores = (jina_scores_norm + bge_scores_norm) / 2
        full_reranked_indices = np.argsort(fused_scores)[::-1]

    phase_2_chunks = [phase_1_chunks[i] for i in full_reranked_indices[:phase_2_k]]

    # Generation
    result, response = rag_pipeline.answer_generator.generate_answer_with_response(
        question=question, chunks=phase_2_chunks
    )
    answer = result.get("answer", "")

    # Consolidate token usage
    prompt_tokens = response.usage.prompt_tokens if response and response.usage else 0
    completion_tokens = (
        response.usage.completion_tokens if response and response.usage else 0
    )
    total_tokens = {
        "prompt_tokens": expansion_prompt_tokens + prompt_tokens,
        "completion_tokens": expansion_completion_tokens + completion_tokens,
        "total_tokens": (
            expansion_prompt_tokens
            + expansion_completion_tokens
            + prompt_tokens
            + completion_tokens
        ),
    }

    retrieved_ids = [chunk["id"] for chunk in phase_2_chunks]

    return answer, retrieved_ids, total_tokens
