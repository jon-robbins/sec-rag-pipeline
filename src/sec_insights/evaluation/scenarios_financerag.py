#!/usr/bin/env python3
"""
Implements the advanced RAG scenario based on the FinanceRAG architecture,
including query expansion and ensemble reranking.
"""
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import tiktoken
from sentence_transformers import CrossEncoder

from sec_insights.rag.client_utils import get_openai_client
from sec_insights.rag.pipeline import RAGPipeline

from .metrics import calculate_retrieval_metrics

logger = logging.getLogger(__name__)


FINQA_QUERY_EXPANSION_PROMPT = """# Role
 Extract keywords from give financial domain query from 10-k Report or financial disclosures :
 - Answer directly without any additional sentences.
 - If possible, extract original words form given query, but If the words are abbreviation use full name, on the other way, if the words can abbreviate use abbreviation
 - Extract proper nouns if existed
 - Number of keywords should be 1~4
 - DO NOT extract year (ex. 2018)
 - use comma(,) between each keyword

 #Example
 What are the respective maintenance revenue in 2018 and 2019?
 maintenance revenue

 What is the percentage change in the non controlling interest share of loss from 2018 to 2019?
 non controlling interest share of loss

 What is the percentage constitution of purchase obligations among the total contractual obligations in 2020?
 constitution, purchase obligations, contractual obligations

 How many categories of liabilities are there?
 categories, liabilities

 if mr . oppenheimer's rsus vest , how many total shares would he then have?
 Mr. Oppenheimer, Restricted Stock Unit, vest

 JP Morgan Segment breakdown
 JPM, Segment, breakdown

 Amazon Acquisitions in 2023?
 AMZN Acquisitions

 at december 31 , 2014 what was the ratio of the debt maturities scheduled for 2015 to 2018
 debt maturities

 # Query
"""


class FinanceRAGScenario:
    """
    Encapsulates the FinanceRAG scenario, including model initialization and execution.
    """

    def __init__(self):
        self._initialize_models()

    def _initialize_models(self):
        """Initializes and caches the reranker models, OpenAI client, and tokenizer."""
        logger.info("Initializing models for FinanceRAG scenario...")
        self.rerankers = {
            "jina": CrossEncoder(
                "jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True
            ),
            "bge": CrossEncoder("BAAI/bge-reranker-base"),
        }
        self.openai_client = get_openai_client()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        """Counts tokens in a string."""
        return len(self.tokenizer.encode(text))

    def get_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Concatenates text from a list of chunks to form a context string."""
        return "\n".join([chunk.get("payload", {}).get("text", "") for chunk in chunks])

    def generate_answer(self, question: str, context: str) -> Tuple[str, int, int]:
        """Generates an answer using the provided context and question."""
        system_prompt = """You are a financial analyst assistant. Your job is to answer questions about SEC filings based ONLY on the provided context from a curated set of document chunks.

IMPORTANT GUIDELINES:
1. Answer based ONLY on the information in the provided context.
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed.
3. If the context doesn't contain enough information, say so clearly.
4. Do not make assumptions or add information not present in the context.

RESPONSE FORMAT:
- Start directly with the answer.
- Do not say "Based on the context" or similar phrases."""
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        answer = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        return answer, prompt_tokens, completion_tokens

    def _expand_query(self, query: str) -> Tuple[str, int, int]:
        """Expands the query using an LLM, similar to FinanceRAG.

        Returns:
            tuple: (expanded_query, prompt_tokens, completion_tokens)
        """
        prompt = f"{FINQA_QUERY_EXPANSION_PROMPT}\n{query}"

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        expanded_keywords = response.choices[0].message.content
        expanded_query = f"{query}\n\n{expanded_keywords}"

        return (
            expanded_query,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    def run(
        self,
        rag_pipeline: RAGPipeline,
        question: str,
        ground_truth_chunks: List[str],
        k_values: List[int],
        phase_1_k: int = 30,
        phase_2_k: int = 10,
        use_rrf: bool = False,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """
        Runs the ensemble reranking RAG scenario.
        """
        # Phase 1: Initial Retrieval
        phase_1_chunks = rag_pipeline.search(query=question, top_k=phase_1_k)

        if not phase_1_chunks:
            return {
                "answer": "Could not retrieve any documents.",
                "retrieval": {},
                "tokens": {},
                "retrieved_chunks": [],
            }

        # Query Expansion
        (
            expanded_query,
            expansion_prompt_tokens,
            expansion_completion_tokens,
        ) = self._expand_query(question)

        # Phase 2: Ensemble Reranking
        rerank_candidates = [
            (expanded_query, chunk.get("payload", {}).get("text", ""))
            for chunk in phase_1_chunks
        ]

        jina_scores = self.rerankers["jina"].predict(
            rerank_candidates, convert_to_numpy=True
        )
        bge_scores = self.rerankers["bge"].predict(
            rerank_candidates, convert_to_numpy=True
        )

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
        context = self.get_context_from_chunks(phase_2_chunks)
        answer, prompt_tokens, completion_tokens = self.generate_answer(
            question=question, context=context
        )

        # Calculate Metrics
        full_reranked_chunk_ids = [
            phase_1_chunks[i]["id"] for i in full_reranked_indices
        ]
        retrieval_metrics = calculate_retrieval_metrics(
            retrieved_chunk_ids=full_reranked_chunk_ids,
            ground_truth_chunk_id=ground_truth_chunks[0],
            k_values=k_values,
            adjacency_map=rag_pipeline.adjacent_map,
            adjacency_bonus=0.5,
        )

        return {
            "answer": answer,
            "retrieval": retrieval_metrics,
            "tokens": {
                "prompt_tokens": expansion_prompt_tokens + prompt_tokens,
                "completion_tokens": expansion_completion_tokens + completion_tokens,
                "total_tokens": (
                    expansion_prompt_tokens
                    + expansion_completion_tokens
                    + prompt_tokens
                    + completion_tokens
                ),
            },
            "contexts": [
                chunk.get("payload", {}).get("text", "") for chunk in phase_2_chunks
            ],
        }


# Keep a single instance of the scenario class
_finance_rag_scenario_instance = None


def get_finance_rag_scenario() -> FinanceRAGScenario:
    """Returns a singleton instance of the FinanceRAGScenario."""
    global _finance_rag_scenario_instance
    if _finance_rag_scenario_instance is None:
        _finance_rag_scenario_instance = FinanceRAGScenario()
    return _finance_rag_scenario_instance


def ensemble_rerank_rag(
    rag_pipeline: RAGPipeline,
    question: str,
    ground_truth_chunks: List[str],
    k_values: List[int],
    phase_1_k: int = 30,
    phase_2_k: int = 10,
    use_rrf: bool = False,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """
    Function to run the ensemble rerank RAG scenario.
    """
    scenario = get_finance_rag_scenario()
    return scenario.run(
        rag_pipeline=rag_pipeline,
        question=question,
        ground_truth_chunks=ground_truth_chunks,
        k_values=k_values,
        phase_1_k=phase_1_k,
        phase_2_k=phase_2_k,
        use_rrf=use_rrf,
        rrf_k=rrf_k,
    )
