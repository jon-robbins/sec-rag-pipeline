#!/usr/bin/env python3
"""
Implements the advanced RAG scenario based on the FinanceRAG architecture,
including query expansion and ensemble reranking.
"""
from typing import Dict, Any, List
import numpy as np
import tiktoken

from rag.pipeline import RAGPipeline
from evaluation.scenarios import format_question_with_context
from rag.reranker import BGEReranker # Assuming this is one of our rerankers
from sentence_transformers import CrossEncoder
from openai import OpenAI

from .metrics import calculate_retrieval_metrics

# Store models globally to avoid re-initialization on every call
_rerankers = {}
_openai_client = None
_tokenizer = None

def _initialize_tokenizer():
    """Initializes the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    """Counts tokens in a string."""
    _initialize_tokenizer()
    return len(_tokenizer.encode(text))

def get_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Concatenates text from a list of chunks to form a context string."""
    return "\n".join([chunk.get("payload", {}).get("text", "") for chunk in chunks])

def generate_answer(
    question: str, context: str, client: OpenAI
) -> (str, int, int):
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
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300
    )
    
    answer = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    return answer, prompt_tokens, completion_tokens

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

def _initialize_rerankers():
    """Initializes and caches the reranker models."""
    global _rerankers
    if not _rerankers:
        print("Initializing Ensemble Rerankers (Jina, BGE)...")
        _rerankers = {
            "jina": CrossEncoder('jinaai/jina-reranker-v2-base-multilingual', trust_remote_code=True),
            "bge": CrossEncoder('BAAI/bge-reranker-base')
        }

def _initialize_openai_client():
    """Initializes and caches the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()

def _expand_query(query: str) -> tuple[str, int, int]:
    """Expands the query using an LLM, similar to FinanceRAG.
    
    Returns:
        tuple: (expanded_query, prompt_tokens, completion_tokens)
    """
    _initialize_openai_client()
    
    prompt = f"{FINQA_QUERY_EXPANSION_PROMPT}\n{query}"
    
    response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    expanded_keywords = response.choices[0].message.content
    expanded_query = f"{query}\n\n{expanded_keywords}"
    
    # Return the expanded query AND token usage
    return expanded_query, response.usage.prompt_tokens, response.usage.completion_tokens

def ensemble_rerank_rag(
    rag_pipeline: RAGPipeline,
    question: str,
    ground_truth_chunks: List[str],
    k_values: List[int],
    top_k_initial: int = 20,
    top_k_final: int = 5,
) -> Dict[str, Any]:
    """
    Runs an evaluation scenario using an ensemble of rerankers.
    
    1.  Initial retrieval from the pipeline's vector store.
    2.  Query expansion using an LLM.
    3.  Reranking of initial results with multiple cross-encoders.
    4.  Fusion of reranker scores.
    5.  Generation of an answer from the final top-k chunks.
    6.  Calculation of retrieval and generation metrics.
    """
    _initialize_rerankers()
    _initialize_openai_client()

    # 1. Initial Retrieval
    retrieved_chunks = rag_pipeline.search(query=question, top_k=top_k_initial)
    
    if not retrieved_chunks:
        return {
            "answer": "Could not retrieve any documents.",
            "retrieval": {},
            "tokens": {},
            "retrieved_chunks": [],
        }

    # 2. Query Expansion - ENABLED FinanceRAG methodology
    # Use LLM to expand the query with relevant financial keywords
    expanded_query, expansion_prompt_tokens, expansion_completion_tokens = _expand_query(question)

    # 3. Ensemble Reranking
    rerank_candidates = [
        (expanded_query, chunk.get("payload", {}).get("text", ""))
        for chunk in retrieved_chunks
    ]
    
    # Get scores from each reranker
    jina_scores = _rerankers["jina"].predict(rerank_candidates, convert_to_numpy=True)
    bge_scores = _rerankers["bge"].predict(rerank_candidates, convert_to_numpy=True)
    
    # 4. Score Fusion (simple averaging)
    jina_scores_norm = (jina_scores - jina_scores.min()) / (jina_scores.max() - jina_scores.min() + 1e-6)
    bge_scores_norm = (bge_scores - bge_scores.min()) / (bge_scores.max() - bge_scores.min() + 1e-6)
    fused_scores = (jina_scores_norm + bge_scores_norm) / 2
    
    # Sort chunks by the new fused score
    reranked_indices = np.argsort(fused_scores)[::-1]
    final_chunks = [retrieved_chunks[i] for i in reranked_indices[:top_k_final]]
    
    # 5. Generation
    context = get_context_from_chunks(final_chunks)
    answer, prompt_tokens, completion_tokens = generate_answer(
        question=question, context=context, client=_openai_client
    )
    
    # 6. Calculate Metrics
    # Calculate metrics on the reranked chunks to properly measure reranking performance
    reranked_chunk_ids = [retrieved_chunks[i]["id"] for i in reranked_indices]
    retrieval_metrics = calculate_retrieval_metrics(
        retrieved_ids=reranked_chunk_ids, 
        true_chunk_id=ground_truth_chunks[0],
        k_values=k_values,
        adjacent_map=rag_pipeline.adjacent_map,
        adjacent_credit=0.5
    )

    return {
        "answer": answer,
        "retrieval": retrieval_metrics,
        "tokens": {
            "prompt_tokens": expansion_prompt_tokens + prompt_tokens,
            "completion_tokens": expansion_completion_tokens + completion_tokens,
            "total_tokens": expansion_prompt_tokens + expansion_completion_tokens + prompt_tokens + completion_tokens,
        },
        "contexts": [chunk.get("payload", {}).get("text", "") for chunk in final_chunks],
    } 