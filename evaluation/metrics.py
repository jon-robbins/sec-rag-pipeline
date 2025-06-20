"""
Evaluation metrics for the RAG pipeline.

Includes ROUGE for answer quality and retrieval metrics like Recall@K and MRR.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from rouge_score import rouge_scorer


def calculate_ndcg(retrieved_ids: List[str], true_chunk_id: str, k: int) -> float:
    """
    Compute NDCG@k for retrieval quality.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs.
        true_chunk_id: The ground truth chunk ID.
        k: The value of k for NDCG.
        
    Returns:
        NDCG@k score.
    """
    relevance = [1 if doc_id == true_chunk_id else 0 for doc_id in retrieved_ids[:k]]
    if not any(relevance):
        return 0.0
        
    dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
    idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True))])
    
    return dcg / idcg if idcg > 0 else 0.0


# --- Adjacent-aware Retrieval Metrics ---

def calculate_retrieval_metrics(
    retrieved_ids: List[str], 
    true_chunk_id: str,
    k_values: List[int] = None,
    adjacent_map: Optional[Dict[str, List[str]]] = None,
    adjacent_credit: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute Recall@K and MRR for retrieval quality for various K.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs.
        true_chunk_id: The ground truth chunk ID.
        k_values: A list of integers for K values to calculate recall for.
        adjacent_map: A dictionary mapping chunk IDs to their adjacent neighbors.
        adjacent_credit: The credit to assign to adjacent neighbors.
        
    Returns:
        Dictionary with recall and MRR scores.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 10]

    correct_position = None
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id == true_chunk_id:
            correct_position = i + 1  # 1-indexed
            break
            
    # Adjacent neighbour handling
    adjacent_position = None
    if adjacent_credit > 0 and adjacent_map and true_chunk_id in adjacent_map:
        neighbour_ids = adjacent_map[true_chunk_id]
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in neighbour_ids:
                adjacent_position = i + 1  # 1-indexed
                break

    metrics = {}
    for k in sorted(k_values):
        exact_hit = 1.0 if (correct_position and correct_position <= k) else 0.0
        metrics[f"recall_at_{k}"] = exact_hit

        # graded recall with adjacent credit
        if adjacent_credit > 0:
            adj_hit = 0.0
            if exact_hit == 0.0 and adjacent_position and adjacent_position <= k:
                adj_hit = adjacent_credit
            metrics[f"adj_recall_at_{k}"] = min(exact_hit + adj_hit, 1.0)

    metrics["mrr"] = 1.0 / correct_position if correct_position else 0.0

    # graded MRR with adjacent credit
    if adjacent_credit > 0:
        if correct_position:
            metrics["adj_mrr"] = 1.0 / correct_position
        elif adjacent_position:
            metrics["adj_mrr"] = adjacent_credit / adjacent_position
        else:
            metrics["adj_mrr"] = 0.0

    if 10 in k_values:
        metrics["ndcg_at_10"] = calculate_ndcg(retrieved_ids, true_chunk_id, 10)
    
    return metrics


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE scores for answer quality.
    
    Args:
        prediction: The generated answer.
        reference: The ground truth answer.
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L f-measures.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    result = {}
    for key, score in scores.items():
        result[key] = {
            "precision": score.precision,
            "recall": score.recall,
            "fmeasure": score.fmeasure,
        }
    return result 