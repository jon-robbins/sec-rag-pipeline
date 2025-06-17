"""
Evaluation metrics for the RAG pipeline.

Includes ROUGE for answer quality and retrieval metrics like Recall@K and MRR.
"""

from typing import List, Dict, Any
from rouge_score import rouge_scorer


def compute_retrieval_metrics(retrieved_ids: List[str], correct_chunk_id: str) -> Dict[str, Any]:
    """
    Compute Recall@K and MRR for retrieval quality.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs.
        correct_chunk_id: The ground truth chunk ID.
        
    Returns:
        Dictionary with recall and MRR scores.
    """
    correct_position = None
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id == correct_chunk_id:
            correct_position = i + 1  # 1-indexed
            break
    
    recall_at_1 = 1.0 if correct_position == 1 else 0.0
    recall_at_3 = 1.0 if correct_position and correct_position <= 3 else 0.0
    recall_at_5 = 1.0 if correct_position and correct_position <= 5 else 0.0
    recall_at_10 = 1.0 if correct_position and correct_position <= 10 else 0.0
    
    mrr = 1.0 / correct_position if correct_position else 0.0
    
    return {
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "retrieved_position": correct_position
    }


def compute_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
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
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }


# --- convenience aliases expected by evaluator ---

def calculate_retrieval_metrics(retrieved_chunk_ids, true_chunk_id, k_values=None):
    return compute_retrieval_metrics(retrieved_chunk_ids, true_chunk_id)

def calculate_rouge_scores(prediction: str, reference: str):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    result = {}
    for key, score in scores.items():
        result[key] = {
            "precision": score.precision,
            "recall": score.recall,
            "fmeasure": score.fmeasure,
        }
    return result 