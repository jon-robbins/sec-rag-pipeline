"""
Evaluation metrics for the RAG pipeline.

Includes ROUGE and BLEU for answer quality and retrieval metrics like Recall@K and MRR.
"""

import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
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

    dcg: float = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
    idcg: float = np.sum(
        [rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True))]
    )

    return dcg / idcg if idcg > 0 else 0.0


# --- Adjacent-aware Retrieval Metrics ---


def calculate_retrieval_metrics(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_id: str,
    k_values: Optional[List[int]] = None,
    adjacency_map: Optional[Dict[str, List[str]]] = None,
    adjacency_bonus: float = 0.0,
) -> Dict[str, float]:
    """
    Compute Recall@K and MRR for retrieval quality for various K.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs.
        ground_truth_chunk_id: The ground truth chunk ID.
        k_values: A list of integers for K values to calculate recall for.
        adjacency_map: A dictionary mapping chunk IDs to their adjacent neighbors.
        adjacency_bonus: The bonus to assign to adjacent neighbors.

    Returns:
        Dictionary with recall and MRR scores.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 10]

    correct_position = None
    for i, chunk_id in enumerate(retrieved_chunk_ids):
        if chunk_id == ground_truth_chunk_id:
            correct_position = i + 1  # 1-indexed
            break

    # Adjacent neighbour handling
    adjacent_position = None
    if adjacency_bonus > 0 and adjacency_map and ground_truth_chunk_id in adjacency_map:
        neighbour_ids = adjacency_map[ground_truth_chunk_id]
        for i, chunk_id in enumerate(retrieved_chunk_ids):
            if chunk_id in neighbour_ids:
                adjacent_position = i + 1  # 1-indexed
                break

    metrics = {}
    for k in sorted(k_values):
        exact_hit = 1.0 if (correct_position and correct_position <= k) else 0.0
        metrics[f"recall_at_{k}"] = exact_hit

        # graded recall with adjacent credit
        if adjacency_bonus > 0:
            adj_hit = 0.0
            if exact_hit == 0.0 and adjacent_position and adjacent_position <= k:
                adj_hit = adjacency_bonus
            metrics[f"adj_recall_at_{k}"] = min(exact_hit + adj_hit, 1.0)

    metrics["mrr"] = 1.0 / correct_position if correct_position else 0.0

    # graded MRR with adjacent credit
    if adjacency_bonus > 0:
        if correct_position:
            metrics["adj_mrr"] = 1.0 / correct_position
        elif adjacent_position:
            metrics["adj_mrr"] = adjacency_bonus / adjacent_position
        else:
            metrics["adj_mrr"] = 0.0

    if 10 in k_values:
        metrics["ndcg_at_10"] = calculate_ndcg(
            retrieved_chunk_ids, ground_truth_chunk_id, 10
        )

    return metrics


def calculate_rouge_scores(
    generated_answer: str, ground_truth_answer: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        generated_answer: The generated answer.
        ground_truth_answer: The ground truth answer.

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores as nested dictionaries.
    """
    if not generated_answer.strip() or not ground_truth_answer.strip():
        return {
            "rouge1": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
            "rouge2": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
            "rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0},
        }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth_answer, generated_answer)

    # Convert Score objects to dictionaries for compatibility
    return {
        key: {
            "fmeasure": value.fmeasure,
            "precision": value.precision,
            "recall": value.recall,
        }
        for key, value in scores.items()
    }


def calculate_bleu_score(
    generated_answer: str, ground_truth_answer: str, max_n: int = 4
) -> Dict[str, float]:
    """
    Calculate BLEU score for text generation evaluation.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
    generated text and reference text, with brevity penalty for short outputs.

    Args:
        generated_answer: The generated answer text.
        ground_truth_answer: The reference/ground truth answer text.
        max_n: Maximum n-gram length to consider (default: 4).

    Returns:
        Dictionary containing BLEU scores and components:
        - bleu: Overall BLEU score
        - bleu_1, bleu_2, bleu_3, bleu_4: Individual n-gram precisions
        - brevity_penalty: Brevity penalty factor
    """
    if not generated_answer.strip() or not ground_truth_answer.strip():
        return {
            "bleu": 0.0,
            "bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
            "brevity_penalty": 0.0,
        }

    # Tokenize (simple whitespace tokenization)
    candidate_tokens = generated_answer.lower().split()
    reference_tokens = ground_truth_answer.lower().split()

    candidate_length = len(candidate_tokens)
    reference_length = len(reference_tokens)

    # Calculate brevity penalty
    if candidate_length == 0:
        brevity_penalty = 0.0
    elif candidate_length >= reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - reference_length / candidate_length)

    # Calculate n-gram precisions
    precisions = []
    bleu_scores = {}

    for n in range(1, min(max_n + 1, candidate_length + 1)):
        # Get n-grams
        candidate_ngrams: Counter[tuple] = Counter()
        reference_ngrams: Counter[tuple] = Counter()

        # Extract candidate n-grams
        for i in range(len(candidate_tokens) - n + 1):
            ngram = tuple(candidate_tokens[i : i + n])
            candidate_ngrams[ngram] += 1

        # Extract reference n-grams
        for i in range(len(reference_tokens) - n + 1):
            ngram = tuple(reference_tokens[i : i + n])
            reference_ngrams[ngram] += 1

        # Calculate clipped precision
        clipped_matches = 0
        total_candidate_ngrams = sum(candidate_ngrams.values())

        for ngram, count in candidate_ngrams.items():
            clipped_matches += min(count, reference_ngrams.get(ngram, 0))

        precision = (
            clipped_matches / total_candidate_ngrams
            if total_candidate_ngrams > 0
            else 0.0
        )
        precisions.append(precision)
        bleu_scores[f"bleu_{n}"] = precision

    # Fill remaining BLEU scores with 0.0 if we have fewer n-grams than max_n
    for n in range(len(precisions) + 1, max_n + 1):
        bleu_scores[f"bleu_{n}"] = 0.0

    # Calculate overall BLEU score (geometric mean of precisions with brevity penalty)
    if all(p > 0 for p in precisions) and len(precisions) > 0:
        geometric_mean = math.exp(
            sum(math.log(p) for p in precisions) / len(precisions)
        )
        bleu_score = brevity_penalty * geometric_mean
    else:
        bleu_score = 0.0

    return {
        "bleu": bleu_score,
        **bleu_scores,
        "brevity_penalty": brevity_penalty,
    }
