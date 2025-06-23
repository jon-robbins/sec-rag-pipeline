"""
Tests for evaluation metrics.
"""

import pytest

from sec_insights.evaluation.metrics import (
    calculate_bleu_score,
    calculate_ndcg,
    calculate_retrieval_metrics,
    calculate_rouge_scores,
)


# Test cases for retrieval metrics
@pytest.mark.parametrize(
    "retrieved_ids, true_chunk_id, k_values, adjacent_map, adjacent_credit, expected_metrics",
    [
        # Exact hit at k=1
        (
            ["c1", "c2", "c3"],
            "c1",
            [1, 3, 5],
            None,
            0.0,
            {"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 1.0},
        ),
        # Exact hit at k=3
        (
            ["c2", "c3", "c1", "c4"],
            "c1",
            [1, 3, 5],
            None,
            0.0,
            {"recall_at_1": 0.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 1 / 3},
        ),
        # No hit
        (
            ["c2", "c3", "c4"],
            "c1",
            [1, 3, 5],
            None,
            0.0,
            {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0},
        ),
        # Adjacent hit with credit
        (
            ["c2", "c3", "c4"],
            "c1",
            [1, 3],
            {"c1": ["c2", "c5"]},
            0.5,
            {
                "recall_at_1": 0.0,
                "adj_recall_at_1": 0.5,
                "recall_at_3": 0.0,
                "adj_recall_at_3": 0.5,
                "mrr": 0.0,
                "adj_mrr": 0.5 / 1.0,
            },
        ),
        # Exact hit takes precedence over adjacent
        (
            ["c1", "c2"],
            "c1",
            [1, 3],
            {"c1": ["c2"]},
            0.5,
            {
                "recall_at_1": 1.0,
                "adj_recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "adj_recall_at_3": 1.0,
                "mrr": 1.0,
                "adj_mrr": 1.0,
            },
        ),
        # NDCG calculation included
        (["c1", "c2", "c3"], "c1", [10], None, 0.0, {"ndcg_at_10": 1.0}),
        (["c2", "c1", "c3"], "c1", [10], None, 0.0, {"ndcg_at_10": 0.63092975}),
    ],
)
def test_calculate_retrieval_metrics(
    retrieved_ids,
    true_chunk_id,
    k_values,
    adjacent_map,
    adjacent_credit,
    expected_metrics,
):
    metrics = calculate_retrieval_metrics(
        retrieved_ids, true_chunk_id, k_values, adjacent_map, adjacent_credit
    )

    # We only check for the keys present in the expected_metrics dictionary
    for key, value in expected_metrics.items():
        assert key in metrics
        assert metrics[key] == pytest.approx(value), f"Metric '{key}' failed"


# Test cases for ROUGE scores
@pytest.mark.parametrize(
    "prediction, reference, expected_f_scores",
    [
        (
            "the cat is on the mat",
            "the cat is on the mat",
            {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0},
        ),
        (
            "the cat is on the mat",
            "a cat is on the mat",
            {"rouge1": 0.8333, "rouge2": 0.8, "rougeL": 0.8333},
        ),
        (
            "this is a test",
            "this is also a test",
            {"rouge1": 0.8888, "rouge2": 0.5714, "rougeL": 0.8888},
        ),
        ("", "this is a test", {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}),
        ("this is a test", "", {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}),
    ],
)
def test_calculate_rouge_scores(prediction, reference, expected_f_scores):
    scores = calculate_rouge_scores(prediction, reference)
    for key, value in expected_f_scores.items():
        assert key in scores
        assert scores[key]["fmeasure"] == pytest.approx(value, abs=0.01)


# Test cases for NDCG
@pytest.mark.parametrize(
    "retrieved_ids, true_chunk_id, k, expected_ndcg",
    [
        (["c1", "c2", "c3"], "c1", 10, 1.0),
        (["c2", "c1", "c3"], "c1", 10, 0.6309),  # 1/log2(3) / 1
        (["c2", "c3", "c1"], "c1", 10, 0.5),  # 1/log2(4) / 1
        (["c2", "c3", "c4"], "c1", 10, 0.0),
    ],
)
def test_calculate_ndcg(retrieved_ids, true_chunk_id, k, expected_ndcg):
    ndcg = calculate_ndcg(retrieved_ids, true_chunk_id, k)
    assert ndcg == pytest.approx(expected_ndcg, abs=0.001)


# Test cases for BLEU scores
@pytest.mark.parametrize(
    "prediction, reference, expected_scores",
    [
        # Perfect match
        (
            "the cat is on the mat",
            "the cat is on the mat",
            {
                "bleu": 1.0,
                "bleu_1": 1.0,
                "bleu_2": 1.0,
                "bleu_3": 1.0,
                "bleu_4": 1.0,
                "brevity_penalty": 1.0,
            },
        ),
        # Partial match with same length
        (
            "the cat is on the table",
            "the cat is on the mat",
            {
                "bleu_1": 0.8333,
                "bleu_2": 0.8,
                "bleu_3": 0.75,
                "bleu_4": 0.6667,
                "brevity_penalty": 1.0,
            },
        ),
        # Shorter prediction (brevity penalty applies)
        (
            "the cat",
            "the cat is on the mat",
            {"bleu_1": 1.0, "bleu_2": 1.0, "brevity_penalty": 0.1353},  # exp(1 - 6/2)
        ),
        # Longer prediction (no brevity penalty)
        (
            "the cat is on the mat today",
            "the cat is on the mat",
            {"bleu_1": 0.8571, "brevity_penalty": 1.0},  # 6/7 unigram precision
        ),
        # Empty strings
        (
            "",
            "this is a test",
            {
                "bleu": 0.0,
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "bleu_4": 0.0,
                "brevity_penalty": 0.0,
            },
        ),
        (
            "this is a test",
            "",
            {
                "bleu": 0.0,
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "bleu_4": 0.0,
                "brevity_penalty": 0.0,
            },
        ),
        # No n-gram overlap
        (
            "completely different words",
            "totally other text",
            {
                "bleu": 0.0,
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "brevity_penalty": 1.0,
            },
        ),
    ],
)
def test_calculate_bleu_score(prediction, reference, expected_scores):
    scores = calculate_bleu_score(prediction, reference)

    # Check that all expected keys are present
    for key in ["bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "brevity_penalty"]:
        assert key in scores

    # Check specific expected values
    for key, value in expected_scores.items():
        assert scores[key] == pytest.approx(
            value, abs=0.001
        ), f"BLEU metric '{key}' failed"


def test_calculate_bleu_score_single_word():
    """Test BLEU calculation with single words (edge case for n-grams)."""
    scores = calculate_bleu_score("cat", "cat")
    assert scores["bleu_1"] == 1.0
    assert scores["bleu_2"] == 0.0  # No bigrams possible
    assert scores["bleu_3"] == 0.0  # No trigrams possible
    assert scores["bleu_4"] == 0.0  # No 4-grams possible
    assert scores["brevity_penalty"] == 1.0


def test_calculate_bleu_score_case_insensitive():
    """Test that BLEU calculation is case-insensitive."""
    scores1 = calculate_bleu_score("The Cat", "the cat")
    scores2 = calculate_bleu_score("the cat", "the cat")

    assert scores1["bleu_1"] == scores2["bleu_1"]
    assert scores1["bleu"] == scores2["bleu"]
