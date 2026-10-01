"""Known-answer tests for ranking metric implementations."""
import pytest
from ml.evaluation.metrics import catalog_coverage_at_k, hit_rate_at_k, ndcg_at_k, recall_at_k

def test_metrics_hand_computed_example():
    """For query 1 A ranks first; query 2 finds C but misses B at K=2."""
    recommendations = [[1, 2], [3, 1, 2]]
    relevant = [{1}, {2, 3}]
    # Recall: (1/1 + 1/2) / 2 = 0.75. Both queries have a hit.
    assert recall_at_k(recommendations, relevant, k=2) == pytest.approx(0.75)
    assert hit_rate_at_k(recommendations, relevant, k=2) == pytest.approx(1.0)
    # NDCG query 1 = 1; query 2 = 1 / (1 + 1/log2(3)); mean those values.
    expected_ndcg = (1 + 1 / (1 + 1 / 1.584962500721156)) / 2
    assert ndcg_at_k(recommendations, relevant, k=2) == pytest.approx(expected_ndcg)
    # Top-2 across both queries covers A, B and C from a four-item catalog.
    assert catalog_coverage_at_k(recommendations, catalog_size=4, k=2) == pytest.approx(0.75)

def test_empty_batches_are_zero():
    """Empty batches should not divide by zero or claim coverage."""
    assert recall_at_k([], []) == 0
    assert ndcg_at_k([], []) == 0
    assert hit_rate_at_k([], []) == 0
    assert catalog_coverage_at_k([], 4) == 0

def test_mismatched_queries_raise():
    """Pairwise metric inputs must have equal batch lengths."""
    with pytest.raises(ValueError):
        recall_at_k([[1]], [])
