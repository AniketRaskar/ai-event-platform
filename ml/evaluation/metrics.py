"""Small, deterministic ranking metrics for implicit-feedback recommenders."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
import math

def _validate(recommendations: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]]) -> None:
    """Reject mismatched batch inputs early."""
    if len(recommendations) != len(relevant):
        raise ValueError("recommendations and relevant must have the same number of queries")

def recall_at_k(recommendations: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]], k: int = 10) -> float:
    """Return mean fraction of each query's relevant items found in its top k."""
    _validate(recommendations, relevant)
    if k < 1:
        raise ValueError("k must be >= 1")
    values = []
    for ranked, truth in zip(recommendations, relevant):
        gold = set(truth)
        values.append(len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0)
    return sum(values) / len(values) if values else 0.0

def ndcg_at_k(recommendations: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]], k: int = 10) -> float:
    """Return mean binary-relevance normalized discounted cumulative gain."""
    _validate(recommendations, relevant)
    if k < 1:
        raise ValueError("k must be >= 1")
    values = []
    for ranked, truth in zip(recommendations, relevant):
        gold = set(truth)
        if not gold:
            values.append(0.0)
            continue
        dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(ranked[:k]) if item in gold)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
        values.append(dcg / ideal if ideal else 0.0)
    return sum(values) / len(values) if values else 0.0

def hit_rate_at_k(recommendations: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]], k: int = 10) -> float:
    """Return the fraction of queries with at least one relevant top-k item."""
    _validate(recommendations, relevant)
    if k < 1:
        raise ValueError("k must be >= 1")
    return sum(bool(set(ranked[:k]) & set(truth)) for ranked, truth in zip(recommendations, relevant)) / len(recommendations) if recommendations else 0.0

def catalog_coverage_at_k(recommendations: Sequence[Sequence[int]], catalog_size: int, k: int = 10) -> float:
    """Return the fraction of catalog items recommended at least once in top k."""
    if catalog_size < 1:
        return 0.0
    if k < 1:
        raise ValueError("k must be >= 1")
    exposed = {item for ranked in recommendations for item in ranked[:k]}
    return len(exposed) / catalog_size

def evaluate_rankings(recommendations: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]], catalog_size: int, k: int = 10) -> dict[str, float]:
    """Compute the four project ranking metrics for a batch of queries."""
    return {"recall": recall_at_k(recommendations, relevant, k), "ndcg": ndcg_at_k(recommendations, relevant, k),
            "hit_rate": hit_rate_at_k(recommendations, relevant, k),
            "catalog_coverage": catalog_coverage_at_k(recommendations, catalog_size, k)}
