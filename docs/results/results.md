# Baseline recommendation evaluation

All interactions and metrics below use the generated **synthetic** dataset. Metrics are from the latest `python -m ml.evaluation.run_experiments` run. Rankings are evaluated at K=10; each held-out positive user/event pair is one query.

- Seed: 42
- Queries: 1325
- Train / validation / test interaction rows: 35518 / 4616 / 6223

## All targets

| Model | Queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 1325 | 0.8792 | 0.4622 | 0.8792 | 0.1640 |
| TF-IDF KNN | 1325 | 0.9155 | 0.5255 | 0.9155 | 0.1680 |
| Matrix Factorization | 1325 | 0.8740 | 0.4251 | 0.8740 | 0.1640 |

## Warm targets

| Model | Queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0 | — | — | — | — |
| TF-IDF KNN | 0 | — | — | — | — |
| Matrix Factorization | 0 | — | — | — | — |

## Cold targets

| Model | Queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 1325 | 0.8792 | 0.4622 | 0.8792 | 0.1640 |
| TF-IDF KNN | 1325 | 0.9155 | 0.5255 | 0.9155 | 0.1680 |
| Matrix Factorization | 1325 | 0.8740 | 0.4251 | 0.8740 | 0.1640 |

## Notes

Each query has one held-out positive target, so Recall@10 and HitRate@10 coincide. Catalog coverage is the share of the full event catalog appearing in the top 10 across that segment's queries.
Candidate pool sizes at test time: min 1, median 12.0, mean 10.71, max 17; 39.4% of queries have at most K candidates.
No warm test targets were available, so warm metrics are shown as unavailable (—), not zero. This follows from the synthetic timing setup: event windows last at most 30 days, while test interactions start 38 days after the training cutoff. This also keeps the top-10 evaluation ceiling high because a substantial share of candidate pools are already at most ten items.
