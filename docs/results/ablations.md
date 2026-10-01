# NeuMF ablations

All figures are measured on the synthetic dataset using held-out event interactions. K=10; model selection used validation NDCG@10.

Embedding backend: `sentence-transformers`.

| Variant | Epochs | Validation NDCG@10 | Test queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No GMF | 5 | 0.4339 | 1325 | 0.8747 | 0.4505 | 0.8747 | 0.1600 |
| No MLP | 5 | 0.4286 | 1325 | 0.8868 | 0.4645 | 0.8868 | 0.1600 |
| No content | 5 | 0.4462 | 1325 | 0.8792 | 0.4622 | 0.8792 | 0.1640 |
| ID dropout 0 | 3 | 0.3929 | 1325 | 0.8853 | 0.4634 | 0.8853 | 0.1640 |
| ID dropout 0.3 | 5 | 0.4418 | 1325 | 0.8792 | 0.4517 | 0.8792 | 0.1600 |
| ID dropout 0.5 | 5 | 0.4373 | 1325 | 0.8823 | 0.4560 | 0.8823 | 0.1600 |
| Negatives 1 | 3 | 0.4420 | 1325 | 0.8891 | 0.4898 | 0.8891 | 0.1680 |
| Negatives 4 | 5 | 0.4418 | 1325 | 0.8792 | 0.4517 | 0.8792 | 0.1600 |
| Negatives 8 | 4 | 0.4468 | 1325 | 0.8883 | 0.4518 | 0.8883 | 0.1600 |

Warm test targets are unavailable under the current synthetic timeline (event open windows are at most 30 days; the test period begins 38 days after the training cutoff). Therefore these test metrics describe the cold targets only. Interpret top-10 scores with the candidate pool sizes in `results.md`: some queries have ten or fewer eligible events.
