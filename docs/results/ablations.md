# NeuMF ablations

All figures are measured on the synthetic dataset using held-out event interactions. K=10; model selection used validation NDCG@10.

Embedding backend: `sentence-transformers`.

| Variant | Epochs | Validation NDCG@10 | Test queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No GMF | 5 | 0.4340 | 1325 | 0.8747 | 0.4500 | 0.8747 | 0.1600 |
| No MLP | 5 | 0.4277 | 1325 | 0.8875 | 0.4648 | 0.8875 | 0.1600 |
| No content | 5 | 0.4462 | 1325 | 0.8792 | 0.4622 | 0.8792 | 0.1640 |
| ID dropout 0 | 3 | 0.3939 | 1325 | 0.8868 | 0.4660 | 0.8868 | 0.1640 |
| ID dropout 0.3 | 5 | 0.4421 | 1325 | 0.8800 | 0.4530 | 0.8800 | 0.1600 |
| ID dropout 0.5 | 5 | 0.4366 | 1325 | 0.8823 | 0.4555 | 0.8823 | 0.1600 |
| Negatives 1 | 3 | 0.4449 | 1325 | 0.8906 | 0.4883 | 0.8906 | 0.1680 |
| Negatives 4 | 5 | 0.4421 | 1325 | 0.8800 | 0.4530 | 0.8800 | 0.1600 |
| Negatives 8 | 4 | 0.4459 | 1325 | 0.8891 | 0.4524 | 0.8891 | 0.1600 |

Warm test targets are unavailable under the current synthetic timeline (event open windows are at most 30 days; the test period begins 38 days after the training cutoff). Therefore these test metrics describe the cold targets only. Interpret top-10 scores with the candidate pool sizes in `results.md`: some queries have ten or fewer eligible events.
