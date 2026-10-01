# RAG evaluation

Results use generated synthetic event data and a generated question set.

- Questions: 60 (50 answerable, 10 unanswerable)
- Dense embedding backend: sentence-transformers

| Retriever | Hit@3 | Hits / answerable | Mean latency (ms) |
|---|---:|---:|---:|
| bm25 | 1.000 | 50 / 50 | 10.11 |
| dense | 0.840 | 42 / 50 | 45.11 |
| hybrid | 1.000 | 50 / 50 | 41.37 |

Hybrid refusal accuracy: 1.000.
Answerable answered/refused: 50/0; unanswerable refused/answered: 10/0.
