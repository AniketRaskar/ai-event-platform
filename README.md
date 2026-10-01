# AI Event Platform v2

A local college event platform combining personalized recommendations with a grounded event-information chatbot. This repository is being built in phases; this first phase provides the scaffold and reproducible synthetic data pipeline.

All generated users, events, friendships, and interactions are synthetic. To create the local data, run `make setup`, then `make data`. Later phases add model training, retrieval, the API, and the web app.

Current generated data is written to `ml/data/generated/` and the SQLite database to `event_platform.db` (both ignored by Git). Model and evaluation results are not available yet; run `make eval` after the evaluation phase is implemented.
