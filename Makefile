.PHONY: setup data train eval index rag-eval api web test
setup:
	python -m pip install -r backend/requirements.txt
	cd frontend && npm install
data:
	python -m ml.data.synthetic_generator
	python -m ml.data.seed_db
train:
	python -m ml.train
eval:
	python -m ml.evaluation.run_experiments
	python -m ml.evaluation.ablations
index:
	python -m rag.ingest
rag-eval:
	python -m rag.eval.run_eval
api:
	python -m uvicorn backend.app.main:app --reload --port 8000
web:
	cd frontend && npm run dev -- --host 127.0.0.1
test:
	python -m pytest -q
