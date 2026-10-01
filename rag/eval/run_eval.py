"""Measure event retrieval hit rate, refusal behavior and latency."""
from __future__ import annotations
import json
import time
from pathlib import Path
from rag.eval.build_testset import OUTPUT, build_testset
from rag.generate import REFUSAL, answer_question
from rag.ingest import INDEX_DIR, rebuild
from rag.retrieve import retrieve
from ml.config import ROOT

def run(output_dir: Path | None = None) -> dict:
    """Run BM25/dense/hybrid retrieval and grounded refusal checks."""
    output_dir = output_dir or ROOT / "docs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (INDEX_DIR / "chunks.json").exists():
        rebuild()
    questions = build_testset()
    # Exclude index deserialization and model startup from per-query latency.
    from rag.retrieve import load_index
    chunks, vectors, index_meta, _, _ = load_index()
    if index_meta.get("backend") == "sentence-transformers":
        from ml.features.embeddings import _sbert
        _sbert(["RAG latency warmup"], show_progress_bar=False)
    retrieval_stats = {}
    all_results = {}
    for mode in ("bm25", "dense", "hybrid"):
        hits = 0
        latencies = []
        results_for_mode = {}
        for index, row in enumerate(questions):
            start = time.perf_counter()
            found = retrieve(row["question"], top_k=3, mode=mode)
            latencies.append((time.perf_counter() - start) * 1000)
            results_for_mode[str(index)] = found
            if row["answerable"] and any(chunk["metadata"].get("event_id") == row["gold_event_id"] for chunk in found):
                hits += 1
        answerable_count = sum(row["answerable"] for row in questions)
        retrieval_stats[mode] = {"hit_at_3": hits / answerable_count if answerable_count else 0,
                                 "hits": hits, "answerable_questions": answerable_count,
                                 "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0}
        all_results[mode] = results_for_mode
    true_positive = true_negative = false_positive = false_negative = 0
    for index, row in enumerate(questions):
        chunks = all_results["hybrid"][str(index)]
        response = answer_question(row["question"], chunks)
        refused = response["answer"] == REFUSAL
        if row["answerable"] and not refused:
            true_positive += 1
        elif row["answerable"] and refused:
            false_negative += 1
        elif not row["answerable"] and refused:
            true_negative += 1
        else:
            false_positive += 1
    refusal_accuracy = (true_positive + true_negative) / len(questions) if questions else 0
    results = {"synthetic_data": True, "question_count": len(questions),
               "answerable_questions": sum(row["answerable"] for row in questions),
               "unanswerable_questions": sum(not row["answerable"] for row in questions),
               "embedding_backend": json.loads((INDEX_DIR / "index_meta.json").read_text(encoding="utf-8")).get("backend"),
               "retrievers": retrieval_stats,
               "refusal_accuracy": refusal_accuracy,
               "refusal_confusion": {"answerable_answered": true_positive, "answerable_refused": false_negative,
                                     "unanswerable_refused": true_negative, "unanswerable_answered": false_positive}}
    (output_dir / "rag_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "rag_results.md").write_text(render_markdown(results), encoding="utf-8")
    print(render_markdown(results))
    print(f"Wrote RAG evaluation to {output_dir / 'rag_results.json'}")
    return results

def render_markdown(results: dict) -> str:
    """Render retrieval evaluation metrics into a compact report."""
    lines = ["# RAG evaluation", "", "Results use generated synthetic event data and a generated question set.",
             f"", f"- Questions: {results['question_count']} ({results['answerable_questions']} answerable, {results['unanswerable_questions']} unanswerable)",
             f"- Dense embedding backend: {results['embedding_backend']}", "", "| Retriever | Hit@3 | Hits / answerable | Mean latency (ms) |", "|---|---:|---:|---:|"]
    for name, metrics in results["retrievers"].items():
        lines.append(f"| {name} | {metrics['hit_at_3']:.3f} | {metrics['hits']} / {metrics['answerable_questions']} | {metrics['mean_latency_ms']:.2f} |")
    lines.extend(["", f"Hybrid refusal accuracy: {results['refusal_accuracy']:.3f}.",
                  f"Answerable answered/refused: {results['refusal_confusion']['answerable_answered']}/{results['refusal_confusion']['answerable_refused']}; unanswerable refused/answered: {results['refusal_confusion']['unanswerable_refused']}/{results['refusal_confusion']['unanswerable_answered']}.", ""])
    return "\n".join(lines)

if __name__ == "__main__":
    run()
