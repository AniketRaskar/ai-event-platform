"""Train baseline recommenders and save actual ranking measurements."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ml.config import ROOT, SEED
from ml.data.splits import DataSplit, build_split
from ml.evaluation.metrics import evaluate_rankings
from ml.models.baselines import MFBaseline, PopularityBaseline, TfidfKnnBaseline

K = 10
SEGMENTS = ("all", "warm", "cold")

def evaluate_model(model, split: DataSplit) -> dict[str, dict[str, float | int]]:
    """Rank each case's eligible candidates and calculate metrics by target type."""
    ranked_by_segment: dict[str, list[list[int]]] = {segment: [] for segment in SEGMENTS}
    truth_by_segment: dict[str, list[set[int]]] = {segment: [] for segment in SEGMENTS}
    for case in split.test_cases:
        if not case.candidates:
            continue
        scores = model.score([case.user_id], list(case.candidates))[0]
        order = np.lexsort((np.asarray(case.candidates), -scores))
        ranked = [case.candidates[int(i)] for i in order[:K]]
        for segment in ("all", case.target_split):
            ranked_by_segment[segment].append(ranked)
            truth_by_segment[segment].append({case.target_event_id})
    result = {}
    for segment in SEGMENTS:
        if ranked_by_segment[segment]:
            result[segment] = {**evaluate_rankings(ranked_by_segment[segment], truth_by_segment[segment], len(split.events), K),
                               "queries": len(ranked_by_segment[segment])}
        else:
            result[segment] = {"recall": None, "ndcg": None, "hit_rate": None,
                               "catalog_coverage": None, "queries": 0}
    return result

def run(output_dir: Path | None = None) -> dict:
    """Fit three baselines, evaluate on held-out interactions, and write JSON/Markdown."""
    output_dir = output_dir or ROOT / "docs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    split = build_split()
    models = {
        "Popularity": PopularityBaseline(),
        "TF-IDF KNN": TfidfKnnBaseline(),
        "Matrix Factorization": MFBaseline(seed=SEED),
    }
    candidate_counts = [len(case.candidates) for case in split.test_cases]
    results = {"synthetic_data": True, "seed": SEED, "k": K,
               "split": {"train_cutoff": split.train_cutoff.isoformat(), "validation_cutoff": split.validation_cutoff.isoformat(),
                         "train_rows": len(split.train), "validation_rows": len(split.validation), "test_rows": len(split.test),
                         "test_queries": len(split.test_cases),
                         "test_candidate_count": {"min": min(candidate_counts, default=0),
                                                  "median": float(np.median(candidate_counts)) if candidate_counts else 0,
                                                  "mean": float(np.mean(candidate_counts)) if candidate_counts else 0,
                                                  "max": max(candidate_counts, default=0),
                                                  "fraction_at_most_k": float(np.mean(np.array(candidate_counts) <= K)) if candidate_counts else 0}},
               "models": {}}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(split)
        metrics = evaluate_model(model, split)
        results["models"][name] = {"fit_and_eval_seconds": round(time.perf_counter() - started, 3), "segments": metrics}
        print(f"Completed {name}: {metrics['all']}")
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path = output_dir / "results.md"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    return results

def render_markdown(results: dict) -> str:
    """Render the result JSON as readable segment-by-model tables."""
    lines = ["# Baseline recommendation evaluation", "", "All interactions and metrics below use the generated **synthetic** dataset. Metrics are from the latest `python -m ml.evaluation.run_experiments` run. Rankings are evaluated at K=10; each held-out positive user/event pair is one query.", "",
             f"- Seed: {results['seed']}", f"- Queries: {results['split']['test_queries']}",
             f"- Train / validation / test interaction rows: {results['split']['train_rows']} / {results['split']['validation_rows']} / {results['split']['test_rows']}", ""]
    for segment in SEGMENTS:
        lines.extend([f"## {segment.title()} targets", "", "| Model | Queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |", "|---|---:|---:|---:|---:|---:|"])
        for name, model_result in results["models"].items():
            m = model_result["segments"][segment]
            values = [f"{m[key]:.4f}" if m[key] is not None else "—" for key in ("recall", "ndcg", "hit_rate", "catalog_coverage")]
            lines.append(f"| {name} | {m['queries']} | " + " | ".join(values) + " |")
        lines.append("")
    pool = results["split"]["test_candidate_count"]
    lines.extend(["## Notes", "", "Each query has one held-out positive target, so Recall@10 and HitRate@10 coincide. Catalog coverage is the share of the full event catalog appearing in the top 10 across that segment's queries.",
                  f"Candidate pool sizes at test time: min {pool['min']}, median {pool['median']:.1f}, mean {pool['mean']:.2f}, max {pool['max']}; {pool['fraction_at_most_k']:.1%} of queries have at most K candidates.",
                  "No warm test targets were available, so warm metrics are shown as unavailable (—), not zero. This follows from the synthetic timing setup: event windows last at most 30 days, while test interactions start 38 days after the training cutoff. This also keeps the top-10 evaluation ceiling high because a substantial share of candidate pools are already at most ten items.", ""])
    return "\n".join(lines)

if __name__ == "__main__":
    run()
