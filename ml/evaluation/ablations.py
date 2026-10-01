"""Run compact NeuMF architecture and training-setting ablations."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ml.config import ROOT, SEED
from ml.data.splits import build_split
from ml.evaluation.run_experiments import evaluate_neumf
from ml.features.embeddings import get_event_embeddings
from ml.train import ordered_content, train_model

def configurations() -> list[tuple[str, dict]]:
    """Return architecture, ID-dropout and negative-sampling variants."""
    base = {"use_gmf": True, "use_mlp": True, "use_content": True,
            "id_dropout": 0.3, "user_side_features": True, "negatives": 4}
    result = []
    for name, values in [("No GMF", {"use_gmf": False}), ("No MLP", {"use_mlp": False}),
                         ("No content", {"use_content": False})]:
        result.append((name, {**base, **values}))
    for value in (0.0, 0.3, 0.5):
        result.append((f"ID dropout {value:g}", {**base, "id_dropout": value}))
    for value in (1, 4, 8):
        result.append((f"Negatives {value}", {**base, "negatives": value}))
    return result

def run(output_dir: Path | None = None) -> dict:
    """Fit every ablation and write the observed metrics to disk."""
    output_dir = output_dir or ROOT / "docs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    split = build_split()
    content, backend = ordered_content(split)
    results = {"synthetic_data": True, "seed": SEED, "k": 10, "embedding_backend": backend, "ablations": {}}
    for name, config in configurations():
        started = time.perf_counter()
        model, history = train_model(split, content, max_epochs=5, patience=2, seed=SEED, **config)
        metrics = evaluate_neumf(model, split)
        results["ablations"][name] = {"settings": config, "epochs": len(history), "best_validation_ndcg10": max((x["validation_ndcg10"] for x in history), default=0),
                                       "seconds": round(time.perf_counter() - started, 3), "segments": metrics}
        print(f"Completed {name}: all={metrics['all']} cold={metrics['cold']}")
    (output_dir / "ablations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "ablations.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"Wrote {output_dir / 'ablations.md'}")
    return results

def render_markdown(results: dict) -> str:
    """Format ablation metrics and call out unavailable target segments."""
    lines = ["# NeuMF ablations", "", "All figures are measured on the synthetic dataset using held-out event interactions. K=10; model selection used validation NDCG@10.",
             f"", f"Embedding backend: `{results['embedding_backend']}`.", "", "| Variant | Epochs | Validation NDCG@10 | Test queries | Recall@10 | NDCG@10 | HitRate@10 | Catalog coverage@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results["ablations"].items():
        m = result["segments"]["all"]
        values = [f"{m[key]:.4f}" if m[key] is not None else "—" for key in ("recall", "ndcg", "hit_rate", "catalog_coverage")]
        lines.append(f"| {name} | {result['epochs']} | {result['best_validation_ndcg10']:.4f} | {m['queries']} | " + " | ".join(values) + " |")
    lines.extend(["", "Warm test targets are unavailable under the current synthetic timeline (event open windows are at most 30 days; the test period begins 38 days after the training cutoff). Therefore these test metrics describe the cold targets only. Interpret top-10 scores with the candidate pool sizes in `results.md`: some queries have ten or fewer eligible events.", ""])
    return "\n".join(lines)

if __name__ == "__main__":
    run()
