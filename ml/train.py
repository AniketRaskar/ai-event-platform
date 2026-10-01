"""Training utilities and entry point for the hybrid NeuMF recommender."""
from __future__ import annotations
import copy
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from ml.config import ARTIFACTS_DIR, CATEGORIES, DEPARTMENTS, ROOT, SEED
from ml.data.splits import DataSplit, EvaluationCase, POSITIVE_TYPES, _cases, build_split
from ml.evaluation.metrics import ndcg_at_k
from ml.features.embeddings import get_event_embeddings
from ml.models.neumf import NeuMF

LOGGER = logging.getLogger(__name__)
torch.set_num_threads(1)

def build_side_features(split: DataSplit) -> np.ndarray:
    """Build user multi-hot interest and department features in mapped order."""
    vectors = np.zeros((len(split.user_id_map), len(CATEGORIES) + len(DEPARTMENTS)), dtype=np.float32)
    categories = {name: i for i, name in enumerate(CATEGORIES)}
    departments = {name: len(CATEGORIES) + i for i, name in enumerate(DEPARTMENTS)}
    for row in split.users.itertuples(index=False):
        index = split.user_id_map[int(row.id)]
        try:
            interests = json.loads(row.interests) if isinstance(row.interests, str) else row.interests
        except (json.JSONDecodeError, TypeError):
            interests = []
        for name in interests or []:
            if name in categories:
                vectors[index, categories[name]] = 1.0
        if row.department in departments:
            vectors[index, departments[row.department]] = 1.0
    return vectors

def ordered_content(split: DataSplit, embedding_cache: Path | None = None) -> tuple[np.ndarray, str]:
    """Load or compute event embeddings and order them by the split's ID map."""
    vectors, backend = get_event_embeddings(split.events, cache_path=embedding_cache) if embedding_cache else get_event_embeddings(split.events)
    ordered = np.zeros((len(split.event_id_map), vectors.shape[1]), dtype=np.float32)
    for eid, vector in zip(split.events["id"].astype(int), vectors):
        ordered[split.event_id_map[int(eid)]] = vector
    return ordered, backend

def make_model(split: DataSplit, event_content: np.ndarray | None, use_gmf: bool = True,
               use_mlp: bool = True, use_content: bool = True, id_dropout: float = 0.3,
               user_side_features: bool = True, latent_dim: int = 32) -> NeuMF:
    """Create a NeuMF instance using aligned user/event IDs and side features."""
    if event_content is None:
        event_content = np.zeros((len(split.event_id_map), 384), dtype=np.float32)
    side = build_side_features(split) if user_side_features else None
    seen = np.zeros(len(split.event_id_map), dtype=bool)
    train_positives = split.train[split.train["type"].isin(POSITIVE_TYPES)]
    for eid in train_positives["event_id"].astype(int).unique():
        seen[split.event_id_map[int(eid)]] = True
    return NeuMF(len(split.user_id_map), len(split.event_id_map), torch.from_numpy(event_content),
                 torch.from_numpy(side) if side is not None else None, torch.from_numpy(seen),
                 latent_dim=latent_dim, use_gmf=use_gmf, use_mlp=use_mlp, use_content=use_content,
                 id_dropout=id_dropout, use_user_side=user_side_features)

def _training_pairs(split: DataSplit, rng: np.random.Generator, negatives: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample epoch training pairs without reading validation or test labels."""
    actions = split.train
    positives = actions[actions["type"].isin(POSITIVE_TYPES)].drop_duplicates(["user_id", "event_id"])
    dislikes = actions[actions["type"] == "dislike"]
    positive_by_user: dict[int, set[int]] = {}
    hard_by_user: dict[int, set[int]] = {}
    for row in positives.itertuples(index=False):
        positive_by_user.setdefault(int(row.user_id), set()).add(int(row.event_id))
    for row in dislikes.itertuples(index=False):
        hard_by_user.setdefault(int(row.user_id), set()).add(int(row.event_id))
    event_ids = np.array(sorted(split.event_id_map), dtype=int)
    user_indices: list[int] = []
    event_indices: list[int] = []
    labels: list[float] = []
    for row in positives.itertuples(index=False):
        uid, eid = int(row.user_id), int(row.event_id)
        user_indices.append(split.user_id_map[uid]); event_indices.append(split.event_id_map[eid]); labels.append(1.0)
        excluded = positive_by_user.get(uid, set()) | hard_by_user.get(uid, set())
        available = np.array([candidate for candidate in event_ids if int(candidate) not in excluded], dtype=int)
        if len(available):
            sampled = rng.choice(available, size=negatives, replace=len(available) < negatives)
            user_indices.extend([split.user_id_map[uid]] * len(sampled))
            event_indices.extend(split.event_id_map[int(candidate)] for candidate in sampled)
            labels.extend([0.0] * len(sampled))
    for uid, events in hard_by_user.items():
        for eid in events:
            user_indices.append(split.user_id_map[uid]); event_indices.append(split.event_id_map[eid]); labels.append(0.0)
    return np.asarray(user_indices), np.asarray(event_indices), np.asarray(labels, dtype=np.float32)

@torch.inference_mode()
def score_cases(model: NeuMF, cases: list[EvaluationCase], split: DataSplit) -> tuple[list[list[int]], list[set[int]]]:
    """Batch-score candidate pairs and turn model scores into case rankings."""
    model.eval()
    flat_users: list[int] = []
    flat_events: list[int] = []
    offsets: list[tuple[int, int]] = []
    for case in cases:
        start = len(flat_users)
        ui = split.user_id_map[case.user_id]
        flat_users.extend([ui] * len(case.candidates))
        flat_events.extend(split.event_id_map[eid] for eid in case.candidates)
        offsets.append((start, len(flat_users)))
    device = next(model.parameters()).device
    user_tensor = torch.tensor(flat_users, dtype=torch.long, device=device)
    event_tensor = torch.tensor(flat_events, dtype=torch.long, device=device)
    scores = model.score_pairs(user_tensor, event_tensor).cpu().numpy()
    ranked, relevant = [], []
    for case, (start, end) in zip(cases, offsets):
        case_scores = scores[start:end]
        order = np.lexsort((np.asarray(case.candidates), -case_scores))
        ranked.append([case.candidates[int(i)] for i in order[:10]])
        relevant.append({case.target_event_id})
    return ranked, relevant

def validation_ndcg(model: NeuMF, split: DataSplit, cases: list[EvaluationCase]) -> float:
    """Calculate NDCG@10 on validation cases for early stopping."""
    ranked, relevant = score_cases(model, cases, split)
    return ndcg_at_k(ranked, relevant, k=10)

def train_model(split: DataSplit, event_content: np.ndarray | None, *, use_gmf: bool = True,
                use_mlp: bool = True, use_content: bool = True, id_dropout: float = 0.3,
                user_side_features: bool = True, negatives: int = 4, max_epochs: int = 8,
                patience: int = 2, seed: int = SEED) -> tuple[NeuMF, list[dict[str, float]]]:
    """Fit NeuMF with epoch-wise negative sampling and validation-NDCG stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(split, event_content, use_gmf, use_mlp, use_content, id_dropout, user_side_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    val_cases = _cases(split.validation, split.events, split.train, split.event_flags)
    best_ndcg = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    waiting = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        user_idx, event_idx, labels = _training_pairs(split, rng, negatives)
        order = rng.permutation(len(labels))
        epoch_loss = 0.0
        for start in range(0, len(order), 2048):
            batch = order[start:start + 2048]
            users_t = torch.as_tensor(user_idx[batch], dtype=torch.long)
            events_t = torch.as_tensor(event_idx[batch], dtype=torch.long)
            targets_t = torch.as_tensor(labels[batch], dtype=torch.float32)
            logits = model(users_t, events_t)
            loss = criterion(logits, targets_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        val_ndcg = validation_ndcg(model, split, val_cases) if val_cases else 0.0
        history.append({"epoch": float(epoch), "train_loss": epoch_loss / max(1, len(labels)), "validation_ndcg10": val_ndcg})
        LOGGER.info("epoch %d/%d loss=%.4f validation_ndcg10=%.4f", epoch, max_epochs, history[-1]["train_loss"], val_ndcg)
        if val_ndcg > best_ndcg + 1e-8:
            best_ndcg = val_ndcg
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(best_weights)
    model.eval()
    LOGGER.info("restored epoch %d with validation NDCG@10 %.4f", best_epoch, max(best_ndcg, 0.0))
    return model, history

def train_final() -> None:
    """Train and persist the headline NeuMF + text-content recommender."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    split = build_split()
    content, backend = ordered_content(split)
    model, history = train_model(split, content, max_epochs=10, patience=3, seed=SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "n_users": len(split.user_id_map), "n_events": len(split.event_id_map),
                  "latent_dim": model.latent_dim, "use_gmf": True, "use_mlp": True, "use_content": True,
                  "id_dropout": 0.3, "use_user_side": True, "embedding_backend": backend}
    torch.save(checkpoint, ARTIFACTS_DIR / "model.pt")
    (ARTIFACTS_DIR / "id_maps.json").write_text(json.dumps({"users": split.user_id_map, "events": split.event_id_map}, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "config.json").write_text(json.dumps({"seed": SEED, "negatives_per_positive": 4, "embedding_backend": backend,
                                                              "training_history": history}, indent=2), encoding="utf-8")
    print(f"Saved final NeuMF checkpoint and mappings in {ARTIFACTS_DIR}; embeddings={backend}; epochs={len(history)}")

if __name__ == "__main__":
    train_final()
