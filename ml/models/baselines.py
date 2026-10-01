"""Popularity, TF-IDF KNN and implicit matrix-factorization baselines."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ml.config import SEED
from ml.data.splits import DataSplit, POSITIVE_TYPES

class PopularityBaseline:
    """Rank events by recent positive-interaction volume."""
    def __init__(self, half_life_days: float = 120.0):
        self.half_life_days = half_life_days
        self.scores: dict[int, float] = {}

    def fit(self, split: DataSplit) -> "PopularityBaseline":
        """Fit exponentially recency-weighted counts on training positives."""
        positives = split.train[split.train["type"].isin(POSITIVE_TYPES)].copy()
        self.scores = {int(eid): 0.0 for eid in split.events["id"]}
        if positives.empty:
            return self
        positives["timestamp"] = pd.to_datetime(positives["timestamp"])
        age = (positives["timestamp"].max() - positives["timestamp"]).dt.total_seconds() / 86400
        positives["weight"] = np.power(0.5, age / self.half_life_days)
        weighted = positives.groupby("event_id")["weight"].sum()
        self.scores.update({int(k): float(v) for k, v in weighted.items()})
        return self

    def score(self, user_ids: list[int] | np.ndarray, event_ids: list[int] | np.ndarray) -> np.ndarray:
        """Return the same popularity ranking for each requested user."""
        vector = np.array([self.scores.get(int(eid), 0.0) for eid in event_ids], dtype=float)
        return np.tile(vector, (len(user_ids), 1))

class TfidfKnnBaseline:
    """Score events by cosine similarity to each user's positive-event profile."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), stop_words="english")
        self.event_ids: list[int] = []
        self.event_matrix = None
        self.user_profiles: dict[int, np.ndarray] = {}

    def fit(self, split: DataSplit) -> "TfidfKnnBaseline":
        """Fit the event text index and average positive history profiles."""
        self.event_ids = [int(i) for i in split.events["id"]]
        event_text = (split.events["title"].fillna("") + " " + split.events["category"].fillna("") + " " + split.events["description"].fillna("")).tolist()
        self.event_matrix = self.vectorizer.fit_transform(event_text)
        row_for_event = {eid: i for i, eid in enumerate(self.event_ids)}
        grouped: dict[int, list[int]] = defaultdict(list)
        for row in split.train[split.train["type"].isin(POSITIVE_TYPES)].itertuples(index=False):
            if int(row.event_id) in row_for_event:
                grouped[int(row.user_id)].append(row_for_event[int(row.event_id)])
        self.user_profiles = {}
        for uid, rows in grouped.items():
            self.user_profiles[uid] = np.asarray(self.event_matrix[rows].mean(axis=0)).ravel()
        return self

    def score(self, user_ids: list[int] | np.ndarray, event_ids: list[int] | np.ndarray) -> np.ndarray:
        """Return user-profile cosine scores for each requested event."""
        event_rows = {eid: row for row, eid in enumerate(self.event_ids)}
        cols = [event_rows.get(int(eid), -1) for eid in event_ids]
        out = np.zeros((len(user_ids), len(event_ids)), dtype=float)
        valid_positions = [j for j, col in enumerate(cols) if col >= 0]
        if self.event_matrix is None or not valid_positions:
            return out
        candidate_matrix = self.event_matrix[[cols[j] for j in valid_positions]]
        for i, uid in enumerate(user_ids):
            profile = self.user_profiles.get(int(uid))
            if profile is not None:
                out[i, valid_positions] = cosine_similarity(profile.reshape(1, -1), candidate_matrix).ravel()
        return out

class MFBaseline:
    """Implicit matrix factorization trained with positive and sampled negative pairs."""
    def __init__(self, factors: int = 32, negatives_per_positive: int = 4, epochs: int = 8,
                 learning_rate: float = 0.01, seed: int = SEED):
        self.factors = factors
        self.negatives_per_positive = negatives_per_positive
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.user_id_map: dict[int, int] = {}
        self.event_id_map: dict[int, int] = {}
        self.user_factors = None
        self.event_factors = None
        self.user_bias = None
        self.event_bias = None

    def fit(self, split: DataSplit) -> "MFBaseline":
        """Optimize a logistic implicit-feedback MF model on the train split."""
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.user_id_map = {int(v): i for i, v in enumerate(sorted(split.users["id"].astype(int).unique()))}
        self.event_id_map = {int(v): i for i, v in enumerate(sorted(split.events["id"].astype(int).unique()))}
        n_users, n_events = len(self.user_id_map), len(self.event_id_map)
        all_positive = split.train[split.train["type"].isin(POSITIVE_TYPES)]
        # Sample negatives from training data only; validation/test labels never
        # influence the model's training pairs.
        known = defaultdict(set)
        for row in all_positive.itertuples(index=False):
            known[int(row.user_id)].add(int(row.event_id))
        positives = [(int(row.user_id), int(row.event_id)) for row in all_positive.itertuples(index=False)]
        if not positives:
            self.user_factors = np.zeros((n_users, self.factors), dtype=np.float32)
            self.event_factors = np.zeros((n_events, self.factors), dtype=np.float32)
            self.user_bias = np.zeros(n_users, dtype=np.float32)
            self.event_bias = np.zeros(n_events, dtype=np.float32)
            return self
        pairs = list(positives)
        labels = [1.0] * len(pairs)
        all_ids = np.array(sorted(self.event_id_map), dtype=int)
        for uid, _ in positives:
            available = np.array([eid for eid in all_ids if eid not in known[uid]], dtype=int)
            if not len(available):
                continue
            sampled = rng.choice(available, size=self.negatives_per_positive, replace=len(available) < self.negatives_per_positive)
            pairs.extend((uid, int(eid)) for eid in sampled)
            labels.extend([0.0] * len(sampled))
        user_idx = torch.tensor([self.user_id_map[u] for u, _ in pairs], dtype=torch.long)
        event_idx = torch.tensor([self.event_id_map[e] for _, e in pairs], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32)
        user_emb = nn.Embedding(n_users, self.factors)
        event_emb = nn.Embedding(n_events, self.factors)
        user_bias = nn.Embedding(n_users, 1)
        event_bias = nn.Embedding(n_events, 1)
        nn.init.normal_(user_emb.weight, std=0.08)
        nn.init.normal_(event_emb.weight, std=0.08)
        optimizer = torch.optim.Adam([*user_emb.parameters(), *event_emb.parameters(), *user_bias.parameters(), *event_bias.parameters()], lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            order = torch.randperm(len(y))
            for batch in order.split(4096):
                logits = (user_emb(user_idx[batch]) * event_emb(event_idx[batch])).sum(dim=1)
                logits = logits + user_bias(user_idx[batch]).squeeze(1) + event_bias(event_idx[batch]).squeeze(1)
                loss = loss_fn(logits, y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.user_factors = user_emb.weight.detach().numpy().copy()
        self.event_factors = event_emb.weight.detach().numpy().copy()
        self.user_bias = user_bias.weight.detach().numpy().ravel().copy()
        self.event_bias = event_bias.weight.detach().numpy().ravel().copy()
        return self

    def score(self, user_ids: list[int] | np.ndarray, event_ids: list[int] | np.ndarray) -> np.ndarray:
        """Return matrix-factorization logits for user/event ID arrays."""
        result = np.zeros((len(user_ids), len(event_ids)), dtype=float)
        if self.user_factors is None or self.event_factors is None:
            return result
        for i, uid in enumerate(user_ids):
            ui = self.user_id_map.get(int(uid))
            if ui is None:
                continue
            for j, eid in enumerate(event_ids):
                ei = self.event_id_map.get(int(eid))
                if ei is not None:
                    result[i, j] = self.user_factors[ui] @ self.event_factors[ei] + self.user_bias[ui] + self.event_bias[ei]
        return result
