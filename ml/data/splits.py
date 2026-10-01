"""Chronological data splits and leakage-safe candidate sets for offline evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from ml.config import GENERATED_DIR

POSITIVE_TYPES = {"like", "register"}

@dataclass(frozen=True)
class EvaluationCase:
    """One test-time ranking request with the catalog open at that instant."""
    user_id: int
    timestamp: pd.Timestamp
    target_event_id: int
    candidates: tuple[int, ...]
    target_split: str

@dataclass
class DataSplit:
    """Prepared train/validation/test records and identifiers for experiments."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    events: pd.DataFrame
    users: pd.DataFrame
    event_flags: pd.DataFrame
    test_cases: list[EvaluationCase]
    user_id_map: dict[int, int]
    event_id_map: dict[int, int]
    train_cutoff: pd.Timestamp
    validation_cutoff: pd.Timestamp

def load_generated(data_dir: Path = GENERATED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the synthetic users, events and interactions from CSV files."""
    needed = [data_dir / f"{name}.csv" for name in ("users", "events", "interactions")]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError("Synthetic CSVs are missing; run `make data` first: " + ", ".join(missing))
    users, events, interactions = (pd.read_csv(path) for path in needed)
    for col in ("posted_at", "start_time", "end_time", "registration_deadline"):
        events[col] = pd.to_datetime(events[col])
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    return users, events, interactions

def _cases(test: pd.DataFrame, events: pd.DataFrame, train: pd.DataFrame, flags: pd.DataFrame) -> list[EvaluationCase]:
    """Create one positive evaluation query per unique held-out user/event pair."""
    train_positive = train[train["type"].isin(POSITIVE_TYPES)]
    seen_by_user = train_positive.groupby("user_id")["event_id"].apply(set).to_dict()
    positive_test = test[test["type"].isin(POSITIVE_TYPES)].sort_values("timestamp")
    # like and register can coexist, but should not count the same response twice.
    positive_test = positive_test.drop_duplicates(["user_id", "event_id"], keep="first")
    flag_map = flags.set_index("event_id")["split"].to_dict()
    cases = []
    for row in positive_test.itertuples(index=False):
        stamp = pd.Timestamp(row.timestamp)
        already_seen = seen_by_user.get(int(row.user_id), set())
        open_events = events[(events["posted_at"] <= stamp) & (events["start_time"] >= stamp)]
        candidates = tuple(sorted(int(eid) for eid in open_events["id"].tolist() if int(eid) not in already_seen))
        target = int(row.event_id)
        if target in candidates:
            cases.append(EvaluationCase(int(row.user_id), stamp, target, candidates, flag_map[target]))
    return cases

def make_split(users: pd.DataFrame | None = None, events: pd.DataFrame | None = None,
               interactions: pd.DataFrame | None = None, data_dir: Path = GENERATED_DIR) -> DataSplit:
    """Split by elapsed timeline: first 70%, next 10%, final 20%."""
    if users is None or events is None or interactions is None:
        users, events, interactions = load_generated(data_dir)
    interactions = interactions.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    events = events.copy()
    for col in ("posted_at", "start_time"):
        events[col] = pd.to_datetime(events[col])
    beginning = interactions["timestamp"].min()
    ending = interactions["timestamp"].max()
    span = ending - beginning
    train_cutoff = beginning + span * 0.70
    validation_cutoff = beginning + span * 0.80
    train = interactions[interactions["timestamp"] < train_cutoff].copy()
    validation = interactions[(interactions["timestamp"] >= train_cutoff) & (interactions["timestamp"] < validation_cutoff)].copy()
    test = interactions[interactions["timestamp"] >= validation_cutoff].copy()
    positives = train[train["type"].isin(POSITIVE_TYPES)].groupby("event_id").size()
    flags = events[["id", "posted_at", "start_time"]].rename(columns={"id": "event_id"}).copy()
    flags["train_positive_count"] = flags["event_id"].map(positives).fillna(0).astype(int)
    flags["split"] = flags["train_positive_count"].map(lambda n: "warm" if n > 0 else "cold")
    cases = _cases(test, events, train, flags)
    return DataSplit(train=train, validation=validation, test=test, events=events, users=users,
                     event_flags=flags, test_cases=cases,
                     user_id_map={int(v): i for i, v in enumerate(sorted(users["id"].astype(int).unique()))},
                     event_id_map={int(v): i for i, v in enumerate(sorted(events["id"].astype(int).unique()))},
                     train_cutoff=pd.Timestamp(train_cutoff), validation_cutoff=pd.Timestamp(validation_cutoff))

def build_split(data_dir: Path = GENERATED_DIR) -> DataSplit:
    """Load generated data and prepare an evaluation split."""
    return make_split(data_dir=data_dir)
