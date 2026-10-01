"""Generate reproducible, synthetic users, events, friendships and interactions."""
from __future__ import annotations
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from ml.config import CATEGORIES, DEPARTMENTS, GENERATED_DIR, N_EVENTS, N_USERS, SEED
from ml.data.event_docs import build_event_copy

# Anchor the twelve-month synthetic window to the day generation runs so the
# demo contains upcoming events instead of aging into an all-past catalog.
BASE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=364)

def _users(rng: random.Random) -> tuple[list[dict], list[set[str]], list[dict]]:
    users: list[dict] = []
    interests: list[set[str]] = []
    group_sizes = [25] * 20
    group_categories = [rng.sample(CATEGORIES, 2) for _ in group_sizes]
    for i in range(N_USERS):
        group = i // 25
        favs = set(group_categories[group])
        if rng.random() > 0.6:
            favs = set(rng.sample(CATEGORIES, rng.randint(1, 3)))
        interests.append(favs)
        users.append({"id": i + 1, "name": f"Student {i + 1:03d}", "email": f"student{i + 1:03d}@synthetic.college.edu",
                      "department": rng.choice(DEPARTMENTS), "year": rng.randint(1, 4),
                      "interests": json.dumps(sorted(favs)), "role": "student"})
    edges: set[tuple[int, int]] = set()
    for group in range(20):
        members = list(range(group * 25, (group + 1) * 25))
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                if rng.random() < 0.12:
                    edges.add((left + 1, right + 1))
    for user in range(N_USERS):
        others = [j for j in range(N_USERS) if j // 25 != user // 25]
        for friend in rng.sample(others, rng.randint(0, 3)):
            edges.add(tuple(sorted((user + 1, friend + 1))))
    friendships = [{"user_id": a, "friend_id": b} for a, b in sorted(edges)]
    friendships += [{"user_id": b, "friend_id": a} for a, b in sorted(edges)]
    return users, interests, friendships

def generate(output_dir: Path = GENERATED_DIR) -> dict[str, pd.DataFrame]:
    """Generate all synthetic CSV tables and print a compact data summary."""
    rng = random.Random(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    users, user_interests, friendships = _users(rng)
    events: list[dict] = []
    for i in range(N_EVENTS):
        category = CATEGORIES[i % len(CATEGORIES)]
        posted = BASE + timedelta(days=rng.randint(0, 364), hours=rng.randint(8, 18))
        start = posted + timedelta(days=rng.randint(7, 30), hours=rng.randint(0, 8))
        copy = build_event_copy(rng, category, f"EVT-{i+1:03d}", posted, start)
        events.append({"id": i + 1, **copy, "category": category, "department": rng.choice(DEPARTMENTS),
                       "start_time": start, "end_time": start + timedelta(hours=rng.randint(2, 8)),
                       "posted_at": posted, "created_by": None,
                       "popularity_weight": 1 / ((i + 1) ** 0.8)})
    # Popularity follows a power law. Friend registrations only influence later encounters.
    interactions: list[dict] = []
    registered_at: dict[int, datetime] = {}
    friend_sets = {u["id"]: set() for u in users}
    for edge in friendships:
        friend_sets[edge["user_id"]].add(edge["friend_id"])
    for event in sorted(events, key=lambda e: e["posted_at"]):
        for user in users:
            uid = user["id"]
            matches = event["category"] in user_interests[uid - 1]
            dept_match = event["department"] == user["department"]
            social = sum(1 for f in friend_sets[uid] if (event["id"] * 1000 + f) in registered_at)
            trend = min(1.0, event["popularity_weight"] * 3)
            engage = min(0.25, 0.018 + 0.105 * matches + 0.035 * dept_match + 0.055 * trend + 0.008 * min(social, 5))
            if rng.random() < min(0.72, 0.18 + engage * 1.7):
                stamp = event["posted_at"] + timedelta(seconds=rng.randint(0, int((event["start_time"] - event["posted_at"]).total_seconds())))
                interactions.append({"user_id": uid, "event_id": event["id"], "type": "view", "rating": None, "comment": None, "timestamp": stamp})
            if rng.random() < engage:
                stamp = event["posted_at"] + timedelta(seconds=rng.randint(0, int((event["start_time"] - event["posted_at"]).total_seconds())))
                interactions.append({"user_id": uid, "event_id": event["id"], "type": "like", "rating": None, "comment": None, "timestamp": stamp})
                if rng.random() < 0.25:
                    reg_stamp = min(event["start_time"], stamp + timedelta(minutes=rng.randint(1, 600)))
                    interactions.append({"user_id": uid, "event_id": event["id"], "type": "register", "rating": None, "comment": None, "timestamp": reg_stamp})
                    registered_at[event["id"] * 1000 + uid] = reg_stamp
            elif rng.random() < engage * 0.35:
                stamp = event["posted_at"] + timedelta(seconds=rng.randint(0, int((event["start_time"] - event["posted_at"]).total_seconds())))
                interactions.append({"user_id": uid, "event_id": event["id"], "type": "register", "rating": None, "comment": None, "timestamp": stamp})
                registered_at[event["id"] * 1000 + uid] = stamp
    tables = {"users": pd.DataFrame(users), "events": pd.DataFrame(events), "friendships": pd.DataFrame(friendships), "interactions": pd.DataFrame(interactions)}
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    _print_summary(tables)
    return tables

def _print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Print counts and distribution checks without claiming model results."""
    users, events, friends, actions = (tables[k] for k in ("users", "events", "friendships", "interactions"))
    pairs = actions[["user_id", "event_id"]].drop_duplicates().shape[0]
    possible = len(users) * len(events)
    print("Synthetic dataset summary (SEED=42)")
    print(f"users={len(users)} events={len(events)} friendship_rows={len(friends)} interactions={len(actions)}")
    print("interactions by type:")
    print(actions["type"].value_counts().sort_index().to_string())
    print(f"user-event sparsity={1 - pairs / possible:.4f} ({pairs}/{possible} pairs observed)")
    print("events per category:")
    print(events["category"].value_counts().sort_index().to_string())
    print(f"friends per user: mean={(len(friends) / len(users)):.2f}, min={friends.groupby('user_id').size().reindex(users.id, fill_value=0).min()}, max={friends.groupby('user_id').size().reindex(users.id, fill_value=0).max()}")

if __name__ == "__main__":
    generate()
