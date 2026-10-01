"""Sanity checks for generated synthetic records."""
import pandas as pd
from ml.config import N_EVENTS, N_USERS
from ml.data.synthetic_generator import generate

def test_generated_counts_and_integrity(tmp_path):
    """Data has expected scale, respects event windows and has unique action keys."""
    tables = generate(tmp_path)
    users, events, actions = tables["users"], tables["events"], tables["interactions"]
    assert len(users) == N_USERS
    assert len(events) == N_EVENTS
    assert actions[["user_id", "event_id", "type"]].duplicated().sum() == 0
    joined = actions.merge(events[["id", "posted_at", "start_time"]], left_on="event_id", right_on="id")
    assert (pd.to_datetime(joined.timestamp) >= pd.to_datetime(joined.posted_at)).all()
    assert (pd.to_datetime(joined.timestamp) <= pd.to_datetime(joined.start_time)).all()
    assert (tmp_path / "users.csv").exists()
