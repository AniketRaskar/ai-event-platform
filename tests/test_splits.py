"""Tests that chronological splits and candidate sets do not leak."""
from ml.data.splits import POSITIVE_TYPES, build_split

def test_chronological_split_has_no_leakage():
    """All train timestamps precede validation/test and cold has no train positives."""
    split = build_split()
    assert split.train.timestamp.max() < split.validation.timestamp.min()
    assert split.validation.timestamp.max() < split.test.timestamp.min()
    cold_ids = set(split.event_flags.loc[split.event_flags.split == "cold", "event_id"])
    train_positive_ids = set(split.train.loc[split.train.type.isin(POSITIVE_TYPES), "event_id"])
    assert cold_ids.isdisjoint(train_positive_ids)

def test_test_candidates_are_open_and_exclude_train_positives():
    """Every case target is open at its query time and its candidates exclude seen positives."""
    split = build_split()
    train_positive = split.train[split.train.type.isin(POSITIVE_TYPES)]
    seen = train_positive.groupby("user_id").event_id.apply(set).to_dict()
    events = split.events.set_index("id")
    assert split.test_cases
    for case in split.test_cases:
        assert case.target_event_id in case.candidates
        assert not (set(case.candidates) & seen.get(case.user_id, set()))
        for event_id in case.candidates:
            row = events.loc[event_id]
            assert row.posted_at <= case.timestamp <= row.start_time
