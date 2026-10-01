"""Tests for the offline event embedding fallback and persistent cache."""
import numpy as np
import pandas as pd
from ml.features.embeddings import DIMENSIONS, get_event_embeddings

def test_tfidf_svd_fallback_is_384_wide_and_cached(tmp_path):
    """Offline vectors align with event IDs and load back from their verified cache."""
    events = pd.DataFrame({"id": [4, 8, 12, 16], "title": ["robotics build", "career workshop", "music festival", "security lab"],
                           "category": ["Robotics", "Career", "Culture", "Security"],
                           "description": ["sensor design project", "resume and interview practice", "student orchestra performance", "network defense exercise"]})
    cache = tmp_path / "event_embeddings.npy"
    first, backend = get_event_embeddings(events, cache_path=cache, prefer_sbert=False)
    second, cached_backend = get_event_embeddings(events, cache_path=cache, prefer_sbert=False)
    assert first.shape == (4, DIMENSIONS)
    assert backend == cached_backend == "tfidf_svd"
    assert np.allclose(first, second)
