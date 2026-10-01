"""Cached 384-dimensional event text embeddings with an offline SVD fallback."""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from ml.config import ARTIFACTS_DIR

LOGGER = logging.getLogger(__name__)
DEFAULT_CACHE = ARTIFACTS_DIR / "event_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384

def event_texts(events: pd.DataFrame) -> list[str]:
    """Combine each event's title, category and description for embedding."""
    return [(f"{row.title} {row.category} {row.description}").strip() for row in events.itertuples()]

def _digest(event_ids: list[int], texts: list[str]) -> str:
    """Hash IDs and text so stale caches are not silently reused."""
    payload = json.dumps(list(zip(event_ids, texts)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _sbert(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode descriptions with the specified compact sentence transformer."""
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
    from huggingface_hub import try_to_load_from_cache
    cached_modules = try_to_load_from_cache(MODEL_NAME, "modules.json")
    if not isinstance(cached_modules, str) and os.getenv("HF_HUB_OFFLINE", "0") != "1":
        try:
            with urlopen(Request("https://huggingface.co", method="HEAD"), timeout=3):
                pass
        except (OSError, URLError, HTTPError) as exc:
            raise RuntimeError("Hugging Face is unreachable and the SBERT model is not cached locally") from exc
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    if vectors.shape[1] != DIMENSIONS:
        raise ValueError(f"Expected {DIMENSIONS} SBERT dimensions, received {vectors.shape[1]}")
    return vectors.astype(np.float32)

def _tfidf_svd(texts: list[str]) -> np.ndarray:
    """Create a 384-wide TF-IDF/SVD fallback, padding the small corpus as needed."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    matrix = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), stop_words="english").fit_transform(texts)
    n_components = min(DIMENSIONS, max(1, matrix.shape[0] - 1), max(1, matrix.shape[1] - 1))
    if min(matrix.shape) <= 1 or matrix.nnz == 0:
        reduced = matrix.toarray()
    else:
        reduced = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42).fit_transform(matrix)
    output = np.zeros((matrix.shape[0], DIMENSIONS), dtype=np.float32)
    output[:, :reduced.shape[1]] = reduced.astype(np.float32)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    output /= np.maximum(norms, 1e-12)
    return output

def get_event_embeddings(events: pd.DataFrame, cache_path: Path = DEFAULT_CACHE,
                         force_rebuild: bool = False, prefer_sbert: bool = True) -> tuple[np.ndarray, str]:
    """Return vectors aligned to events, using a verified cache or computing anew."""
    event_ids = [int(v) for v in events["id"].tolist()]
    texts = event_texts(events)
    checksum = _digest(event_ids, texts)
    metadata_path = cache_path.with_suffix(".json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not force_rebuild and cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path)
            if metadata.get("sha256") == checksum and metadata.get("event_ids") == event_ids and cached.shape == (len(event_ids), DIMENSIONS):
                return cached.astype(np.float32, copy=False), str(metadata.get("backend", "unknown"))
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.warning("Ignoring unreadable event embedding cache; rebuilding it.")
    backend = "tfidf_svd"
    if prefer_sbert:
        try:
            vectors = _sbert(texts)
            backend = "sentence-transformers"
        except Exception as exc:  # model downloads can fail offline or be unavailable
            LOGGER.warning("SBERT could not be loaded (%s). Falling back to TF-IDF + TruncatedSVD (padded to %d dimensions).", exc, DIMENSIONS)
            vectors = _tfidf_svd(texts)
    else:
        LOGGER.warning("Using TF-IDF + TruncatedSVD fallback as requested; SBERT was not attempted.")
        vectors = _tfidf_svd(texts)
    np.save(cache_path, vectors.astype(np.float32))
    metadata_path.write_text(json.dumps({"event_ids": event_ids, "sha256": checksum, "backend": backend,
                                         "dimensions": DIMENSIONS, "model": MODEL_NAME if backend == "sentence-transformers" else None}, indent=2), encoding="utf-8")
    return vectors.astype(np.float32), backend
