"""Turn event records and sample campus guides into a persisted retrieval index."""
from __future__ import annotations
import json
import logging
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from ml.config import ARTIFACTS_DIR, DATABASE_PATH, ROOT

LOGGER = logging.getLogger(__name__)
RAG_DIR = Path(__file__).resolve().parent
GENERAL_DOCS_DIR = RAG_DIR / "data" / "general_docs"
INDEX_DIR = RAG_DIR / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_WRITE_LOCK = threading.RLock()

def _value(event: Any, name: str, default=None):
    """Read an event field from a SQLAlchemy row or mapping."""
    return event.get(name, default) if isinstance(event, dict) else getattr(event, name, default)

def _iso(value) -> str | None:
    """Serialize dates in a stable ISO form."""
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)

def event_to_chunks(event: Any) -> list[dict]:
    """Create a header, rules and one FAQ chunk per event FAQ item."""
    event_id = int(_value(event, "id"))
    start_time = _value(event, "start_time")
    common = {"event_id": event_id, "category": _value(event, "category"), "department": _value(event, "department"),
              "start_time": _iso(start_time), "fee": float(_value(event, "fee", 0) or 0)}
    title = _value(event, "title", "Untitled event")
    fee = common["fee"]
    header = (f"{title}. Event date: {_iso(start_time)}. Venue: {_value(event, 'venue')}. "
              f"Fee: {'free' if fee == 0 else f'₹{fee:g}'}. Category: {common['category']}. "
              f"Department: {common['department']}. Eligibility: {_value(event, 'eligibility')} "
              f"Description: {_value(event, 'description')}")
    chunks = [{"text": header, "metadata": {**common, "doc_type": "event_header", "title": title}},
              {"text": f"{title} rules: {_value(event, 'rules', '')}", "metadata": {**common, "doc_type": "event_rules", "title": title}}]
    faq = _value(event, "faq", []) or []
    if isinstance(faq, str):
        try:
            faq = json.loads(faq)
        except json.JSONDecodeError:
            faq = []
    for number, item in enumerate(faq, start=1):
        question = item.get("q", "") if isinstance(item, dict) else ""
        answer = item.get("a", "") if isinstance(item, dict) else ""
        chunks.append({"text": f"{title} FAQ {number}. Question: {question} Answer: {answer}",
                       "metadata": {**common, "doc_type": "event_faq", "title": title, "faq_number": number}})
    return chunks

def general_doc_chunks(directory: Path = GENERAL_DOCS_DIR) -> list[dict]:
    """Read sample campus markdown guides as independently citable chunks."""
    chunks = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            chunks.append({"text": text, "metadata": {"event_id": None, "category": None, "department": None,
                          "start_time": None, "fee": None, "doc_type": "general", "doc_id": path.stem,
                          "title": path.stem.replace("_", " ").title()}})
    return chunks

def make_chunks(events: Iterable[Any], general_dir: Path = GENERAL_DOCS_DIR) -> list[dict]:
    """Build the complete event and general-document corpus."""
    chunks = []
    for event in events:
        chunks.extend(event_to_chunks(event))
    chunks.extend(general_doc_chunks(general_dir))
    return chunks

def tokenize(text: str) -> list[str]:
    """Normalize English words, event codes and common technical tokens."""
    return re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", text.lower())

def _encode(texts: list[str], backend: str | None = None, vectorizer=None) -> tuple[np.ndarray, str, Any]:
    """Encode corpus text with SBERT, with TF-IDF cosine as an offline fallback."""
    if backend == "tfidf" and vectorizer is not None:
        return vectorizer.transform(texts).toarray().astype(np.float32), "tfidf", vectorizer
    try:
        from ml.features.embeddings import _sbert
        return _sbert(texts), "sentence-transformers", None
    except Exception as exc:
        if backend == "sentence-transformers":
            raise RuntimeError("The saved RAG index uses SBERT, but its model could not be loaded") from exc
        LOGGER.warning("SBERT unavailable for RAG index (%s); using TF-IDF cosine vectors.", exc)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = vectorizer or TfidfVectorizer(max_features=12000, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(texts) if not hasattr(vectorizer, "vocabulary_") else vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32), "tfidf", vectorizer

def _write_index(chunks: list[dict], vectors: np.ndarray, backend: str, index_dir: Path, vectorizer=None) -> None:
    """Persist chunks, normalized dense vectors and index settings atomically enough for local use."""
    index_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk["metadata"] = {key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in chunk["metadata"].items()}
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(index_dir / "dense_vectors.npy", vectors.astype(np.float32))
    (index_dir / "index_meta.json").write_text(json.dumps({"backend": backend, "model": MODEL_NAME if backend == "sentence-transformers" else None,
                                                               "vector_width": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
                                                               "chunk_count": len(chunks)}, indent=2), encoding="utf-8")
    vector_path = index_dir / "tfidf_vectorizer.pkl"
    if backend == "tfidf" and vectorizer is not None:
        with vector_path.open("wb") as handle:
            pickle.dump(vectorizer, handle)
    elif vector_path.exists():
        vector_path.unlink()

def rebuild(events: Iterable[Any] | None = None, index_dir: Path = INDEX_DIR,
            general_dir: Path = GENERAL_DOCS_DIR) -> list[dict]:
    """Recreate and persist the full RAG index from SQLite and sample guides."""
    if events is None:
        from sqlalchemy import select
        from backend.app.database import SessionLocal
        from backend.app.models import Event
        db = SessionLocal()
        try:
            events = list(db.scalars(select(Event).order_by(Event.id)))
        finally:
            db.close()
    chunks = make_chunks(events, general_dir)
    if not chunks:
        raise ValueError("The RAG corpus is empty; run `make data` and add general documents first")
    with _WRITE_LOCK:
        vectors, backend, vectorizer = _encode([chunk["text"] for chunk in chunks])
        _write_index(chunks, vectors, backend, index_dir, vectorizer)
    print(f"Indexed {len(chunks)} chunks with {backend} embeddings at {index_dir}")
    return chunks

def add_event(event: Any, index_dir: Path = INDEX_DIR) -> list[dict]:
    """Add or replace one event's chunks, then refresh persisted retrieval structures."""
    with _WRITE_LOCK:
        chunks_path = index_dir / "chunks.json"
        if not chunks_path.exists():
            # First-time initialization must index every event already in SQLite,
            # not just the newly created one.
            return rebuild(index_dir=index_dir)
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        event_id = int(_value(event, "id"))
        chunks = [chunk for chunk in chunks if chunk["metadata"].get("event_id") != event_id]
        new_chunks = event_to_chunks(event)
        chunks.extend(new_chunks)
        metadata = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        backend = metadata.get("backend", "sentence-transformers")
        vectorizer_path = index_dir / "tfidf_vectorizer.pkl"
        vectorizer = None
        if backend == "tfidf" and vectorizer_path.exists():
            with vectorizer_path.open("rb") as handle:
                vectorizer = pickle.load(handle)
        existing_vectors = np.load(index_dir / "dense_vectors.npy")
        existing_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        # Keep vectors aligned with the retained old chunk order and embed only the new chunks.
        kept_indices = [i for i, chunk in enumerate(existing_chunks) if chunk["metadata"].get("event_id") != event_id]
        kept_vectors = existing_vectors[kept_indices]
        new_vectors, actual_backend, vectorizer = _encode([chunk["text"] for chunk in new_chunks], backend, vectorizer)
        if actual_backend != backend:
            # An index created with a model that is no longer available needs a consistent full refresh.
            vectors, backend, vectorizer = _encode([chunk["text"] for chunk in chunks])
        else:
            vectors = np.vstack([kept_vectors, new_vectors])
        _write_index(chunks, vectors, backend, index_dir, vectorizer)
    return chunks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rebuild()
