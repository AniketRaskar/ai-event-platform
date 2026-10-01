"""BM25, dense cosine and reciprocal-rank-fusion retrieval with metadata filters."""
from __future__ import annotations
import json
import logging
import os
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from rag.ingest import INDEX_DIR, MODEL_NAME, _encode, tokenize
from rag.query_rewrite import rewrite_query

LOGGER = logging.getLogger(__name__)
RRF_K = 60
DEFAULT_TOP_K = 5

@lru_cache(maxsize=2)
def _load(index_dir_string: str, fingerprint: int):
    """Load one index snapshot and build BM25 over its saved chunk text."""
    del fingerprint
    index_dir = Path(index_dir_string)
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(index_dir / "dense_vectors.npy").astype(np.float32)
    metadata = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
    vectorizer = None
    path = index_dir / "tfidf_vectorizer.pkl"
    if path.exists():
        with path.open("rb") as handle:
            vectorizer = pickle.load(handle)
    return chunks, vectors, metadata, bm25, vectorizer

def load_index(index_dir: Path = INDEX_DIR):
    """Load a persisted RAG index, invalidating the in-process cache on file updates."""
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"RAG index not found at {index_dir}; run `make index` first")
    fingerprint = max((path.stat().st_mtime_ns for path in index_dir.glob("*")), default=0)
    return _load(str(index_dir.resolve()), fingerprint)

def _query_vector(query: str, backend: str, vectorizer, width: int) -> np.ndarray:
    """Encode a query using the same backend that built the persisted corpus vectors."""
    if backend == "sentence-transformers":
        from ml.features.embeddings import _sbert
        vector = _sbert([query], show_progress_bar=False)[0]
    elif vectorizer is not None:
        vector = vectorizer.transform([query]).toarray()[0].astype(np.float32)
    else:
        vector = np.zeros(width, dtype=np.float32)
    if len(vector) != width:
        raise ValueError(f"Query vector width {len(vector)} does not match index width {width}")
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def _as_datetime(value) -> datetime | None:
    """Parse optional metadata-filter date values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)

def _matches(meta: dict, filters: dict) -> bool:
    """Check whether a chunk matches all event metadata constraints."""
    if filters.get("category") and str(meta.get("category", "")).casefold() != str(filters["category"]).casefold():
        return False
    if filters.get("department") and str(meta.get("department", "")).casefold() != str(filters["department"]).casefold():
        return False
    fee = meta.get("fee")
    if "fee" in filters and (fee is None or float(fee) != float(filters["fee"])):
        return False
    if "fee_gt" in filters and (fee is None or float(fee) <= float(filters["fee_gt"])):
        return False
    start = _as_datetime(meta.get("start_time"))
    after, before = _as_datetime(filters.get("start_after")), _as_datetime(filters.get("start_before"))
    if after and (start is None or start < after):
        return False
    if before and (start is None or start >= before):
        return False
    return True

def _rank(scores: np.ndarray, indices: list[int]) -> list[int]:
    """Sort document indexes by descending score with stable ID tie-breaking."""
    return sorted(indices, key=lambda index: (-float(scores[index]), index))

def retrieve(query: str, top_k: int = DEFAULT_TOP_K, filters: dict | None = None,
             mode: str = "hybrid", now: datetime | None = None, index_dir: Path = INDEX_DIR) -> list[dict]:
    """Retrieve matching chunks with `bm25`, `dense` or `hybrid` ranking."""
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError("mode must be bm25, dense, or hybrid")
    if top_k < 1:
        return []
    chunks, vectors, metadata, bm25, vectorizer = load_index(index_dir)
    if not chunks:
        return []
    merged_filters = {**rewrite_query(query, now=now), **(filters or {})}
    allowed = [i for i, chunk in enumerate(chunks) if _matches(chunk["metadata"], merged_filters)]
    fell_back = False
    if not allowed and merged_filters:
        allowed = list(range(len(chunks)))
        fell_back = True
    bm25_scores = bm25.get_scores(tokenize(query)) if bm25 is not None else np.zeros(len(chunks))
    bm25_rank = _rank(bm25_scores, allowed)
    if mode == "bm25":
        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        dense_rank = bm25_rank
    else:
        query_vector = _query_vector(query, metadata.get("backend", "tfidf"), vectorizer, vectors.shape[1])
        dense_scores = vectors @ query_vector
        dense_rank = _rank(dense_scores, allowed)
    bm25_norm = max(0.0, float(bm25_scores.max()))
    dense_norm = max(0.0, float(dense_scores.max()))
    bm25_relevance = bm25_scores / bm25_norm if bm25_norm > 0 else np.zeros(len(chunks))
    dense_relevance = np.clip(dense_scores, 0, 1)
    bm25_positions = {doc_id: position for position, doc_id in enumerate(bm25_rank, start=1)}
    dense_positions = {doc_id: position for position, doc_id in enumerate(dense_rank, start=1)}
    if mode == "bm25":
        final_rank = bm25_rank
    elif mode == "dense":
        final_rank = dense_rank
    else:
        fused = {i: 1 / (RRF_K + bm25_positions[i]) + 1 / (RRF_K + dense_positions[i]) for i in allowed}
        final_rank = sorted(allowed, key=lambda i: (-fused[i], i))
    results = []
    for doc_id in final_rank[:top_k]:
        chunk = chunks[doc_id]
        confidence = 0.55 * float(dense_relevance[doc_id]) + 0.45 * float(bm25_relevance[doc_id])
        rrf_score = 1 / (RRF_K + bm25_positions[doc_id]) + 1 / (RRF_K + dense_positions[doc_id])
        results.append({"text": chunk["text"], "metadata": chunk["metadata"], "score": float(rrf_score if mode == "hybrid" else (dense_scores[doc_id] if mode == "dense" else bm25_scores[doc_id])),
                        "confidence": confidence, "bm25_score": float(bm25_scores[doc_id]),
                        "dense_score": float(dense_scores[doc_id]), "filter_fallback": fell_back})
    if os.getenv("RAG_RERANK", "false").casefold() == "true" and results:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            pairs = [(query, result["text"]) for result in results]
            rerank_scores = reranker.predict(pairs)
            for result, score in zip(results, rerank_scores):
                result["rerank_score"] = float(score)
            results.sort(key=lambda item: -item["rerank_score"])
        except Exception as exc:
            LOGGER.warning("RAG_RERANK is enabled but cross-encoder reranking failed: %s", exc)
    return results

def retrieve_bm25(query: str, top_k: int = DEFAULT_TOP_K, **kwargs) -> list[dict]:
    """Convenience entry point for BM25-only evaluation."""
    return retrieve(query, top_k=top_k, mode="bm25", **kwargs)

def retrieve_dense(query: str, top_k: int = DEFAULT_TOP_K, **kwargs) -> list[dict]:
    """Convenience entry point for dense-only evaluation."""
    return retrieve(query, top_k=top_k, mode="dense", **kwargs)
