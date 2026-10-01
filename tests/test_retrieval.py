"""Tests for exact-name retrieval and rule-based metadata filters."""
from datetime import datetime
from types import SimpleNamespace
import pytest
from rag.ingest import add_event, rebuild
from rag.generate import REFUSAL, answer_question
from rag.query_rewrite import rewrite_query
from rag.retrieve import retrieve

@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory):
    """Build a small event index once for BM25 and hybrid checks."""
    directory = tmp_path_factory.mktemp("rag-index")
    docs = tmp_path_factory.mktemp("rag-docs")
    event = SimpleNamespace(id=7, title="HackPVG 2026", category="Web/Software", department="Computer Science",
                            start_time=datetime(2026, 10, 24, 10), fee=0, venue="Innovation Lab",
                            eligibility="Open to all currently enrolled students.",
                            description="A campus build challenge with coding workshops and a student demo.",
                            rules="Bring student ID and form a team of up to four.",
                            faq=[{"q": "What is the team size?", "a": "Teams may have up to four students."}])
    rebuild(events=[event], index_dir=directory, general_dir=docs)
    return directory

def test_exact_event_name_is_found_by_bm25_and_hybrid(tiny_index):
    """Exact event codes should be retrievable with lexical and fused ranking."""
    for mode in ("bm25", "hybrid"):
        results = retrieve("HackPVG 2026", top_k=3, mode=mode, index_dir=tiny_index)
        assert results
        assert any(result["metadata"]["event_id"] == 7 for result in results)

def test_live_event_update_refreshes_retrieval_index(tiny_index):
    """Replacing an event refreshes chunks and dense vectors in the persisted index."""
    updated = SimpleNamespace(id=7, title="HackPVG 2026", category="Web/Software", department="Computer Science",
                              start_time=datetime(2026, 10, 24, 10), fee=0, venue="Updated Innovation Lab",
                              eligibility="Open to all currently enrolled students.",
                              description="Updated schedule includes an additional design workshop.",
                              rules="Bring student ID.", faq=[])
    add_event(updated, index_dir=tiny_index)
    results = retrieve("HackPVG updated schedule", top_k=3, mode="bm25", index_dir=tiny_index)
    assert any("additional design workshop" in result["text"] for result in results)

def test_free_this_weekend_extracts_fee_and_date_filters():
    """Relative dates are deterministic and free phrasing produces fee=0."""
    filters = rewrite_query("free events this weekend", now=datetime(2026, 10, 2, 12))
    assert filters["fee"] == 0.0
    assert filters["start_after"] == datetime(2026, 10, 3)
    assert filters["start_before"] == datetime(2026, 10, 5)

def test_dense_metadata_filters_keep_free_events(tiny_index):
    """An explicit fee filter retains only free event chunks."""
    results = retrieve("HackPVG registration fee", filters={"fee": 0}, mode="hybrid", index_dir=tiny_index)
    assert results
    assert all(result["metadata"]["fee"] == 0 for result in results)

def test_grounded_answer_cites_the_event_field():
    """The offline answer extracts the requested brochure fact and cites it."""
    chunks = [{"text": "HackPVG. Venue: Innovation Lab. Fee: free.", "confidence": 0.8,
               "metadata": {"event_id": 7, "title": "HackPVG", "doc_type": "event_header", "fee": 0}}]
    result = answer_question("Where will HackPVG take place?", chunks)
    assert "Innovation Lab" in result["answer"]
    assert "[E7]" in result["answer"]
    assert result["citations"] == [7]

def test_unsupported_specific_field_is_refused():
    """A high-similarity event header cannot authorize an absent keynote fact."""
    chunks = [{"text": "HackPVG. Venue: Innovation Lab. Fee: free.", "confidence": 0.8,
               "metadata": {"event_id": 7, "title": "HackPVG", "doc_type": "event_header", "fee": 0}}]
    result = answer_question("Who is the keynote speaker at HackPVG?", chunks)
    assert result["answer"] == REFUSAL

def test_empty_date_filter_fallback_does_not_answer_out_of_range():
    """Unfiltered retrieval fallback is not presented as an in-range event."""
    chunks = [{"text": "An unrelated past event.", "confidence": 0.8, "filter_fallback": True,
               "metadata": {"event_id": 9, "title": "Past Event", "doc_type": "event_header"}}]
    result = answer_question("free events this weekend", chunks)
    assert result["answer"] == REFUSAL
