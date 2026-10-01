"""Grounded answer prompting, confidence-based refusal and citation extraction."""
from __future__ import annotations
import os
import re
from rag.llm import generate as llm_generate
from rag.query_rewrite import rewrite_query

REFUSAL = "I don't have that information."
DEFAULT_MIN_CONFIDENCE = 0.18
UNSUPPORTED_FIELD_GROUPS = {
    ("speaker", "keynote", "presenter", "panelist"): ("speaker", "keynote", "presenter", "panelist"),
    ("menu", "meal", "food", "snack"): ("menu", "meal", "food", "snack", "refreshment"),
    ("parking", "parking lot"): ("parking",),
    ("wifi", "wi-fi", "wireless password"): ("wifi", "wi-fi", "wireless"),
    ("livestream", "live stream", "streaming link"): ("livestream", "live stream", "streaming link"),
    ("shuttle", "bus pickup", "transport pickup"): ("shuttle", "pickup", "transport"),
    ("judges", "judging panel", "jury"): ("judge", "judging", "jury"),
    ("refund", "money back"): ("refund", "money back"),
    ("dress code", "what should i wear"): ("dress code", "wear"),
    ("wifi password", "password for the wifi"): ("password",),
}

def citation_for(metadata: dict) -> str:
    """Format the citation token for an event or general guide chunk."""
    event_id = metadata.get("event_id")
    return f"[E{event_id}]" if event_id is not None else f"[G:{metadata.get('doc_id', 'guide')}]"

def grounded_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a prompt that restricts claims to cited retrieved context."""
    contexts = []
    for index, chunk in enumerate(context_chunks, start=1):
        contexts.append(f"[{index}] {citation_for(chunk['metadata'])}\n{chunk['text']}")
    return ("Answer the question using only the context below. Do not add outside facts or guess. "
            "Cite every factual statement with its supplied citation token. If the context does not answer the question, say you do not have that information.\n\n"
            f"Question: {question}\n\nContext:\n" + "\n\n".join(contexts))

def _requested_field_is_supported(question: str, chunks: list[dict]) -> bool:
    """Refuse specific field questions when the retrieved evidence lacks that field."""
    lower_question = question.casefold()
    corpus = " ".join(chunk.get("text", "").casefold() for chunk in chunks)
    for question_terms, evidence_terms in UNSUPPORTED_FIELD_GROUPS.items():
        if any(term in lower_question for term in question_terms) and not any(term in corpus for term in evidence_terms):
            return False
    return True

def answer_question(question: str, context_chunks: list[dict] | None = None,
                    min_confidence: float | None = None) -> dict:
    """Answer from pre-retrieved chunks, or refuse when evidence is weak."""
    chunks = context_chunks or []
    threshold = min_confidence if min_confidence is not None else float(os.getenv("RAG_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE))
    if not chunks or float(chunks[0].get("confidence", 0.0)) < threshold:
        return {"answer": REFUSAL, "citations": [], "confidence": float(chunks[0].get("confidence", 0.0)) if chunks else 0.0}
    if rewrite_query(question) and all(chunk.get("filter_fallback", False) for chunk in chunks):
        return {"answer": REFUSAL, "citations": [], "confidence": float(chunks[0].get("confidence", 0.0))}
    if not _requested_field_is_supported(question, chunks):
        return {"answer": REFUSAL, "citations": [], "confidence": float(chunks[0].get("confidence", 0.0))}
    prompt = grounded_prompt(question, chunks)
    answer = llm_generate(prompt, chunks).strip()
    if not answer or answer == REFUSAL:
        return {"answer": REFUSAL, "citations": [], "confidence": float(chunks[0].get("confidence", 0.0))}
    citation_tokens = re.findall(r"\[(?:E\d+|G:[A-Za-z0-9_-]+)\]", answer)
    if not citation_tokens:
        citation_tokens = [citation_for(chunks[0]["metadata"])]
        answer = f"{answer} {citation_tokens[0]}"
    unique = list(dict.fromkeys(citation_tokens))
    citations = []
    for token in unique:
        event_match = re.fullmatch(r"\[E(\d+)\]", token)
        citations.append(int(event_match.group(1)) if event_match else token[3:-1])
    return {"answer": answer, "citations": citations, "confidence": float(chunks[0].get("confidence", 0.0))}

def generate_answer(question: str, top_k: int = 5, **retrieve_options) -> dict:
    """Retrieve context and produce a grounded response in one call."""
    from rag.retrieve import retrieve
    contexts = retrieve(question, top_k=top_k, **retrieve_options)
    return answer_question(question, contexts)
