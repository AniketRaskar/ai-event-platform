"""Grounded answer prompting, confidence-based refusal and citation extraction."""
from __future__ import annotations
import os
import re
from rag.llm import generate as llm_generate
from rag.query_rewrite import rewrite_query

REFUSAL = "I don't have that information."
DEFAULT_MIN_CONFIDENCE = 0.18
EVENT_FIELD_TERMS = {
    "deadline": ("deadline", "registration closes", "register by", "last day to register"),
    "date": ("date", "when", "what day", "schedule", "start", "time"),
    "venue": ("venue", "where", "location", "place", "held", "take place"),
    "fee": ("fee", "cost", "price", "paid", "free", "how much"),
}
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


def _normal_words(value: str) -> str:
    """Normalize titles and questions for punctuation-insensitive name matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _asks_for(lower_question: str, terms: tuple[str, ...]) -> bool:
    """Match field words as whole words so they don't match inside event names."""
    return any(re.search(rf"\b{re.escape(term)}\b", lower_question) for term in terms)


def _event_named_in_question(question: str):
    """Find events whose full or non-code title appears in a question."""
    from sqlalchemy import select
    from backend.app.database import SessionLocal
    from backend.app.models import Event

    normalized_question = _normal_words(question)
    with SessionLocal() as db:
        events = list(db.scalars(select(Event).order_by(Event.id)))
    matches = []
    for event in events:
        title = _normal_words(event.title)
        if not title:
            continue
        # Synthetic titles end in an ID such as EVT-024. Matching the title
        # without that code is useful, but may be ambiguous across many events.
        base_title = re.sub(r"\s+evt\s+\d+$", "", title)
        if f" {title} " in f" {normalized_question} " or (
                base_title != title and f" {base_title} " in f" {normalized_question} "):
            matches.append(event)
    if not matches:
        return None, []
    # A full title including its code is a stronger, unambiguous match.
    full_matches = [event for event in matches if f" {_normal_words(event.title)} " in f" {normalized_question} "]
    return (full_matches or matches)[0] if len(full_matches or matches) == 1 else None, (full_matches or matches)


def _answer_named_event_field(question: str) -> dict | None:
    """Answer date, venue, fee, or deadline from the live event database."""
    lower = question.casefold()
    field = next((name for name, terms in EVENT_FIELD_TERMS.items() if _asks_for(lower, terms)), None)
    if field is None:
        return None
    event, matches = _event_named_in_question(question)
    if event is None:
        if not matches:
            return None
        choices = ", ".join(f"{item.title} [E{item.id}]" for item in matches[:5])
        more = " There are more matches; include the event code to narrow it down." if len(matches) > 5 else ""
        return {"answer": f"I found multiple events matching that name: {choices}.{more} Please include the full event name or event code.",
                "citations": [item.id for item in matches[:5]], "confidence": 1.0}

    citation = f"[E{event.id}]"
    if field == "date":
        answer = f"{event.title} starts on {event.start_time.strftime('%A, %d %B %Y at %I:%M %p')}. {citation}"
    elif field == "venue":
        answer = f"{event.title} will be held at {event.venue}. {citation}"
    elif field == "fee":
        answer = f"{event.title} is free to attend. {citation}" if event.fee == 0 else f"The fee for {event.title} is ₹{event.fee:g}. {citation}"
    else:
        answer = f"Registration for {event.title} closes on {event.registration_deadline.strftime('%A, %d %B %Y at %I:%M %p')}. {citation}"
    return {"answer": answer, "citations": [event.id], "confidence": 1.0}

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
    named_answer = _answer_named_event_field(question)
    if named_answer is not None:
        return named_answer
    from rag.retrieve import retrieve
    contexts = retrieve(question, top_k=top_k, **retrieve_options)
    return answer_question(question, contexts)
