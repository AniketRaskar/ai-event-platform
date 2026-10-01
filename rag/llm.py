"""Offline extractive and optional Ollama/OpenAI-compatible text generation."""
from __future__ import annotations
import logging
import os
import re
from typing import Any
import requests

LOGGER = logging.getLogger(__name__)

def _extractive(prompt: str, context_chunks: list[dict] | None = None) -> str:
    """Return the strongest context sentence with a citation, without inventing facts."""
    chunks = context_chunks or []
    if not chunks:
        return "I don't have that information."
    question_match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, flags=re.IGNORECASE)
    question = question_match.group(1).lower() if question_match else ""
    query_terms = set(re.findall(r"[a-z0-9]+", question))
    structured = _answer_event_field(question, chunks)
    if structured:
        return structured
    candidates = []
    for chunk in chunks:
        text = chunk.get("text", "")
        sentences = re.split(r"(?<=[!?])\s+|(?<!\d)(?<=\.)\s+(?=[A-Z])", text)
        for sentence in sentences:
            terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            score = len(terms & query_terms) / max(1, len(query_terms))
            candidates.append((score, sentence.strip(), chunk.get("metadata", {})))
    if not candidates:
        return "I don't have that information."
    _, sentence, metadata = max(candidates, key=lambda item: item[0])
    event_id = metadata.get("event_id")
    citation = f"[E{event_id}]" if event_id is not None else f"[G:{metadata.get('doc_id', 'guide')}]"
    title = metadata.get("title")
    lead = f"{title}: " if title else ""
    return f"{lead}{sentence} {citation}"

def _answer_event_field(question: str, chunks: list[dict]) -> str | None:
    """Answer common event fields directly from structured brochure text."""
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("event_id") is None:
            continue
        text = chunk.get("text", "")
        title = metadata.get("title", "This event")
        citation = f"[E{metadata['event_id']}]"
        lower = question.casefold()
        if any(term in lower for term in ("venue", "where", "take place", "location")):
            match = re.search(r"Venue:\s*([^.]*)", text, flags=re.IGNORECASE)
            if match:
                return f"{title} will be held at {match.group(1).strip()} {citation}"
        if any(term in lower for term in ("fee", "cost", "price", "paid", "free")):
            if metadata.get("fee") is not None:
                amount = float(metadata["fee"])
                return f"{title} is free to attend." + f" {citation}" if amount == 0 else f"The fee for {title} is ₹{amount:g}. {citation}"
        if any(term in lower for term in ("deadline", "registration close", "register by", "last day to register")):
            match = re.search(r"Registration closes\s+([^.]*)", text, flags=re.IGNORECASE)
            if match:
                return f"Registration for {title} closes {match.group(1).strip()} {citation}"
        if "team" in lower or "team size" in lower:
            match = re.search(r"(?:up to |include )(\d+) (?:participants?|members?)", text, flags=re.IGNORECASE)
            if match:
                return f"Teams for {title} may have up to {match.group(1)} participants. {citation}"
        if any(term in lower for term in ("eligible", "eligibility", "who can", "who is allowed")):
            match = re.search(r"Eligibility:\s*(.*?)\s+Description:", text, flags=re.IGNORECASE)
            if match:
                return f"Eligibility for {title}: {match.group(1).strip()} {citation}"
        if any(term in lower for term in ("when is", "what date", "event date")):
            if metadata.get("start_time"):
                return f"{title} starts at {metadata['start_time']} {citation}"
    return None

def generate(prompt: str, context_chunks: list[dict] | None = None) -> str:
    """Generate from context using the configured provider; default to offline extraction."""
    provider = os.getenv("LLM_PROVIDER", "extractive").casefold()
    if provider == "extractive":
        return _extractive(prompt, context_chunks)
    model = os.getenv("LLM_MODEL", "llama3.2")
    try:
        if provider == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            response = requests.post(f"{base}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=60)
            response.raise_for_status()
            return response.json()["response"].strip()
        if provider == "openai_compatible":
            base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            response = requests.post(f"{base}/chat/completions", headers=headers,
                                     json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    except Exception as exc:
        LOGGER.warning("LLM provider %s failed (%s); using the offline extractive response.", provider, exc)
        return _extractive(prompt, context_chunks)
