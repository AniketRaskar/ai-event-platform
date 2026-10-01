"""Rule-based chat routing for event information and event recommendations."""
from __future__ import annotations
import re
from rag.query_rewrite import rewrite_query

RECOMMEND_PATTERNS = ("suggest", "recommend", "what should i attend", "for me", "what can i go to", "find me an event")

def classify_route(message: str) -> str:
    """Return INFO or RECOMMEND using common recommendation phrases."""
    lower = message.casefold()
    return "RECOMMEND" if any(phrase in lower for phrase in RECOMMEND_PATTERNS) else "INFO"

def route_message(message: str, user_id: int | None = None, top_k: int = 5) -> dict:
    """Dispatch a user message to the recommender tool or grounded RAG flow."""
    route = classify_route(message)
    filters = rewrite_query(message)
    if route == "RECOMMEND":
        from rag.tools import recommend_events
        events = recommend_events(user_id=user_id, filters=filters, limit=top_k)
        if not events:
            return {"answer": "I couldn't find upcoming events that match those filters.", "citations": [], "route": route,
                    "used_tool": True, "events": [], "filters": filters}
        lines = [f"Here are a few upcoming events:"]
        for event in events:
            fee = "free" if event["fee"] == 0 else f"₹{event['fee']:g}"
            lines.append(f"• {event['title']} — {event['category']}, {event['start_time']}, {fee}. {event['reason']} [E{event['id']}]")
        return {"answer": "\n".join(lines), "citations": [event["id"] for event in events], "route": route,
                "used_tool": True, "events": events, "filters": filters}
    from rag.generate import generate_answer
    result = generate_answer(message)
    return {**result, "route": route, "used_tool": False, "events": [], "filters": filters}

def route(message: str, user_id: int | None = None, top_k: int = 5) -> dict:
    """Alias for route_message, convenient for clients and scripts."""
    return route_message(message, user_id=user_id, top_k=top_k)
