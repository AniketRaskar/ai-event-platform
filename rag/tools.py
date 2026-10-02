"""Tools that let the chat route look up events or offer simple popular picks."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import func, select
from backend.app.database import SessionLocal
from backend.app.models import Event, Interaction

def _event_payload(event: Event, score: float = 0.0, reason: str = "") -> dict:
    """Serialize the small event payload used by the chat tool."""
    return {"id": event.id, "title": event.title, "category": event.category, "department": event.department,
            "venue": event.venue, "start_time": event.start_time.isoformat(), "fee": event.fee,
            "description": event.description, "score": score, "reason": reason}

def get_event(event_id: int) -> dict | None:
    """Fetch a single event as a plain dictionary."""
    with SessionLocal() as db:
        event = db.get(Event, event_id)
        return _event_payload(event) if event else None

def recommend_events(user_id: int | None = None, filters: dict | None = None, limit: int = 5) -> list[dict]:
    """Return personalized event picks when a user is known, otherwise popular picks."""
    filters = filters or {}
    now = datetime.now()
    query = select(Event, func.count(Interaction.id).label("popularity")).outerjoin(
        Interaction, (Interaction.event_id == Event.id) & Interaction.type.in_(["like", "register"])
    ).where(Event.start_time >= now)
    if filters.get("category"):
        query = query.where(Event.category == filters["category"])
    if filters.get("department"):
        query = query.where(Event.department == filters["department"])
    if "fee" in filters:
        query = query.where(Event.fee == float(filters["fee"]))
    if "fee_gt" in filters:
        query = query.where(Event.fee > float(filters["fee_gt"]))
    if filters.get("start_after"):
        query = query.where(Event.start_time >= filters["start_after"])
    if filters.get("start_before"):
        query = query.where(Event.start_time < filters["start_before"])
    query = query.group_by(Event.id).order_by(func.count(Interaction.id).desc(), Event.start_time.asc()).limit(limit)
    with SessionLocal() as db:
        if user_id is not None:
            from backend.app.models import User
            from backend.app.services.recommender import recommend_for_user
            user = db.get(User, user_id)
            if user is not None:
                rows = recommend_for_user(db, user, limit=limit, filters=filters)
                return [_event_payload(row["event"], row["score"], row["reason"]) for row in rows]
        rows = db.execute(query).all()
        return [_event_payload(event, float(popularity), "Popular with students") for event, popularity in rows]
