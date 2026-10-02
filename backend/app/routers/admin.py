"""Admin-only registration lists and event analytics."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ..database import get_db
from ..deps import require_admin
from ..models import Event, EventActionLog, Interaction, User

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/events/{event_id}/registrations")
def registrations(event_id: int, db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """List registered student profiles for one event."""
    if db.get(Event, event_id) is None:
        raise HTTPException(status_code=404, detail="Event not found")
    rows = db.execute(select(User.id, User.name, User.email, User.department, User.year, Interaction.timestamp)
                      .join(Interaction, Interaction.user_id == User.id)
                      .where(Interaction.event_id == event_id, Interaction.type == "register")
                      .order_by(Interaction.timestamp)).all()
    return [{"id": row.id, "name": row.name, "email": row.email, "department": row.department,
             "year": row.year, "registered_at": row.timestamp} for row in rows]


@router.get("/analytics")
def analytics(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """Return simple event registration/like summaries and impression placeholders."""
    registrations = db.execute(select(Event.id, Event.title, func.count(Interaction.id).label("count"))
        .outerjoin(Interaction, (Interaction.event_id == Event.id) & (Interaction.type == "register"))
        .group_by(Event.id).order_by(func.count(Interaction.id).desc(), Event.id)).all()
    likes = db.execute(select(Event.category, func.count(Interaction.id).label("count"))
        .join(Interaction, (Interaction.event_id == Event.id) & (Interaction.type == "like"))
        .group_by(Event.category).order_by(func.count(Interaction.id).desc())).all()
    top_events = [{"event_id": row.id, "title": row.title, "registrations": row.count} for row in registrations[:10]]
    impressions = db.scalar(select(func.count(EventActionLog.id)).where(EventActionLog.action == "recommend_impression")) or 0
    clicks = db.scalar(select(func.count(EventActionLog.id)).where(EventActionLog.action == "view",
                                                                   EventActionLog.source == "recommend")) or 0
    return {"registrations_per_event": [{"event_id": row.id, "title": row.title, "registrations": row.count}
                                         for row in registrations],
            "likes_per_category": [{"category": row.category, "likes": row.count} for row in likes],
            "top_events": top_events,
            "recommendation_click_through": {"impressions": impressions, "clicks": clicks,
                                              "rate": round(clicks / impressions, 4) if impressions else None}}
