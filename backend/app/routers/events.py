"""Event discovery, admin publishing and student action endpoints."""
from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from ..database import get_db
from ..deps import get_current_user, require_admin
from ..models import Event, EventActionLog, Interaction, Notification, User
from ..schemas import EventCreate, EventResponse, EventUpdate, FeedbackCreate

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["events"])


def _record(db: Session, user_id: int, event_id: int, action: str, source: str = "events") -> None:
    """Append an auditable event action without losing repeat impressions."""
    db.add(EventActionLog(user_id=user_id, event_id=event_id, action=action, source=source))


def _upsert_interaction(db: Session, user_id: int, event_id: int, kind: str, *, rating=None, comment=None) -> Interaction:
    """Maintain one training interaction per user/event/action while logging repeats."""
    row = db.scalar(select(Interaction).where(Interaction.user_id == user_id,
                      Interaction.event_id == event_id, Interaction.type == kind))
    if row is None:
        row = Interaction(user_id=user_id, event_id=event_id, type=kind, timestamp=datetime.now())
        db.add(row)
    else:
        row.timestamp = datetime.now()
    if rating is not None:
        row.rating = rating
    if comment is not None:
        row.comment = comment
    return row


@router.get("/events", response_model=list[EventResponse])
def list_events(category: str | None = None, department: str | None = None, free: bool | None = None,
                q: str | None = None, upcoming: bool = False, limit: int = Query(100, ge=1, le=250),
                offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    """Search and filter events; no login is needed for the public catalogue."""
    query = select(Event)
    if category:
        query = query.where(Event.category == category)
    if department:
        query = query.where(Event.department == department)
    if free is True:
        query = query.where(Event.fee == 0)
    elif free is False:
        query = query.where(Event.fee > 0)
    if q and q.strip():
        pattern = f"%{q.strip()}%"
        query = query.where(or_(Event.title.ilike(pattern), Event.description.ilike(pattern),
                                Event.category.ilike(pattern), Event.organizer.ilike(pattern)))
    if upcoming:
        query = query.where(Event.start_time >= datetime.now())
    return list(db.scalars(query.order_by(Event.start_time.asc()).offset(offset).limit(limit)))


@router.get("/events/{event_id}", response_model=EventResponse)
def get_event(event_id: int, db: Session = Depends(get_db)):
    """Return complete public event details."""
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def _notify_matching_users(db: Session, event: Event) -> None:
    """Notify users in the top matching preference group after an event is posted."""
    users = list(db.scalars(select(User).where(User.role == "student")))
    scored = [(user, int(event.category in (user.interests or [])) + int(event.department == user.department))
              for user in users]
    positive = [score for _, score in scored if score > 0]
    if not positive:
        return
    cutoff = sorted(positive)[max(0, int(len(positive) * 0.8) - 1)]
    for user, score in scored:
        if score >= cutoff and score > 0:
            db.add(Notification(user_id=user.id, event_id=event.id,
                                message=f"New event that may match your interests: {event.title}"))


def _update_rag_index(event: Event) -> None:
    """Refresh the persisted chatbot index after an event is created or edited."""
    try:
        from rag.ingest import add_event
        add_event(event)
    except Exception:
        LOGGER.exception("Event %s was saved but its RAG index update failed", event.id)


@router.post("/events", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
def create_event(payload: EventCreate, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    """Create an event, notify interested students and update RAG immediately."""
    if payload.end_time <= payload.start_time:
        raise HTTPException(status_code=422, detail="end_time must be after start_time")
    if payload.registration_deadline > payload.start_time:
        raise HTTPException(status_code=422, detail="registration_deadline must not be after start_time")
    values = payload.model_dump()
    event = Event(**values, posted_at=datetime.now(), created_by=admin.id)
    db.add(event)
    db.flush()
    _notify_matching_users(db, event)
    db.commit()
    db.refresh(event)
    _update_rag_index(event)
    return event


@router.put("/events/{event_id}", response_model=EventResponse)
def update_event(event_id: int, payload: EventUpdate, db: Session = Depends(get_db),
                 admin: User = Depends(require_admin)):
    """Update event fields and replace its RAG chunks."""
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    changes = payload.model_dump(exclude_unset=True)
    if any(value is None for value in changes.values()):
        raise HTTPException(status_code=422, detail="Event fields cannot be set to null")
    start = changes.get("start_time", event.start_time)
    end = changes.get("end_time", event.end_time)
    deadline = changes.get("registration_deadline", event.registration_deadline)
    if end <= start or deadline > start:
        raise HTTPException(status_code=422, detail="Event dates are inconsistent")
    for key, value in changes.items():
        setattr(event, key, value)
    db.commit()
    db.refresh(event)
    _update_rag_index(event)
    return event


@router.post("/events/{event_id}/register")
def register_for_event(event_id: int, response: Response, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    """Register the current student if the event is open and has capacity."""
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    now = datetime.now()
    if event.registration_deadline < now or event.start_time < now:
        raise HTTPException(status_code=409, detail="Registration for this event is closed")
    existing = db.scalar(select(Interaction).where(Interaction.user_id == user.id,
                             Interaction.event_id == event_id, Interaction.type == "register"))
    if existing:
        return {"ok": True, "message": "You are already registered"}
    registered = db.scalar(select(func.count()).select_from(Interaction).where(
        Interaction.event_id == event_id, Interaction.type == "register")) or 0
    if registered >= event.capacity:
        raise HTTPException(status_code=409, detail="This event has reached capacity")
    _upsert_interaction(db, user.id, event_id, "register")
    _record(db, user.id, event_id, "register")
    db.commit()
    response.status_code = status.HTTP_201_CREATED
    return {"ok": True, "message": "Registration confirmed"}


def _action(event_id: int, kind: str, db: Session, user: User, source: str = "events"):
    """Record a like, dislike or view after checking the event exists."""
    if db.get(Event, event_id) is None:
        raise HTTPException(status_code=404, detail="Event not found")
    _upsert_interaction(db, user.id, event_id, kind)
    _record(db, user.id, event_id, kind, source)
    db.commit()
    return {"ok": True, "action": kind}


@router.post("/events/{event_id}/like")
def like_event(event_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Like an event."""
    return _action(event_id, "like", db, user)


@router.post("/events/{event_id}/dislike")
def dislike_event(event_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Dislike an event; this is retained as a recommender hard negative."""
    return _action(event_id, "dislike", db, user)


@router.post("/events/{event_id}/view")
def view_event(event_id: int, source: str = Query("events", pattern="^(events|recommend)$"),
               db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Log a click/view and update the interaction used in future training."""
    return _action(event_id, "view", db, user, source)


@router.post("/events/{event_id}/feedback")
def feedback_event(event_id: int, payload: FeedbackCreate, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    """Save event feedback and its rating/comment for later evaluation."""
    if db.get(Event, event_id) is None:
        raise HTTPException(status_code=404, detail="Event not found")
    _upsert_interaction(db, user.id, event_id, "feedback", rating=payload.rating, comment=payload.comment or "")
    _record(db, user.id, event_id, "feedback")
    db.commit()
    return {"ok": True, "rating": payload.rating}
