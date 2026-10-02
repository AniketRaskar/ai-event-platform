"""Private notification inbox routes."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import get_db
from ..deps import get_current_user
from ..models import Notification, User

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("")
def list_notifications(unread_only: bool = False, limit: int = Query(50, ge=1, le=200),
                       db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """List only the current account's notifications, newest first."""
    query = select(Notification).where(Notification.user_id == user.id)
    if unread_only:
        query = query.where(Notification.read.is_(False))
    return list(db.scalars(query.order_by(Notification.created_at.desc()).limit(limit)))


@router.post("/{notification_id}/read")
def mark_read(notification_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Mark one of the current user's notifications as read."""
    note = db.scalar(select(Notification).where(Notification.id == notification_id,
                                                Notification.user_id == user.id))
    if note is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    note.read = True
    db.commit()
    return {"ok": True, "id": note.id, "read": note.read}
