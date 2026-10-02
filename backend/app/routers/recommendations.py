"""Personalized recommendation endpoint."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..services.recommender import recommend_for_user

router = APIRouter(tags=["recommendations"])


@router.get("/recommend")
def recommend(k: int = Query(10, ge=1, le=50), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Return a personalized ranked list and log each recommendation impression."""
    rows = recommend_for_user(db, user, limit=k)
    return [{"event": row["event"], "score": row["score"], "reason": row["reason"]} for row in rows]
