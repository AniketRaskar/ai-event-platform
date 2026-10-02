"""Authenticated grounded chatbot endpoint."""
from fastapi import APIRouter, Depends
from ..deps import get_current_user
from ..models import User
from ..schemas import ChatRequest
from rag.router import route_message

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat(payload: ChatRequest, user: User = Depends(get_current_user)):
    """Answer an event question or call recommendations for this user."""
    del payload.history  # Rule-based retrieval is stateless; history is accepted for client compatibility.
    return route_message(payload.message, user_id=user.id)
