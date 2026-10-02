"""Personalized event ranking with NeuMF inference and safe cold-start fallback."""
from __future__ import annotations
from datetime import datetime
import json
import logging
import torch
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ml.config import ARTIFACTS_DIR
from ..models import Event, Friendship, Interaction, User

LOGGER = logging.getLogger(__name__)
torch.set_num_threads(1)
_model_state: dict | None = None


def _load_neumf(user_id: int | None = None) -> dict | None:
    """Load the trained recommender lazily; return None when artifacts are absent."""
    global _model_state
    if _model_state is not None:
        return _model_state
    checkpoint_path = ARTIFACTS_DIR / "model.pt"
    maps_path = ARTIFACTS_DIR / "id_maps.json"
    if not checkpoint_path.exists() or not maps_path.exists():
        return None
    try:
        maps = json.loads(maps_path.read_text(encoding="utf-8"))
        user_map = {int(k): int(v) for k, v in maps["users"].items()}
        if user_id is not None and user_id not in user_map:
            return None
        from ml.data.splits import build_split
        from ml.train import make_model, ordered_content
        split = build_split()
        content, _ = ordered_content(split)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = make_model(split, content, latent_dim=checkpoint.get("latent_dim", 32),
                           use_gmf=checkpoint.get("use_gmf", True), use_mlp=checkpoint.get("use_mlp", True),
                           use_content=checkpoint.get("use_content", True), id_dropout=checkpoint.get("id_dropout", 0.3),
                           user_side_features=checkpoint.get("use_user_side", True))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _model_state = {"model": model, "users": user_map,
                        "events": {int(k): int(v) for k, v in maps["events"].items()}}
        return _model_state
    except Exception:
        LOGGER.exception("Could not load NeuMF artifacts; using profile/popularity recommendation")
        _model_state = {}
        return None


def _matches_filters(event: Event, filters: dict) -> bool:
    """Return whether an event satisfies parsed metadata filters."""
    if filters.get("category") and event.category.casefold() != str(filters["category"]).casefold():
        return False
    if filters.get("department") and event.department.casefold() != str(filters["department"]).casefold():
        return False
    if "fee" in filters and event.fee != float(filters["fee"]):
        return False
    if "fee_gt" in filters and event.fee <= float(filters["fee_gt"]):
        return False
    if filters.get("start_after") and event.start_time < datetime.fromisoformat(str(filters["start_after"])):
        return False
    if filters.get("start_before") and event.start_time >= datetime.fromisoformat(str(filters["start_before"])):
        return False
    return True


def recommend_for_user(db: Session, user: User, *, limit: int = 10, filters: dict | None = None) -> list[dict]:
    """Rank upcoming eligible events and write one impression row per returned item."""
    from ..models import EventActionLog
    filters = filters or {}
    now = datetime.now()
    past_actions = list(db.execute(select(Interaction.event_id, Interaction.type).where(Interaction.user_id == user.id)))
    positive = {int(eid) for eid, kind in past_actions if kind in {"like", "register"}}
    disliked = {int(eid) for eid, kind in past_actions if kind == "dislike"}
    candidates = [event for event in db.scalars(select(Event).where(Event.start_time >= now,
                     Event.registration_deadline >= now).order_by(Event.start_time))
                  if event.id not in positive and event.id not in disliked and _matches_filters(event, filters)]
    if not candidates:
        return []

    popularity_rows = db.execute(select(Interaction.event_id, func.count(Interaction.id)).where(
        Interaction.type.in_(["like", "register"])).group_by(Interaction.event_id)).all()
    popularity = {int(event_id): int(count) for event_id, count in popularity_rows}
    friend_ids = set(db.scalars(select(Friendship.friend_id).where(Friendship.user_id == user.id)))
    friend_counts = {}
    if friend_ids:
        rows = db.execute(select(Interaction.event_id, func.count(Interaction.id)).where(
            Interaction.user_id.in_(friend_ids), Interaction.type == "register").group_by(Interaction.event_id)).all()
        friend_counts = {int(event_id): int(count) for event_id, count in rows}

    model_scores = {}
    state = _load_neumf(user.id)
    user_index = state.get("users", {}).get(user.id) if state else None
    if state and user_index is not None:
        known = [(event, state["events"][event.id]) for event in candidates if event.id in state["events"]]
        if known:
            event_indices = torch.tensor([index for _, index in known], dtype=torch.long)
            user_indices = torch.full((len(known),), user_index, dtype=torch.long)
            scores = torch.sigmoid(state["model"].score_pairs(user_indices, event_indices)).cpu().numpy()
            model_scores = {event.id: float(score) for (event, _), score in zip(known, scores)}

    interests = set(user.interests or [])
    max_pop = max(popularity.values(), default=0) or 1
    max_friends = max(friend_counts.values(), default=0) or 1
    ranked = []
    for event in candidates:
        category_match = event.category in interests
        department_match = event.department == user.department
        social = friend_counts.get(event.id, 0) / max_friends
        popular = popularity.get(event.id, 0) / max_pop
        profile_score = 0.45 * float(category_match) + 0.20 * float(department_match) + 0.20 * social + 0.15 * popular
        score = model_scores.get(event.id, profile_score)
        reasons = []
        if category_match:
            reasons.append(f"Matches your interest in {event.category}")
        if department_match:
            reasons.append(f"Relevant to {event.department}")
        if friend_counts.get(event.id, 0):
            reasons.append(f"{friend_counts[event.id]} friend(s) registered")
        if not reasons:
            reasons.append("Popular with students" if popularity.get(event.id, 0) else "Upcoming event")
        ranked.append((event, score, "; ".join(reasons)))
    ranked.sort(key=lambda row: (-row[1], row[0].start_time, row[0].id))
    output = []
    for event, score, reason in ranked[:limit]:
        db.add(EventActionLog(user_id=user.id, event_id=event.id, action="recommend_impression", source="recommend"))
        output.append({"event": event, "score": score, "reason": reason})
    db.commit()
    return output
