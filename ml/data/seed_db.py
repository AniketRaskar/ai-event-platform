"""Load generated CSVs into the local SQLite database and add demo accounts."""
from __future__ import annotations
import ast
import hashlib
import json
import secrets
import pandas as pd
from sqlalchemy import select
from backend.app.database import Base, SessionLocal, engine
from backend.app.models import Event, EventActionLog, Friendship, Interaction, Notification, User
from ml.config import DATABASE_PATH, GENERATED_DIR

def hash_password(password: str) -> str:
    """Hash a password with scrypt and a per-password random salt."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"

def _json(value):
    """Decode list/dict values stored in CSV cells."""
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return ast.literal_eval(value)

def seed() -> None:
    """Create tables and populate them from generated synthetic CSV files."""
    required = [GENERATED_DIR / f"{name}.csv" for name in ("users", "events", "friendships", "interactions")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run `python -m ml.data.synthetic_generator` first; missing: " + ", ".join(missing))
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        # A clean seed is deterministic and avoids duplicate rows on repeated `make data` runs.
        for model in (EventActionLog, Notification, Interaction, Friendship, Event, User):
            db.query(model).delete()
        db.commit()
        user_df, event_df, friend_df, interaction_df = [pd.read_csv(p) for p in required]
        for row in user_df.to_dict("records"):
            db.add(User(id=int(row["id"]), name=row["name"], email=row["email"], password_hash=hash_password(secrets.token_urlsafe(18)),
                        role=row["role"], department=row["department"], year=int(row["year"]), interests=_json(row["interests"])))
        db.flush()
        for row in event_df.to_dict("records"):
            db.add(Event(id=int(row["id"]), title=row["title"], description=row["description"], category=row["category"],
                         department=row["department"], organizer=row["organizer"], venue=row["venue"],
                         start_time=pd.Timestamp(row["start_time"]).to_pydatetime(), end_time=pd.Timestamp(row["end_time"]).to_pydatetime(),
                         registration_deadline=pd.Timestamp(row["registration_deadline"]).to_pydatetime(), fee=float(row["fee"]),
                         capacity=int(row["capacity"]), eligibility=row["eligibility"], rules=row["rules"], faq=_json(row["faq"]),
                         posted_at=pd.Timestamp(row["posted_at"]).to_pydatetime(), created_by=None))
        db.flush()
        db.bulk_save_objects([Friendship(user_id=int(r.user_id), friend_id=int(r.friend_id)) for r in friend_df.itertuples()])
        db.bulk_save_objects([Interaction(user_id=int(r.user_id), event_id=int(r.event_id), type=r.type,
                                          rating=None if pd.isna(r.rating) else int(r.rating),
                                          comment=None if pd.isna(r.comment) else str(r.comment),
                                          timestamp=pd.Timestamp(r.timestamp).to_pydatetime()) for r in interaction_df.itertuples()])
        db.flush()
        for email, name, role, password in [("admin@college.edu", "Demo Admin", "admin", "admin123"),
                                             ("student@college.edu", "Demo Student", "student", "student123")]:
            existing = db.scalar(select(User).where(User.email == email))
            if not existing:
                db.add(User(name=name, email=email, role=role, password_hash=hash_password(password),
                            department="Computer Science", year=2, interests=["AI/ML", "Web/Software"]))
        db.commit()
        print(f"Seeded SQLite database: {DATABASE_PATH}")
        print(f"users={db.query(User).count()} events={db.query(Event).count()} friendships={db.query(Friendship).count()} interactions={db.query(Interaction).count()} notifications={db.query(Notification).count()}")
        print("Demo accounts: admin@college.edu / admin123; student@college.edu / student123")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
