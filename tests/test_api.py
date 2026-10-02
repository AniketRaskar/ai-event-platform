"""Smoke tests for the authenticated API and its SQLite dependencies."""
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import Event, EventActionLog, User
from backend.app.security import hash_password
from backend.app.routers import events as event_routes
from backend.app.services import recommender


@pytest.fixture
def api_client(monkeypatch):
    """Use an isolated in-memory database for each HTTP test."""
    test_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(test_engine)
    test_session = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        with test_session() as db:
            yield db

    monkeypatch.setitem(app.dependency_overrides, get_db, override_get_db)
    monkeypatch.setattr(event_routes, "_update_rag_index", lambda _event: None)
    monkeypatch.setattr(recommender, "_load_neumf", lambda _user_id=None: None)
    with TestClient(app) as client:
        yield client, test_session
    app.dependency_overrides.pop(get_db, None)
    test_engine.dispose()


def _seed_user(session_factory, *, email: str, role: str, name: str):
    """Insert an account into the isolated test database."""
    with session_factory() as db:
        user = User(name=name, email=email, password_hash=hash_password("test-password-123"), role=role,
                    department="Computer Science", year=2, interests=["AI/ML"])
        db.add(user)
        db.commit()
        return user.id


def _seed_event(session_factory, *, title: str, capacity: int = 5):
    """Insert a future event that remains open for registration."""
    now = datetime.now()
    with session_factory() as db:
        event = Event(title=title, description="A test event", category="AI/ML", department="Computer Science",
            organizer="Test Club", venue="Test Hall", start_time=now + timedelta(days=3),
            end_time=now + timedelta(days=3, hours=1), registration_deadline=now + timedelta(days=2), fee=0,
            capacity=capacity, eligibility="Students", rules="Be respectful", faq=[], posted_at=now,
            created_by=None)
        db.add(event)
        db.commit()
        return event.id


def test_registration_login_and_auth_profile(api_client):
    """A new student can register, log in, and update onboarding interests."""
    client, _ = api_client
    response = client.post("/auth/register", json={"name": "Test Student", "email": "test@example.edu",
        "password": "test-password-123", "department": "Computer Science", "year": 2,
        "interests": ["AI/ML"]})
    assert response.status_code == 201
    login = client.post("/auth/login", json={"email": "TEST@example.edu", "password": "test-password-123"})
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.get("/auth/me", headers=headers).json()["email"] == "test@example.edu"
    updated = client.put("/auth/me/interests", headers=headers, json={"interests": ["Robotics/IoT"]})
    assert updated.status_code == 200
    assert updated.json()["interests"] == ["Robotics/IoT"]
    assert client.get("/auth/me").status_code == 401


def test_admin_actions_recommendation_and_impression_logging(api_client):
    """Admin-only event creation is enforced and student clicks/impressions persist."""
    client, session_factory = api_client
    _seed_user(session_factory, email="admin@example.edu", role="admin", name="Admin")
    _seed_user(session_factory, email="student@example.edu", role="student", name="Student")
    first_event = _seed_event(session_factory, title="First event", capacity=1)
    second_event = _seed_event(session_factory, title="Second event")
    admin_token = client.post("/auth/login", json={"email": "admin@example.edu", "password": "test-password-123"}).json()["access_token"]
    student_token = client.post("/auth/login", json={"email": "student@example.edu", "password": "test-password-123"}).json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    student_headers = {"Authorization": f"Bearer {student_token}"}

    now = datetime.now()
    create_payload = {"title": "New event", "description": "Details", "category": "AI/ML",
        "department": "Computer Science", "organizer": "Test Club", "venue": "Test Hall",
        "start_time": (now + timedelta(days=5)).isoformat(), "end_time": (now + timedelta(days=5, hours=1)).isoformat(),
        "registration_deadline": (now + timedelta(days=4)).isoformat(), "fee": 0, "capacity": 10,
        "eligibility": "Students", "rules": "Be respectful", "faq": []}
    assert client.post("/events", headers=student_headers, json=create_payload).status_code == 403
    created = client.post("/events", headers=admin_headers, json=create_payload)
    assert created.status_code == 201
    new_event_id = created.json()["id"]
    edited = client.put(f"/events/{new_event_id}", headers=admin_headers, json={"venue": "Updated Hall"})
    assert edited.status_code == 200
    assert edited.json()["venue"] == "Updated Hall"
    assert client.get("/events").status_code == 200
    assert client.post(f"/events/{first_event}/register", headers=student_headers).status_code == 201
    assert client.post(f"/events/{first_event}/register", headers=student_headers).status_code == 200
    assert client.post(f"/events/{second_event}/view?source=recommend", headers=student_headers).status_code == 200
    assert len(client.get("/notifications", headers=student_headers).json()) == 1
    recommendations = client.get("/recommend?k=5", headers=student_headers)
    assert recommendations.status_code == 200
    assert recommendations.json()[0]["event"]["id"] == second_event
    with session_factory() as db:
        assert db.query(EventActionLog).filter_by(action="recommend_impression").count() == 2
        assert db.query(EventActionLog).filter_by(action="view", source="recommend").count() == 1
    assert client.get("/admin/analytics", headers=student_headers).status_code == 403
    assert client.get("/admin/analytics", headers=admin_headers).status_code == 200
