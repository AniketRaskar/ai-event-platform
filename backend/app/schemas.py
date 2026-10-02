"""Validated request and response models for the HTTP API."""
from datetime import datetime
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _valid_email(value: str) -> str:
    """Normalize a basic email address and reject malformed values."""
    value = value.strip().lower()
    if not _EMAIL.fullmatch(value):
        raise ValueError("Enter a valid email address")
    return value


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str
    password: str = Field(min_length=8, max_length=128)
    department: str = Field(min_length=1, max_length=100)
    year: int = Field(ge=1, le=8)
    interests: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _valid_email(value)


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _valid_email(value)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    role: str
    department: str
    year: int
    interests: list[str]


class InterestsUpdate(BaseModel):
    interests: list[str] = Field(max_length=20)


class EventCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str
    category: str = Field(min_length=1, max_length=100)
    department: str = Field(min_length=1, max_length=100)
    organizer: str = Field(min_length=1, max_length=160)
    venue: str = Field(min_length=1, max_length=200)
    start_time: datetime
    end_time: datetime
    registration_deadline: datetime
    fee: float = Field(ge=0)
    capacity: int = Field(ge=1)
    eligibility: str
    rules: str
    faq: list[dict[str, str]] = Field(default_factory=list)


class EventUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    category: str | None = None
    department: str | None = None
    organizer: str | None = None
    venue: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    registration_deadline: datetime | None = None
    fee: float | None = Field(default=None, ge=0)
    capacity: int | None = Field(default=None, ge=1)
    eligibility: str | None = None
    rules: str | None = None
    faq: list[dict[str, str]] | None = None


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=30)


class MarkRead(BaseModel):
    read: bool = True


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    description: str
    category: str
    department: str
    organizer: str
    venue: str
    start_time: datetime
    end_time: datetime
    registration_deadline: datetime
    fee: float
    capacity: int
    eligibility: str
    rules: str
    faq: list[dict]
    posted_at: datetime
