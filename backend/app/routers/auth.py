"""Registration, login and profile endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..deps import get_current_user
from ..database import get_db
from ..models import User
from ..schemas import InterestsUpdate, LoginRequest, TokenResponse, UserCreate, UserResponse
from ..security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a student account and return its public profile."""
    if db.scalar(select(User.id).where(User.email == payload.email.lower())):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = User(name=payload.name.strip(), email=payload.email.lower(), password_hash=hash_password(payload.password),
                role="student", department=payload.department.strip(), year=payload.year,
                interests=list(dict.fromkeys(item.strip() for item in payload.interests if item.strip())))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate a demo or registered user and return a signed token."""
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    """Return the authenticated account's public profile."""
    return user


@router.put("/me/interests", response_model=UserResponse)
def update_interests(payload: InterestsUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Replace the interest list used for cold-start recommendations."""
    user.interests = list(dict.fromkeys(item.strip() for item in payload.interests if item.strip()))
    db.commit()
    db.refresh(user)
    return user
