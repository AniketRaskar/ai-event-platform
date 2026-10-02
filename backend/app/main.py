"""FastAPI application entry point for the AI Event Platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from .config import CORS_ORIGINS
from .database import Base, engine
from . import models  # noqa: F401 - register SQLAlchemy table metadata
from .routers import admin, auth, chat, events, notifications, recommendations


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create any newly added local tables before serving requests."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="AI Event Platform API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
for route_module in (auth, events, recommendations, chat, admin, notifications):
    app.include_router(route_module.router)


@app.get("/health", tags=["health"])
def health():
    """Check that the API can reach its configured database."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}
