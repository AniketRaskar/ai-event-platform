"""Application settings loaded from environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'event_platform.db'}")
SECRET_KEY = os.getenv("SECRET_KEY", "development-only-change-me")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")
