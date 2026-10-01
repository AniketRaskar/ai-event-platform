"""Shared configuration for the synthetic event data and ML pipeline."""
from pathlib import Path

SEED = 42
N_USERS = 500
N_EVENTS = 250
ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = Path(__file__).resolve().parent / "data" / "generated"
DATABASE_PATH = ROOT / "event_platform.db"
CATEGORIES = ["AI/ML", "Web/Software", "Cybersecurity", "Robotics/IoT", "Sports", "Cultural", "Entrepreneurship", "Higher Studies/Career"]
DEPARTMENTS = ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"]
