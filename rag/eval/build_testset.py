"""Build reproducible answerable and deliberately unanswerable event questions."""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from ml.config import GENERATED_DIR, ROOT, SEED

OUTPUT = Path(__file__).resolve().parent / "qa_testset.json"

def build_testset(output_path: Path = OUTPUT, data_dir: Path = GENERATED_DIR) -> list[dict]:
    """Create 50 field questions with gold event IDs and 10 absent-field queries."""
    events_path = data_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError("Run `make data` before building the RAG evaluation set")
    events = pd.read_csv(events_path).sample(frac=1, random_state=SEED).head(10)
    rows = []
    for event in events.to_dict("records"):
        title = str(event["title"])
        eid = int(event["id"])
        team_match = re.search(r"include (\d+) participants?", str(event["eligibility"]))
        team_size = int(team_match.group(1)) if team_match else 1
        questions = [
            (f"Where will {title} take place?", "venue"),
            (f"What is the registration fee for {title}?", "fee"),
            (f"When does registration close for {title}?", "deadline"),
            (f"Who is eligible to join {title}?", "eligibility"),
            (f"What is the maximum team size for {title}?", "team_size"),
        ]
        rows.extend({"question": question, "answerable": True, "gold_event_id": eid, "field": field} for question, field in questions)
    absent_fields = ["keynote speaker", "event menu", "parking availability", "Wi-Fi password", "livestream link",
                     "shuttle pickup time", "judging panel", "refund policy", "dress code", "meal options"]
    sample_events = events.head(len(absent_fields))
    for field, event in zip(absent_fields, sample_events.to_dict("records")):
        rows.append({"question": f"What is the {field} for {event['title']}?", "answerable": False,
                     "gold_event_id": None, "field": field})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows

if __name__ == "__main__":
    result = build_testset()
    print(f"Wrote {len(result)} RAG evaluation questions to {OUTPUT}")
