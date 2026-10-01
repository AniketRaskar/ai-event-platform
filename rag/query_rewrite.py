"""Extract deterministic metadata filters from ordinary event questions."""
from __future__ import annotations
from datetime import datetime, time, timedelta
from ml.config import CATEGORIES, DEPARTMENTS

def rewrite_query(query: str, now: datetime | None = None) -> dict:
    """Map time, fee, department and category language to metadata filters."""
    now = now or datetime.now()
    lower = query.casefold()
    filters: dict = {}
    if "free" in lower or "no fee" in lower or "without a fee" in lower:
        filters["fee"] = 0.0
    elif "paid" in lower or "fee-based" in lower:
        filters["fee_gt"] = 0.0
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "today" in lower:
        filters.update(start_after=day_start, start_before=day_start + timedelta(days=1))
    elif "tomorrow" in lower:
        start = day_start + timedelta(days=1)
        filters.update(start_after=start, start_before=start + timedelta(days=1))
    elif "this weekend" in lower or "weekend" in lower:
        saturday = day_start + timedelta(days=(5 - day_start.weekday()) % 7)
        if saturday < day_start:
            saturday += timedelta(days=7)
        sunday_end = saturday + timedelta(days=2)
        filters.update(start_after=saturday, start_before=sunday_end)
    elif "next week" in lower:
        monday = day_start + timedelta(days=(7 - day_start.weekday()) % 7 or 7)
        filters.update(start_after=monday, start_before=monday + timedelta(days=7))
    for department in DEPARTMENTS:
        if department.casefold() in lower:
            filters["department"] = department
            break
    aliases = {"artificial intelligence": "AI/ML", "machine learning": "AI/ML", "ai/ml": "AI/ML",
               "software": "Web/Software", "web development": "Web/Software", "cyber": "Cybersecurity",
               "robotics": "Robotics/IoT", "iot": "Robotics/IoT", "culture": "Cultural",
               "startup": "Entrepreneurship", "entrepreneur": "Entrepreneurship", "career": "Higher Studies/Career"}
    for keyword, category in aliases.items():
        if keyword in lower:
            filters["category"] = category
            break
    if "category" not in filters:
        for category in CATEGORIES:
            if category.casefold() in lower:
                filters["category"] = category
                break
    return filters

def rewrite(query: str, now: datetime | None = None) -> tuple[str, dict]:
    """Compatibility helper returning the original query and parsed filters."""
    return query, rewrite_query(query, now=now)
