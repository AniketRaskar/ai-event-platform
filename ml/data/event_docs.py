"""Templates that make clearly synthetic, brochure-style event records."""
from datetime import datetime, timedelta
import random

TITLE_BITS = {
    "AI/ML": ["Model Sprint", "Vision Lab Challenge", "Data Jam", "Promptathon"],
    "Web/Software": ["HackPVG", "Build Night", "Open Source Day", "Full Stack Forge"],
    "Cybersecurity": ["Capture the Flag", "Blue Team Lab", "Cyber Range"],
    "Robotics/IoT": ["RoboRally", "IoT Maker Day", "Autonomous Bot Challenge"],
    "Sports": ["Interdepartment Cup", "Campus Run", "Court Clash"],
    "Cultural": ["Sargam Night", "Rangotsav", "Campus Stage"],
    "Entrepreneurship": ["Venture Pitch Day", "Startup Studio", "Founder Forum"],
    "Higher Studies/Career": ["Career Connect", "Grad School Clinic", "Alumni Pathways"],
}
CLUBS = ["ByteForge", "Aarohan Society", "Robotics Circle", "Entrepreneurship Cell", "Athletics Council", "Cultural Collective", "CyberCell", "IEEE Student Branch", "Design Guild", "Eco Club", "Career Forum", "Open Source Community", "Literary Circle", "Photography Club", "Innovation Lab"]
VENUES = ["Main Auditorium", "Innovation Lab, Block C", "Seminar Hall 2", "Central Sports Ground", "Library Conference Room", "Student Activity Centre"]
CONTACTS = ["events@college.edu", "student.affairs@college.edu", "clubs@college.edu"]

def build_event_copy(rng: random.Random, category: str, event_code: str, posted_at: datetime, start_time: datetime) -> dict:
    """Return realistic synthetic event copy and operational details."""
    title = f"{rng.choice(TITLE_BITS[category])} {event_code}"
    fee = rng.choice([0, 0, 0, 50, 100, 150, 250])
    team_size = rng.choice([1, 2, 3, 4])
    prize = rng.choice(["₹5,000 in prizes", "mentorship sessions", "certificates and sponsor kits", "a featured project showcase"])
    venue = rng.choice(VENUES)
    deadline = start_time.replace(hour=23) - timedelta(days=rng.randint(1, 4))
    article = "an" if category.startswith(("AI/ML", "Entrepreneurship")) else "a"
    eligibility = f"Open to currently enrolled students from all five departments. Teams may include {team_size} participant{'s' if team_size > 1 else ''}; beginners are welcome."
    description = (f"{title} is {article} {category} campus gathering hosted by {rng.choice(CLUBS)}. "
                   f"Join hands-on sessions, meet student mentors, and take part in a guided challenge. "
                   f"The programme begins at {start_time:%I:%M %p} at {venue}; selected projects receive {prize}. "
                   f"Registration closes {deadline:%d %b %Y}. Contact {rng.choice(CONTACTS)} for accessibility or schedule questions.")
    rules = (f"Bring your student ID and arrive 15 minutes early. Teams must have {team_size} or fewer members and submit original work. "
             f"Respect venue staff, follow the safety briefing, and do not share another participant's work without permission. "
             f"The registration fee is {'free' if fee == 0 else f'₹{fee} per participant'}; late arrivals may lose their reserved seat.")
    faq = [
        {"q": "Who can attend?", "a": eligibility},
        {"q": "What is the team size?", "a": f"Teams can have up to {team_size} participant(s); solo entries are welcome."},
        {"q": "Is there a fee?", "a": "Admission is free." if fee == 0 else f"The fee is ₹{fee} per participant."},
        {"q": "Where is the event?", "a": f"Meet at {venue}. Check in at the welcome desk."},
        {"q": "What should I bring?", "a": "Bring your student ID and a laptop if your team plans to build or present a project."},
    ]
    return {"title": title, "description": description, "organizer": rng.choice(CLUBS), "venue": venue,
            "fee": fee, "capacity": rng.randint(35, 240), "eligibility": eligibility, "rules": rules,
            "faq": faq, "registration_deadline": deadline}
