"""Ask a question against the local RAG index from the command line."""
from __future__ import annotations
import argparse
import sys
from rag.ingest import INDEX_DIR, rebuild
from rag.router import route_message

def main() -> None:
    """Ensure an index exists, route the supplied question and print the answer."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Ask the local AI Event Platform assistant")
    parser.add_argument("question", help="Your question about campus events or sample campus guidance")
    parser.add_argument("--user-id", type=int, default=None, help="Optional user ID for recommendation routing")
    args = parser.parse_args()
    if not (INDEX_DIR / "chunks.json").exists():
        rebuild()
    result = route_message(args.question, user_id=args.user_id)
    print(f"[{result['route']}] {result['answer']}")

if __name__ == "__main__":
    main()
