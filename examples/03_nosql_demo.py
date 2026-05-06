"""
Demo 3: NoSQL Document Store
==============================
Shows flexible schema documents for user watch sessions and event streams.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from rich.console import Console
from rich.syntax import Syntax
from databaseai.nosql_db import UserSessionStore

console = Console()


def main():
    console.rule("[bold cyan]NoSQL Document Store Demo[/bold cyan]")

    store = UserSessionStore()

    # Simulate a realistic user session
    console.print("\n[bold]Simulating alice_w watch session on Smart TV...[/bold]")
    sid = store.create_session("u01", {"type": "smart_tv", "model": "Samsung QN90B", "os": "Tizen 7"})

    events = [
        {"type": "home_view", "section": "continue_watching"},
        {"type": "search", "query": "mind bending sci-fi", "results_count": 12},
        {"type": "hover", "movie_id": "m01", "duration_ms": 2300},
        {"type": "play", "movie_id": "m01", "quality": "4K_HDR"},
        {"type": "pause", "movie_id": "m01", "position_s": 1823},
        {"type": "resume", "movie_id": "m01", "position_s": 1823},
        {"type": "seek", "movie_id": "m01", "from_s": 3600, "to_s": 3500},
        {"type": "complete", "movie_id": "m01", "watch_percent": 99.2},
        {"type": "rate", "movie_id": "m01", "score": 5, "thumbs": "up"},
        # A/B test event — no schema change needed in document store
        {"type": "ab_impression", "experiment": "rec_algo_v2", "variant": "B", "position": 1},
    ]

    for event in events:
        store.add_event("u01", sid, event)

    store.end_session("u01", sid)

    session = store.get_session("u01", sid)
    console.print(f"\n[green]✓[/green] Session captured {len(session['events'])} events")
    console.print(f"  Device: {session['device']}")
    console.print(f"  Started: {session['started_at']}")
    console.print(f"  Ended:   {session['ended_at']}")

    # Show the raw document
    console.print("\n[bold]Raw session document (first 4 events):[/bold]")
    preview = {**session, "events": session["events"][:4]}
    syntax = Syntax(json.dumps(preview, indent=2), "json", theme="monokai", line_numbers=False)
    console.print(syntax)

    # Analytics
    console.print("\n[bold]Event analytics:[/bold]")
    counts = store.get_event_counts("u01")
    for etype, count in sorted(counts.items()):
        console.print(f"  {etype:25s}: {count}")

    console.print(f"\n  Watch history: {store.get_watch_history('u01')}")

    # Multi-user
    console.print("\n[bold]Multi-user session isolation:[/bold]")
    for uid in ["u02", "u03"]:
        s = store.create_session(uid, {"type": "mobile"})
        store.add_event(uid, s, {"type": "play", "movie_id": "m07"})

    console.print(f"  Total sessions across all users: {store.session_count()}")
    console.print(f"  u02 watch history: {store.get_watch_history('u02')}")
    console.print(f"  u03 watch history: {store.get_watch_history('u03')}")
    console.print(f"  u01 watch history: {store.get_watch_history('u01')}")
    console.print("\n  [green]User data is completely isolated — no cross-contamination[/green]")


if __name__ == "__main__":
    main()
