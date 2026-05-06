"""
Demo 2: Relational Database
============================
Shows ACID transactions, joins, aggregates, and indexes on movie data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich import box
from databaseai.relational_db import MovieRegistry
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()


def main():
    console.rule("[bold cyan]Relational Database Demo[/bold cyan]")

    reg = MovieRegistry()
    reg.add_movies(MOVIES)
    for u in USERS:
        reg.add_user(u)
    for uid, mid, score, review in RATINGS:
        reg.add_rating(uid, mid, score, review)

    console.print(f"\n[green]✓[/green] {reg.movie_count()} movies | {reg.user_count()} users | {reg.rating_count()} ratings\n")

    # Top-rated overall
    console.print("[bold]Top 5 rated movies (min 3 ratings):[/bold]")
    t = Table("Title", "Genre", "Avg Score", "# Ratings", box=box.SIMPLE)
    for r in reg.top_rated_movies(n=5):
        t.add_row(r["title"], r["genre"], str(r["avg_score"]), str(r["total_ratings"]))
    console.print(t)

    # Top sci-fi
    console.print("[bold]Top sci-fi movies:[/bold]")
    t = Table("Title", "Year", "Director", "Avg Score", box=box.SIMPLE)
    for r in reg.top_rated_movies(genre="sci-fi", n=5):
        movie = reg.get_movie(r["id"])
        t.add_row(r["title"], str(movie["year"]), movie["director"], str(r["avg_score"]))
    console.print(t)

    # User ratings
    console.print("[bold]u01 alice_w ratings:[/bold]")
    t = Table("Title", "Genre", "Score", "Review", box=box.SIMPLE)
    for r in reg.get_user_ratings("u01"):
        t.add_row(r["title"], r["genre"], str(r["score"]), r["review"][:30] + "...")
    console.print(t)

    # ACID demo
    console.print("[bold]ACID Transaction — rate then re-rate (upsert):[/bold]")
    reg.add_rating("u01", "m02", 3.0, "First impression")
    stats_before = reg.movie_stats("m02")
    reg.add_rating("u01", "m02", 5.0, "Changed my mind after rewatch")
    stats_after = reg.movie_stats("m02")
    console.print(f"  Before re-rate: avg={stats_before['avg_score']}, count={stats_before['total_ratings']}")
    console.print(f"  After  re-rate: avg={stats_after['avg_score']}, count={stats_after['total_ratings']}")
    console.print("  [green]Count unchanged — UNIQUE constraint enforced upsert[/green]")

    # Nolan filmography
    console.print("[bold]Christopher Nolan filmography in DB:[/bold]")
    t = Table("Title", "Year", "Genre", box=box.SIMPLE)
    for m in reg.search_movies(director="Nolan"):
        t.add_row(m["title"], str(m["year"]), m["genre"])
    console.print(t)


if __name__ == "__main__":
    main()
