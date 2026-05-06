"""
Demo 4: Feature Store
======================
Shows offline feature computation, online serving, and point-in-time correctness.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich import box
from databaseai.feature_store import FeatureStore
from databaseai.relational_db import MovieRegistry
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()


def main():
    console.rule("[bold cyan]Feature Store Demo[/bold cyan]")

    # Build registry first (source of truth for raw ratings)
    reg = MovieRegistry()
    reg.add_movies(MOVIES)
    for u in USERS:
        reg.add_user(u)
    for uid, mid, score, review in RATINGS:
        reg.add_rating(uid, mid, score, review)

    # ---- Offline: compute features (batch job) ----
    fs = FeatureStore()
    console.print("\n[bold]Computing user features (offline batch)...[/bold]")
    for u in USERS:
        raw_ratings = reg.get_user_ratings(u["id"])
        feats = fs.compute_user_features(
            user_id=u["id"],
            ratings=raw_ratings,
            watch_count_7d=5,
            watch_count_30d=20,
        )
        console.print(f"  {u['username']:12s}  avg_rating={feats['avg_rating']:.2f}  "
                      f"fav_genre={feats['fav_genre']:12s}  ratings={feats['total_ratings']}")

    console.print("\n[bold]Computing movie features (offline batch)...[/bold]")
    for m in MOVIES:
        stats = reg.movie_stats(m["id"])
        rlist = ([stats["avg_score"]] * int(stats["total_ratings"])
                 if stats["total_ratings"] else [])
        fs.compute_movie_features(m["id"], m["genre"], rlist)
    counts = fs.feature_count()
    console.print(f"  {counts['movie_features']} movie feature vectors written")

    # ---- Online: serve features (real-time path) ----
    console.print("\n[bold]Online serving — fetch user features at inference time:[/bold]")
    t = Table("User", "Avg Rating", "Fav Genre", "Watch 7d", "Watch 30d", box=box.SIMPLE)
    for u in USERS:
        f = fs.get_user_features(u["id"])
        t.add_row(u["username"], str(f["avg_rating"]), f["fav_genre"],
                  str(f["watch_count_7d"]), str(f["watch_count_30d"]))
    console.print(t)

    # ---- Top movies by popularity ----
    console.print("[bold]Top 5 movies by popularity score:[/bold]")
    t = Table("Movie ID", "Popularity", "Avg Score", "# Ratings", box=box.SIMPLE)
    for r in fs.get_top_movies_by_popularity(n=5):
        t.add_row(r["movie_id"], f"{r['popularity_score']:.2f}",
                  f"{r['avg_score']:.2f}", str(r["total_ratings"]))
    console.print(t)

    # ---- Genre vector ----
    console.print("\n[bold]Genre feature vector for Inception (m01):[/bold]")
    mf = fs.get_movie_features("m01")
    for genre, weight in mf["genre_vector"].items():
        bar = "█" * int(weight * 20)
        console.print(f"  {genre:12s} {bar or '─'} ({weight:.1f})")

    # ---- Point-in-time training snapshot ----
    console.print("\n[bold]Point-in-time training snapshot (prevents future data leakage):[/bold]")
    snapshot = fs.get_training_snapshot("2099-01-01T00:00:00+00:00")
    console.print(f"  Snapshot contains {len(snapshot)} user feature records")
    console.print("  [green]Same features used for training will be served at inference — no skew[/green]")


if __name__ == "__main__":
    main()
