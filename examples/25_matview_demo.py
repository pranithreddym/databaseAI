"""
Demo 25: Materialized Views & Refresh Strategies
=================================================
A regular SQL VIEW is a saved query — it re-runs on every access.  A
MATERIALIZED VIEW is a snapshot of that query's result stored as a real
table.  The trade-off is freshness vs. speed: the snapshot can be stale,
but reads are instant regardless of how expensive the underlying aggregation
would be.

This demo builds two materialized views over the CineAI seed data and
explores four refresh strategies side-by-side:
  1. Full eager refresh — rebuild inside the write transaction; readers are
     always current, writers pay the cost.
  2. Lazy refresh — mark the view stale on write, rebuild only when a read
     arrives and finds the stale flag set.
  3. Incremental (partial) refresh — recompute only the slice of the MV
     that actually changed, without touching unaffected rows.
  4. Scheduled batch refresh — accumulate many writes, then rebuild all
     stale views in one pass, amortising the refresh cost.

Real-world parallel: Netflix pre-computes genre carousels ("Top 10 in Sci-Fi",
"New Arrivals in Drama") as materialized snapshots in EVCache/Cassandra.  A
Spark job refreshes them every 15 minutes; between refreshes all homepage
loads read sub-millisecond snapshots — decoupling read throughput from the
cost of aggregating billions of rating events.  PostgreSQL REFRESH MATERIALIZED
VIEW CONCURRENTLY, Snowflake DYNAMIC TABLE, and dbt incremental models each
implement variations of the same pattern at different layers of the stack.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import random

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.matview import MaterializedViewStore
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()
random.seed(42)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int = 32) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _bar(value: float, max_val: float, width: int = 14) -> str:
    if max_val <= 0:
        return "░" * width
    filled = max(0, int(round(value / max_val * width)))
    return "[cyan]" + "█" * filled + "░" * (width - filled) + "[/cyan]"


def _ms(value: float) -> str:
    return f"{value:.3f} ms"


# ── Setup ─────────────────────────────────────────────────────────────────────

def _build_store() -> MaterializedViewStore:
    store = MaterializedViewStore()
    store.load_seed(MOVIES, RATINGS)
    return store


# ── Main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule(
        "[bold cyan]Database Demo 25 — Materialized Views & Refresh Strategies[/bold cyan]"
    )
    console.print(
        "[dim]Real-world parallel: Netflix genre carousels are pre-computed "
        "materialized snapshots refreshed every 15 minutes by Spark jobs. "
        "Between refreshes, millions of homepage loads read sub-millisecond "
        "snapshots without touching the live ratings tables.[/dim]\n"
    )

    store = _build_store()

    # ── Section 1: What the MVs contain ──────────────────────────────────────
    console.print(Panel("[bold]1. Schema — Two Materialized Views[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Source tables:  movies (20 rows)  +  ratings (24 rows)[/dim]\n"
        "  [dim]mv_genre_stats: one row per genre — avg rating, movie count, "
        "top title.[/dim]\n"
        "  [dim]mv_top_movies:  ranked list of movies by avg rating — instant "
        "top-N without aggregating ratings.[/dim]\n"
        "  [dim]mv_meta:        one control row per view — last_refreshed, "
        "is_stale flag, refresh_count, total_refresh_ms.[/dim]"
    )

    # Initial full refresh
    elapsed_g = store.refresh_genre_stats()
    elapsed_t = store.refresh_top_movies()
    console.print(
        f"\n  [green]✓[/green] Initial full refresh complete — "
        f"mv_genre_stats in {_ms(elapsed_g)}, "
        f"mv_top_movies in {_ms(elapsed_t)}"
    )

    # Show genre stats MV
    console.print()
    console.print(Panel("[bold]2. mv_genre_stats — Snapshot of Genre Aggregates[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]One JOIN + GROUP BY over movies × ratings computed once; "
        "all subsequent reads return this table at O(1) per genre.[/dim]"
    )
    t = Table("Genre", "Avg Rating", "Movies", "Ratings", "Top Movie", box=box.SIMPLE_HEAD)
    for row in store.get_genre_stats():
        t.add_row(
            row["genre"],
            f"{row['avg_rating']:.4f}",
            str(row["movie_count"]),
            str(row["rating_count"]),
            _truncate(row["top_movie_title"] or "—"),
        )
    console.print(t)

    # Show top movies MV
    console.print()
    console.print(Panel("[bold]3. mv_top_movies — Pre-Ranked Leaderboard[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Ranking by AVG(rating) with ROW_NUMBER() is computed at refresh "
        "time; reads are a simple primary-key scan of mv_top_movies.[/dim]"
    )
    max_r = 5.0
    t = Table("Rank", "Title", "Genre", "Avg Rating", "Ratings", "Bar", box=box.SIMPLE_HEAD)
    for row in store.get_top_movies(n=10):
        t.add_row(
            str(row["rank"]),
            _truncate(row["title"]),
            row["genre"],
            f"{row['avg_rating']:.4f}",
            str(row["rating_count"]),
            _bar(row["avg_rating"], max_r),
        )
    console.print(t)

    # ── Section 2: Eager refresh ──────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]4. Eager Refresh — Rebuild Inside the Write Transaction[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]With eager=True, add_rating() inserts the new row and immediately "
        "rebuilds both MVs in the same logical write.  The next reader always "
        "sees current data; the writer pays the rebuild cost.[/dim]"
    )

    t0 = time.perf_counter()
    store.add_rating("u01", "m02", 4.8, eager=True)
    eager_write_ms = (time.perf_counter() - t0) * 1000

    meta = store.get_meta(MaterializedViewStore.VIEW_GENRE_STATS)
    console.print(
        f"\n  [green]✓[/green] Rating written + both MVs rebuilt in {_ms(eager_write_ms)}\n"
        f"  [dim]mv_genre_stats refresh count: {meta['refresh_count']}  "
        f"| is_stale: {'yes' if meta['is_stale'] else 'no'}[/dim]"
    )
    stale_after_eager = store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
    console.print(
        f"  [green]✓[/green] MV is_stale after eager write: "
        f"[{'red' if stale_after_eager else 'green'}]{stale_after_eager}[/]"
    )

    # ── Section 3: Lazy refresh ───────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]5. Lazy Refresh — Rebuild on First Stale Read[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]With eager=False (default), writes are fast — only the stale flag "
        "is set.  The first reader that finds is_stale=1 triggers the rebuild "
        "transparently with lazy=True, then returns fresh data.[/dim]"
    )

    store.add_rating("u02", "m03", 4.2, eager=False)
    stale_before = store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
    console.print(
        f"\n  After lazy write → is_stale: "
        f"[{'red' if stale_before else 'green'}]{stale_before}[/]"
    )

    t0 = time.perf_counter()
    rows = store.get_genre_stats(lazy=True)
    lazy_read_ms = (time.perf_counter() - t0) * 1000
    stale_after = store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)

    console.print(
        f"  First stale read (lazy=True) → refreshed + returned {len(rows)} rows "
        f"in {_ms(lazy_read_ms)}"
    )
    console.print(
        f"  [green]✓[/green] is_stale after lazy read: "
        f"[{'red' if stale_after else 'green'}]{stale_after}[/]"
    )

    t0 = time.perf_counter()
    store.get_genre_stats(lazy=True)
    second_read_ms = (time.perf_counter() - t0) * 1000
    console.print(
        f"  Second read (MV already fresh):  {_ms(second_read_ms)}"
    )

    # ── Section 4: Incremental refresh ───────────────────────────────────────
    console.print()
    console.print(Panel("[bold]6. Incremental Refresh — Recompute Only Changed Genres[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]When a new rating targets a single genre, only that genre's row in "
        "mv_genre_stats needs updating.  refresh_genre_for(['sci-fi']) runs the "
        "aggregation WHERE genre='sci-fi' and UPSERTS one row — far cheaper than "
        "a full DELETE + re-INSERT of every genre.[/dim]"
    )

    store.add_rating("u03", "m04", 5.0, eager=False)
    store.mark_stale(MaterializedViewStore.VIEW_GENRE_STATS)

    t_full = store.refresh_genre_stats()
    store.mark_stale(MaterializedViewStore.VIEW_GENRE_STATS)
    t_incr = store.refresh_genre_for(["sci-fi"])

    console.print(
        f"\n  Full refresh (all genres):          {_ms(t_full)}\n"
        f"  Incremental refresh (sci-fi only):  {_ms(t_incr)}\n"
        f"  [dim]At scale (thousands of genres), the incremental path avoids "
        f"recomputing genres that had no new activity.[/dim]"
    )

    rows = store.get_genre_stats()
    scifi = next((r for r in rows if r["genre"] == "sci-fi"), None)
    if scifi:
        console.print(
            f"  [green]✓[/green] Updated sci-fi: avg={scifi['avg_rating']:.4f}  "
            f"ratings={scifi['rating_count']}  "
            f"top_movie={_truncate(scifi['top_movie_title'] or '—')}"
        )

    # ── Section 5: Benchmark — live query vs MV read ──────────────────────────
    console.print()
    console.print(Panel("[bold]7. Benchmark — Live Query vs. Materialized View Read[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Run the full GROUP BY aggregation N times on live tables, then "
        "read from the pre-computed MV N times.  The MV read is a simple "
        "SELECT * FROM mv_genre_stats with no join or aggregation.[/dim]"
    )

    bm = store.benchmark(n_queries=300)
    speedup_color = "green" if bm["speedup_x"] >= 1.0 else "yellow"
    t = Table(box=box.SIMPLE_HEAD)
    t.add_column("Metric",    style="dim")
    t.add_column("Live Query")
    t.add_column("MV Read")
    t.add_column("Delta")
    t.add_row(
        f"Total ({bm['n_queries']} queries)",
        _ms(bm["live_total_ms"]),
        _ms(bm["mv_total_ms"]),
        f"[{speedup_color}]{bm['speedup_x']}× faster[/]",
    )
    t.add_row(
        "Per query (avg)",
        _ms(bm["live_avg_ms"]),
        _ms(bm["mv_avg_ms"]),
        "",
    )
    console.print(t)
    console.print(
        f"  [dim]Live query: JOIN movies × ratings + GROUP BY genre every time.\n"
        f"  MV read: SELECT * FROM mv_genre_stats — one indexed table scan.[/dim]"
    )

    # ── Section 6: Scheduled batch refresh ───────────────────────────────────
    console.print()
    console.print(Panel("[bold]8. Scheduled Batch Refresh — Amortise Cost Across Many Writes[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Simulate a burst of 15 writes (no eager rebuild) then refresh all "
        "stale views in one pass.  The amortised cost per write is the total "
        "refresh time divided by the number of writes that triggered the stale "
        "flag — typically far lower than rebuilding on every write.[/dim]"
    )

    extra_users  = [u["id"] for u in USERS]
    extra_movies = [m["id"] for m in MOVIES if m["id"] not in {"m01", "m02", "m03"}]
    write_count  = 15
    for i in range(write_count):
        uid = extra_users[i % len(extra_users)]
        mid = extra_movies[i % len(extra_movies)]
        rating = round(3.0 + random.random() * 2.0, 1)
        store.add_rating(uid, mid, rating, eager=False)

    both_stale = (
        store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS) and
        store.is_stale(MaterializedViewStore.VIEW_TOP_MOVIES)
    )
    console.print(
        f"\n  After {write_count} writes — both MVs stale: "
        f"[{'red' if both_stale else 'green'}]{both_stale}[/]"
    )

    t0 = time.perf_counter()
    times = store.refresh_all()
    batch_ms = (time.perf_counter() - t0) * 1000

    console.print(
        f"  Batch refresh (both views) in {_ms(batch_ms)}:\n"
        f"    mv_genre_stats: {_ms(times[MaterializedViewStore.VIEW_GENRE_STATS])}\n"
        f"    mv_top_movies:  {_ms(times[MaterializedViewStore.VIEW_TOP_MOVIES])}\n"
        f"  Amortised cost per write: {_ms(batch_ms / write_count)}"
    )

    meta_g = store.get_meta(MaterializedViewStore.VIEW_GENRE_STATS)
    meta_t = store.get_meta(MaterializedViewStore.VIEW_TOP_MOVIES)
    console.print(
        f"  [green]✓[/green] Total refreshes — mv_genre_stats: {meta_g['refresh_count']}  "
        f"| mv_top_movies: {meta_t['refresh_count']}"
    )

    # ── Section 7: Final state of top-movies MV ───────────────────────────────
    console.print()
    console.print(Panel("[bold]9. Final mv_top_movies State After All Writes[/bold]", box=box.ROUNDED))
    t = Table("Rank", "Title", "Genre", "Avg Rating", "Ratings", "Bar", box=box.SIMPLE_HEAD)
    for row in store.get_top_movies(n=10):
        t.add_row(
            str(row["rank"]),
            _truncate(row["title"]),
            row["genre"],
            f"{row['avg_rating']:.4f}",
            str(row["rating_count"]),
            _bar(row["avg_rating"], 5.0),
        )
    console.print(t)

    # ── Architecture summary ──────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold]Production Architecture Notes[/bold]\n\n"
            "[dim]• Eager refresh: suitable for read-heavy workloads where stale data is "
            "unacceptable (payment summaries, inventory counts).\n"
            "• Lazy refresh: suitable when occasional write-then-read latency spikes are "
            "tolerable — e.g., recommendation carousels where a 50 ms delay on the first "
            "post-write read is invisible to the user.\n"
            "• Incremental refresh: critical at scale; PostgreSQL REFRESH MATERIALIZED VIEW "
            "CONCURRENTLY and Snowflake DYNAMIC TABLE implement this with change-data-capture "
            "streams to avoid locking readers during rebuild.\n"
            "• Scheduled refresh: the Netflix pattern — Spark/Flink jobs rebuild carousels "
            "on a 15-minute cadence, accepting up to 15 min of staleness in exchange for "
            "decoupling homepage read latency from real-time aggregation cost.[/dim]",
            box=box.ROUNDED,
        )
    )

    console.print("\n[bold green]Demo complete.[/bold green]\n")


if __name__ == "__main__":
    main()
