"""
Demo 9: Caching Layer
======================
Builds an LRU cache with TTL on top of a relational DB.  Caches expensive
aggregate queries (top-rated movies by genre, per-genre stats) and measures
the latency difference between cold (cache miss) and hot (cache hit) calls.

Real-world parallel: Redis caching Netflix homepage carousels — the same
GROUP-BY + AVG query that powers "Top Sci-Fi Picks" is served from an
in-process cache on every subsequent page reload instead of hitting Postgres.
"""

import sys
import os
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.cache_layer import LRUCache, CachedMovieDB

console = Console()

SIMULATED_DB_DELAY = 0.002


def _build_db(cache_capacity=32, ttl=60.0, delay=SIMULATED_DB_DELAY):
    db = CachedMovieDB(cache_capacity=cache_capacity, ttl_seconds=ttl, query_delay=delay)
    db.seed(MOVIES, RATINGS)
    return db


def _bench(fn, n: int) -> tuple[float, Any]:
    result = None
    t0 = time.perf_counter()
    for _ in range(n):
        result = fn()
    return time.perf_counter() - t0, result


def main():
    console.rule("[bold cyan]Caching Layer Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Redis caching Netflix homepage carousels — "
        "expensive aggregate queries served from in-memory LRU instead of Postgres[/dim]\n"
    )

    db = _build_db()

    console.print(Panel("[bold]1. Dataset Loaded into SQLite[/bold]", box=box.ROUNDED))
    console.print(
        f"  [green]✓[/green] {db.movie_count()} movies  |  "
        f"{db.rating_count()} ratings  |  "
        f"Cache capacity: {db.cache.capacity} entries  |  "
        f"Simulated DB latency: {SIMULATED_DB_DELAY*1000:.0f} ms per query\n"
        f"  [dim]Each cache miss incurs one SQL GROUP BY + JOIN; "
        f"subsequent hits are pure Python dict lookups.[/dim]"
    )

    console.print()
    console.print(Panel("[bold]2. Cache Miss vs Cache Hit — genre_stats()[/bold]", box=box.ROUNDED))
    REPS = 50
    db.cache.clear()
    miss_time, _ = _bench(db.genre_stats, 1)
    db.cache.reset_stats()
    hit_time, stats_result = _bench(db.genre_stats, REPS)
    t = Table("Query", "Calls", "Total (ms)", "Avg per call (µs)", box=box.SIMPLE_HEAD)
    t.add_row("genre_stats() — MISS", "1", f"{miss_time*1000:.2f}", f"{miss_time*1_000_000:.0f}")
    t.add_row("genre_stats() — HIT",  str(REPS), f"{hit_time*1000:.2f}", f"{hit_time/REPS*1_000_000:.1f}")
    console.print(t)
    speedup = miss_time / (hit_time / REPS) if hit_time else float("inf")
    console.print(f"  Cache hit is [bold green]{speedup:.0f}×[/bold green] faster per call "
                  f"(miss ≈ {miss_time*1000:.2f} ms, hit ≈ {hit_time/REPS*1000:.3f} ms)")

    console.print()
    console.print(Panel("[bold]3. Cached Genre Stats — What the Query Returns[/bold]", box=box.ROUNDED))
    t = Table("Genre", "Movies", "Ratings", "Avg Score", box=box.SIMPLE_HEAD)
    for row in stats_result:
        t.add_row(row["genre"] or "(none)", str(row["movie_count"]),
                  str(row["rating_count"]), str(row["avg_rating"]) if row["avg_rating"] else "—")
    console.print(t)

    console.print()
    console.print(Panel("[bold]4. Top-Rated Carousel — top_rated_by_genre()[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Netflix: \"Best Sci-Fi\" carousel = SELECT … WHERE genre='sci-fi' ORDER BY avg_score DESC\n"
        "  The carousel is built once and cached; all 250 M users share the same cached row.[/dim]"
    )
    db.cache.clear()
    db.cache.reset_stats()
    genres = ["sci-fi", "action", "thriller", "drama", "animation"]
    t = Table("Genre", "Miss (ms)", "Hit (ms)", "Speedup", "Top Movie", box=box.SIMPLE_HEAD)
    for genre in genres:
        t0 = time.perf_counter()
        movies = db.top_rated_by_genre(genre, limit=3)
        miss_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        db.top_rated_by_genre(genre, limit=3)
        hit_ms = (time.perf_counter() - t0) * 1000
        sp = f"{miss_ms/hit_ms:.0f}×" if hit_ms > 0 else "∞"
        t.add_row(genre, f"{miss_ms:.2f}", f"{hit_ms:.3f}", sp, movies[0]["title"] if movies else "—")
    console.print(t)

    console.print()
    console.print(Panel("[bold]5. Cache Statistics After Section 4[/bold]", box=box.ROUNDED))
    s = db.cache.stats()
    t = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t.add_row("Cache size",    f"{s['size']} / {s['capacity']} entries")
    t.add_row("Total lookups", str(s["total"]))
    t.add_row("Hits",          str(s["hits"]))
    t.add_row("Misses",        str(s["misses"]))
    t.add_row("Hit rate",      f"{s['hit_rate']*100:.1f}%")
    console.print(t)
    console.print(f"  Current keys in cache (LRU → MRU):\n    [cyan]{', '.join(db.cache.keys())}[/cyan]")

    console.print()
    console.print(Panel("[bold]6. LRU Eviction — Capacity-3 Cache[/bold]", box=box.ROUNDED))
    lru = LRUCache(capacity=3, ttl_seconds=0)
    t = Table("Action", "Keys after (LRU→MRU)", "Eviction", box=box.SIMPLE_HEAD)
    lru.put("sci-fi", "sci-fi data")
    lru.put("action", "action data")
    lru.put("thriller", "thriller data")
    t.add_row("put(sci-fi, action, thriller)", ", ".join(lru.keys()), "—")
    lru.get("sci-fi")
    t.add_row("get(sci-fi) — promote to MRU", ", ".join(lru.keys()), "—")
    lru.put("drama", "drama data")
    t.add_row("put(drama) — capacity full", ", ".join(lru.keys()), "action evicted")
    lru.put("horror", "horror data")
    t.add_row("put(horror)", ", ".join(lru.keys()), "thriller evicted")
    console.print(t)

    console.print()
    console.print(Panel("[bold]7. TTL Expiration — Stale Cache Entries[/bold]", box=box.ROUNDED))
    short_db = _build_db(ttl=0.05, delay=0.0)
    short_db.cache.clear()
    short_db.cache.reset_stats()
    t = Table("Call #", "Time after seed", "Result", "Cache state", box=box.SIMPLE_HEAD)
    t0 = time.perf_counter(); short_db.genre_stats(); ms1 = (time.perf_counter() - t0) * 1000
    t.add_row("1 (miss)", "0 ms", f"fetched from DB ({ms1:.2f} ms)", "1 entry, fresh")
    t0 = time.perf_counter(); short_db.genre_stats(); ms2 = (time.perf_counter() - t0) * 1000
    t.add_row("2 (hit)", f"~{ms1:.0f} ms", f"from cache ({ms2:.3f} ms)", "1 entry, fresh")
    time.sleep(0.06)
    t0 = time.perf_counter(); short_db.genre_stats(); ms3 = (time.perf_counter() - t0) * 1000
    t.add_row("3 (miss)", "~60 ms", f"TTL expired → re-fetched ({ms3:.2f} ms)", "1 entry, fresh")
    console.print(t)

    console.print()
    console.print(Panel("[bold]8. Targeted Cache Invalidation on Write[/bold]", box=box.ROUNDED))
    inv_db = _build_db(delay=0.0)
    inv_db.genre_stats()
    inv_db.top_rated_by_genre("sci-fi")
    inv_db.top_rated_by_genre("action")
    keys_before = set(inv_db.cache.keys())
    inv_db.add_rating("u01", "m04", 4.8, "Still a classic")
    keys_after = set(inv_db.cache.keys())
    evicted  = keys_before - keys_after
    retained = keys_before & keys_after
    t = Table("Cache key", "Status", "Reason", box=box.SIMPLE_HEAD)
    for k in sorted(keys_before):
        if k in evicted:
            t.add_row(k, "[red]EVICTED[/red]",    "aggregates over modified data")
        else:
            t.add_row(k, "[green]RETAINED[/green]", "unaffected by this write")
    console.print(t)
    console.print(f"  Evicted: [bold red]{len(evicted)}[/bold red]  |  Retained: [bold green]{len(retained)}[/bold green]")

    console.print()
    console.print("[bold green]Key Caching Layer Takeaways:[/bold green]")
    console.print("  • [cyan]LRU eviction[/cyan]             — OrderedDict gives O(1) promote and O(1) evict; same structure as functools.lru_cache")
    console.print("  • [cyan]TTL[/cyan]                      — bounds staleness without an external scheduler; lazy expiry on read keeps the hot path lock-free")
    console.print("  • [cyan]Write-around + invalidation[/cyan] — targeted key removal preserves unrelated hot entries while preventing stale aggregates")
    console.print("  • [cyan]Hit rate[/cyan]                 — the primary cache health metric; below ~80% the miss penalty dominates")
    console.print("  [dim]Production: Redis (in-process or remote), Memcached (shared across processes), Caffeine (JVM, W-TinyLFU policy)[/dim]")


if __name__ == "__main__":
    main()
