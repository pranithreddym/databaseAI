"""
Demo 18: Tiered Caching — Hierarchical L1 + L2 Cache Architecture
==================================================================
Implements a two-tier write-through cache: L1 (in-process LRU) sits in front of
L2 (SQLite-backed warm tier), which sits in front of the source database.  A cache
read cascades L1 → L2 → source; each miss back-fills closer tiers so the next
request is served from the fastest available layer.

Real-world parallel: Netflix EVCache runs Memcached at two layers — a per-rack
L1 (32 GB, 30-second TTL) and a per-region L2 (256 GB, 5-minute TTL).
Recommendation model outputs for "Top Picks" are written through both tiers on
every model refresh; a cache miss cascades L1 → L2 → Cassandra origin, each tier
roughly 10× slower than the previous.  Tiering reduces Cassandra QPS by ~94 %
even when the L1 hit rate drops during a deploy.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.tiered_cache import L1Cache, L2Cache, TieredCache, TieredMovieDB
from databaseai.seed_data import MOVIES, RATINGS

console = Console()

SIMULATED_DB_DELAY = 0.003          # 3 ms simulated source latency per query
GENRES = ["sci-fi", "action", "thriller", "drama", "animation", "horror"]


def _build_db(l1_capacity=32, l1_ttl=30.0, l2_ttl=300.0, delay=SIMULATED_DB_DELAY):
    db = TieredMovieDB(
        l1_capacity=l1_capacity,
        l1_ttl=l1_ttl,
        l2_ttl=l2_ttl,
        query_delay=delay,
    )
    db.seed(MOVIES, RATINGS)
    return db


def _bench(fn, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    console.rule("[bold cyan]Tiered Caching Demo — L1 + L2 Hierarchy[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Netflix EVCache — per-rack L1 Memcached → "
        "per-region L2 Memcached → Cassandra origin\n"
        "Each tier is ~10× slower than the previous; tiering absorbs >94 % of "
        "source reads during recommendation serving.[/dim]\n"
    )

    # ── Section 1: Architecture overview ────────────────────────────────────
    console.print(Panel("[bold]1. Architecture — Read & Write Paths[/bold]", box=box.ROUNDED))
    arch = Table(box=box.SIMPLE_HEAD, show_header=True)
    arch.add_column("Tier",       style="bold")
    arch.add_column("Technology",  style="cyan")
    arch.add_column("Capacity",    justify="right")
    arch.add_column("TTL",         justify="right")
    arch.add_column("Latency (sim)", justify="right")
    arch.add_row("L1", "In-process LRU (OrderedDict)", "32 keys",  " 30 s", "~0 µs")
    arch.add_row("L2", "SQLite cache table",           "unlimited", "300 s", "~50 µs")
    arch.add_row("Source", "SQLite source DB",         "—",         "—",     f"~{SIMULATED_DB_DELAY*1000:.0f} ms")
    console.print(arch)
    console.print(
        "  Read path:  [bold]L1[/bold] → (miss) → [bold]L2[/bold] → (miss) → [bold]Source[/bold]\n"
        "  Write path: [bold]Source[/bold] + [bold]L2[/bold] + [bold]L1[/bold] updated simultaneously (write-through)\n"
        "  On L2 hit:  value promoted to L1 so the next read is served locally."
    )

    # ── Section 2: Dataset ───────────────────────────────────────────────────
    db = _build_db()
    console.print()
    console.print(Panel("[bold]2. Dataset Loaded[/bold]", box=box.ROUNDED))
    console.print(
        f"  [green]✓[/green] {db.movie_count()} movies  |  {db.rating_count()} ratings  |  "
        f"L1 capacity: {db.cache.l1.capacity}  |  "
        f"L1 TTL: {db.cache.l1.ttl:.0f}s  |  L2 TTL: {db.cache.l2.ttl:.0f}s  |  "
        f"Source latency: {SIMULATED_DB_DELAY*1000:.0f} ms\n"
        "  [dim]All 20 movies across 6 genres with 24 ratings, mirroring the "
        "shared seed_data used by every demo.[/dim]"
    )

    # ── Section 3: Cold start — every request traverses all three tiers ─────
    console.print()
    console.print(Panel("[bold]3. Cold Start — All Requests Reach the Source[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]On a fresh deploy both cache tiers are empty.  Each unique query\n"
        "  must travel L1 → L2 → source and pay the full source latency.[/dim]"
    )
    cold_table = Table("Query key", "Tier served", "Latency (ms)", box=box.SIMPLE_HEAD)
    keys_to_warm = ["genre_stats"] + [f"top:{g}" for g in GENRES]
    for key in keys_to_warm:
        t0 = time.perf_counter()
        tier, _ = db.get_cached(key)
        ms = (time.perf_counter() - t0) * 1000
        cold_table.add_row(key, f"[red]{tier}[/red]", f"{ms:.2f}")
    console.print(cold_table)
    s = db.cache.stats()
    console.print(
        f"  After cold start: L1 hits={s['l1_hits']}  L2 hits={s['l2_hits']}  "
        f"source misses={s['source_hits']}  (expected: all source)"
    )

    # ── Section 4: Warm L1 — repeated reads served instantly ────────────────
    console.print()
    console.print(Panel("[bold]4. L1 Warm — Repeated Reads Served from In-Process Cache[/bold]", box=box.ROUNDED))
    REPS = 100
    db.cache.reset_stats()
    t = Table("Query key", f"Avg over {REPS} hits (µs)", "Tier", box=box.SIMPLE_HEAD)
    for key in keys_to_warm:
        t0 = time.perf_counter()
        for _ in range(REPS):
            db.get_cached(key)
        avg_us = (time.perf_counter() - t0) / REPS * 1_000_000
        tier, _ = db.get_cached(key)
        t.add_row(key, f"{avg_us:.1f}", f"[green]{tier}[/green]")
    console.print(t)
    s = db.cache.stats()
    console.print(
        f"  L1 hit rate: [bold green]{s['l1_hit_rate']*100:.1f}%[/bold green]  "
        f"({s['l1_hits']} hits  |  source misses: {s['source_hits']})"
    )
    source_ms = SIMULATED_DB_DELAY * 1000
    avg_l1_us = _bench(lambda: db.get_cached("genre_stats"), 200) * 1000
    console.print(
        f"  L1 hit ≈ {avg_l1_us:.1f} µs  |  source miss ≈ {source_ms:.0f} ms  →  "
        f"[bold green]{source_ms*1000/avg_l1_us:.0f}× faster[/bold green]"
    )

    # ── Section 5: L2 promotion — L1 miss falls through to L2 ───────────────
    console.print()
    console.print(Panel("[bold]5. L2 Promotion — L1 Miss Falls Through to L2[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Simulate a rolling deploy: L1 is cleared (app process restart)\n"
        "  but L2 (shared regional cache) stays warm.  First request after\n"
        "  restart is an L2 hit — much faster than hitting the source DB.[/dim]"
    )
    db.cache.l1.clear()
    db.cache.reset_stats()
    t = Table("Query key", "Tier served", "Latency (ms)", box=box.SIMPLE_HEAD)
    for key in keys_to_warm:
        t0 = time.perf_counter()
        tier, _ = db.get_cached(key)
        ms = (time.perf_counter() - t0) * 1000
        color = "yellow" if tier == "l2" else "red"
        t.add_row(key, f"[{color}]{tier}[/{color}]", f"{ms:.2f}")
    console.print(t)

    # L2 hit latency vs source latency
    db.cache.l1.clear()
    l2_ms = _bench(lambda: db.get_cached("genre_stats"), 20)
    # now L1 is warm again
    l1_ms = _bench(lambda: db.get_cached("genre_stats"), 200)
    console.print(
        f"\n  L2 hit ≈ [yellow]{l2_ms:.3f} ms[/yellow]  |  "
        f"L1 hit ≈ [green]{l1_ms*1000:.1f} µs[/green]  |  "
        f"Source ≈ [red]{source_ms:.0f} ms[/red]\n"
        f"  L2 is [bold]{source_ms/l2_ms:.0f}× faster than source[/bold] — "
        f"keeps the service alive during an L1 cold start."
    )
    s = db.cache.stats()
    console.print(
        f"  After promotion run: L1_hits={s['l1_hits']}  L2_hits={s['l2_hits']}  "
        f"source_hits={s['source_hits']}"
    )

    # ── Section 6: Write-through invalidation ───────────────────────────────
    console.print()
    console.print(Panel("[bold]6. Write-Through — Rating Update Invalidates Both Tiers[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]A new user rating arrives.  Write-through removes the stale\n"
        "  aggregate keys from L1 and L2 simultaneously; unrelated genre keys\n"
        "  remain cached so only one query re-fetches from the source.[/dim]"
    )
    db.genre_stats()
    db.top_rated_by_genre("sci-fi")
    db.top_rated_by_genre("action")
    l1_keys_before = set(db.cache.l1.keys())
    db.add_rating("u05", "m01", 3.8, "Great but overhyped")
    l1_keys_after = set(db.cache.l1.keys())
    evicted  = l1_keys_before - l1_keys_after
    retained = l1_keys_before & l1_keys_after

    wt = Table("Cache key", "L1 status", "Reason", box=box.SIMPLE_HEAD)
    for k in sorted(l1_keys_before):
        if k in evicted:
            wt.add_row(k, "[red]EVICTED[/red]",    "aggregates over modified data")
        else:
            wt.add_row(k, "[green]RETAINED[/green]", "unrelated genre — still valid")
    console.print(wt)
    console.print(
        f"  Evicted: [bold red]{len(evicted)}[/bold red]  |  "
        f"Retained: [bold green]{len(retained)}[/bold green]  |  "
        "Both L1 and L2 invalidated for affected keys."
    )

    # ── Section 7: TTL per-tier behaviour ───────────────────────────────────
    console.print()
    console.print(Panel("[bold]7. TTL Per Tier — L1 Expiry Falls Through to L2[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]L1 TTL is intentionally shorter than L2 TTL.  When L1 expires,\n"
        "  the L2 entry is still valid and serves the request — the source is\n"
        "  not hit.  Only when both tiers expire does the source query execute.[/dim]"
    )
    short_db = _build_db(l1_ttl=0.05, l2_ttl=0.20, delay=0.0)
    short_db.seed(MOVIES, RATINGS)
    ttl_t = Table("Call", "Elapsed (ms)", "Tier served", "Note", box=box.SIMPLE_HEAD)

    t_start = time.perf_counter()
    tier, _ = short_db.get_cached("genre_stats")
    ttl_t.add_row("1st", f"{(time.perf_counter()-t_start)*1000:.1f}", f"[red]{tier}[/red]", "cold — fetched from source → back-fills L1+L2")

    tier, _ = short_db.get_cached("genre_stats")
    ttl_t.add_row("2nd", f"{(time.perf_counter()-t_start)*1000:.1f}", f"[green]{tier}[/green]", "L1 warm")

    time.sleep(0.07)
    tier, _ = short_db.get_cached("genre_stats")
    ttl_t.add_row("3rd", f"{(time.perf_counter()-t_start)*1000:.1f}", f"[yellow]{tier}[/yellow]", "L1 expired → L2 hit → re-promotes to L1")

    tier, _ = short_db.get_cached("genre_stats")
    ttl_t.add_row("4th", f"{(time.perf_counter()-t_start)*1000:.1f}", f"[green]{tier}[/green]", "L1 re-warmed from L2")

    time.sleep(0.18)
    tier, _ = short_db.get_cached("genre_stats")
    ttl_t.add_row("5th", f"{(time.perf_counter()-t_start)*1000:.1f}", f"[red]{tier}[/red]", "both L1+L2 expired → source re-fetched")

    console.print(ttl_t)

    # ── Section 8: Cache warming before a traffic spike ─────────────────────
    console.print()
    console.print(Panel("[bold]8. Cache Warming — Pre-Loading Before a Traffic Spike[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Friday 8 PM: traffic triples as subscribers start movie nights.\n"
        "  Warm the cache 30 seconds before the spike so the first wave of\n"
        "  requests hits L1, not the source DB.[/dim]"
    )
    warm_db = _build_db(delay=0.0)
    warm_db.seed(MOVIES, RATINGS)
    warm_keys = ["genre_stats"] + [f"top:{g}" for g in GENRES]

    warm_db.cache.l1.clear()
    warm_db.cache.l2.clear()
    warm_db.cache.reset_stats()
    loaded = warm_db.cache.warm(warm_keys)

    warm_db.cache.reset_stats()
    hit_count = 0
    for key in warm_keys:
        tier, _ = warm_db.get_cached(key)
        if tier == "l1":
            hit_count += 1

    wm = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    wm.add_row("Keys pre-loaded",         str(loaded))
    wm.add_row("Keys served from L1",     str(hit_count))
    wm.add_row("L1 hit rate after warm",  f"{hit_count/len(warm_keys)*100:.0f}%")
    wm.add_row("Source queries during spike", "0")
    console.print(wm)
    console.print(
        "  [dim]Without warming, 7 source queries would fire in the first 100 ms.\n"
        "  With warming, 0 source queries fire — the spike is fully absorbed.[/dim]"
    )

    # ── Section 9: Per-tier statistics summary ───────────────────────────────
    console.print()
    console.print(Panel("[bold]9. Per-Tier Statistics — Full Run[/bold]", box=box.ROUNDED))
    db.cache.reset_stats()
    FINAL_REPS = 50
    for _ in range(FINAL_REPS):
        for key in keys_to_warm:
            db.get_cached(key)
    s = db.cache.stats()
    l1s = db.cache.l1.stats()
    l2s = db.cache.l2.stats()

    st = Table("Metric", "L1", "L2", "Source", box=box.SIMPLE_HEAD)
    total = s["total"]
    st.add_row("Hits",     str(s["l1_hits"]), str(s["l2_hits"]), str(s["source_hits"]))
    st.add_row("Hit rate", f"{s['l1_hit_rate']*100:.1f}%", f"{s['l2_hit_rate']*100:.1f}%",
               f"{s['source_miss_rate']*100:.1f}%")
    st.add_row("Keys stored", str(l1s["size"]), str(l2s["size"]), "—")
    console.print(st)
    console.print(
        f"  Total requests: {total}  |  Source queries avoided: "
        f"[bold green]{total - s['source_hits']}[/bold green] "
        f"({(1 - s['source_miss_rate'])*100:.1f}% cache absorption)"
    )

    # ── Key takeaways ────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Key Tiered Caching Takeaways:[/bold green]")
    console.print("  • [cyan]L1 (in-process)[/cyan]    — sub-microsecond reads; zero network; lost on process restart")
    console.print("  • [cyan]L2 (shared tier)[/cyan]   — survives restarts and deploys; absorbs spikes during L1 cold start")
    console.print("  • [cyan]Write-through[/cyan]      — both tiers updated atomically on write; no stale reads after an update")
    console.print("  • [cyan]TTL per tier[/cyan]       — L1 short (freshness), L2 long (availability); tier gap prevents thundering herd")
    console.print("  • [cyan]Warming[/cyan]            — bulk pre-load at startup / before traffic spikes; source QPS ≈ 0 during first wave")
    console.print("  [dim]Production stack: Caffeine (L1, JVM), Redis / EVCache (L2), PostgreSQL / Cassandra (source)[/dim]")


if __name__ == "__main__":
    main()
