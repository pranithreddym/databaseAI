"""
Demo 26: Adaptive Connection Pool — Dynamic Resizing with Health Checks
========================================================================
Demos 10 and 20 showed fixed-size pools and read-replica routing.  This demo
tackles the sizing problem itself: a fixed pool forces a painful trade-off
between wasting server RAM during quiet hours (oversized) and hitting
PoolExhaustedError during traffic spikes (undersized).

An adaptive pool resolves this by observing utilisation and resizing between
min_size and max_size on the fly:

  Scale-up:   When utilisation exceeds a threshold, the pool batch-creates
              new connections (grow_step) up to max_size.
  Scale-down: When utilisation drops below a lower threshold for a cooldown
              window, idle connections are closed down to min_size.
  Health:     Validate-on-borrow catches stale connections; max-age eviction
              prevents slow memory leaks from long-lived sessions.

Real-world parallel: HikariCP in a Spring Boot microservice — minimumIdle
grows to maximumPoolSize under load, then shrinks back during off-peak.
Aurora Serverless v2 applies the same principle at the database layer,
auto-scaling ACUs between a configured min and max based on CPU and
connection demand, billed per second.
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.adaptive_pool import AdaptivePool, PoolExhaustedError, AdaptiveMovieDB

console = Console()

CONN_OVERHEAD_S = 0.015
QUERY_HOLD_S = 0.010


def _make_pool(
    min_size: int = 2,
    max_size: int = 8,
    cooldown_s: float = 0.1,
    max_conn_age_s: float = 300.0,
    timeout: float = 5.0,
) -> AdaptivePool:
    return AdaptivePool(
        db_factory=AdaptivePool.__module__ and __import__(
            "databaseai.adaptive_pool.adaptive", fromlist=["sqlite_factory"]
        ).sqlite_factory(":memory:"),
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        scale_up_threshold=0.75,
        scale_down_threshold=0.25,
        grow_step=2,
        cooldown_s=cooldown_s,
        max_conn_age_s=max_conn_age_s,
        connection_overhead=CONN_OVERHEAD_S,
    )


def _simple_pool(min_size=2, max_size=8, cooldown_s=0.1, max_conn_age_s=300.0):
    from databaseai.adaptive_pool.adaptive import sqlite_factory
    return AdaptivePool(
        db_factory=sqlite_factory(":memory:"),
        min_size=min_size,
        max_size=max_size,
        timeout=5.0,
        scale_up_threshold=0.75,
        scale_down_threshold=0.25,
        grow_step=2,
        cooldown_s=cooldown_s,
        max_conn_age_s=max_conn_age_s,
        connection_overhead=CONN_OVERHEAD_S,
    )


def main() -> None:
    console.rule("[bold cyan]Database Demo 26 — Adaptive Connection Pool[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: HikariCP dynamically sizes between minimumIdle and\n"
        "  maximumPoolSize based on demand. Aurora Serverless v2 applies the same\n"
        "  principle at the database layer, auto-scaling ACUs per second.[/dim]\n"
    )

    # ── Section 1: Pool Configuration ────────────────────────────────────────
    console.print(Panel("[bold]1. Adaptive Pool Configuration[/bold]", box=box.ROUNDED))

    pool = _simple_pool(min_size=2, max_size=8, cooldown_s=0.05)
    db = AdaptiveMovieDB(pool)
    db.seed(MOVIES, RATINGS)

    s = pool.stats()
    t = Table("Parameter", "Value", box=box.SIMPLE_HEAD)
    t.add_row("min_size (floor)", str(pool.min_size))
    t.add_row("max_size (ceiling)", str(pool.max_size))
    t.add_row("current capacity", str(pool.capacity))
    t.add_row("scale_up_threshold", "75% utilisation")
    t.add_row("scale_down_threshold", "25% utilisation")
    t.add_row("grow_step", "2 connections per scale-up")
    t.add_row("cooldown", "50 ms (shortened for demo)")
    t.add_row("connection overhead (simulated)", f"{CONN_OVERHEAD_S*1000:.0f} ms")
    t.add_row("movies loaded", str(db.movie_count()))
    t.add_row("ratings loaded", str(db.rating_count()))
    console.print(t)
    console.print(
        "  [dim]The pool starts at min_size=2 and will grow towards max_size=8\n"
        "  as utilisation climbs past 75%.[/dim]"
    )

    # ── Section 2: Scale-Up Under Burst Load ─────────────────────────────
    console.print()
    console.print(Panel("[bold]2. Scale-Up Under Burst Load — 6 Concurrent Workers[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]6 threads start simultaneously against a pool that begins at capacity=2.\n"
        "  The pool detects utilisation > 75% and batch-creates new connections\n"
        "  (grow_step=2) until all workers are served.[/dim]"
    )

    pool.reset_stats()
    capacity_before = pool.capacity

    results = []
    lock = threading.Lock()

    def burst_worker(wid: int):
        t0 = time.monotonic()
        try:
            with pool.connection() as conn:
                acquired_ms = (time.monotonic() - t0) * 1000
                conn.execute("SELECT 1")
                time.sleep(QUERY_HOLD_S)
            with lock:
                results.append({
                    "worker": wid,
                    "wait_ms": acquired_ms,
                    "status": "OK",
                })
        except PoolExhaustedError:
            with lock:
                results.append({
                    "worker": wid,
                    "wait_ms": (time.monotonic() - t0) * 1000,
                    "status": "TIMEOUT",
                })

    threads = [threading.Thread(target=burst_worker, args=(i,)) for i in range(6)]
    wall_t0 = time.monotonic()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall_ms = (time.monotonic() - wall_t0) * 1000

    capacity_after = pool.capacity

    results.sort(key=lambda r: r["wait_ms"])
    t = Table("Worker", "Status", "Acquire Wait (ms)", box=box.SIMPLE_HEAD)
    for r in results:
        colour = "green" if r["status"] == "OK" else "red"
        t.add_row(f"W{r['worker']}", f"[{colour}]{r['status']}[/{colour}]",
                  f"{r['wait_ms']:.1f}")
    console.print(t)

    s = pool.stats()
    console.print(
        f"\n  Capacity: [yellow]{capacity_before}[/yellow] → [green]{capacity_after}[/green]  "
        f"(scale-ups triggered: {s['scale_ups']})"
    )
    console.print(
        f"  Connections created: [bold]{s['total_connections_created']}[/bold]  |  "
        f"Wall time: {wall_ms:.0f} ms"
    )
    console.print(
        "  [dim]The pool grew automatically — no manual resize or restart needed.\n"
        "  HikariCP logs 'Added connection' at INFO level during scale-up so\n"
        "  operators can correlate pool growth with traffic spikes.[/dim]"
    )

    # ── Section 3: Scale-Down During Idle Period ─────────────────────────
    console.print()
    console.print(Panel("[bold]3. Scale-Down During Idle Period[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]After the burst, all connections are returned. Utilisation drops\n"
        "  below 25%. After the cooldown window, the pool closes idle connections\n"
        "  back towards min_size, freeing server-side resources.[/dim]"
    )

    capacity_pre_idle = pool.capacity
    time.sleep(0.1)

    removed = pool.force_scale_down(count=4)
    while pool.capacity > pool.min_size and pool.idle > 0:
        pool.force_scale_down(1)

    capacity_post_idle = pool.capacity

    console.print(
        f"  Capacity before idle: [yellow]{capacity_pre_idle}[/yellow]\n"
        f"  Capacity after  idle: [green]{capacity_post_idle}[/green]\n"
        f"  Connections closed:   {capacity_pre_idle - capacity_post_idle}"
    )
    console.print(
        "  [dim]HikariCP's idleTimeout (default 10 min) serves the same purpose —\n"
        "  connections that sit idle beyond the timeout are retired, reducing\n"
        "  server-side memory to match actual demand.[/dim]"
    )

    # ── Section 4: Validate-on-Borrow ──────────────────────────────────────
    console.print()
    console.print(Panel("[bold]4. Validate-on-Borrow — Catching Stale Connections[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Before handing a connection to a caller, the pool runs SELECT 1.\n"
        "  If the probe fails (firewall drop, server restart), the connection\n"
        "  is discarded and a fresh one is created — transparent to the caller.[/dim]"
    )

    fresh_pool = _simple_pool(min_size=2, max_size=4, cooldown_s=0.05)
    fresh_db = AdaptiveMovieDB(fresh_pool)
    fresh_db.seed(MOVIES, RATINGS)

    with fresh_pool.connection() as conn:
        conn.execute("SELECT 1")
    console.print("  [green]✓[/green] Healthy connection validated and returned successfully")

    with fresh_pool.connection() as conn:
        result = conn.validate()
    console.print(f"  [green]✓[/green] validate() probe returned: {result}")

    s = fresh_pool.stats()
    console.print(
        f"  Health check failures: {s['health_failures']}  |  "
        f"Connections served: {s['total_served']}"
    )
    console.print(
        "  [dim]Django 4.1+ added CONN_HEALTH_CHECKS = True, which runs the same\n"
        "  validate-on-borrow probe. Without it, a stale connection surfaces as\n"
        "  a cryptic 'server closed the connection unexpectedly' on the first\n"
        "  query after a firewall idle timeout.[/dim]"
    )
    fresh_pool.close()

    # ── Section 5: Max-Age Eviction ────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]5. Max-Age Eviction — Connection Lifetime Limits[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Connections older than max_conn_age_s are closed on return rather\n"
        "  than recycled. This bounds exposure to server-side state drift (e.g.\n"
        "  PostgreSQL parameter changes, timezone reloads) and limits damage from\n"
        "  slow memory leaks in the DB driver.[/dim]"
    )

    short_life_pool = _simple_pool(
        min_size=1, max_size=4, cooldown_s=0.05, max_conn_age_s=0.05
    )
    short_db = AdaptiveMovieDB(short_life_pool)
    short_db.seed(MOVIES, RATINGS)

    with short_life_pool.connection() as conn:
        conn_id_first = conn.conn_id

    time.sleep(0.1)

    with short_life_pool.connection() as conn:
        conn_id_second = conn.conn_id
    evicted = conn_id_first != conn_id_second

    s = short_life_pool.stats()
    console.print(
        f"  First connection ID:  {conn_id_first}\n"
        f"  Second connection ID: {conn_id_second}  "
        f"({'[green]new — old was evicted[/green]' if evicted else '[yellow]reused[/yellow]'})\n"
        f"  Age evictions: {s['age_evictions']}"
    )
    console.print(
        "  [dim]HikariCP's maxLifetime (default 30 min) retires connections before\n"
        "  they hit the server-side wait_timeout, preventing the 'connection has\n"
        "  been closed' surprise. Best practice: set maxLifetime a few seconds\n"
        "  shorter than the server's wait_timeout.[/dim]"
    )
    short_life_pool.close()

    # ── Section 6: Concurrent Throughput — Fixed vs Adaptive ───────────────
    console.print()
    console.print(Panel("[bold]6. Throughput — Fixed Pool (2) vs Adaptive Pool (2→8)[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]8 workers, 5 queries each. A fixed pool of 2 forces heavy queuing.\n"
        "  The adaptive pool starts at 2 but scales to 8, reducing contention.[/dim]"
    )

    WORKERS = 8
    QUERIES_PER_WORKER = 5

    from databaseai.adaptive_pool.adaptive import sqlite_factory

    fixed_pool = AdaptivePool(
        db_factory=sqlite_factory(":memory:"),
        min_size=2, max_size=2,
        timeout=10.0,
        scale_up_threshold=1.0,
        scale_down_threshold=0.0,
        cooldown_s=999,
        connection_overhead=CONN_OVERHEAD_S,
    )
    fixed_db = AdaptiveMovieDB(fixed_pool)
    fixed_db.seed(MOVIES, RATINGS)

    adaptive_pool = _simple_pool(min_size=2, max_size=8, cooldown_s=0.02)
    adaptive_db = AdaptiveMovieDB(adaptive_pool)
    adaptive_db.seed(MOVIES, RATINGS)

    def run_throughput(p: AdaptivePool) -> float:
        errors = []

        def w():
            try:
                for _ in range(QUERIES_PER_WORKER):
                    with p.connection() as conn:
                        conn.execute("SELECT 1")
                        time.sleep(QUERY_HOLD_S)
            except Exception as e:
                errors.append(e)

        ths = [threading.Thread(target=w) for _ in range(WORKERS)]
        t0 = time.monotonic()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return (time.monotonic() - t0) * 1000

    fixed_ms = run_throughput(fixed_pool)
    adaptive_ms = run_throughput(adaptive_pool)

    fs = fixed_pool.stats()
    as_ = adaptive_pool.stats()

    t = Table("Metric", "Fixed (size=2)", "Adaptive (2→8)", box=box.SIMPLE_HEAD)
    t.add_row("Wall time", f"{fixed_ms:.0f} ms", f"{adaptive_ms:.0f} ms")
    t.add_row("Capacity (final)", str(fs["capacity"]), str(as_["capacity"]))
    t.add_row("Connections created", str(fs["total_connections_created"]),
              str(as_["total_connections_created"]))
    t.add_row("Avg wait (ms)", f"{fs['avg_wait_ms']:.1f}", f"{as_['avg_wait_ms']:.1f}")
    t.add_row("Max wait (ms)", f"{fs['max_wait_ms']:.1f}", f"{as_['max_wait_ms']:.1f}")
    console.print(t)

    if adaptive_ms > 0:
        speedup = fixed_ms / adaptive_ms
        colour = "green" if speedup >= 1.2 else "yellow"
        console.print(
            f"  Adaptive pool throughput: [{colour}]{speedup:.1f}× faster[/{colour}]"
        )

    console.print(
        "  [dim]The adaptive pool created more connections but finished faster because\n"
        "  workers spent less time queuing. The trade-off: more server-side RAM\n"
        "  during spikes, but only while load demands it.[/dim]"
    )

    fixed_pool.close()
    adaptive_pool.close()

    # ── Section 7: Top-Rated Query Via Adaptive Pool ─────────────────────
    console.print()
    console.print(Panel("[bold]7. Top-Rated Movies — Query Through the Adaptive Pool[/bold]", box=box.ROUNDED))

    demo_pool = _simple_pool(min_size=2, max_size=6, cooldown_s=0.05)
    demo_db = AdaptiveMovieDB(demo_pool)
    demo_db.seed(MOVIES, RATINGS)

    top = demo_db.top_rated(limit=8)
    t = Table("Rank", "Title", "Genre", "Avg Score", "Votes", box=box.SIMPLE_HEAD)
    for i, row in enumerate(top, 1):
        t.add_row(str(i), row["title"], row["genre"],
                  str(row["avg_score"]), str(row["votes"]))
    console.print(t)
    console.print(
        "  [dim]Each query borrows one connection, runs the GROUP BY + JOIN, and\n"
        "  returns it — same transaction-mode pattern as PgBouncer, but the\n"
        "  pool behind it resizes automatically.[/dim]"
    )

    demo_pool.close()

    # ── Section 8: Stats Summary ─────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]8. Adaptive Pool Stats Summary[/bold]", box=box.ROUNDED))

    s = pool.stats()
    t = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t.add_row("Total connections created", str(s["total_connections_created"]))
    t.add_row("Total requests served", str(s["total_served"]))
    t.add_row("Scale-ups triggered", str(s["scale_ups"]))
    t.add_row("Scale-downs triggered", str(s["scale_downs"]))
    t.add_row("Health check failures", str(s["health_failures"]))
    t.add_row("Age evictions", str(s["age_evictions"]))
    t.add_row("Timeouts", str(s["total_timeouts"]))
    t.add_row("Avg utilisation", f"{s['avg_utilisation']:.2%}")
    console.print(t)

    pool.close()

    # ── Architecture Notes ─────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold]Production Architecture Notes[/bold]\n\n"
            "[dim]Fixed pool sizing leaves performance on the table in two directions:\n"
            "  too small → PoolExhaustedError and 503s during spikes\n"
            "  too large → wasted server RAM and connection-slot starvation for\n"
            "              other services sharing the same PostgreSQL instance\n\n"
            "Adaptive pooling resolves the tension:\n"
            "  HikariCP:      minimumIdle grows to maximumPoolSize on demand;\n"
            "                 idleTimeout (10 min) and maxLifetime (30 min) retire\n"
            "                 unused and aged-out connections automatically.\n"
            "  Aurora v2:     ACUs scale between configured min/max; billed per\n"
            "                 second — the database equivalent of container autoscaling.\n"
            "  Pgpool-II:     dynamically forks/kills worker processes to match load.\n"
            "  Django 4.1+:   CONN_HEALTH_CHECKS validates on borrow; CONN_MAX_AGE\n"
            "                 evicts stale connections — together they implement\n"
            "                 the same health policies shown in this demo.\n\n"
            "The key insight: pool sizing is not a deployment-time constant — it is a\n"
            "runtime variable that should track demand with appropriate damping\n"
            "(cooldown) to avoid thrashing.[/dim]",
            box=box.ROUNDED,
        )
    )

    console.print("\n[bold green]Demo complete.[/bold green]\n")


if __name__ == "__main__":
    main()
