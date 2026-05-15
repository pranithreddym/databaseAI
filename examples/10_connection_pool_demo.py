"""
Demo 10: Connection Pooling
============================
Simulates a connection pool with a fixed max size, FIFO wait queue, and
configurable timeout.  Compares per-request connection-creation overhead
against pool reuse under sequential and concurrent load, then demonstrates
what happens when the pool is fully exhausted.

Real-world parallel: PgBouncer running in transaction mode in front of
PostgreSQL — Netflix, Shopify, and GitHub all use connection poolers to
serve millions of requests through a bounded set of server connections,
preventing the "too many clients" error that would otherwise occur when
hundreds of Lambda functions or Gunicorn workers all try to connect at once.
"""

import sys
import os
import time
import threading
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.connection_pool import (
    ConnectionPool, PoolExhaustedError, PooledMovieDB, sqlite_factory
)

console = Console()

# Simulated per-connection overhead (TCP handshake + TLS + auth on a real DB).
# PgBouncer eliminates this by keeping server connections open permanently;
# the client only connects to the pooler (< 1 ms local socket), not Postgres.
CONN_OVERHEAD_S = 0.020   # 20 ms

# Simulated time each worker holds the connection (represents query + CPU time).
QUERY_HOLD_S = 0.012      # 12 ms


def _make_pool(db_path: str, max_size: int, min_size: int = 0,
               timeout: float = 5.0) -> ConnectionPool:
    return ConnectionPool(
        db_factory=sqlite_factory(db_path),
        max_size=max_size,
        min_size=min_size,
        timeout=timeout,
        connection_overhead=CONN_OVERHEAD_S,
    )


def _no_pool_query(db_path: str) -> float:
    """Open a brand-new connection, run a lightweight query, close it.
    Returns wall-clock time in ms.  Simulates CONN_OVERHEAD_S per call."""
    import sqlite3
    t0 = time.monotonic()
    time.sleep(CONN_OVERHEAD_S)          # TCP + auth latency
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("SELECT COUNT(*) FROM movies").fetchone()
    time.sleep(QUERY_HOLD_S)
    conn.close()
    return (time.monotonic() - t0) * 1000


def main():
    console.rule("[bold cyan]Connection Pooling Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: PgBouncer in transaction mode in front of PostgreSQL —\n"
        "  bounding open server connections while queuing excess callers fairly.[/dim]\n"
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    DB_PATH = tmp.name
    tmp.close()

    # ---------------------------------------------------------------
    # Section 1: Pool Setup & Data Load
    # ---------------------------------------------------------------
    console.print(Panel("[bold]1. Pool Setup & Data Load[/bold]", box=box.ROUNDED))
    pool = _make_pool(DB_PATH, max_size=3, min_size=3)
    db = PooledMovieDB(pool)
    db.seed(MOVIES, RATINGS)

    s = pool.stats()
    t = Table("Parameter", "Value", box=box.SIMPLE_HEAD)
    t.add_row("pool max_size",                   str(s["max_size"]))
    t.add_row("pool min_size  (pre-warmed)",      str(s["min_size"]))
    t.add_row("connections created at startup",   str(s["total_created"]))
    t.add_row("connections available now",        str(s["available"]))
    t.add_row("simulated connection overhead",    f"{CONN_OVERHEAD_S*1000:.0f} ms per open")
    t.add_row("simulated query hold time",        f"{QUERY_HOLD_S*1000:.0f} ms per request")
    t.add_row("movies loaded",                    str(db.movie_count()))
    t.add_row("ratings loaded",                   str(db.rating_count()))
    console.print(t)
    console.print(
        "  [dim]min_size=3 pays the connection cost at startup, not during peak traffic.\n"
        "  PgBouncer calls this 'server_idle_timeout' + 'min_pool_size'.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 2: Sequential Overhead — No-Pool vs Pool
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. Sequential Overhead — No-Pool vs Pool[/bold]", box=box.ROUNDED))
    console.print(
        f"  [dim]8 sequential queries. No-pool opens a new connection for each"
        f" ({CONN_OVERHEAD_S*1000:.0f} ms overhead/call).\n"
        f"  Pool reuses its 3 pre-warmed connections; acquire wait ≈ 0 ms.[/dim]"
    )

    N = 8

    no_pool_start = time.monotonic()
    for _ in range(N):
        _no_pool_query(DB_PATH)
    no_pool_ms = (time.monotonic() - no_pool_start) * 1000

    pool.reset_stats()
    pool_start = time.monotonic()
    for _ in range(N):
        with pool.connection() as conn:
            conn.execute("SELECT COUNT(*) FROM movies").fetchone()
            time.sleep(QUERY_HOLD_S)
    pool_ms = (time.monotonic() - pool_start) * 1000

    t = Table("Approach", "Connections Opened", "Conn Overhead Total",
              "Total Wall Time", box=box.SIMPLE_HEAD)
    t.add_row("No pool",           str(N),       f"{N * CONN_OVERHEAD_S * 1000:.0f} ms",
              f"{no_pool_ms:.0f} ms")
    t.add_row("Pool (pre-warmed)", "0 (reused)", "0 ms",
              f"{pool_ms:.0f} ms")
    console.print(t)
    speedup = no_pool_ms / pool_ms if pool_ms else float("inf")
    console.print(
        f"  Pool is [bold green]{speedup:.1f}×[/bold green] faster for {N} sequential queries "
        f"({no_pool_ms:.0f} ms → {pool_ms:.0f} ms)."
    )

    # ---------------------------------------------------------------
    # Section 3: Concurrent Load — 8 Workers, Pool of 3
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]3. Concurrent Load — 8 Workers, Pool Size 3[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]8 threads start simultaneously.  Only 3 connections exist in the pool.\n"
        "  Workers 4–8 enter the FIFO queue and wait for a slot to be released.[/dim]"
    )

    WORKERS = 8
    conc_pool = _make_pool(DB_PATH, max_size=3, min_size=3, timeout=10.0)

    results = []
    lock = threading.Lock()

    def worker(wid: int):
        t0 = time.monotonic()
        with conc_pool.connection() as conn:
            acquired_ms = (time.monotonic() - t0) * 1000
            conn.execute("SELECT COUNT(*) FROM movies").fetchone()
            time.sleep(QUERY_HOLD_S)
        done_ms = (time.monotonic() - t0) * 1000
        with lock:
            results.append({
                "worker": wid,
                "wait_ms": acquired_ms,
                "total_ms": done_ms,
                "conn_id": conn.conn_id,
            })

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    wall_t0 = time.monotonic()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall_ms = (time.monotonic() - wall_t0) * 1000

    results.sort(key=lambda r: r["wait_ms"])
    t = Table("Worker", "Conn ID", "Queue Wait (ms)", "Total Time (ms)", "Path",
              box=box.SIMPLE_HEAD)
    for r in results:
        queued = r["wait_ms"] > 5
        wait_s = f"[yellow]{r['wait_ms']:.1f}[/yellow]" if queued else f"[green]{r['wait_ms']:.1f}[/green]"
        path = "queued" if queued else "immediate"
        t.add_row(f"W{r['worker']}", str(r["conn_id"]), wait_s,
                  f"{r['total_ms']:.0f}", path)
    console.print(t)

    cs = conc_pool.stats()
    console.print(f"\n  All {WORKERS} workers succeeded. Wall-clock time: [bold]{wall_ms:.0f} ms[/bold]")
    console.print(f"  Connections ever opened: [bold]{cs['total_created']}[/bold] "
                  f"(pool bounded to {conc_pool.max_size}, not {WORKERS})")
    console.print(f"  Max queue wait: [yellow]{cs['max_wait_ms']:.0f} ms[/yellow]  |  "
                  f"Avg wait: {cs['avg_wait_ms']:.1f} ms")

    # ---------------------------------------------------------------
    # Section 4: Top-Rated Query Via the Pool
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]4. Top-Rated Movies Query Via Pool[/bold]", box=box.ROUNDED))
    top = db.top_rated(limit=5)
    t = Table("Rank", "Title", "Genre", "Avg Score", "Votes", box=box.SIMPLE_HEAD)
    for i, row in enumerate(top, 1):
        t.add_row(str(i), row["title"], row["genre"],
                  str(row["avg_score"]), str(row["votes"]))
    console.print(t)
    console.print(
        "  [dim]Each call acquires one connection from the pool, runs the GROUP BY + JOIN,\n"
        "  and immediately releases the connection — PgBouncer transaction mode.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 5: Pool Exhaustion — Timeout Demonstration
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]5. Pool Exhaustion — 100 ms Timeout[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Pool size = 1, timeout = 100 ms.  Three workers start simultaneously.\n"
        "  Worker 0 acquires immediately and holds the connection for 200 ms.\n"
        "  Workers 1 & 2 time out and receive PoolExhaustedError.[/dim]"
    )

    tiny_pool = ConnectionPool(
        db_factory=sqlite_factory(DB_PATH),
        max_size=1,
        min_size=1,
        timeout=0.10,
        connection_overhead=0.0,
    )
    barrier = threading.Barrier(3)
    exh_results = []

    def exhaustion_worker(wid: int):
        barrier.wait()
        t0 = time.monotonic()
        try:
            with tiny_pool.connection():
                time.sleep(0.20)          # hold for 200 ms; others timeout at 100 ms
            exh_results.append({"worker": wid, "status": "OK",
                                 "ms": (time.monotonic() - t0) * 1000})
        except PoolExhaustedError:
            exh_results.append({"worker": wid, "status": "TIMEOUT",
                                 "ms": (time.monotonic() - t0) * 1000})

    eth = [threading.Thread(target=exhaustion_worker, args=(i,)) for i in range(3)]
    for th in eth:
        th.start()
    for th in eth:
        th.join()

    exh_results.sort(key=lambda r: r["worker"])
    t = Table("Worker", "Result", "Time (ms)", "Reason", box=box.SIMPLE_HEAD)
    for r in exh_results:
        if r["status"] == "OK":
            t.add_row(f"W{r['worker']}", "[green]OK[/green]",
                      f"{r['ms']:.0f}", "acquired immediately")
        else:
            t.add_row(f"W{r['worker']}", "[red]TIMEOUT[/red]",
                      f"{r['ms']:.0f}", "pool exhausted — 100 ms deadline exceeded")
    console.print(t)

    ts = tiny_pool.stats()
    console.print(
        f"\n  Timeouts: [bold red]{ts['total_timeouts']}[/bold red] / 3 workers\n"
        "  [dim]PoolExhaustedError exposes overload immediately rather than silently\n"
        "  queueing forever — the caller can return HTTP 503 and shed load.[/dim]"
    )
    tiny_pool.close()

    # ---------------------------------------------------------------
    # Section 6: Pool Stats Comparison
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. Pool Stats Summary[/bold]", box=box.ROUNDED))
    seq_s  = pool.stats()
    conc_s = conc_pool.stats()
    t = Table("Metric", "Sequential Pool (§2)", "Concurrent Pool (§3)", box=box.SIMPLE_HEAD)
    t.add_row("max_size",                str(seq_s["max_size"]),       str(conc_s["max_size"]))
    t.add_row("total connections created",
              str(seq_s["total_created"]), str(conc_s["total_created"]))
    t.add_row("total requests served",   str(seq_s["total_served"]),   str(conc_s["total_served"]))
    t.add_row("total timeouts",          str(seq_s["total_timeouts"]), str(conc_s["total_timeouts"]))
    t.add_row("avg acquire wait (ms)",   str(seq_s["avg_wait_ms"]),    str(conc_s["avg_wait_ms"]))
    t.add_row("max acquire wait (ms)",   str(seq_s["max_wait_ms"]),    str(conc_s["max_wait_ms"]))
    console.print(t)
    console.print(
        f"  Both pools opened exactly [bold]{seq_s['total_created']}[/bold] connections regardless of "
        f"how many requests were served — that is the core guarantee of pooling."
    )

    # ---------------------------------------------------------------
    # Section 7: Key Takeaways
    # ---------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Connection Pooling Takeaways:[/bold green]")
    console.print("  • [cyan]Bounded connections[/cyan]    — max_size caps open connections; prevents DB max_connections overflow")
    console.print("  • [cyan]Amortised overhead[/cyan]     — TCP + TLS + auth paid once at startup, not on every request")
    console.print("  • [cyan]FIFO queue[/cyan]             — excess callers wait fairly; bounded timeout prevents silent runaway queuing")
    console.print("  • [cyan]Transaction mode[/cyan]       — release immediately after the query (not at session end) to maximise reuse")
    console.print("  • [cyan]Pre-warming (min_size)[/cyan] — eliminate cold-start latency spikes on first burst of traffic")
    console.print("  • [cyan]PoolExhaustedError[/cyan]     — surfaces overload so callers can return 503 rather than pile up silently")
    console.print("  [dim]Production: PgBouncer (PostgreSQL), HikariCP (Java/Spring Boot), "
                  "SQLAlchemy QueuePool (Python), AWS RDS Proxy[/dim]")

    pool.close()
    conc_pool.close()
    try:
        os.unlink(DB_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    main()
