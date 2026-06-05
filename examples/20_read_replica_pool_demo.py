"""
Demo 20: Read-Replica Connection Pool — Primary / Replica Routing
=================================================================
Demonstrates how a streaming platform routes database traffic across one
write primary and three read replicas, using weighted round-robin, a
simulated replication write log, automatic failover, and a fallback to
primary when all replicas are down.

Sections:
  1. Architecture overview — pool topology and routing rules.
  2. Read/Write split — writes go to primary, reads fan out to replicas.
  3. Replication lag — writes are not immediately visible on replicas;
     stale reads are quantified before and after propagation.
  4. Weighted routing — replica-2 gets double weight, attracts 50 % of reads.
  5. Replica failover — take replica-1 offline; traffic redistributes.
  6. Total primary fallback — all replicas down; reads go to primary.
  7. Throughput comparison — sequential reads on primary vs. three replicas.
  8. Routing statistics summary.

Real-world parallel: Netflix EVCache / recommendation DB architecture.
Every "Top Picks" carousel read is served from a regional read replica
(often Cassandra or a MySQL standby) while write-path events (play start,
rating submitted) hit the primary cluster.  ProxySQL / RDS Proxy / Vitess
handle the routing transparently, applying weights per replica based on
observed latency and replication lag.
"""

import sys
import os
import random
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.read_replica_pool import Replica, PrimaryReplicaRouter
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()

SYNTHETIC_RATINGS = 200      # extra ratings injected after initial seed
CONCURRENT_READERS = 8       # threads used in throughput comparison
READS_PER_THREAD = 20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str, subtitle: str = ""):
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def _ok(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def _warn(msg: str):
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def _info(msg: str):
    console.print(f"  [dim]→[/dim] {msg}")


def _make_router(weights=(1.0, 1.0, 1.0), lags_ms=(0.0, 0.0, 0.0)) -> PrimaryReplicaRouter:
    replicas = [
        Replica(replica_id=i + 1, weight=weights[i], lag_ms=lags_ms[i])
        for i in range(3)
    ]
    return PrimaryReplicaRouter(replicas=replicas, primary_pool_size=4)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Architecture overview
# ─────────────────────────────────────────────────────────────────────────────

def section_architecture():
    _section(
        "1 · Architecture Overview",
        "Pool topology and routing rules",
    )

    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True)
    tbl.add_column("Node",       style="bold")
    tbl.add_column("Role",       style="cyan")
    tbl.add_column("Pool size")
    tbl.add_column("Handles",    style="italic")
    tbl.add_column("Real-world analogy")

    tbl.add_row("Primary",    "Write primary",     "4",  "INSERT / UPDATE / DELETE",  "RDS primary / Postgres leader")
    tbl.add_row("Replica-1",  "Read replica",       "3",  "SELECT (weight 1×)",        "Aurora reader / Postgres standby")
    tbl.add_row("Replica-2",  "Read replica",       "3",  "SELECT (weight 1×)",        "ProxySQL backend #2")
    tbl.add_row("Replica-3",  "Read replica",       "3",  "SELECT (weight 1×)",        "Cross-AZ read replica")
    console.print(tbl)

    console.print(
        Panel(
            "[bold]Routing rule[/bold]\n"
            "  • Writes  →  Primary pool (guaranteed up-to-date)\n"
            "  • Reads   →  Weighted round-robin over healthy replicas\n"
            "  • Failover →  Unhealthy replica weight excluded; if all down → primary\n\n"
            "[bold]Replication model[/bold]\n"
            "  Each write is logged with a [italic]replicate_after[/italic] timestamp.\n"
            "  propagate() drains the log and applies due entries to every\n"
            "  healthy replica — simulating WAL shipping / binlog replication.",
            title="Routing Policy",
            border_style="dim",
            padding=(0, 2),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Basic read/write split
# ─────────────────────────────────────────────────────────────────────────────

def section_rw_split(router: PrimaryReplicaRouter):
    _section(
        "2 · Read / Write Splitting",
        "Writes land on primary; reads fan out to three replicas",
    )

    # Seed all nodes with identical snapshot
    router.seed(MOVIES, RATINGS)
    _ok(f"Seeded primary + 3 replicas with {len(MOVIES)} movies, {len(RATINGS)} ratings")

    # Issue writes via primary
    inserts = [
        ("INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
         ("s01", "Dune", "sci-fi", 2021, "Denis Villeneuve")),
        ("INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
         ("s02", "Tenet", "sci-fi", 2020, "Christopher Nolan")),
    ]
    for sql, params in inserts:
        router.execute_write(sql, params, replica_lag_ms=0)

    router.propagate()
    _ok("Wrote 2 new movies to primary and propagated to replicas immediately")

    # Read from replicas
    rows = router.execute_read("SELECT title, year FROM movies ORDER BY year DESC LIMIT 5")
    tbl = Table("Title", "Year", box=box.SIMPLE)
    for r in rows:
        tbl.add_row(r["title"], str(r["year"]))
    console.print(tbl)

    stats = router.routing_stats
    _info(f"After seeding: {stats['total_writes']} writes to primary, "
          f"{stats['reads_to_replicas']} reads served by replicas")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Replication lag and stale reads
# ─────────────────────────────────────────────────────────────────────────────

def section_replication_lag():
    _section(
        "3 · Replication Lag & Stale Reads",
        "Writes on primary are invisible on replicas until the lag window expires",
    )

    router = _make_router(lags_ms=(200.0, 200.0, 200.0))  # 200 ms lag each
    router.seed(MOVIES, RATINGS)

    rng = random.Random(42)
    NEW_RATING_SQL = (
        "INSERT OR REPLACE INTO ratings (user_id, movie_id, score, review) "
        "VALUES (?, ?, ?, ?)"
    )
    # 10 brand-new ratings written to primary only
    new_ratings = [
        ("u99", f"m{rng.randint(1, 15):02d}", round(rng.uniform(3.5, 5.0), 1), "new review")
        for _ in range(10)
    ]
    for r in new_ratings:
        router.execute_write(NEW_RATING_SQL, r)  # uses replica's own lag_ms (200 ms)

    pending = router.pending_replication_count()
    _warn(f"{pending} write(s) pending replication — replicas still show old data")

    # Read before propagation
    before_rows = router.execute_read(
        "SELECT COUNT(*) AS cnt FROM ratings WHERE user_id = 'u99'"
    )
    before_count = before_rows[0]["cnt"]
    _info(f"Replica read BEFORE propagate():  u99 rating count = {before_count}  ← stale")

    # Advance time past all lag windows and propagate
    future_ts = time.perf_counter() + 1.0   # 1 second past now covers 200 ms lag
    applied = router.propagate(until_ts=future_ts)
    _ok(f"propagate() applied {applied} write entries to replicas")

    after_rows = router.execute_read(
        "SELECT COUNT(*) AS cnt FROM ratings WHERE user_id = 'u99'"
    )
    after_count = after_rows[0]["cnt"]
    _info(f"Replica read AFTER  propagate():  u99 rating count = {after_count}  ← fresh")

    tbl = Table("Metric", "Value", box=box.SIMPLE)
    tbl.add_row("New writes sent to primary",  str(len(new_ratings)))
    tbl.add_row("Simulated replica lag",        "200 ms per write")
    tbl.add_row("Replica count before propagate", str(before_count))
    tbl.add_row("Replica count after propagate",  str(after_count))
    console.print(tbl)

    router.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Weighted routing
# ─────────────────────────────────────────────────────────────────────────────

def section_weighted_routing():
    _section(
        "4 · Weighted Routing",
        "Replica-2 gets 2× weight → attracts ~50 % of reads",
    )

    router = _make_router(weights=(1.0, 2.0, 1.0))
    router.seed(MOVIES, RATINGS)

    QUERY = "SELECT title FROM movies ORDER BY year DESC LIMIT 3"
    for _ in range(80):
        router.execute_read(QUERY)

    stats = router.replica_stats
    total_reads = sum(r["queries_served"] for r in stats)

    tbl = Table("Replica", "Weight", "Queries Served", "Share (%)", box=box.SIMPLE_HEAVY)
    for r in stats:
        share = 100 * r["queries_served"] / total_reads if total_reads else 0
        bar = "█" * round(share / 5)
        colour = "green" if r["id"] == 2 else "white"
        tbl.add_row(
            f"[{colour}]replica-{r['id']}[/{colour}]",
            f"[{colour}]{r['weight']:.1f}[/{colour}]",
            str(r["queries_served"]),
            f"{share:.0f} %  {bar}",
        )
    console.print(tbl)
    _info("Replica-2 (weight=2.0) receives roughly twice the traffic of replica-1 and replica-3")
    router.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Replica failover
# ─────────────────────────────────────────────────────────────────────────────

def section_failover():
    _section(
        "5 · Replica Failover",
        "Marking a replica unhealthy redistributes its share to the remaining replicas",
    )

    router = _make_router()
    router.seed(MOVIES, RATINGS)

    QUERY = "SELECT AVG(score) AS avg_score FROM ratings"

    # Phase A: all replicas healthy
    for _ in range(30):
        router.execute_read(QUERY)

    _ok("Phase A — 3 healthy replicas, 30 reads issued")
    tbl_a = Table("Replica", "Status", "Queries", box=box.SIMPLE)
    for r in router.replica_stats:
        tbl_a.add_row(f"replica-{r['id']}", "[green]healthy[/green]", str(r["queries_served"]))
    console.print(tbl_a)

    # Take replica-1 offline
    router.mark_unhealthy(1)
    _warn("replica-1 marked UNHEALTHY (network timeout / replication stopped)")

    # Phase B: 2 healthy replicas
    for _ in range(30):
        router.execute_read(QUERY)

    _ok("Phase B — 2 healthy replicas, 30 more reads issued")
    tbl_b = Table("Replica", "Status", "Queries (total)", box=box.SIMPLE)
    for r in router.replica_stats:
        status = "[red]UNHEALTHY[/red]" if not r["healthy"] else "[green]healthy[/green]"
        tbl_b.add_row(f"replica-{r['id']}", status, str(r["queries_served"]))
    console.print(tbl_b)

    rs = router.routing_stats
    _info(f"Total reads served by replicas: {rs['reads_to_replicas']}  "
          f"primary fallback: {rs['reads_to_primary_fallback']}")
    router.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Full primary fallback
# ─────────────────────────────────────────────────────────────────────────────

def section_primary_fallback():
    _section(
        "6 · Full Primary Fallback",
        "When all replicas are unhealthy, reads fall back to primary",
    )

    router = _make_router()
    router.seed(MOVIES, RATINGS)

    for rid in (1, 2, 3):
        router.mark_unhealthy(rid)
    _warn("All 3 replicas marked UNHEALTHY")

    QUERY = "SELECT title FROM movies ORDER BY year LIMIT 5"
    for _ in range(10):
        router.execute_read(QUERY)

    rs = router.routing_stats
    _info(f"Reads to replicas: {rs['reads_to_replicas']}  "
          f"reads to primary (fallback): {rs['reads_to_primary_fallback']}")
    _ok("All 10 reads routed to primary — service degraded but uninterrupted")

    # Restore replicas
    for rid in (1, 2, 3):
        router.mark_healthy(rid)
    _ok("Replicas restored — routing returns to normal")
    router.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Throughput comparison: primary-only vs 3 replicas
# ─────────────────────────────────────────────────────────────────────────────

def section_throughput():
    _section(
        "7 · Throughput Comparison",
        f"{CONCURRENT_READERS} threads × {READS_PER_THREAD} reads — primary-only vs 3 replicas",
    )

    QUERY = "SELECT title, AVG(score) AS avg FROM movies JOIN ratings ON movies.id=ratings.movie_id GROUP BY movies.id ORDER BY avg DESC LIMIT 5"

    def _run(router: PrimaryReplicaRouter, use_replicas: bool) -> float:
        results = []
        barrier = threading.Barrier(CONCURRENT_READERS)

        def worker():
            barrier.wait()
            t0 = time.perf_counter()
            for _ in range(READS_PER_THREAD):
                if use_replicas:
                    router.execute_read(QUERY)
                else:
                    with router._primary.connection() as conn:
                        conn.execute(QUERY).fetchall()
            results.append(time.perf_counter() - t0)

        threads = [threading.Thread(target=worker) for _ in range(CONCURRENT_READERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return max(results)   # wall-clock time = slowest thread

    # Primary-only run
    router_primary = _make_router()
    router_primary.seed(MOVIES, RATINGS)
    t_primary = _run(router_primary, use_replicas=False)
    router_primary.close()

    # Replica run
    router_replica = _make_router()
    router_replica.seed(MOVIES, RATINGS)
    t_replica = _run(router_replica, use_replicas=True)
    router_replica.close()

    total_reads = CONCURRENT_READERS * READS_PER_THREAD
    tbl = Table("Scenario", "Wall time (s)", "Throughput (reads/s)", box=box.SIMPLE_HEAVY)
    tbl.add_row(
        "Primary-only", f"{t_primary:.3f}", f"{total_reads / t_primary:,.0f}"
    )
    tbl.add_row(
        "[bold green]3 Replicas (round-robin)[/bold green]",
        f"[bold green]{t_replica:.3f}[/bold green]",
        f"[bold green]{total_reads / t_replica:,.0f}[/bold green]",
    )
    console.print(tbl)
    _info(
        f"With 3 replicas the {CONCURRENT_READERS} concurrent readers contend "
        f"for 9 connections instead of 4 → less queuing, higher throughput"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Final routing statistics
# ─────────────────────────────────────────────────────────────────────────────

def section_final_stats():
    _section(
        "8 · Routing Statistics Summary",
        "Aggregate counters across this demo session",
    )

    router = _make_router()
    router.seed(MOVIES, RATINGS)

    # Simulate a typical workload: 5 writes, 50 reads
    for i in range(5):
        router.execute_write(
            "INSERT OR REPLACE INTO ratings (user_id,movie_id,score,review) VALUES (?,?,?,?)",
            (f"u0{i+1}", "m01", round(3.0 + i * 0.4, 1), "demo rating"),
            replica_lag_ms=0,
        )
    router.propagate()
    for _ in range(50):
        router.execute_read("SELECT title, year FROM movies LIMIT 10")

    rs = router.routing_stats
    replica_detail = router.replica_stats

    summary_tbl = Table("Metric", "Value", box=box.SIMPLE)
    summary_tbl.add_row("Total writes (primary)", str(rs["total_writes"]))
    summary_tbl.add_row("Total reads (replicas)", str(rs["reads_to_replicas"]))
    summary_tbl.add_row("Primary fallback reads", str(rs["reads_to_primary_fallback"]))
    summary_tbl.add_row("Pending replication",    str(rs["pending_replication"]))
    console.print(summary_tbl)

    detail_tbl = Table("Replica", "Healthy", "Queries", "Avg latency (µs)", box=box.SIMPLE_HEAVY)
    for r in replica_detail:
        status = "[green]✓[/green]" if r["healthy"] else "[red]✗[/red]"
        detail_tbl.add_row(
            f"replica-{r['id']}",
            status,
            str(r["queries_served"]),
            f"{r['avg_latency_us']:.1f}",
        )
    console.print(detail_tbl)
    router.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print(
        Panel.fit(
            "[bold white]Demo 20 — Read-Replica Connection Pool[/bold white]\n"
            "[dim]Primary / Replica Routing · Replication Lag · Weighted Round-Robin · Failover[/dim]",
            border_style="cyan",
        )
    )

    section_architecture()

    router = _make_router()
    section_rw_split(router)
    router.close()

    section_replication_lag()
    section_weighted_routing()
    section_failover()
    section_primary_fallback()
    section_throughput()
    section_final_stats()

    console.print()
    console.print(
        Panel(
            "[bold]Key takeaways[/bold]\n\n"
            "  1. [cyan]Read/Write split[/cyan] lets you scale read throughput independently "
            "of the primary.\n"
            "  2. [cyan]Replication lag[/cyan] means replicas can return stale data; "
            "design your reads around acceptable staleness.\n"
            "  3. [cyan]Weighted round-robin[/cyan] directs traffic proportionally — "
            "useful when replicas have different hardware specs.\n"
            "  4. [cyan]Failover[/cyan] removes unhealthy nodes transparently; "
            "primary fallback preserves availability at the cost of replica isolation.\n"
            "  5. [cyan]Connection pools[/cyan] cap the number of open sockets per node, "
            "preventing the 'too many clients' error under bursty load.",
            title="Summary",
            border_style="green",
            padding=(0, 2),
        )
    )


if __name__ == "__main__":
    main()
