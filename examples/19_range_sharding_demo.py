"""
Demo 19: Range-Based Sharding — Partition Pruning vs Scatter-Gather
====================================================================
Partitions movies and their ratings across four year-range SQLite shards:

  shard-0 : ≤ 1999  (classics)
  shard-1 : 2000–2009
  shard-2 : 2010–2019  (naturally the hottest shard in our seed data)
  shard-3 : 2020+

Demonstrates three key mechanics that contrast with the consistent-hash
sharding shown in Demo 11:

  1. Partition pruning   — a query filtered by year only opens the shard(s)
                           whose range overlaps the predicate.
  2. Scatter-gather      — a query filtered by score (not the shard key) must
                           fan out to every shard and merge results in memory.
  3. Hot-shard detection and splitting — the 2010–2019 shard accumulates
     most rows given the seed data distribution; a midpoint split bisects it
     into 2010–2014 and 2015–2019, rebalancing the load.

Real-world parallel: ClickHouse PARTITION BY toYYYYMM(event_date), BigQuery
PARTITION BY DATE(timestamp), and Snowflake CLUSTER BY date all implement
range partitioning so that OLAP queries with date predicates physically skip
partitions outside the window.  Netflix's engagement pipeline partitions
24 months of play events into monthly shards; a query for "what did users
watch in January 2024" reads 1 of 24 partitions — ~4 % of stored data.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.range_sharding import ShardedMovieDB, Shard, _OPEN_HIGH
from databaseai.seed_data import MOVIES, RATINGS

console = Console()

SYNTHETIC_RATINGS = 800   # extra ratings seeded for timing contrast
TIMING_ITERATIONS = 60    # repeat queries this many times for stable averages


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "—"
    p = 100 * part / total
    colour = "green" if p <= 30 else "yellow" if p <= 60 else "red"
    return f"[{colour}]{p:.0f} %[/{colour}]"


def _bar(count: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "░" * width
    filled = round(width * count / total)
    return "[cyan]" + "█" * filled + "[/cyan]" + "░" * (width - filled)


def _time_query(db: ShardedMovieDB, kind: str, **kwargs) -> float:
    total = 0.0
    for _ in range(TIMING_ITERATIONS):
        if kind == "pruned":
            _, _, ms = db.query_range_pruned(**kwargs)
        elif kind == "full_scan":
            _, _, ms = db.query_range_full_scan(**kwargs)
        else:
            _, _, ms = db.query_scatter_gather(**kwargs)
        total += ms
    return total / TIMING_ITERATIONS


def main() -> None:
    console.rule("[bold cyan]Range-Based Sharding Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: ClickHouse / BigQuery range partitioning by date —\n"
        "  a 'WHERE event_date = today' query skips every shard except today's\n"
        "  partition, reducing I/O from 100 % to 1/N of stored data.[/dim]\n"
    )

    db = ShardedMovieDB()

    # ---------------------------------------------------------------
    # Section 1: Partition Map
    # ---------------------------------------------------------------
    console.print(Panel("[bold]1. Partition Map — Four Year-Range Shards[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Each shard owns a contiguous, non-overlapping year interval.\n"
        "  A router table maps every year to exactly one shard.  Unlike\n"
        "  consistent hashing (Demo 11), adjacent years always land on the\n"
        "  same shard, enabling efficient range scans without fan-out.[/dim]\n"
    )

    t = Table("Shard", "Year Range", "Analogy", box=box.SIMPLE_HEAD)
    t.add_row("shard-0", "≤ 1999",    "ClickHouse 'pre-2000' partition")
    t.add_row("shard-1", "2000–2009", "ClickHouse '2000s' partition")
    t.add_row("shard-2", "2010–2019", "ClickHouse '2010s' partition  ← hot")
    t.add_row("shard-3", "2020+",     "ClickHouse 'current' partition")
    console.print(t)

    # ---------------------------------------------------------------
    # Section 2: Seed data — routing
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. Seeding — Routing Movies to Shards by Year[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Every movie is routed once to the shard that owns its release year.\n"
        "  Its ratings follow the same routing key so all data for a given\n"
        "  movie is co-located on one shard — no cross-shard JOINs needed for\n"
        "  'top movies from the 2010s' queries.[/dim]\n"
    )

    db.seed(MOVIES, RATINGS)
    db.seed_synthetic(n=SYNTHETIC_RATINGS)

    stats = db.shard_stats()
    total_movies   = sum(s["movies"]  for s in stats)
    total_ratings  = sum(s["ratings"] for s in stats)

    t = Table("Shard", "Year Range", "Movies", "Ratings",
              "Rating share", "Distribution", box=box.SIMPLE_HEAD)
    for s in stats:
        label_parts = s["label"].split(" ", 1)
        shard_label = label_parts[0]
        year_range  = label_parts[1] if len(label_parts) > 1 else ""
        t.add_row(
            shard_label,
            year_range,
            str(s["movies"]),
            str(s["ratings"]),
            _pct(s["ratings"], total_ratings),
            _bar(s["ratings"], total_ratings),
        )
    console.print(t)
    console.print(
        f"  [dim]Total: {total_movies} movies, {total_ratings} ratings "
        f"({SYNTHETIC_RATINGS} synthetic + {len(RATINGS)} seed).\n"
        f"  The 2010–2019 shard dominates because 11 of 20 seed movies\n"
        f"  were released in that decade — a common real-world skew.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 3: Pruned Range Query
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]3. Partition Pruning — Range Query Skips 3 of 4 Shards[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]Query: 'ratings for 2010–2019 movies with score ≥ 4.0'\n"
        "  The router checks which shard ranges overlap [2010, 2019] and\n"
        "  finds exactly one match: shard-2.  Shards 0, 1, and 3 are never\n"
        "  opened — partition pruning eliminates 75 % of I/O.[/dim]\n"
    )

    pruned_results, pruned_shards, pruned_ms = db.query_range_pruned(2010, 2019)
    high_rated = [r for r in pruned_results if r["score"] >= 4.0]

    console.print(f"  Shards contacted : [green]{pruned_shards}[/green] of "
                  f"[dim]{db.manager.shard_count}[/dim]")
    console.print(f"  Total rows returned : {len(pruned_results)}")
    console.print(f"  High-rated rows (score ≥ 4.0) : {len(high_rated)}\n")

    if high_rated:
        t = Table("Title", "Year", "User", "Score", box=box.SIMPLE_HEAD)
        for r in sorted(high_rated, key=lambda x: x["score"], reverse=True)[:8]:
            t.add_row(r["title"], str(r["year"]), r["user_id"], f"{r['score']:.1f}")
        console.print(t)

    # ---------------------------------------------------------------
    # Section 4: Scatter-Gather Query
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]4. Scatter-Gather — Non-Shard-Key Filter Hits All Shards[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]Query: 'all ratings with score ≥ 4.5 regardless of release year'\n"
        "  Score is not the shard key, so the router has no pruning\n"
        "  information.  Every shard is contacted; results are merged in\n"
        "  memory (merge-sort by score) before returning.\n\n"
        "  In production this pattern triggers 'full scan across all partitions'\n"
        "  warnings in ClickHouse EXPLAIN PIPELINE and BigQuery query plans.[/dim]\n"
    )

    sg_results, sg_shards, sg_ms = db.query_scatter_gather(min_score=4.5)

    console.print(f"  Shards contacted : [yellow]{sg_shards}[/yellow] of "
                  f"[dim]{db.manager.shard_count}[/dim]  (all shards)")
    console.print(f"  Rows returned    : {len(sg_results)}\n")

    if sg_results:
        t = Table("Title", "Year", "Shard", "Score", box=box.SIMPLE_HEAD)
        for r in sg_results[:8]:
            shard = db.manager.shard_for_key(r["year"])
            t.add_row(r["title"], str(r["year"]),
                      shard.label() if shard else "?",
                      f"{r['score']:.1f}")
        console.print(t)

    # ---------------------------------------------------------------
    # Section 5: Latency Comparison
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]5. Latency — Pruned vs Scatter-Gather ({TIMING_ITERATIONS} iterations)[/bold]",
        box=box.ROUNDED))
    console.print(
        "  [dim]Each query runs multiple times; the averages expose the I/O cost\n"
        "  difference between opening 1 shard vs. 4 shards for an equal-sized\n"
        "  result set.  The advantage grows linearly with N shard count.[/dim]\n"
    )

    avg_pruned    = _time_query(db, "pruned",    year_low=2010, year_high=2019)
    avg_full_scan = _time_query(db, "full_scan", year_low=2010, year_high=2019)

    shards_pruned = len(db.manager.shards_for_range(2010, 2019))
    shards_all    = db.manager.shard_count

    t = Table("Query", "Shards touched", "Avg latency (μs)", "Overhead",
              box=box.SIMPLE_HEAD)
    t.add_row(
        "Pruned  (year 2010–2019, with pruning)",
        f"[green]{shards_pruned} / {shards_all}[/green]",
        f"{avg_pruned * 1000:.1f}",
        "[green]baseline[/green]",
    )
    t.add_row(
        "Full scan (same predicate, no pruning)",
        f"[yellow]{shards_all} / {shards_all}[/yellow]",
        f"{avg_full_scan * 1000:.1f}",
        f"[yellow]{avg_full_scan / avg_pruned:.1f}×[/yellow]" if avg_pruned > 0 else "—",
    )
    console.print(t)
    console.print(
        "  [dim]Both queries return identical results — the difference is that the\n"
        "  full-scan version opens all 4 shards and discards empty result sets\n"
        "  from shards 0, 1, and 3.  At Netflix scale (24 monthly partitions)\n"
        "  a non-pruned query is up to 24× slower for the same output.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 6: Hot Shard Detection
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. Hot-Shard Detection[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]A 'hot shard' receives disproportionately many reads and writes.\n"
        "  In range sharding this almost always correlates with a key range\n"
        "  that covers a disproportionately large slice of the data —\n"
        "  e.g. the 'current year' partition that every new write targets.\n\n"
        "  Detection: compare each shard's row count against a threshold\n"
        "  (e.g. 1.5× the average).  Mitigation: split the hot shard at its\n"
        "  midpoint, creating two child shards each owning half the interval.[/dim]\n"
    )

    avg_ratings = total_ratings / db.manager.shard_count if db.manager.shard_count else 0
    threshold = max(int(avg_ratings * 1.5), 5)
    hot = db.manager.hot_shards(threshold)

    t = Table("Shard", "Ratings", "Avg", "Threshold", "Status", box=box.SIMPLE_HEAD)
    for s in db.manager.all_shards():
        cnt = s.row_count("ratings")
        status = "[bold red]HOT[/bold red]" if s in hot else "[green]OK[/green]"
        t.add_row(s.label(), str(cnt), f"{avg_ratings:.0f}", str(threshold), status)
    console.print(t)

    if hot:
        console.print(f"\n  [yellow]Hot shards detected:[/yellow] "
                      f"{', '.join(s.label() for s in hot)}")
    else:
        console.print("\n  [green]No hot shards detected at this threshold.[/green]")

    # ---------------------------------------------------------------
    # Section 7: Shard Splitting
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]7. Shard Splitting — Bisect the Hot Partition[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]Splitting bisects the hot shard's year range at its midpoint.\n"
        "  All movies and ratings from the parent are migrated to two child\n"
        "  shards — no data is lost, and the manager atomically swaps the\n"
        "  parent entry for the two children.\n\n"
        "  HBase splits a region when it exceeds ~10 GB; Spanner splits a\n"
        "  tablet when it exceeds ~1 GB.  Both use the same midpoint strategy\n"
        "  and keep serving reads from the parent during the migration.[/dim]\n"
    )

    if hot:
        target_shard = hot[0]
        before_label = target_shard.label()
        before_ratings = target_shard.row_count("ratings")
        before_movies  = target_shard.row_count("movies")

        console.print(f"  Splitting: [bold]{before_label}[/bold]  "
                      f"({before_movies} movies, {before_ratings} ratings)")

        lower, upper = db.manager.split_shard(target_shard)

        console.print(f"  → [green]{lower.label()}[/green]  "
                      f"({lower.row_count('movies')} movies, "
                      f"{lower.row_count('ratings')} ratings)")
        console.print(f"  → [green]{upper.label()}[/green]  "
                      f"({upper.row_count('movies')} movies, "
                      f"{upper.row_count('ratings')} ratings)")

        console.print(f"\n  Shard count before: [dim]{db.manager.shard_count - 1}[/dim]  "
                      f"→  after: [green]{db.manager.shard_count}[/green]\n")

        stats_after = db.shard_stats()
        total_after = sum(s["ratings"] for s in stats_after)
        t = Table("Shard", "Year Range", "Movies", "Ratings", "Share",
                  box=box.SIMPLE_HEAD)
        for s in stats_after:
            parts = s["label"].split(" ", 1)
            t.add_row(
                parts[0],
                parts[1] if len(parts) > 1 else "",
                str(s["movies"]),
                str(s["ratings"]),
                _pct(s["ratings"], total_after),
            )
        console.print(t)

        total_after_sum = sum(s["ratings"] for s in stats_after)
        console.print(
            f"  [dim]Integrity check: {total_after_sum} ratings across "
            f"{db.manager.shard_count} shards (was {total_ratings} across "
            f"{db.manager.shard_count - 1} shards before split).[/dim]"
        )
    else:
        console.print(
            "  [dim]No hot shards to split in this run.  Increase SYNTHETIC_RATINGS\n"
            "  or lower the hot-shard threshold to trigger a split.[/dim]"
        )

    # ---------------------------------------------------------------
    # Section 8: Key Takeaways
    # ---------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Range-Sharding Takeaways:[/bold green]")
    console.print("  • [cyan]Range vs hash[/cyan]       — hash distributes evenly but destroys range locality; "
                  "range preserves locality but risks skew")
    console.print("  • [cyan]Partition pruning[/cyan]   — queries filtered by the shard key skip irrelevant "
                  "shards, reducing I/O linearly with shard count")
    console.print("  • [cyan]Scatter-gather[/cyan]      — queries on non-shard-key columns always fan out to "
                  "every shard; minimise these at schema design time")
    console.print("  • [cyan]Shard key choice[/cyan]    — pick the column used most often in WHERE clauses "
                  "(date/timestamp for analytics, user_id for transactional)")
    console.print("  • [cyan]Hot-shard detection[/cyan] — monitor row counts and throughput per partition; "
                  "alert when one shard exceeds 1.5× the average")
    console.print("  • [cyan]Midpoint splitting[/cyan]  — bisect the hot range; rows migrate to two child "
                  "shards; zero data loss")
    console.print(
        "  [dim]Production: ClickHouse SYSTEM DROP PARTITION prunes cold data;\n"
        "  BigQuery INFORMATION_SCHEMA.PARTITIONS shows partition sizes;\n"
        "  HBase hbase shell> split 'table','splitkey' triggers an explicit split.[/dim]"
    )

    db.close()


if __name__ == "__main__":
    main()
