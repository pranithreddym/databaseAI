"""
Demo 12: Indexing Strategies
=============================
Demonstrates four index types in SQLite — B-tree, composite, covering, and
partial — using EXPLAIN QUERY PLAN to show structural query plan changes and
timed queries to show wall-clock latency improvements.

Real-world parallel: PostgreSQL index tuning for a Netflix-style recommendation
engine.  Every query that filters ratings by score, genre, or user_id maps
directly to one of the four strategies shown here.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.indexing import (
    IndexingDemo,
    QUERY_BTREE,
    QUERY_COMPOSITE_BOTH,
    QUERY_COMPOSITE_LEFT,
    QUERY_COMPOSITE_RIGHT_ONLY,
    QUERY_COVERING,
    QUERY_PARTIAL_MATCH,
    QUERY_PARTIAL_NO_MATCH,
)

console = Console()

LARGE_RATINGS = 4000   # synthetic rows added on top of seed data


def _plan_style(plan: str) -> str:
    """Colour-code the query plan line for the console."""
    if "USING COVERING INDEX" in plan:
        return f"[bold green]{plan}[/bold green]"
    if "USING INDEX" in plan:
        return f"[green]{plan}[/green]"
    if "SCAN" in plan:
        return f"[yellow]{plan}[/yellow]"
    return plan


def _speedup(before: float, after: float) -> str:
    if after <= 0:
        return "—"
    ratio = before / after
    colour = "green" if ratio >= 1.5 else "yellow"
    return f"[{colour}]{ratio:.1f}×[/{colour}]"


def main() -> None:
    console.rule("[bold cyan]Indexing Strategies Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: PostgreSQL index tuning for a recommendation\n"
        "  engine — choosing the right index type for each query pattern.[/dim]\n"
    )

    demo = IndexingDemo()
    demo.seed(MOVIES, RATINGS)
    demo.seed_large(n_ratings=LARGE_RATINGS)
    demo.analyze()

    n_movies   = demo.row_count("movies_idx")
    n_ratings  = demo.row_count("ratings_idx")

    t = Table("Table", "Rows", box=box.SIMPLE_HEAD)
    t.add_row("movies_idx",  str(n_movies))
    t.add_row("ratings_idx", str(n_ratings))
    console.print(t)
    console.print(
        f"  [dim]{n_ratings} ratings across {n_movies} movies — large enough for the\n"
        f"  optimizer to prefer index scans over full table scans.[/dim]\n"
    )

    # ---------------------------------------------------------------
    # Section 1: Full Table Scan (no indexes)
    # ---------------------------------------------------------------
    console.print(Panel("[bold]1. Baseline — Full Table Scan (No Indexes)[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]Without an index every read must visit every row.  O(N) regardless\n"
        "  of how selective the predicate is.  At Netflix's 250B+ rating events\n"
        "  this is completely untenable for latency-sensitive recommendations.[/dim]\n"
    )

    baseline_plan = demo.explain(QUERY_BTREE)
    baseline_time = demo.time_query(QUERY_BTREE)

    t = Table("Query", "Plan", "Avg latency", box=box.SIMPLE_HEAD)
    t.add_row(
        "WHERE score >= 4.5",
        _plan_style(baseline_plan.strip()),
        f"{baseline_time:.1f} μs",
    )
    console.print(t)

    # ---------------------------------------------------------------
    # Section 2: B-tree Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. B-tree Index — Single Column Range[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A B-tree index on ratings_idx.score keeps score values sorted in a\n"
        "  balanced tree.  A range predicate (score >= 4.5) descends to the first\n"
        "  matching leaf then scans forward — O(log N + K) where K is the result\n"
        "  set size, not the total row count.[/dim]\n"
    )

    demo.create_btree_index()
    btree_plan = demo.explain(QUERY_BTREE)
    btree_time = demo.time_query(QUERY_BTREE)

    t = Table("State", "Query Plan", "Avg latency", "Speedup", box=box.SIMPLE_HEAD)
    t.add_row("No index",    _plan_style(baseline_plan.strip()), f"{baseline_time:.1f} μs", "—")
    t.add_row("B-tree idx",  _plan_style(btree_plan.strip()),    f"{btree_time:.1f} μs",    _speedup(baseline_time, btree_time))
    console.print(t)
    console.print(
        "  [dim]The plan now shows USING INDEX — the engine descends the B-tree once\n"
        "  and reads only matching leaf pages.  Index pages are much smaller than\n"
        "  the full table, so the improvement grows with table size.[/dim]"
    )

    demo.drop_btree_index()

    # ---------------------------------------------------------------
    # Section 3: Composite Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]3. Composite Index — Leftmost Prefix Rule[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A composite index on (genre, year) sorts entries by genre first, then\n"
        "  by year within each genre.  The optimizer can use it for:\n"
        "    ✓ WHERE genre = ?                (left prefix)\n"
        "    ✓ WHERE genre = ? AND year >= ?  (full prefix, most selective)\n"
        "    ✗ WHERE year >= ?                (right column alone — year values are\n"
        "                                      interleaved across genres in the index)\n"
        "  PostgreSQL's multi-column index pages obey the same rule.[/dim]\n"
    )

    plan_both_before  = demo.explain(QUERY_COMPOSITE_BOTH,  ("sci-fi", 2010))
    plan_left_before  = demo.explain(QUERY_COMPOSITE_LEFT,  ("sci-fi",))
    plan_right_before = demo.explain(QUERY_COMPOSITE_RIGHT_ONLY, (2010,))

    demo.create_composite_index()

    plan_both_after   = demo.explain(QUERY_COMPOSITE_BOTH,  ("sci-fi", 2010))
    plan_left_after   = demo.explain(QUERY_COMPOSITE_LEFT,  ("sci-fi",))
    plan_right_after  = demo.explain(QUERY_COMPOSITE_RIGHT_ONLY, (2010,))

    t = Table("Query", "Without Index", "With Composite Index", box=box.SIMPLE_HEAD)
    t.add_row(
        "genre = ? AND year >= ?",
        _plan_style(plan_both_before.strip()),
        _plan_style(plan_both_after.strip()),
    )
    t.add_row(
        "genre = ?  (left prefix)",
        _plan_style(plan_left_before.strip()),
        _plan_style(plan_left_after.strip()),
    )
    t.add_row(
        "year >= ?  (right only)",
        _plan_style(plan_right_before.strip()),
        _plan_style(plan_right_after.strip()),
    )
    console.print(t)
    console.print(
        "  [dim]Year-only queries still scan — the index B-tree is sorted by genre,\n"
        "  so years from different genres are interleaved.  A separate index on year\n"
        "  would be needed to accelerate that pattern.[/dim]"
    )

    demo.drop_composite_index()

    # ---------------------------------------------------------------
    # Section 4: Covering Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]4. Covering Index — Eliminate Table Lookups[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A query is 'covered' when every column it projects and filters is\n"
        "  present in the index.  The engine reads only index pages — the main\n"
        "  table heap is never touched.  This halves or more the I/O for\n"
        "  read-heavy recommendation queries that project a few columns.\n\n"
        "  Query: SELECT user_id, score FROM ratings_idx WHERE user_id = ?\n"
        "  Covering index: (user_id, score)  ← both columns live in the index.[/dim]\n"
    )

    sample_user = "ux0050"
    plan_cov_before = demo.explain(QUERY_COVERING, (sample_user,))
    time_cov_before = demo.time_query(QUERY_COVERING, (sample_user,))

    demo.create_covering_index()

    plan_cov_after  = demo.explain(QUERY_COVERING, (sample_user,))
    time_cov_after  = demo.time_query(QUERY_COVERING, (sample_user,))

    t = Table("State", "Plan", "Avg latency", "Speedup", box=box.SIMPLE_HEAD)
    t.add_row("No index",       _plan_style(plan_cov_before.strip()), f"{time_cov_before:.1f} μs", "—")
    t.add_row("Covering index", _plan_style(plan_cov_after.strip()),  f"{time_cov_after:.1f} μs",  _speedup(time_cov_before, time_cov_after))
    console.print(t)
    console.print(
        "  [dim]'USING COVERING INDEX' confirms zero table-heap I/O.  In PostgreSQL\n"
        "  this appears as 'Index Only Scan' in EXPLAIN ANALYZE output.[/dim]"
    )

    demo.drop_covering_index()

    # ---------------------------------------------------------------
    # Section 5: Partial Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]5. Partial Index — Index a Meaningful Subset[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A partial index stores only the rows satisfying a fixed predicate.\n"
        "  Index: CREATE INDEX idx_partial_high_score ON ratings_idx(score)\n"
        "         WHERE score >= 4.0\n\n"
        "  Benefits:\n"
        "    • Index is ~30 % smaller (only high-scoring rows)\n"
        "    • Writes to low-score rows never update this index\n"
        "    • VACUUM is faster (fewer index pages to process)\n\n"
        "  The optimizer uses it ONLY when the query predicate is a provable\n"
        "  subset of the index predicate:\n"
        "    ✓ WHERE score >= 4.5  (4.5 >= 4.0 → all matching rows are indexed)\n"
        "    ✗ WHERE score >= 3.0  (3.0 < 4.0 → rows with 3.0-4.0 not in index)[/dim]\n"
    )

    plan_match_before    = demo.explain(QUERY_PARTIAL_MATCH)
    plan_no_match_before = demo.explain(QUERY_PARTIAL_NO_MATCH)
    time_match_before    = demo.time_query(QUERY_PARTIAL_MATCH)
    time_no_match_before = demo.time_query(QUERY_PARTIAL_NO_MATCH)

    demo.create_partial_index()

    plan_match_after    = demo.explain(QUERY_PARTIAL_MATCH)
    plan_no_match_after = demo.explain(QUERY_PARTIAL_NO_MATCH)
    time_match_after    = demo.time_query(QUERY_PARTIAL_MATCH)
    time_no_match_after = demo.time_query(QUERY_PARTIAL_NO_MATCH)

    t = Table("Query", "Before partial index", "After partial index",
              "Latency before", "Latency after", "Speedup", box=box.SIMPLE_HEAD)
    t.add_row(
        "score >= 4.0  ✓",
        _plan_style(plan_match_before.strip()),
        _plan_style(plan_match_after.strip()),
        f"{time_match_before:.1f} μs",
        f"{time_match_after:.1f} μs",
        _speedup(time_match_before, time_match_after),
    )
    t.add_row(
        "score >= 3.0  ✗",
        _plan_style(plan_no_match_before.strip()),
        _plan_style(plan_no_match_after.strip()),
        f"{time_no_match_before:.1f} μs",
        f"{time_no_match_after:.1f} μs",
        "[dim]—[/dim]",
    )
    console.print(t)
    console.print(
        "  [dim]The partial index is invisible to score >= 3.0 queries — the engine\n"
        "  falls back to a full scan, same as if the index did not exist.[/dim]"
    )

    demo.drop_partial_index()

    # ---------------------------------------------------------------
    # Section 6: All Four Indexes Together
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. All Four Indexes — Index Catalog[/bold]",
                        box=box.ROUNDED))

    demo.create_btree_index()
    demo.create_composite_index()
    demo.create_covering_index()
    demo.create_partial_index()

    indexes = demo.list_indexes()
    t = Table("Index Name", "Table", "Type", "Purpose", box=box.SIMPLE_HEAD)
    meta = {
        "idx_btree_score":          ("ratings_idx", "B-tree",    "Range queries on score"),
        "idx_composite_genre_year": ("movies_idx",  "Composite", "Genre + year filter (leftmost prefix)"),
        "idx_covering_user_score":  ("ratings_idx", "Covering",  "user_id lookup with score projection"),
        "idx_partial_high_score":   ("ratings_idx", "Partial",   "Only rows with score >= 4.0"),
    }
    for idx in indexes:
        name = idx["name"]
        itype, purpose = meta.get(name, ("?", "?"))[1], meta.get(name, ("?", "?"))[2]
        t.add_row(name, idx["table_name"], itype, purpose)
    console.print(t)

    # ---------------------------------------------------------------
    # Section 7: Key Takeaways
    # ---------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Indexing Takeaways:[/bold green]")
    console.print("  • [cyan]B-tree index[/cyan]      — O(log N) lookup for equality and range; the safe default")
    console.print("  • [cyan]Composite index[/cyan]   — leftmost-prefix rule: lead with the most selective column")
    console.print("  • [cyan]Covering index[/cyan]    — project only indexed columns to eliminate table-heap I/O")
    console.print("  • [cyan]Partial index[/cyan]     — index only the hot subset; smaller, faster writes on cold rows")
    console.print("  • [cyan]EXPLAIN QUERY PLAN[/cyan]— always verify the plan changed; the optimizer can surprise you")
    console.print("  • [cyan]Write amplification[/cyan]— each index adds a write per INSERT/UPDATE; balance read vs. write")
    console.print(
        "  [dim]Production: PostgreSQL EXPLAIN (ANALYZE, BUFFERS) for buffer-hit stats;\n"
        "  MySQL EXPLAIN FORMAT=JSON for key_len and filtered %; pg_stat_user_indexes\n"
        "  to find indexes with zero scans (drop them to reclaim write throughput).[/dim]"
    )

    demo.close()


if __name__ == "__main__":
    main()
