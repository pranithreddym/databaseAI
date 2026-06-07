"""
Demo 21: Indexing Strategies — Catalog Browse & Discovery
==========================================================
Demo 12 indexed the recommendation path (filter ratings by score, genre, or
user_id).  This demo indexes the *browse* path — the genre rows, "Top Rated"
shelf, and "New & Trending" rail a viewer scrolls through on the home screen,
long before any ranking model runs.  Same four index types — B-tree,
composite, covering, partial — applied to a different access pattern, shown
with EXPLAIN QUERY PLAN and timed queries.

Real-world parallel: the PostgreSQL / Elasticsearch indexes backing a
streaming service's "Browse" home screen — composite indexes for genre rows,
covering indexes for carousel cards, and partial indexes for the "New &
Trending" rail that only ever queries recent releases.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.catalog_index import (
    CatalogIndexDemo,
    QUERY_BTREE_TOP_RATED,
    QUERY_COMPOSITE_BOTH,
    QUERY_COMPOSITE_LEFT,
    QUERY_COMPOSITE_RIGHT_ONLY,
    QUERY_COVERING_CAROUSEL,
    QUERY_PARTIAL_MATCH,
    QUERY_PARTIAL_NO_MATCH,
)

console = Console()

LARGE_TITLES = 4000   # synthetic rows added on top of seed data


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
    console.rule("[bold cyan]Catalog Browse Indexing Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: the index choices behind a streaming service's\n"
        "  Browse home screen — genre rows, Top Rated, New & Trending — a\n"
        "  completely different access pattern from the recommendation engine.[/dim]\n"
    )

    demo = CatalogIndexDemo()
    demo.seed(MOVIES, RATINGS)
    demo.seed_large(n_titles=LARGE_TITLES)
    demo.analyze()

    n_titles = demo.row_count("catalog")

    t = Table("Table", "Rows", box=box.SIMPLE_HEAD)
    t.add_row("catalog", str(n_titles))
    console.print(t)
    console.print(
        f"  [dim]{n_titles} browsable titles — large enough for the optimizer to\n"
        f"  prefer index scans over full table scans.[/dim]\n"
    )

    # ---------------------------------------------------------------
    # Section 1: Full Table Scan (no indexes)
    # ---------------------------------------------------------------
    console.print(Panel("[bold]1. Baseline — Full Table Scan (No Indexes)[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]Without an index, the 'Top Rated' shelf — WHERE rating_avg >= 4.5 —\n"
        "  must visit every row in the catalog. O(N) regardless of how few titles\n"
        "  actually qualify. At catalog scale that's untenable for a home screen\n"
        "  that has to render in well under 100 ms.[/dim]\n"
    )

    baseline_plan = demo.explain(QUERY_BTREE_TOP_RATED)
    baseline_time = demo.time_query(QUERY_BTREE_TOP_RATED)

    t = Table("Query", "Plan", "Avg latency", box=box.SIMPLE_HEAD)
    t.add_row(
        "WHERE rating_avg >= 4.5",
        _plan_style(baseline_plan.strip()),
        f"{baseline_time:.1f} µs",
    )
    console.print(t)

    # ---------------------------------------------------------------
    # Section 2: B-tree Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. B-tree Index — The Top Rated Shelf[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A B-tree index on catalog.rating_avg keeps scores sorted in a\n"
        "  balanced tree.  The range predicate (rating_avg >= 4.5) descends to\n"
        "  the first qualifying leaf and scans forward — O(log N + K), where K\n"
        "  is the shelf size, not the catalog size.[/dim]\n"
    )

    demo.create_btree_index()
    btree_plan = demo.explain(QUERY_BTREE_TOP_RATED)
    btree_time = demo.time_query(QUERY_BTREE_TOP_RATED)

    t = Table("State", "Query Plan", "Avg latency", "Speedup", box=box.SIMPLE_HEAD)
    t.add_row("No index",   _plan_style(baseline_plan.strip()), f"{baseline_time:.1f} µs", "—")
    t.add_row("B-tree idx", _plan_style(btree_plan.strip()),    f"{btree_time:.1f} µs",    _speedup(baseline_time, btree_time))
    console.print(t)
    console.print(
        "  [dim]The plan now shows USING INDEX — the engine descends the B-tree once\n"
        "  and reads only the leaf pages holding rating_avg >= 4.5.  The shelf\n"
        "  renders from a handful of index pages instead of the full catalog.[/dim]"
    )

    demo.drop_btree_index()

    # ---------------------------------------------------------------
    # Section 3: Composite Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]3. Composite Index — Genre Rows Filtered by Decade[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A composite index on (genre, year) sorts entries by genre first,\n"
        "  then by year within each genre — exactly the shape of a 'Sci-Fi:\n"
        "  2010s and later' browse row.  The optimizer can use it for:\n"
        "    ✓ WHERE genre = ?                (left prefix — the genre row itself)\n"
        "    ✓ WHERE genre = ? AND year >= ?  (full prefix, most selective)\n"
        "    ✗ WHERE year >= ?                (right column alone — years from every\n"
        "                                      genre are interleaved in the index)[/dim]\n"
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
        "genre = ? AND year >= ?  (Sci-Fi, 2010s+)",
        _plan_style(plan_both_before.strip()),
        _plan_style(plan_both_after.strip()),
    )
    t.add_row(
        "genre = ?  (genre row, left prefix)",
        _plan_style(plan_left_before.strip()),
        _plan_style(plan_left_after.strip()),
    )
    t.add_row(
        "year >= ?  (right only — e.g. 'Decade' filter)",
        _plan_style(plan_right_before.strip()),
        _plan_style(plan_right_after.strip()),
    )
    console.print(t)
    console.print(
        "  [dim]A 'browse by decade across all genres' filter still scans — the\n"
        "  index B-tree is sorted by genre first, so a separate index on year\n"
        "  alone would be needed to accelerate that pattern.[/dim]"
    )

    demo.drop_composite_index()

    # ---------------------------------------------------------------
    # Section 4: Covering Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]4. Covering Index — Rendering a Carousel Without Heap I/O[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]A genre carousel renders only a title and a score per card — never\n"
        "  the full row.  An index on (genre, title, rating_avg) satisfies that\n"
        "  projection entirely from index pages; the engine never seeks into\n"
        "  the table heap to fetch the rest of the row.\n\n"
        "  Query: SELECT title, rating_avg FROM catalog WHERE genre = ?\n"
        "  Covering index: (genre, title, rating_avg)  ← every projected and\n"
        "  filtered column lives in the index.[/dim]\n"
    )

    sample_genre = "sci-fi"
    plan_cov_before = demo.explain(QUERY_COVERING_CAROUSEL, (sample_genre,))
    time_cov_before = demo.time_query(QUERY_COVERING_CAROUSEL, (sample_genre,))

    demo.create_covering_index()

    plan_cov_after  = demo.explain(QUERY_COVERING_CAROUSEL, (sample_genre,))
    time_cov_after  = demo.time_query(QUERY_COVERING_CAROUSEL, (sample_genre,))

    t = Table("State", "Plan", "Avg latency", "Speedup", box=box.SIMPLE_HEAD)
    t.add_row("No index",       _plan_style(plan_cov_before.strip()), f"{time_cov_before:.1f} µs", "—")
    t.add_row("Covering index", _plan_style(plan_cov_after.strip()),  f"{time_cov_after:.1f} µs",  _speedup(time_cov_before, time_cov_after))
    console.print(t)
    console.print(
        "  [dim]'USING COVERING INDEX' confirms zero table-heap I/O — the same thing\n"
        "  PostgreSQL reports as 'Index Only Scan' in EXPLAIN ANALYZE.  At carousel\n"
        "  render volume (millions of row-loads per minute at peak) this is the\n"
        "  difference between an index-page cache hit and a heap seek per card.[/dim]"
    )

    demo.drop_covering_index()

    # ---------------------------------------------------------------
    # Section 5: Partial Index
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]5. Partial Index — The New & Trending Rail[/bold]",
                        box=box.ROUNDED))
    console.print(
        "  [dim]The 'New & Trending' rail only ever queries recent releases, so a\n"
        "  partial index stores just that slice of the catalog:\n"
        "  CREATE INDEX idx_partial_new_releases ON catalog(year)\n"
        "         WHERE year >= 2020\n\n"
        "  Benefits:\n"
        "    • The index covers a small, bounded fraction of the full catalog\n"
        "    • Titles outside the window never trigger an index write on insert\n"
        "    • VACUUM / index maintenance touches far fewer pages\n\n"
        "  SQLite proves a query is answerable from a partial index only when its\n"
        "  WHERE clause is syntactically IDENTICAL to the index's WHERE clause —\n"
        "  it does not attempt the more general 'is this a logical subset?' proof:\n"
        "    ✓ WHERE year >= 2020  (matches the index predicate exactly)\n"
        "    ✗ WHERE year >= 2010  (looser — rows from 2010-2019 are simply absent\n"
        "                           from the index, so the engine cannot trust it)[/dim]\n"
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
        "year >= 2020  ✓ (New & Trending)",
        _plan_style(plan_match_before.strip()),
        _plan_style(plan_match_after.strip()),
        f"{time_match_before:.1f} µs",
        f"{time_match_after:.1f} µs",
        _speedup(time_match_before, time_match_after),
    )
    t.add_row(
        "year >= 2010  ✗ (Decade browse)",
        _plan_style(plan_no_match_before.strip()),
        _plan_style(plan_no_match_after.strip()),
        f"{time_no_match_before:.1f} µs",
        f"{time_no_match_after:.1f} µs",
        "[dim]—[/dim]",
    )
    console.print(t)
    console.print(
        "  [dim]The year >= 2010 query is logically broader than the index's WHERE\n"
        "  clause, and SQLite only trusts a partial index when the query's predicate\n"
        "  is syntactically identical to its own — so the engine falls back to a\n"
        "  full scan, exactly as if the index did not exist.[/dim]"
    )

    demo.drop_partial_index()

    # ---------------------------------------------------------------
    # Section 6: All Four Indexes Together
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. All Four Indexes — The Browse Page Index Catalog[/bold]",
                        box=box.ROUNDED))

    demo.create_btree_index()
    demo.create_composite_index()
    demo.create_covering_index()
    demo.create_partial_index()

    indexes = demo.list_indexes()
    t = Table("Index Name", "Table", "Type", "Browse Surface", box=box.SIMPLE_HEAD)
    meta = {
        "idx_btree_rating":           ("catalog", "B-tree",    "Top Rated shelf (rating_avg range)"),
        "idx_composite_genre_year":   ("catalog", "Composite", "Genre rows filtered by decade"),
        "idx_covering_genre_carousel": ("catalog", "Covering",  "Genre carousel cards (title + score)"),
        "idx_partial_new_releases":   ("catalog", "Partial",   "New & Trending rail (year >= 2020)"),
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
    console.print("[bold green]Key Takeaways — Browse vs. Recommendation Indexing:[/bold green]")
    console.print("  • [cyan]Access pattern drives index choice[/cyan] — Browse filters by genre/year/recency;")
    console.print("    recommendation filters by score/user_id. Same four index types, different columns.")
    console.print("  • [cyan]B-tree index[/cyan]      — O(log N) range scans power the Top Rated shelf")
    console.print("  • [cyan]Composite index[/cyan]   — leftmost-prefix rule shapes genre+decade browse rows")
    console.print("  • [cyan]Covering index[/cyan]    — carousel cards render straight from index pages")
    console.print("  • [cyan]Partial index[/cyan]     — New & Trending stays small as the full catalog grows")
    console.print(
        "  [dim]Production: PostgreSQL composite indexes back every horizontal row on\n"
        "  the Netflix-style home screen; covering indexes avoid heap fetches at\n"
        "  hundreds of millions of card-renders per day; partial indexes / BRIN\n"
        "  keep the 'recent releases' surface cheap to maintain as the catalog\n"
        "  grows into the hundreds of thousands of titles.[/dim]"
    )

    demo.close()


if __name__ == "__main__":
    main()
