"""
Demo 23: Star Schema & Dimensional Modeling
===========================================
Third Normal Form (3NF) is optimal for writes — minimal redundancy, tight
constraints, fast point lookups.  But the same normalisation that protects
OLTP integrity becomes friction for analytics: a single "average rating by
genre" report must traverse three or more join hops before aggregating.

This demo builds both sides of that trade-off in the same SQLite file:
  1. A normalised 3NF OLTP schema (movies / users / ratings tables)
  2. A star-schema warehouse (fact_plays + dim_movie + dim_user + dim_date)
  3. A one-pass ETL pipeline that moves OLTP rows into the warehouse
  4. Identical analytical queries answered from both schemas — side-by-side

Real-world parallel: Netflix's data warehouse (Apache Iceberg on S3,
catalogued by Metacat) is populated nightly by Spark ETL jobs that read
from MySQL Vitess OLTP shards and write denormalized dim_title /
dim_profile / fact_play_event tables consumed by the recommendation
model-training pipelines and the Content Intelligence dashboards.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.star_schema import StarSchemaDemo
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rating_bar(avg: float, max_val: float = 5.0, width: int = 20) -> str:
    filled = int(round(avg / max_val * width))
    return "[cyan]" + "█" * filled + "░" * (width - filled) + "[/cyan]"


def _genre_color(genre: str) -> str:
    palette = {
        "sci-fi": "cyan", "action": "red", "thriller": "yellow",
        "drama": "green", "animation": "magenta", "horror": "red",
    }
    color = palette.get(genre, "white")
    return f"[{color}]{genre}[/{color}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold cyan]Star Schema & Dimensional Modeling[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Netflix data warehouse — nightly Spark ETL from\n"
        "  MySQL Vitess OLTP shards → Apache Iceberg star-schema facts consumed\n"
        "  by recommendation model training and Content Intelligence dashboards.[/dim]\n"
    )

    demo = StarSchemaDemo()

    # -----------------------------------------------------------------------
    # Section 1: Seed the OLTP schema
    # -----------------------------------------------------------------------
    console.print(Panel(
        "[bold]1. Normalized 3NF OLTP Schema — Source of Truth[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Three tables in Third Normal Form:\n"
        "    oltp_movies  — movie_id PK, title, genre, year, director\n"
        "    oltp_users   — user_id PK, username, email\n"
        "    oltp_ratings — rating_id PK, user_id FK, movie_id FK, score, review\n\n"
        "  Every update anomaly is prevented: changing a director's name means\n"
        "  one UPDATE on one row in oltp_movies.  Perfect for transactional writes.\n"
        "  But every analytical query requires multi-table JOINs.[/dim]\n"
    )

    oltp_seed = [(u, m, s, r) for u, m, s, r in RATINGS]
    counts = demo.seed_oltp(MOVIES, USERS, oltp_seed)

    t = Table("OLTP Table", "Row Count", box=box.SIMPLE_HEAD)
    t.add_row("oltp_movies",  str(counts["movies"]))
    t.add_row("oltp_users",   str(counts["users"]))
    t.add_row("oltp_ratings", str(counts["ratings"]))
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 2: OLTP analytical query (3NF)
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]2. OLTP Query — Average Rating by Genre (3NF, Multi-Join)[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]SQL:\n"
        "    SELECT m.genre, COUNT(*) AS plays, AVG(r.score) AS avg_rating\n"
        "    FROM   oltp_ratings r\n"
        "    JOIN   oltp_movies  m ON m.movie_id = r.movie_id\n"
        "    GROUP  BY m.genre\n"
        "    ORDER  BY avg_rating DESC\n\n"
        "  Two tables joined before the GROUP BY.  In a fully normalised schema\n"
        "  (separate genres table, separate directors table) this grows to 4-5\n"
        "  joins.  Every additional join hop is another hash/nested-loop to plan.[/dim]\n"
    )

    oltp_results = demo.oltp_avg_rating_by_genre()
    t = Table("Genre", "Plays", "Avg Rating", "Unique Titles", box=box.SIMPLE_HEAD)
    for row in oltp_results:
        t.add_row(
            _genre_color(row["genre"]),
            str(row["play_count"]),
            f"{row['avg_rating']:.2f}",
            str(row["unique_titles"]),
        )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 3: ETL — OLTP → Star Schema
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]3. ETL Pipeline — Normalised OLTP → Star Schema Warehouse[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Four dimensions/fact tables built in one INSERT … SELECT pass:\n\n"
        "    dim_movie  — carries all movie attributes + pre-computed 'decade'\n"
        "                 (a derived attribute stored in the dimension for free;\n"
        "                  adding it to OLTP would violate 3NF).\n\n"
        "    dim_user   — flat copy of user attributes (username, email)\n\n"
        "    dim_date   — one row per distinct rating date; carries year/month/\n"
        "                 day/quarter so date arithmetic never touches fact rows.\n\n"
        "    fact_plays — grain: one rating submitted by one user for one movie\n"
        "                 on one date.  Stores measures (rating, has_review)\n"
        "                 and surrogate-key FKs to the three dimensions.\n\n"
        "  Surrogate keys (INTEGER AUTOINCREMENT) insulate the fact table from\n"
        "  natural-key changes — a user rename updates dim_user in one place,\n"
        "  fact_plays never changes.[/dim]\n"
    )

    etl_counts = demo.run_etl()

    t = Table("Warehouse Table", "Rows Loaded", "Role", box=box.SIMPLE_HEAD)
    t.add_row("dim_movie",   str(etl_counts["dim_movie"]),  "Dimension — movie attributes + decade")
    t.add_row("dim_user",    str(etl_counts["dim_user"]),   "Dimension — user profile")
    t.add_row("dim_date",    str(etl_counts["dim_date"]),   "Dimension — calendar attributes")
    t.add_row("fact_plays",  str(etl_counts["fact_plays"]), "Fact — rating events with measures")
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 4: Same query on star schema (fewer joins)
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]4. Star Schema Query — Same Result, Fewer Joins[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]SQL:\n"
        "    SELECT dm.genre, COUNT(*) AS plays, AVG(fp.rating) AS avg_rating\n"
        "    FROM   fact_plays fp\n"
        "    JOIN   dim_movie  dm ON dm.movie_sk = fp.movie_sk\n"
        "    GROUP  BY dm.genre\n"
        "    ORDER  BY avg_rating DESC\n\n"
        "  Two tables total (one fact + one dimension).  Because dim_movie already\n"
        "  carries genre, no additional lookup is needed.  In Redshift / BigQuery,\n"
        "  the planner uses zone maps on the narrow dim_movie to prune blocks before\n"
        "  reading a single fact row — the star topology maps directly to columnar\n"
        "  execution.[/dim]\n"
    )

    star_results = demo.star_avg_rating_by_genre()
    t = Table("Genre", "Plays", "Avg Rating", "Unique Titles", "Bar", box=box.SIMPLE_HEAD)
    for row in star_results:
        t.add_row(
            _genre_color(row["genre"]),
            str(row["play_count"]),
            f"{row['avg_rating']:.2f}",
            str(row["unique_titles"]),
            _rating_bar(row["avg_rating"]),
        )
    console.print(t)

    console.print(
        "  [dim]Results are identical to the OLTP query — the star schema is a\n"
        "  materialized re-shape of the same data, not a new source of truth.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 5: OLAP Slice — restrict to one genre
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]5. OLAP Slice — Restrict Fact Space to One Genre[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]A 'slice' fixes one dimension attribute to a single value,\n"
        "  exposing only the subset of the fact table that matches.\n"
        "  Here we slice by genre = 'sci-fi' to see per-movie detail.[/dim]\n"
    )

    scifi_rows = demo.slice_by_genre("sci-fi")
    t = Table(
        "Title", "Year", "Director", "Plays", "Avg Rating", "Reviews",
        box=box.SIMPLE_HEAD,
    )
    for row in scifi_rows:
        t.add_row(
            row["title"],
            str(row["year"]),
            row["director"],
            str(row["plays"]),
            f"{row['avg_rating']:.2f}",
            str(row["reviews"]),
        )
    console.print(t)

    drama_rows = demo.slice_by_genre("drama")
    console.print(f"\n  [dim]Genre = drama ({len(drama_rows)} titles):[/dim]")
    t2 = Table("Title", "Director", "Plays", "Avg Rating", box=box.SIMPLE_HEAD)
    for row in drama_rows:
        t2.add_row(row["title"], row["director"], str(row["plays"]), f"{row['avg_rating']:.2f}")
    console.print(t2)

    # -----------------------------------------------------------------------
    # Section 6: OLAP Drill-Down — genre → director
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]6. OLAP Drill-Down — Genre → Director (Adding a Dimension Attribute)[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Drill-down adds a finer-grained attribute from the same dimension.\n"
        "  Going from genre to director requires NO new JOIN — the director column\n"
        "  is already in dim_movie.  This is why wide, denormalized dimensions\n"
        "  outperform normalised snowflake schemas for interactive BI tools.\n\n"
        "  In a snowflake schema, director would live in a separate dim_director\n"
        "  table, requiring an extra join hop for every drill-down.[/dim]\n"
    )

    drill = demo.drill_down_by_director()
    t = Table("Genre", "Director", "Titles", "Plays", "Avg Rating", box=box.SIMPLE_HEAD)
    current_genre = None
    for row in drill:
        genre_cell = ""
        if row["genre"] != current_genre:
            genre_cell = _genre_color(row["genre"])
            current_genre = row["genre"]
        t.add_row(
            genre_cell,
            row["director"],
            str(row["titles"]),
            str(row["plays"]),
            f"{row['avg_rating']:.2f}",
        )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 7: OLAP Roll-Up — year → decade
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]7. OLAP Roll-Up — Individual Years Collapsed to Decades[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Roll-up aggregates a fine-grained attribute (year) to a coarser\n"
        "  level (decade).  The 'decade' column was pre-computed during ETL:\n"
        "    decade = (year / 10) * 10\n"
        "  and stored in dim_movie — no runtime arithmetic in the query.\n\n"
        "  This is an example of a 'junk' / pre-computed dimension attribute.\n"
        "  Adding it to the OLTP movies table would create a derived-data\n"
        "  maintenance problem; storing it in the warehouse dimension is safe\n"
        "  because the warehouse is rebuilt from OLTP on each ETL run.[/dim]\n"
    )

    decades = demo.rollup_by_decade()
    t = Table("Decade", "Titles Rated", "Total Plays", "Avg Rating", box=box.SIMPLE_HEAD)
    for row in decades:
        t.add_row(
            f"{row['decade']}s",
            str(row["titles"]),
            str(row["total_plays"]),
            f"{row['avg_rating']:.2f}",
        )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 8: Window function — Top-N per genre
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]8. Window Function — Top-3 Movies per Genre[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]ROW_NUMBER() OVER (PARTITION BY genre ORDER BY avg_rating DESC)\n"
        "  ranks movies within each genre in a single scan of fact_plays + dim_movie.\n"
        "  Columnar engines (Redshift, BigQuery) execute window functions after the\n"
        "  GROUP BY aggregation, so the cost is proportional to the number of\n"
        "  distinct (genre, movie) pairs — not the number of fact rows.[/dim]\n"
    )

    top_movies = demo.top_movies_by_genre(top_n=2)
    t = Table("Genre", "Rank", "Title", "Director", "Year", "Plays", "Avg Rating", box=box.SIMPLE_HEAD)
    for row in top_movies:
        rank_cell = (
            f"[gold1]#{row['rn']}[/gold1]" if row["rn"] == 1 else f"[dim]#{row['rn']}[/dim]"
        )
        t.add_row(
            _genre_color(row["genre"]),
            rank_cell,
            row["title"],
            row["director"],
            str(row["year"]),
            str(row["plays"]),
            f"{row['avg_rating']:.2f}",
        )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 9: Dimensional Modeling Trade-offs
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]9. OLTP vs Star Schema — When to Use Each[/bold]",
        box=box.ROUNDED,
    ))

    t = Table("Property", "3NF OLTP", "Star Schema", box=box.SIMPLE_HEAD)
    t.add_row("Write speed",         "[green]Fast — narrow rows[/green]",       "[red]Slow — wide dims, ETL lag[/red]")
    t.add_row("Update anomalies",    "[green]Prevented by normalisation[/green]","[yellow]ETL must be idempotent[/yellow]")
    t.add_row("Read / analytics",    "[red]Multi-join, complex SQL[/red]",       "[green]1-2 joins, simple SQL[/green]")
    t.add_row("Query plan cost",     "[red]Hash/NL join chains[/red]",           "[green]Fact scan + dim lookup[/green]")
    t.add_row("Columnar engines",    "[yellow]Partial benefit[/yellow]",         "[green]Full benefit (wide dims)[/green]")
    t.add_row("Derived attributes",  "[red]Violates normal form[/red]",          "[green]Free in dimension row[/green]")
    t.add_row("Real-time freshness", "[green]Immediate on commit[/green]",       "[yellow]Depends on ETL cadence[/yellow]")
    console.print(t)

    # -----------------------------------------------------------------------
    # Key Takeaways
    # -----------------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Dimensional Modeling Takeaways:[/bold green]")
    console.print("  • [cyan]Fact table grain[/cyan]           — the most important design decision; one row = one rating event")
    console.print("  • [cyan]Wide dimensions[/cyan]            — carry all descriptive attributes so facts stay narrow")
    console.print("  • [cyan]Surrogate keys[/cyan]             — insulate facts from natural-key changes in the OLTP source")
    console.print("  • [cyan]Pre-computed attributes[/cyan]    — decade, quarter stored at ETL time; free at query time")
    console.print("  • [cyan]Slice / dice / drill-down[/cyan]  — all possible with no extra joins when attrs live in dims")
    console.print("  • [cyan]Window functions[/cyan]           — rank within partition without self-joins")
    console.print("  • [cyan]ETL idempotency[/cyan]            — INSERT OR IGNORE + surrogate keys make reruns safe")
    console.print(
        "  [dim]Production: Netflix Spark ETL → Iceberg facts; Spotify Beam → BigQuery;\n"
        "  Airbnb Druid star schema; Snowflake / Redshift columnar execution model.[/dim]"
    )

    demo.close()


if __name__ == "__main__":
    main()
