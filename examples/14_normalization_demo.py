"""
Demo 14: Database Normalization
================================
Walks through UNF → 1NF → 2NF → 3NF using a movie-ratings dataset, showing
the DDL changes at each step and the update/insertion/deletion anomalies that
each normal form eliminates.

Real-world parallel: migrating a legacy flat-file dataset — the same progression
a data engineering team follows when ingesting a CSV export from a spreadsheet
tool into a production relational database.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, USERS, RATINGS
from databaseai.normalization import NormalizationDemo, DIRECTOR_METADATA

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yes(flag: bool, yes: str = "YES", no: str = "NO") -> str:
    return f"[green]{yes}[/green]" if flag else f"[red]{no}[/red]"


def _check(ok: bool) -> str:
    return "[bold green]✓ ELIMINATED[/bold green]" if ok else "[bold red]✗ PRESENT[/bold red]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold cyan]Database Normalization — UNF → 1NF → 2NF → 3NF[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: migrating a legacy flat-file dataset — the same\n"
        "  steps a data engineering team follows when ingesting a spreadsheet CSV\n"
        "  export into a production relational database.[/dim]\n"
    )

    demo = NormalizationDemo()
    demo.seed(MOVIES, USERS, RATINGS)

    # -----------------------------------------------------------------------
    # Section 1: The Problem — Unnormalized Form (UNF)
    # -----------------------------------------------------------------------
    console.print(Panel("[bold]1. Unnormalized Form (UNF) — The Legacy Spreadsheet[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Imagine a data analyst exporting a movie catalogue to a CSV file.\n"
        "  Each row represents one movie and all its ratings are jammed into a single\n"
        "  pipe-delimited field:  'u01:alice_w:5.0|u02:bob_k:4.5|...'\n\n"
        "  This violates First Normal Form: the ratings_csv column is non-atomic —\n"
        "  it holds a variable-length list of values.  Queries must parse strings at\n"
        "  runtime (LIKE '%u01%') rather than using indexed lookups.[/dim]\n"
    )

    info = demo.count_non_atomic_values()

    t = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t.add_row("Rows in movies_unf (one per movie)", str(info["total_unf_movies"]))
    t.add_row("Movies with at least one rating",    str(info["movies_with_ratings"]))
    t.add_row("Movies with [bold]multiple[/bold] ratings in ratings_csv",
              f"[red]{info['movies_with_multiple_ratings']}[/red]")
    t.add_row("Rows that would exist in 1NF",       f"[green]{info['rows_in_1nf']}[/green]")
    console.print(t)
    console.print(
        f"  [dim]Columns in movies_unf: {', '.join(demo.table_columns('movies_unf'))}[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 2: 1NF — Atomic Values, Composite Primary Key
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. First Normal Form (1NF) — Atomic Values, No Repeating Groups[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]1NF rule: every cell holds exactly one value.  We explode ratings_csv\n"
        "  into individual rows — one row per (movie, user) pair — and define a\n"
        "  composite primary key (movie_id, user_id).\n\n"
        "  Problem that remains: user_name appears once per (movie, user) row.\n"
        "  Alice rates 5 movies → her username is stored 5 times.  This is a\n"
        "  partial dependency — user_name depends on user_id alone, not on the\n"
        "  full composite key (movie_id, user_id).[/dim]\n"
    )

    red = demo.count_redundant_user_data_in_1nf()

    t = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t.add_row("Rows in movies_1nf (one per rating)",       str(red["total_1nf_rows"]))
    t.add_row("Unique (user_id, user_name) pairs",         str(red["unique_user_name_pairs"]))
    t.add_row("[bold]Redundant[/bold] user_name rows",     f"[red]{red['redundant_user_name_rows']}[/red]")
    t.add_row("Rows in users_2nf after fix",               f"[green]{red['nf2_user_rows']}[/green]")
    console.print(t)
    console.print(
        "  [dim]Columns in movies_1nf: "
        f"{', '.join(demo.table_columns('movies_1nf'))}[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 3: 2NF — Partial Dependencies Removed
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]3. Second Normal Form (2NF) — Partial Dependencies Eliminated[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]2NF rule: every non-key column must depend on the WHOLE primary key.\n"
        "  Extracting user_name → Users table satisfies 2NF.\n\n"
        "  Problem that remains: director_birth_year and director_nationality sit\n"
        "  inside movies_2nf but they do NOT depend on movie_id.  They depend on\n"
        "  director_name → a transitive dependency.  Christopher Nolan directed\n"
        "  3 films: his birth year is stored 3 times.  A mis-typed nationality on\n"
        "  any one row creates a contradiction.[/dim]\n"
    )

    rdd = demo.count_redundant_director_data_in_2nf()

    t = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t.add_row("Rows in movies_2nf",                  str(rdd["total_movies_2nf"]))
    t.add_row("Unique directors",                    str(rdd["unique_directors"]))
    t.add_row("[bold]Redundant[/bold] director-metadata rows",
              f"[red]{rdd['redundant_director_rows']}[/red]")
    t.add_row("Rows in directors_3nf after fix",     f"[green]{rdd['nf3_director_rows']}[/green]")
    console.print(t)

    if rdd["multi_movie_directors"]:
        console.print()
        t2 = Table("Director with multiple films", "Movie count", box=box.SIMPLE_HEAD)
        for entry in sorted(rdd["multi_movie_directors"], key=lambda r: -r["movie_count"]):
            t2.add_row(entry["name"], str(entry["movie_count"]))
        console.print(t2)

    # -----------------------------------------------------------------------
    # Section 4: 3NF Schema — Four Clean Tables
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]4. Third Normal Form (3NF) — Transitive Dependencies Eliminated[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]3NF rule: every non-key column must depend on 'the key, the whole key,\n"
        "  and nothing but the key.'  Extracting director metadata → Directors table\n"
        "  satisfies 3NF.\n\n"
        "  The final schema has four independent tables with clear ownership:\n"
        "    directors_3nf  — who made the film\n"
        "    movies_3nf     — what the film is (title, genre, year, director FK)\n"
        "    users_3nf      — who rates films\n"
        "    ratings_3nf    — the rating event (movie FK, user FK, score)[/dim]\n"
    )

    summary = demo.table_counts_summary()

    t = Table("Normal Form", "Table", "Rows", "Columns", box=box.SIMPLE_HEAD)
    t.add_row("UNF",  "movies_unf",    str(summary["movies_unf"]),
              str(len(demo.table_columns("movies_unf"))))
    t.add_row("1NF",  "movies_1nf",    str(summary["movies_1nf"]),
              str(len(demo.table_columns("movies_1nf"))))
    t.add_row("2NF",  "movies_2nf",    str(summary["movies_2nf"]),
              str(len(demo.table_columns("movies_2nf"))))
    t.add_row("2NF",  "users_2nf",     str(summary["users_2nf"]),
              str(len(demo.table_columns("users_2nf"))))
    t.add_row("2NF",  "ratings_2nf",   str(summary["ratings_2nf"]),
              str(len(demo.table_columns("ratings_2nf"))))
    t.add_row("3NF",  "directors_3nf", str(summary["directors_3nf"]),
              str(len(demo.table_columns("directors_3nf"))))
    t.add_row("3NF",  "movies_3nf",    str(summary["movies_3nf"]),
              str(len(demo.table_columns("movies_3nf"))))
    t.add_row("3NF",  "users_3nf",     str(summary["users_3nf"]),
              str(len(demo.table_columns("users_3nf"))))
    t.add_row("3NF",  "ratings_3nf",   str(summary["ratings_3nf"]),
              str(len(demo.table_columns("ratings_3nf"))))
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 5: Update Anomaly — 2NF vs 3NF
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]5. Update Anomaly — Correcting a Director's Nationality[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Denis Villeneuve directed Arrival (m11) and Blade Runner 2049 (m18).\n"
        "  In movies_2nf his nationality is stored in both rows.  A correction must\n"
        "  touch both rows atomically — miss one and the database contains two\n"
        "  contradictory facts about the same person.  In directors_3nf there is\n"
        "  exactly one row to update, regardless of filmography size.[/dim]\n"
    )

    ua = demo.demonstrate_update_anomaly("Denis Villeneuve")

    t = Table("Stage", "Rows touched to change nationality", "Inconsistency risk", box=box.SIMPLE_HEAD)
    t.add_row(
        "2NF (movies_2nf)",
        f"[red]{ua['rows_actually_updated_in_2nf']}[/red]",
        _yes(ua["inconsistency_risk_in_2nf"], "YES — multi-row update required", "NO"),
    )
    t.add_row(
        "3NF (directors_3nf)",
        f"[green]{ua['rows_actually_updated_in_3nf']}[/green]",
        "[green]NONE — single authoritative row[/green]",
    )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 6: Insertion Anomaly — New Director, No Film Yet
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. Insertion Anomaly — Cataloguing a Director Before Any Films[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]A studio researcher wants to record Alejandro González Iñárritu's\n"
        "  bio (birth year: 1963, Mexican) before his next film is catalogued.\n\n"
        "  In 2NF, director metadata lives in movies_2nf — there is no way to\n"
        "  store a director without a movie row, because movie_id is the primary\n"
        "  key.  The information cannot be saved.\n\n"
        "  In 3NF, directors_3nf is an independent table — the director can be\n"
        "  inserted immediately and associated with movies later via FK.[/dim]\n"
    )

    ia = demo.demonstrate_insertion_anomaly()

    t = Table("Stage", "Can insert director without movie?", box=box.SIMPLE_HEAD)
    t.add_row(
        "2NF (movies_2nf)",
        _yes(not ia["insertion_anomaly_in_2nf"], "YES", "NO — insertion anomaly"),
    )
    t.add_row(
        "3NF (directors_3nf)",
        _yes(ia["can_insert_director_in_3nf_without_movie"], "YES — no anomaly", "NO"),
    )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 7: Deletion Anomaly — Last Film by a Director
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]7. Deletion Anomaly — Removing a Director's Only Film[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Ari Aster directed only Hereditary (m17) in our catalogue.\n"
        "  His birth year (1986) and nationality (American) exist solely because\n"
        "  movies_2nf has a row for Hereditary.\n\n"
        "  In 2NF: deleting Hereditary destroys Ari Aster's entire biographical\n"
        "  record from the database — a deletion anomaly.\n\n"
        "  In 3NF: the movie row in movies_3nf is deleted, but directors_3nf\n"
        "  retains his record independently.[/dim]\n"
    )

    da = demo.demonstrate_deletion_anomaly("m17")

    t = Table("Stage", "Director record after deleting last film", box=box.SIMPLE_HEAD)
    t.add_row(
        "2NF (movies_2nf)",
        _yes(not da["director_info_lost_in_2nf"], "PRESERVED", "LOST — deletion anomaly"),
    )
    t.add_row(
        "3NF (directors_3nf)",
        _yes(da["director_info_preserved_in_3nf"], "PRESERVED — no anomaly", "LOST"),
    )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 8: Anomaly Scorecard
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]8. Anomaly Scorecard — Which Normal Form Fixes What[/bold]", box=box.ROUNDED))

    t = Table(
        "Normal Form", "Update Anomaly", "Insertion Anomaly", "Deletion Anomaly",
        box=box.SIMPLE_HEAD,
    )
    t.add_row("UNF",  "[red]✗ present[/red]", "[red]✗ present[/red]", "[red]✗ present[/red]")
    t.add_row("1NF",  "[red]✗ present[/red]", "[red]✗ present[/red]", "[red]✗ present[/red]")
    t.add_row("2NF",  "[yellow]~ reduced[/yellow]", "[red]✗ present[/red]", "[red]✗ present[/red]")
    t.add_row("3NF",  "[green]✓ eliminated[/green]", "[green]✓ eliminated[/green]", "[green]✓ eliminated[/green]")
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 9: Key Takeaways
    # -----------------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Normalization Takeaways:[/bold green]")
    console.print("  • [cyan]UNF → 1NF[/cyan]  — make every cell atomic; eliminate repeating groups; define a primary key")
    console.print("  • [cyan]1NF → 2NF[/cyan]  — remove partial dependencies; every non-key column must need the WHOLE key")
    console.print("  • [cyan]2NF → 3NF[/cyan]  — remove transitive dependencies; non-key columns depend only on the key")
    console.print("  • [cyan]Update anomaly[/cyan]   — redundant storage; one logical fact stored in N rows")
    console.print("  • [cyan]Insertion anomaly[/cyan] — entities coupled; can't record X without Y")
    console.print("  • [cyan]Deletion anomaly[/cyan]  — destroying one entity silently destroys another")
    console.print("  • [cyan]Denormalization[/cyan]   — intentional reversal for read-heavy OLAP workloads (star schema, wide tables)")
    console.print(
        "  [dim]Production: Django ORM uses SeparateDatabaseAndState to extract a\n"
        "  new model from an existing one; PostgreSQL COPY + INSERT … SELECT\n"
        "  normalises staging tables in a single atomic transaction.[/dim]"
    )

    demo.close()


if __name__ == "__main__":
    main()
