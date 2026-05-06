"""
Demo 6: MySQL Relational Database
===================================
Shows the same MovieRegistry interface running against MySQL instead of SQLite.
Highlights the SQL dialect differences between the two backends.

Requirements:
  A running MySQL server. Start one with Docker:

    docker run -d --name mysql-cineai \\
      -e MYSQL_ROOT_PASSWORD=cineai \\
      -e MYSQL_DATABASE=cineai \\
      -p 3306:3306 mysql:8

  Then run:
    python examples/06_mysql_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pymysql
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from databaseai.relational_db import MovieRegistry, MySQLMovieRegistry
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()

MYSQL_CONFIG = dict(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="cineai",
    database="cineai",
)


def check_mysql() -> bool:
    try:
        conn = pymysql.connect(**MYSQL_CONFIG, connect_timeout=2)
        conn.close()
        return True
    except Exception as e:
        console.print(f"[yellow]MySQL not available:[/yellow] {e}")
        return False


def load_registry(reg) -> None:
    reg.add_movies(MOVIES)
    for u in USERS:
        reg.add_user(u)
    for uid, mid, score, review in RATINGS:
        reg.add_rating(uid, mid, score, review)


def show_comparison(sqlite_reg: MovieRegistry, mysql_reg) -> None:
    console.print()
    console.print(Panel(
        "[bold]SQLite vs MySQL — Same Interface, Different Backends[/bold]",
        box=box.ROUNDED
    ))

    t = Table("Query", "SQLite Result", "MySQL Result", box=box.SIMPLE_HEAD)

    t.add_row(
        "movie_count()",
        str(sqlite_reg.movie_count()),
        str(mysql_reg.movie_count()),
    )
    t.add_row(
        "user_count()",
        str(sqlite_reg.user_count()),
        str(mysql_reg.user_count()),
    )
    t.add_row(
        "rating_count()",
        str(sqlite_reg.rating_count()),
        str(mysql_reg.rating_count()),
    )

    sqlite_top = sqlite_reg.top_rated_movies(n=1)
    mysql_top = mysql_reg.top_rated_movies(n=1)
    t.add_row(
        "top_rated_movies(n=1)",
        f"{sqlite_top[0]['title']} ({sqlite_top[0]['avg_score']})" if sqlite_top else "—",
        f"{mysql_top[0]['title']} ({mysql_top[0]['avg_score']})" if mysql_top else "—",
    )
    console.print(t)


def show_dialect_diff() -> None:
    console.print()
    console.print("[bold]SQL Dialect Differences[/bold]")
    t = Table("Operation", "SQLite", "MySQL", box=box.SIMPLE_HEAD)
    t.add_row("Parameter style",    "`?`",                   "`%s`")
    t.add_row("Upsert",
              "ON CONFLICT(col) DO UPDATE SET ...",
              "ON DUPLICATE KEY UPDATE col=VALUES(col)")
    t.add_row("Insert ignore",      "INSERT OR IGNORE INTO", "INSERT IGNORE INTO")
    t.add_row("Replace",            "INSERT OR REPLACE INTO","REPLACE INTO")
    t.add_row("Auto-increment",     "INTEGER PRIMARY KEY AUTOINCREMENT",
              "INT AUTO_INCREMENT PRIMARY KEY")
    t.add_row("Current timestamp",  "datetime('now')",       "NOW()")
    t.add_row("Foreign key enable", "PRAGMA foreign_keys=ON","ON by default (InnoDB)")
    t.add_row("DDL batch",          "conn.executescript(sql)","execute() per statement")
    t.add_row("Row result type",    "sqlite3.Row (named)",   "dict (DictCursor)")
    t.add_row("GROUP BY",           "Flexible (any column)", "Must list all non-agg cols")
    console.print(t)


def main():
    console.rule("[bold cyan]MySQL Relational Database Demo[/bold cyan]")

    # ---- SQLite (always available) ----
    console.print("\n[bold green]▶ SQLite backend (zero-config)[/bold green]")
    sqlite_reg = MovieRegistry()
    load_registry(sqlite_reg)
    console.print(f"  Loaded: {sqlite_reg.movie_count()} movies, "
                  f"{sqlite_reg.user_count()} users, {sqlite_reg.rating_count()} ratings")

    # ---- MySQL (requires server) ----
    if check_mysql():
        console.print("\n[bold green]▶ MySQL backend (127.0.0.1:3306)[/bold green]")
        mysql_reg = MySQLMovieRegistry(**MYSQL_CONFIG)
        mysql_reg.teardown()
        mysql_reg._init_schema()
        load_registry(mysql_reg)
        console.print(f"  Loaded: {mysql_reg.movie_count()} movies, "
                      f"{mysql_reg.user_count()} users, {mysql_reg.rating_count()} ratings")

        # Top-rated
        console.print("\n[bold]Top 5 movies by avg rating (MySQL):[/bold]")
        t = Table("Title", "Genre", "Avg Score", "# Ratings", box=box.SIMPLE)
        for r in mysql_reg.top_rated_movies(n=5):
            t.add_row(r["title"], r["genre"], str(r["avg_score"]), str(r["total_ratings"]))
        console.print(t)

        # Search
        console.print("[bold]Sci-fi movies from 2010+ (MySQL):[/bold]")
        t = Table("Title", "Year", "Director", box=box.SIMPLE)
        for r in mysql_reg.search_movies(genre="sci-fi", min_year=2010):
            t.add_row(r["title"], str(r["year"]), r["director"])
        console.print(t)

        # Side-by-side comparison
        show_comparison(sqlite_reg, mysql_reg)

        mysql_reg.teardown()
    else:
        console.print("\n[dim]Skipping MySQL demo — server not running.[/dim]")
        console.print("  Start MySQL with:")
        console.print("  [bold]docker run -d --name mysql-cineai "
                      "-e MYSQL_ROOT_PASSWORD=cineai "
                      "-e MYSQL_DATABASE=cineai -p 3306:3306 mysql:8[/bold]")

    show_dialect_diff()


if __name__ == "__main__":
    main()
