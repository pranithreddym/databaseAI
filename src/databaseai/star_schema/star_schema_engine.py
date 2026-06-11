"""
Star Schema & Dimensional Modeling — OLTP → Data Warehouse
===========================================================

DB Architect notes:
  Third Normal Form (3NF) is ideal for OLTP: minimal redundancy, fast
  point writes, referential integrity via foreign keys.  But analytical
  queries over a 3NF schema require deep JOIN chains that are expensive
  and hard to write — a single "top-rated movies by genre" report might
  join five tables before the planner can even start aggregating.

  Dimensional modeling (Ralph Kimball, 1996) inverts the priority: it
  intentionally denormalizes reference data into wide, flat DIMENSION
  tables, and stores the measurable events in a central FACT table that
  references the dimensions by surrogate key.  The resulting star
  topology (one fact table surrounded by dimension tables) reduces join
  depth, makes query plans predictable, and aligns perfectly with
  columnar engines (Redshift, BigQuery, Snowflake) that read entire
  columns in bulk.

  Key concepts implemented here:
    - Fact table (fact_plays): one row per rating/play event; numeric
      measures (rating, review_count) + foreign keys to dimensions.
    - Dimension tables: dim_movie, dim_user, dim_date.  Dimensions carry
      all descriptive attributes so the fact table stays narrow.
    - Surrogate keys: integer PKs in each dimension insulate the fact
      table from natural-key changes (e.g. a user renaming themselves).
    - Slowly Changing Dimension (SCD-0): dimensions here are static;
      in production SCD-2 rows track history with effective_from/to
      dates — see Kimball's "Data Warehouse Toolkit" for full treatment.
    - ETL pipeline: a simple INSERT INTO ... SELECT that reads from the
      normalized OLTP schema and populates the warehouse in one pass.

  Query comparison (Section 4):
    3NF query for "avg rating by genre" requires joining movies +
    ratings + users — three tables, two JOIN conditions.
    Star schema query for the same result: one fact table + one slim
    dimension join — planner reads far fewer rows.

  Grain:
    The fact_plays grain here is "one rating submitted by one user for
    one movie on one calendar date."  Choosing the right grain is the
    most consequential decision in dimensional design.

Production parallels:
  - Netflix Data Warehouse (Metacat / Iceberg): Spark ETL jobs run
    nightly to materialize denormalized "play events" facts from raw
    event streams.  Dimension tables (dim_title, dim_profile) are
    rebuilt from the OLTP source-of-truth (MySQL Vitess shards).
  - Spotify Beam pipelines: stream listening events into BigQuery fact
    tables keyed on dim_track and dim_user surrogate IDs, enabling the
    "Wrapped" year-end analytics.
  - Airbnb Airflow → Druid: booking fact tables + host/property
    dimension tables power the real-time analytics dashboard.
  - Snowflake / Redshift: both engines implement micro-partition /
    zone-map pruning that exploits the star schema's narrow join width
    to skip entire data blocks during dimension filter scans.
"""

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_OLTP_SCHEMA = """
CREATE TABLE IF NOT EXISTS oltp_movies (
    movie_id   TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    genre      TEXT NOT NULL,
    year       INTEGER NOT NULL,
    director   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS oltp_users (
    user_id    TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    email      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS oltp_ratings (
    rating_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL REFERENCES oltp_users(user_id),
    movie_id   TEXT NOT NULL REFERENCES oltp_movies(movie_id),
    score      REAL NOT NULL,
    review     TEXT,
    rated_on   TEXT NOT NULL DEFAULT (date('now'))
);
"""

_WAREHOUSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_movie (
    movie_sk   INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id   TEXT NOT NULL UNIQUE,
    title      TEXT NOT NULL,
    genre      TEXT NOT NULL,
    year       INTEGER NOT NULL,
    director   TEXT NOT NULL,
    decade     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_user (
    user_sk    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL UNIQUE,
    username   TEXT NOT NULL,
    email      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_sk    INTEGER PRIMARY KEY AUTOINCREMENT,
    full_date  TEXT NOT NULL UNIQUE,
    year       INTEGER NOT NULL,
    month      INTEGER NOT NULL,
    day        INTEGER NOT NULL,
    quarter    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_plays (
    play_sk    INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_sk   INTEGER NOT NULL REFERENCES dim_movie(movie_sk),
    user_sk    INTEGER NOT NULL REFERENCES dim_user(user_sk),
    date_sk    INTEGER NOT NULL REFERENCES dim_date(date_sk),
    rating     REAL NOT NULL,
    has_review INTEGER NOT NULL DEFAULT 0,
    UNIQUE(movie_sk, user_sk, date_sk)
);
"""


class StarSchemaDemo:
    """
    Demonstrates the OLTP → Star Schema pipeline using two SQLite databases
    living in the same file (OLTP tables and warehouse tables co-located for
    simplicity; in production they are separate systems).

    Each public method returns a plain dict so the demo script and tests can
    assert exact values without coupling to the DB connection.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            fd, self._db_path = tempfile.mkstemp(suffix=".db", prefix="star_")
            os.close(fd)
            self._owns_file = True
        else:
            self._db_path = db_path
            self._owns_file = False

        with self._conn() as conn:
            conn.executescript(_OLTP_SCHEMA)
            conn.executescript(_WAREHOUSE_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        if self._owns_file:
            for ext in ("", "-wal", "-shm"):
                try:
                    os.unlink(self._db_path + ext)
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    # OLTP seed
    # ------------------------------------------------------------------

    def seed_oltp(
        self,
        movies: List[Dict],
        users: List[Dict],
        ratings: List[Tuple],
    ) -> Dict[str, int]:
        """
        Populate the normalized OLTP tables from the standard seed data.

        ratings is a list of (user_id, movie_id, score, review) tuples.
        Returns row counts for each table.
        """
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO oltp_movies "
                "(movie_id, title, genre, year, director) "
                "VALUES (:id, :title, :genre, :year, :director)",
                movies,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO oltp_users "
                "(user_id, username, email) "
                "VALUES (:id, :username, :email)",
                users,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO oltp_ratings "
                "(user_id, movie_id, score, review, rated_on) "
                "VALUES (?, ?, ?, ?, date('now'))",
                ratings,
            )

        with self._conn() as conn:
            return {
                "movies": conn.execute("SELECT COUNT(*) FROM oltp_movies").fetchone()[0],
                "users":  conn.execute("SELECT COUNT(*) FROM oltp_users").fetchone()[0],
                "ratings": conn.execute("SELECT COUNT(*) FROM oltp_ratings").fetchone()[0],
            }

    # ------------------------------------------------------------------
    # ETL: OLTP → Star Schema
    # ------------------------------------------------------------------

    def run_etl(self) -> Dict[str, int]:
        """
        Single-pass ETL from the normalized OLTP schema into the warehouse.

        dim_movie: enriched with a pre-computed 'decade' attribute — a
          derived attribute that would be expensive to add to OLTP but is
          free in a denormalized dimension (no normalisation penalty).

        dim_date: populated from the distinct rated_on dates in oltp_ratings.
          In production a date dimension covers years of dates, generated
          once and never updated.

        fact_plays: one row per rating, joining the three dimensions to
          resolve surrogate keys.
        """
        with self._conn() as conn:
            # dim_movie
            conn.execute(
                """
                INSERT OR IGNORE INTO dim_movie
                    (movie_id, title, genre, year, director, decade)
                SELECT movie_id, title, genre, year, director,
                       (year / 10) * 10
                FROM   oltp_movies
                """
            )

            # dim_user
            conn.execute(
                """
                INSERT OR IGNORE INTO dim_user (user_id, username, email)
                SELECT user_id, username, email FROM oltp_users
                """
            )

            # dim_date — one row per distinct rating date
            conn.execute(
                """
                INSERT OR IGNORE INTO dim_date (full_date, year, month, day, quarter)
                SELECT DISTINCT
                    rated_on,
                    CAST(strftime('%Y', rated_on) AS INTEGER),
                    CAST(strftime('%m', rated_on) AS INTEGER),
                    CAST(strftime('%d', rated_on) AS INTEGER),
                    (CAST(strftime('%m', rated_on) AS INTEGER) + 2) / 3
                FROM oltp_ratings
                """
            )

            # fact_plays
            conn.execute(
                """
                INSERT OR IGNORE INTO fact_plays
                    (movie_sk, user_sk, date_sk, rating, has_review)
                SELECT
                    dm.movie_sk,
                    du.user_sk,
                    dd.date_sk,
                    r.score,
                    CASE WHEN r.review IS NOT NULL AND r.review != '' THEN 1 ELSE 0 END
                FROM   oltp_ratings r
                JOIN   dim_movie dm ON dm.movie_id = r.movie_id
                JOIN   dim_user  du ON du.user_id  = r.user_id
                JOIN   dim_date  dd ON dd.full_date = r.rated_on
                """
            )

            return {
                "dim_movie": conn.execute("SELECT COUNT(*) FROM dim_movie").fetchone()[0],
                "dim_user":  conn.execute("SELECT COUNT(*) FROM dim_user").fetchone()[0],
                "dim_date":  conn.execute("SELECT COUNT(*) FROM dim_date").fetchone()[0],
                "fact_plays": conn.execute("SELECT COUNT(*) FROM fact_plays").fetchone()[0],
            }

    # ------------------------------------------------------------------
    # 3NF OLTP analytical query (verbose — many joins)
    # ------------------------------------------------------------------

    def oltp_avg_rating_by_genre(self) -> List[Dict[str, Any]]:
        """
        Average rating grouped by genre, answered from the normalized 3NF
        OLTP schema.

        Requires a three-table join: oltp_ratings → oltp_movies (to resolve
        genre) + implicit grouping.  In a real OLTP schema with additional
        normalisation (e.g. a separate genres table, a separate directors
        table) the join depth grows further.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT   m.genre,
                         COUNT(r.rating_id)              AS play_count,
                         ROUND(AVG(r.score), 2)          AS avg_rating,
                         COUNT(DISTINCT r.movie_id)      AS unique_titles
                FROM     oltp_ratings r
                JOIN     oltp_movies  m ON m.movie_id = r.movie_id
                GROUP BY m.genre
                ORDER BY avg_rating DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Star schema analytical query (compact — one fact + one dim join)
    # ------------------------------------------------------------------

    def star_avg_rating_by_genre(self) -> List[Dict[str, Any]]:
        """
        The same average-rating-by-genre query answered from the star schema.

        Only TWO tables are touched: fact_plays + dim_movie.  The dimension
        already carries the genre attribute so no further joins are needed.
        In Redshift / BigQuery the planner can push the genre filter as a
        zone-map / partition prune on dim_movie before touching the fact.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT   dm.genre,
                         COUNT(fp.play_sk)               AS play_count,
                         ROUND(AVG(fp.rating), 2)        AS avg_rating,
                         COUNT(DISTINCT fp.movie_sk)     AS unique_titles
                FROM     fact_plays fp
                JOIN     dim_movie  dm ON dm.movie_sk = fp.movie_sk
                GROUP BY dm.genre
                ORDER BY avg_rating DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # OLAP slice: filter fact by a single dimension attribute
    # ------------------------------------------------------------------

    def slice_by_genre(self, genre: str) -> List[Dict[str, Any]]:
        """
        OLAP slice — restrict the fact space to one genre value.

        "Slicing" fixes one dimension attribute and returns all fact rows
        that match, aggregated by movie.  Equivalent to a WHERE clause on
        the dimension joined back to the fact.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT   dm.title,
                         dm.year,
                         dm.director,
                         COUNT(fp.play_sk)          AS plays,
                         ROUND(AVG(fp.rating), 2)   AS avg_rating,
                         SUM(fp.has_review)         AS reviews
                FROM     fact_plays fp
                JOIN     dim_movie  dm ON dm.movie_sk = fp.movie_sk
                WHERE    dm.genre = ?
                GROUP BY dm.movie_sk
                ORDER BY avg_rating DESC
                """,
                (genre,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # OLAP drill-down: genre → director (adding a dimension attribute)
    # ------------------------------------------------------------------

    def drill_down_by_director(self) -> List[Dict[str, Any]]:
        """
        OLAP drill-down — start at the genre grain, then add the director
        attribute to break each genre total into per-director subtotals.

        Drill-down never requires additional JOIN tables when the dimension
        already contains the finer-grained attribute — this is the payoff
        of a wide dimension table.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT   dm.genre,
                         dm.director,
                         COUNT(fp.play_sk)          AS plays,
                         ROUND(AVG(fp.rating), 2)   AS avg_rating,
                         COUNT(DISTINCT fp.movie_sk) AS titles
                FROM     fact_plays fp
                JOIN     dim_movie  dm ON dm.movie_sk = fp.movie_sk
                GROUP BY dm.genre, dm.director
                ORDER BY dm.genre, avg_rating DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # OLAP roll-up: decade-level aggregation
    # ------------------------------------------------------------------

    def rollup_by_decade(self) -> List[Dict[str, Any]]:
        """
        OLAP roll-up — collapse individual year values into decades.

        The 'decade' column was pre-computed during ETL (year / 10 * 10) and
        stored in dim_movie.  Roll-up queries use it directly without any
        runtime arithmetic — another benefit of denormalized dimensions.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT   dm.decade,
                         COUNT(DISTINCT dm.movie_sk) AS titles,
                         COUNT(fp.play_sk)           AS total_plays,
                         ROUND(AVG(fp.rating), 2)    AS avg_rating
                FROM     fact_plays fp
                JOIN     dim_movie  dm ON dm.movie_sk = fp.movie_sk
                GROUP BY dm.decade
                ORDER BY dm.decade
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Top-N per dimension value
    # ------------------------------------------------------------------

    def top_movies_by_genre(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Return the top_n highest-rated movie per genre from the star schema.

        Uses a window function (ROW_NUMBER OVER PARTITION BY genre) to rank
        titles within each genre — a common star-schema analytical pattern.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        dm.genre,
                        dm.title,
                        dm.director,
                        dm.year,
                        COUNT(fp.play_sk)        AS plays,
                        ROUND(AVG(fp.rating), 2) AS avg_rating,
                        ROW_NUMBER() OVER (
                            PARTITION BY dm.genre
                            ORDER BY AVG(fp.rating) DESC
                        ) AS rn
                    FROM fact_plays fp
                    JOIN dim_movie dm ON dm.movie_sk = fp.movie_sk
                    GROUP BY dm.genre, dm.movie_sk
                )
                SELECT genre, title, director, year, plays, avg_rating, rn
                FROM   ranked
                WHERE  rn <= ?
                ORDER BY genre, rn
                """,
                (top_n,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Dimension completeness helpers (used by tests)
    # ------------------------------------------------------------------

    def dim_genres(self) -> List[str]:
        """Return sorted list of genres present in dim_movie."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT genre FROM dim_movie ORDER BY genre"
            ).fetchall()
        return [r["genre"] for r in rows]

    def fact_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM fact_plays").fetchone()[0]

    def dim_count(self, table: str) -> int:
        with self._conn() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def oltp_count(self, table: str) -> int:
        with self._conn() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
