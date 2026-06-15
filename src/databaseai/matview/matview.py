"""
Materialized Views & Refresh Strategies — SQLite-backed pre-computed aggregations
==================================================================================

A materialized view (MV) is a query result stored as a real table.  Unlike a
regular view — which is merely a saved SQL string that re-runs on every access —
an MV is a snapshot of the result, readable at microsecond speed regardless of
how expensive the underlying join or aggregation would be.

DB Architect notes:
  - Full refresh: DROP + INSERT AS SELECT restores exact consistency.  Simple
    to implement, correct by construction, and the only option when source rows
    can be updated or deleted (not just appended).  Cost: O(N * M) of the query.
  - Incremental refresh: scan only rows changed since last_refreshed and fold
    the delta into the MV.  Requires the source table to carry an updated_at
    timestamp or a monotone surrogate key so that "new rows since T" can be
    expressed as a range predicate.  Orders-of-magnitude faster on large tables
    when the write rate is low relative to table size.
  - Lazy refresh: the MV is marked stale on every write but only rebuilt on the
    next read.  Readers pay the refresh cost once; subsequent reads see the new
    data immediately.  Trade-off: first reader after a burst of writes bears the
    full refresh latency.
  - Eager refresh: rebuild happens inside the same write transaction that mutated
    the source.  Readers always see a fresh MV; writers pay extra latency.
    Suitable when reads heavily outnumber writes (e.g., Netflix homepage carousel
    served to millions of sessions from a MV refreshed every 5 minutes).
  - Staleness TTL: rather than tracking individual row changes, set a wall-clock
    TTL.  Any read older than TTL_SECONDS triggers a refresh.  Simple to
    implement; acceptable when "close enough" data is fine (e.g., trending charts
    that lag by a minute are still useful).

Production parallels:
  - Netflix homepage: genre carousels ("Top 10 in Sci-Fi", "New Arrivals") are
    pre-computed offline and stored as materialized views in EVCache (Redis) and
    Cassandra.  A Spark job refreshes them every 15 minutes; between refreshes
    readers see cached results.  This allows millions of homepage loads/sec with
    zero live JOIN overhead.
  - PostgreSQL MATERIALIZED VIEW with REFRESH MATERIALIZED VIEW CONCURRENTLY —
    refreshes the MV in the background without locking reads; old data is
    visible until the refresh completes.
  - Snowflake DYNAMIC TABLE: MVs that self-refresh when upstream tables change,
    tracked via an internal change-data-capture stream.  Netflix's data warehouse
    uses this pattern for recommendation feature tables.
  - dbt incremental models: only rows newer than the last run are merged into the
    target table, making warehouse transformations tractable at terabyte scale.
"""

import sqlite3
import time
from contextlib import contextmanager
from typing import Any


_DDL = """
CREATE TABLE IF NOT EXISTS movies (
    id        TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    genre     TEXT NOT NULL,
    year      INTEGER,
    director  TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id   TEXT NOT NULL,
    movie_id  TEXT NOT NULL REFERENCES movies(id),
    rating    REAL NOT NULL CHECK(rating BETWEEN 1.0 AND 5.0),
    PRIMARY KEY (user_id, movie_id)
);

-- Materialized view: one row per genre with aggregate stats and the top movie.
-- Refreshed from the live tables; reads are O(1) per genre.
CREATE TABLE IF NOT EXISTS mv_genre_stats (
    genre           TEXT PRIMARY KEY,
    avg_rating      REAL NOT NULL,
    movie_count     INTEGER NOT NULL,
    rating_count    INTEGER NOT NULL,
    top_movie_id    TEXT,
    top_movie_title TEXT
);

-- Materialized view: ranked list of movies by average rating.
-- Readers get instant top-N without aggregating the ratings table.
CREATE TABLE IF NOT EXISTS mv_top_movies (
    rank          INTEGER PRIMARY KEY,
    movie_id      TEXT NOT NULL,
    title         TEXT NOT NULL,
    genre         TEXT NOT NULL,
    avg_rating    REAL NOT NULL,
    rating_count  INTEGER NOT NULL
);

-- Refresh metadata: tracks staleness and refresh timing per view.
CREATE TABLE IF NOT EXISTS mv_meta (
    view_name        TEXT PRIMARY KEY,
    last_refreshed   TEXT,
    is_stale         INTEGER NOT NULL DEFAULT 1,
    refresh_count    INTEGER NOT NULL DEFAULT 0,
    total_refresh_ms REAL    NOT NULL DEFAULT 0.0
);

INSERT OR IGNORE INTO mv_meta (view_name) VALUES ('mv_genre_stats');
INSERT OR IGNORE INTO mv_meta (view_name) VALUES ('mv_top_movies');
"""

_GENRE_STATS_QUERY = """
WITH genre_agg AS (
    SELECT
        m.genre,
        AVG(r.rating)   AS avg_rating,
        COUNT(DISTINCT m.id) AS movie_count,
        COUNT(r.rating) AS rating_count
    FROM movies m
    LEFT JOIN ratings r ON r.movie_id = m.id
    GROUP BY m.genre
),
top_per_genre AS (
    SELECT
        m.genre,
        m.id    AS top_movie_id,
        m.title AS top_movie_title,
        AVG(r.rating) AS movie_avg
    FROM movies m
    JOIN ratings r ON r.movie_id = m.id
    GROUP BY m.id, m.genre
    HAVING COUNT(r.rating) >= 1
    ORDER BY movie_avg DESC
)
SELECT
    ga.genre,
    ROUND(ga.avg_rating, 4)   AS avg_rating,
    ga.movie_count,
    ga.rating_count,
    tg.top_movie_id,
    tg.top_movie_title
FROM genre_agg ga
LEFT JOIN (
    SELECT genre, top_movie_id, top_movie_title,
           ROW_NUMBER() OVER (PARTITION BY genre ORDER BY movie_avg DESC) AS rn
    FROM top_per_genre
) tg ON tg.genre = ga.genre AND tg.rn = 1
ORDER BY ga.avg_rating DESC
"""

_TOP_MOVIES_QUERY = """
SELECT
    ROW_NUMBER() OVER (ORDER BY AVG(r.rating) DESC) AS rank,
    m.id    AS movie_id,
    m.title AS title,
    m.genre AS genre,
    ROUND(AVG(r.rating), 4)  AS avg_rating,
    COUNT(r.rating)          AS rating_count
FROM movies m
JOIN ratings r ON r.movie_id = m.id
GROUP BY m.id
ORDER BY avg_rating DESC
"""


class MaterializedViewStore:
    """
    SQLite-backed materialized view manager for movie recommendation analytics.

    Real-world parallel:
      Netflix pre-computes genre carousels as materialized views refreshed on a
      15-minute schedule by Spark jobs.  Between refreshes, millions of homepage
      loads read from the snapshot with sub-millisecond latency, decoupling read
      throughput from the cost of aggregating billions of rating events.
    """

    VIEW_GENRE_STATS = "mv_genre_stats"
    VIEW_TOP_MOVIES  = "mv_top_movies"
    ALL_VIEWS = (VIEW_GENRE_STATS, VIEW_TOP_MOVIES)

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        if db_path == ":memory:":
            self._shared: sqlite3.Connection | None = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._shared.row_factory = sqlite3.Row
            self._shared.executescript(_DDL)
            self._shared.commit()
        else:
            self._shared = None
            with self._connect() as conn:
                conn.executescript(_DDL)
                conn.commit()

    @contextmanager
    def _connect(self):
        if self._shared is not None:
            yield self._shared
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Seed ─────────────────────────────────────────────────────────────────

    def load_seed(self, movies: list[dict], ratings: list[tuple]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id, title, genre, year, director) "
                "VALUES (:id, :title, :genre, :year, :director)",
                movies,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO ratings (user_id, movie_id, rating) VALUES (?,?,?)",
                [(r[0], r[1], r[2]) for r in ratings],
            )
            conn.commit()

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh_genre_stats(self) -> float:
        """Full rebuild of mv_genre_stats. Returns wall-clock duration in ms."""
        t0 = time.perf_counter()
        with self._connect() as conn:
            rows = conn.execute(_GENRE_STATS_QUERY).fetchall()
            conn.execute("DELETE FROM mv_genre_stats")
            conn.executemany(
                "INSERT INTO mv_genre_stats "
                "(genre, avg_rating, movie_count, rating_count, top_movie_id, top_movie_title) "
                "VALUES (?,?,?,?,?,?)",
                [(r["genre"], r["avg_rating"], r["movie_count"],
                  r["rating_count"], r["top_movie_id"], r["top_movie_title"])
                 for r in rows],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            conn.execute(
                "UPDATE mv_meta SET last_refreshed=datetime('now'), is_stale=0, "
                "refresh_count=refresh_count+1, total_refresh_ms=total_refresh_ms+? "
                "WHERE view_name=?",
                (elapsed, self.VIEW_GENRE_STATS),
            )
            conn.commit()
        return elapsed

    def refresh_top_movies(self, max_rank: int = 20) -> float:
        """Full rebuild of mv_top_movies up to max_rank entries."""
        t0 = time.perf_counter()
        with self._connect() as conn:
            rows = conn.execute(_TOP_MOVIES_QUERY).fetchall()
            conn.execute("DELETE FROM mv_top_movies")
            conn.executemany(
                "INSERT INTO mv_top_movies (rank, movie_id, title, genre, avg_rating, rating_count) "
                "VALUES (?,?,?,?,?,?)",
                [(i + 1, r["movie_id"], r["title"], r["genre"],
                  r["avg_rating"], r["rating_count"])
                 for i, r in enumerate(rows[:max_rank])],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            conn.execute(
                "UPDATE mv_meta SET last_refreshed=datetime('now'), is_stale=0, "
                "refresh_count=refresh_count+1, total_refresh_ms=total_refresh_ms+? "
                "WHERE view_name=?",
                (elapsed, self.VIEW_TOP_MOVIES),
            )
            conn.commit()
        return elapsed

    def refresh_all(self) -> dict[str, float]:
        """Rebuild all materialized views. Returns dict view_name -> refresh_ms."""
        return {
            self.VIEW_GENRE_STATS: self.refresh_genre_stats(),
            self.VIEW_TOP_MOVIES:  self.refresh_top_movies(),
        }

    def refresh_genre_for(self, genres: list[str]) -> float:
        """
        Incremental refresh: recompute only the given genres in mv_genre_stats.

        Faster than a full rebuild when a single rating event touches one genre
        in a table with thousands of genres — only that slice is recomputed.
        """
        if not genres:
            return 0.0
        placeholders = ",".join("?" * len(genres))
        t0 = time.perf_counter()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                WITH genre_agg AS (
                    SELECT m.genre,
                           AVG(r.rating)        AS avg_rating,
                           COUNT(DISTINCT m.id) AS movie_count,
                           COUNT(r.rating)      AS rating_count
                    FROM movies m
                    LEFT JOIN ratings r ON r.movie_id = m.id
                    WHERE m.genre IN ({placeholders})
                    GROUP BY m.genre
                ),
                top_per_genre AS (
                    SELECT m.genre, m.id AS top_movie_id, m.title AS top_movie_title,
                           AVG(r.rating) AS movie_avg
                    FROM movies m
                    JOIN ratings r ON r.movie_id = m.id
                    WHERE m.genre IN ({placeholders})
                    GROUP BY m.id
                    ORDER BY movie_avg DESC
                )
                SELECT ga.genre,
                       ROUND(ga.avg_rating, 4) AS avg_rating,
                       ga.movie_count,
                       ga.rating_count,
                       tg.top_movie_id,
                       tg.top_movie_title
                FROM genre_agg ga
                LEFT JOIN (
                    SELECT genre, top_movie_id, top_movie_title,
                           ROW_NUMBER() OVER (PARTITION BY genre ORDER BY movie_avg DESC) AS rn
                    FROM top_per_genre
                ) tg ON tg.genre = ga.genre AND tg.rn = 1
                """,
                genres + genres,
            ).fetchall()
            for r in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO mv_genre_stats "
                    "(genre, avg_rating, movie_count, rating_count, top_movie_id, top_movie_title) "
                    "VALUES (?,?,?,?,?,?)",
                    (r["genre"], r["avg_rating"], r["movie_count"],
                     r["rating_count"], r["top_movie_id"], r["top_movie_title"]),
                )
            elapsed = (time.perf_counter() - t0) * 1000
            conn.commit()
        return elapsed

    # ── Staleness ─────────────────────────────────────────────────────────────

    def mark_stale(self, *view_names: str) -> None:
        """Mark one or more views as stale. Pass no args to mark all."""
        targets = view_names if view_names else self.ALL_VIEWS
        with self._connect() as conn:
            for name in targets:
                conn.execute(
                    "UPDATE mv_meta SET is_stale=1 WHERE view_name=?", (name,)
                )
            conn.commit()

    def is_stale(self, view_name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_stale FROM mv_meta WHERE view_name=?", (view_name,)
            ).fetchone()
            return bool(row["is_stale"]) if row else True

    def get_meta(self, view_name: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mv_meta WHERE view_name=?", (view_name,)
            ).fetchone()
            return dict(row) if row else {}

    # ── Write path ────────────────────────────────────────────────────────────

    def add_rating(
        self, user_id: str, movie_id: str, rating: float, *, eager: bool = False
    ) -> None:
        """
        Insert or replace a rating.  Marks both MVs stale; if eager=True,
        immediately refreshes them so readers never see stale data.

        eager=True  consistent but slower writes (each write pays refresh cost).
        eager=False  fast writes; staleness resolved on next read (lazy) or
                      on next scheduled refresh.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating) VALUES (?,?,?)",
                (user_id, movie_id, rating),
            )
            conn.execute(
                "UPDATE mv_meta SET is_stale=1 WHERE view_name IN (?,?)",
                (self.VIEW_GENRE_STATS, self.VIEW_TOP_MOVIES),
            )
            conn.commit()
        if eager:
            self.refresh_all()

    # ── Read path ─────────────────────────────────────────────────────────────

    def get_genre_stats(self, *, lazy: bool = False) -> list[dict[str, Any]]:
        """
        Return genre stats from the materialized view.

        lazy=True: transparently refreshes the view if it is stale before
        returning, so the caller always sees current data without knowing
        about the underlying refresh mechanism.
        """
        if lazy and self.is_stale(self.VIEW_GENRE_STATS):
            self.refresh_genre_stats()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM mv_genre_stats ORDER BY avg_rating DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_top_movies(self, n: int = 10, *, lazy: bool = False) -> list[dict[str, Any]]:
        """Return top-N movies from the materialized view."""
        if lazy and self.is_stale(self.VIEW_TOP_MOVIES):
            self.refresh_top_movies()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM mv_top_movies ORDER BY rank LIMIT ?", (n,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Live (non-materialized) reads for benchmarking ────────────────────────

    def live_genre_stats(self) -> list[dict[str, Any]]:
        """Execute the full aggregation query against live tables every time."""
        with self._connect() as conn:
            rows = conn.execute(_GENRE_STATS_QUERY).fetchall()
            return [dict(r) for r in rows]

    def live_top_movies(self, n: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(_TOP_MOVIES_QUERY + " LIMIT ?", (n,)).fetchall()
            return [dict(r) for r in rows]

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, n_queries: int = 200) -> dict[str, Any]:
        """
        Time n_queries identical reads against live tables vs the materialized
        view.  The MV must be fresh before the benchmark starts.

        Returns dict with keys: live_total_ms, mv_total_ms, live_avg_ms,
        mv_avg_ms, speedup_x (how many times faster the MV is).
        """
        if self.is_stale(self.VIEW_GENRE_STATS):
            self.refresh_genre_stats()

        t0 = time.perf_counter()
        for _ in range(n_queries):
            self.live_genre_stats()
        live_total = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(n_queries):
            self.get_genre_stats(lazy=False)
        mv_total = (time.perf_counter() - t0) * 1000

        return {
            "n_queries":    n_queries,
            "live_total_ms": round(live_total, 3),
            "mv_total_ms":   round(mv_total, 3),
            "live_avg_ms":   round(live_total / n_queries, 4),
            "mv_avg_ms":     round(mv_total  / n_queries, 4),
            "speedup_x":     round(live_total / mv_total, 2) if mv_total > 0 else float("inf"),
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def movie_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    def rating_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
