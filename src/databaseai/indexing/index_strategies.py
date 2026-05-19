"""
Indexing Strategies — B-tree, Composite, Covering, and Partial Indexes
=======================================================================

DB Architect notes:
  Without an index, every query that filters or sorts must perform a full table
  scan — reading every row regardless of how many match the predicate.  At
  Netflix scale (hundreds of billions of rating events) that becomes untenable.
  An index is a separate, automatically-maintained data structure that trades a
  small write-time overhead for dramatically faster reads.

  B-tree index (default in every RDBMS):
    Stores column values in a balanced tree sorted by the indexed expression.
    Lookup is O(log N) instead of O(N).  Range predicates (score >= 4.5) walk
    the leaf pages sequentially after a single tree descent — ideal for ordered
    scans and range queries.

  Composite index (multi-column B-tree):
    Sorts by (col_A, col_B, …) together.  Queries must match the LEFTMOST
    prefix of the column list to use the index efficiently.  The index on
    (genre, year) can accelerate `WHERE genre = ? AND year >= ?` and
    `WHERE genre = ?` but NOT `WHERE year >= ?` alone because year values for
    different genres are interleaved in the index B-tree.

  Covering index:
    A query is "covered" when every column it touches — both in WHERE and in
    SELECT — is present in the index.  The engine never visits the main table
    heap: all data comes directly from the index pages.  This halves or more the
    I/O for read-heavy recommendation queries that project only a few columns.

  Partial index (WHERE clause on the index):
    Indexes only the rows matching a fixed predicate, keeping the index small.
    A partial index on ratings WHERE score >= 4.0 stores ~30 % of the rows —
    writes to low-score rows never update this index, and the index itself fits
    in fewer memory pages.  The optimizer uses it only when the query's filter
    is provably a subset of the index predicate.

Production parallels:
  - PostgreSQL: B-tree (default), Hash, GiST (geometry/full-text), GIN
    (arrays/JSONB), BRIN (time-series append-only heaps).
  - Netflix recommendation pipeline: composite indexes on (user_id, timestamp)
    in Cassandra for ordered watch-history lookups; partial indexes in
    PostgreSQL on high-engagement content to keep VACUUM fast on cold rows.
  - MySQL InnoDB: covering indexes via "index-only scan" when EXPLAIN shows
    "Using index" in the Extra column — same concept as SQLite's USING COVERING
    INDEX.
  - Elasticsearch inverted-index: the extreme case of a covering structure for
    full-text search over movie descriptions and tags.
"""

import random
import sqlite3
import time
from typing import Dict, List

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=OFF;

CREATE TABLE IF NOT EXISTS movies_idx (
    id       TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    genre    TEXT,
    year     INTEGER,
    director TEXT
);

CREATE TABLE IF NOT EXISTS ratings_idx (
    row_id   INTEGER PRIMARY KEY,
    user_id  TEXT NOT NULL,
    movie_id TEXT NOT NULL,
    score    REAL NOT NULL,
    review   TEXT
);
"""

# Canonical queries — each tied to one index type for consistent demo/test use
QUERY_BTREE = (
    "SELECT user_id, movie_id, score FROM ratings_idx WHERE score >= 4.5"
)
QUERY_COMPOSITE_BOTH = (
    "SELECT id, title FROM movies_idx WHERE genre = ? AND year >= ?"
)
QUERY_COMPOSITE_LEFT = (
    "SELECT id, title FROM movies_idx WHERE genre = ?"
)
QUERY_COMPOSITE_RIGHT_ONLY = (
    "SELECT id, title FROM movies_idx WHERE year >= ?"
)
QUERY_COVERING = (
    "SELECT user_id, score FROM ratings_idx WHERE user_id = ?"
)
QUERY_PARTIAL_MATCH = (
    "SELECT user_id, score FROM ratings_idx WHERE score >= 4.0"
)
QUERY_PARTIAL_NO_MATCH = (
    "SELECT user_id, score FROM ratings_idx WHERE score >= 3.0"
)


class IndexingDemo:
    """
    Manages a SQLite database to demonstrate the effect of each index type on
    query plans (via EXPLAIN QUERY PLAN) and on raw execution latency.

    The instance intentionally starts with NO user-defined indexes so that
    before/after comparisons are clear.  Each create_*_index / drop_*_index
    pair lets the caller install or remove a single index type in isolation.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def seed(self, movies: list, ratings: list) -> None:
        """Load seed data from seed_data.py into the indexing demo tables."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO movies_idx (id, title, genre, year, director) "
            "VALUES (:id, :title, :genre, :year, :director)",
            movies,
        )
        # ratings tuples: (user_id, movie_id, score, review)
        self._conn.executemany(
            "INSERT OR IGNORE INTO ratings_idx (user_id, movie_id, score, review) "
            "VALUES (?, ?, ?, ?)",
            ratings,
        )
        self._conn.commit()

    def seed_large(self, n_ratings: int = 4000, rng_seed: int = 42) -> None:
        """
        Generate a reproducible synthetic dataset large enough for the query
        optimizer to prefer index scans over full table scans.
        """
        rng = random.Random(rng_seed)
        genres = ["sci-fi", "action", "thriller", "drama", "animation", "horror"]

        movie_ids = [f"mx{i:04d}" for i in range(200)]
        self._conn.executemany(
            "INSERT OR IGNORE INTO movies_idx (id, title, genre, year, director) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (mid, f"Synthetic Movie {i}", rng.choice(genres),
                 rng.randint(1990, 2024), f"Director_{i % 20}")
                for i, mid in enumerate(movie_ids)
            ],
        )

        user_ids = [f"ux{i:04d}" for i in range(500)]
        rows = [
            (rng.choice(user_ids), rng.choice(movie_ids),
             round(rng.uniform(1.0, 5.0), 1), "")
            for _ in range(n_ratings)
        ]
        self._conn.executemany(
            "INSERT INTO ratings_idx (user_id, movie_id, score, review) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def analyze(self) -> None:
        """Run ANALYZE to update table statistics used by the query planner."""
        self._conn.execute("ANALYZE")
        self._conn.commit()

    # ------------------------------------------------------------------
    # EXPLAIN QUERY PLAN helpers
    # ------------------------------------------------------------------

    def explain(self, sql: str, params: tuple = ()) -> str:
        """Return the EXPLAIN QUERY PLAN output as a single string."""
        # Temporarily disable row_factory so row[-1] works on plain tuples.
        old_factory = self._conn.row_factory
        self._conn.row_factory = None
        try:
            rows = self._conn.execute(
                f"EXPLAIN QUERY PLAN {sql}", params
            ).fetchall()
        finally:
            self._conn.row_factory = old_factory
        return "\n".join(str(row[-1]) for row in rows)

    def uses_index(self, sql: str, params: tuple = ()) -> bool:
        """Return True if the query plan references any user-defined index."""
        plan = self.explain(sql, params)
        return "USING INDEX" in plan or "USING COVERING INDEX" in plan

    def uses_covering_index(self, sql: str, params: tuple = ()) -> bool:
        """Return True when the plan shows USING COVERING INDEX."""
        return "USING COVERING INDEX" in self.explain(sql, params)

    def uses_full_scan(self, sql: str, params: tuple = ()) -> bool:
        """Return True when no index is used (full table scan)."""
        return not self.uses_index(sql, params)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def time_query(self, sql: str, params: tuple = (), repeat: int = 300) -> float:
        """Return average execution time in microseconds over `repeat` runs."""
        start = time.perf_counter()
        for _ in range(repeat):
            self._conn.execute(sql, params).fetchall()
        return (time.perf_counter() - start) / repeat * 1_000_000

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def create_btree_index(self) -> None:
        """B-tree index on ratings_idx.score — accelerates range queries."""
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_btree_score ON ratings_idx(score)"
        )
        self._conn.commit()

    def drop_btree_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_btree_score")
        self._conn.commit()

    def create_composite_index(self) -> None:
        """Composite B-tree on (genre, year) — supports genre+year filters."""
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_composite_genre_year "
            "ON movies_idx(genre, year)"
        )
        self._conn.commit()

    def drop_composite_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_composite_genre_year")
        self._conn.commit()

    def create_covering_index(self) -> None:
        """
        Covering index on (user_id, score): both columns needed by QUERY_COVERING
        live in the index, so the engine never touches the main table heap.
        """
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_covering_user_score "
            "ON ratings_idx(user_id, score)"
        )
        self._conn.commit()

    def drop_covering_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_covering_user_score")
        self._conn.commit()

    def create_partial_index(self) -> None:
        """
        Partial index on score WHERE score >= 4.0: stores only high-rating rows.
        The engine uses it only when the query predicate is a subset of
        'score >= 4.0', e.g., 'WHERE score >= 4.5' qualifies; 'WHERE score >= 3.0'
        does not because it would need rows outside the partial index.
        """
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_partial_high_score "
            "ON ratings_idx(score) WHERE score >= 4.0"
        )
        self._conn.commit()

    def drop_partial_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_partial_high_score")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_indexes(self) -> List[Dict]:
        """Return all user-created indexes (SQLite auto-indexes excluded)."""
        rows = self._conn.execute(
            "SELECT name, tbl_name AS table_name "
            "FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def row_count(self, table: str) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()
