"""
Catalog Browse Indexing — B-tree, Composite, Covering, and Partial Indexes
==========================================================================

DB Architect notes:
  Demo 12 indexed the *recommendation* path — queries that filter ratings by
  score or user_id to rank what to show next.  This module indexes the
  *browse* path instead: the "Genres", "Top Rated", and "New & Trending" rows
  a viewer scrolls through before a recommendation model ever runs.  Same four
  index types, different access pattern — and that difference changes which
  index actually gets chosen by the planner.

  B-tree index (default in every RDBMS):
    A balanced tree sorted by the indexed expression.  The "Top Rated" shelf
    runs `WHERE rating_avg >= 4.5` — a range predicate that walks consecutive
    leaf pages after one O(log N) descent instead of scanning the whole table.

  Composite index (multi-column B-tree):
    Sorts by (col_A, col_B, …) together.  A genre row filtered to a decade —
    `WHERE genre = ? AND year >= ?` — matches the (genre, year) index because
    genre is the LEFTMOST column.  `WHERE genre = ?` alone still uses it
    (left prefix); `WHERE year >= ?` alone cannot, because year values for
    different genres are interleaved inside the index.

  Covering index:
    A genre carousel renders only title and score — never the full row.  An
    index on (genre, title, rating_avg) satisfies that projection entirely
    from index pages; the engine never seeks into the table heap.  This is
    the difference between "USING INDEX" and "USING COVERING INDEX" in
    EXPLAIN QUERY PLAN.

  Partial index (WHERE clause baked into the index):
    The "New & Trending" shelf only ever asks for recent releases, so an
    index on year WHERE year >= 2020 stores a small fraction of the catalog.
    SQLite proves a query is safe to answer from a partial index only when the
    query's WHERE clause is *syntactically identical* to the index's WHERE
    clause — `WHERE year >= 2020` matches and uses the index, while a looser
    `WHERE year >= 2010` (which the index cannot fully answer — rows from
    2010-2019 are simply absent from it) falls back to a full table scan.

Production parallels:
  - Netflix / Disney+ Browse API: PostgreSQL composite indexes on
    (genre_id, release_year) and (genre_id, popularity_rank) back every
    horizontal row on the home screen; covering indexes on the card
    projection (id, title, art_asset_id, score) avoid heap fetches at the
    scale of hundreds of millions of row-renders per day.
  - Elasticsearch / OpenSearch: the catalog search box uses inverted indexes
    (the text-search analogue of a covering index — match AND rank from the
    index alone) over title and synopsis fields.
  - "New & Trending" / "Coming Soon" rows map directly onto partial indexes —
    BRIN or partial B-tree in Postgres, TTL-bounded materialized views in
    ClickHouse — because the predicate (recent release) is fixed and the
    indexed subset stays small even as the full catalog grows.
"""

import random
import sqlite3
import time
from typing import Dict, List

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS catalog (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    genre      TEXT,
    year       INTEGER,
    rating_avg REAL,
    popularity INTEGER
);
"""

# Canonical queries — each tied to one index type for consistent demo/test use
QUERY_BTREE_TOP_RATED = (
    "SELECT id, title FROM catalog WHERE rating_avg >= 4.5"
)
QUERY_COMPOSITE_BOTH = (
    "SELECT id, title FROM catalog WHERE genre = ? AND year >= ?"
)
QUERY_COMPOSITE_LEFT = (
    "SELECT id, title FROM catalog WHERE genre = ?"
)
QUERY_COMPOSITE_RIGHT_ONLY = (
    "SELECT id, title FROM catalog WHERE year >= ?"
)
QUERY_COVERING_CAROUSEL = (
    "SELECT title, rating_avg FROM catalog WHERE genre = ?"
)
QUERY_PARTIAL_MATCH = (
    "SELECT id, title FROM catalog WHERE year >= 2020"
)
QUERY_PARTIAL_NO_MATCH = (
    "SELECT id, title FROM catalog WHERE year >= 2010"
)

_NEW_RELEASES_CUTOFF = 2020


class CatalogIndexDemo:
    """
    Manages a SQLite "browse catalog" table to demonstrate how index choice
    follows the *access pattern* of a streaming service's discovery surface
    (genre rows, Top Rated, New & Trending) rather than its recommendation
    pipeline.

    The instance starts with NO user-defined indexes so before/after
    comparisons are visible.  Each create_*_index / drop_*_index pair installs
    or removes exactly one index type in isolation.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def seed(self, movies: list, ratings: list) -> None:
        """Build the browse catalog from seed_data.py movies and ratings."""
        sums: Dict[str, List[float]] = {}
        for uid, mid, score, _ in ratings:
            sums.setdefault(mid, []).append(score)

        rows = []
        for i, movie in enumerate(movies):
            scores = sums.get(movie["id"], [])
            avg = round(sum(scores) / len(scores), 2) if scores else None
            popularity = len(scores) * 10 + (i % 7)
            rows.append((
                movie["id"], movie["title"], movie["genre"], movie["year"],
                avg, popularity,
            ))
        self._conn.executemany(
            "INSERT OR IGNORE INTO catalog (id, title, genre, year, rating_avg, popularity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def seed_large(self, n_titles: int = 4000, rng_seed: int = 42) -> None:
        """
        Generate a reproducible synthetic catalog large enough for the query
        optimizer to prefer index scans over full table scans.
        """
        rng = random.Random(rng_seed)
        genres = ["sci-fi", "action", "thriller", "drama", "animation", "horror"]

        rows = [
            (
                f"cx{i:05d}",
                f"Synthetic Title {i}",
                rng.choice(genres),
                rng.randint(1970, 2024),
                round(rng.uniform(1.0, 5.0), 1),
                rng.randint(0, 1000),
            )
            for i in range(n_titles)
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO catalog (id, title, genre, year, rating_avg, popularity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
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
        """B-tree index on catalog.rating_avg — accelerates the Top Rated shelf."""
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_btree_rating ON catalog(rating_avg)"
        )
        self._conn.commit()

    def drop_btree_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_btree_rating")
        self._conn.commit()

    def create_composite_index(self) -> None:
        """Composite B-tree on (genre, year) — supports genre+decade browse rows."""
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_composite_genre_year "
            "ON catalog(genre, year)"
        )
        self._conn.commit()

    def drop_composite_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_composite_genre_year")
        self._conn.commit()

    def create_covering_index(self) -> None:
        """
        Covering index on (genre, title, rating_avg): every column the genre
        carousel projects lives in the index, so the engine never seeks into
        the table heap to render the row.
        """
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_covering_genre_carousel "
            "ON catalog(genre, title, rating_avg)"
        )
        self._conn.commit()

    def drop_covering_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_covering_genre_carousel")
        self._conn.commit()

    def create_partial_index(self) -> None:
        """
        Partial index on year WHERE year >= 2020: stores only the slice of the
        catalog the "New & Trending" shelf ever queries.  SQLite's planner uses
        it only when the query's WHERE clause is syntactically identical to the
        index's — 'year >= 2020' qualifies; a looser 'year >= 2010' does not,
        because the index simply does not contain rows from 2010-2019.
        """
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_partial_new_releases "
            f"ON catalog(year) WHERE year >= {_NEW_RELEASES_CUTOFF}"
        )
        self._conn.commit()

    def drop_partial_index(self) -> None:
        self._conn.execute("DROP INDEX IF EXISTS idx_partial_new_releases")
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

    def row_count(self, table: str = "catalog") -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()
