"""
Range-Based Sharding — partition data by key ranges with pruning
================================================================

DB Architect notes:
  Consistent-hash sharding distributes data evenly by transforming every key
  into a numeric token, but it makes range queries expensive: "give me all
  ratings from 2010–2014" touches every shard because years are not
  clustered in hash space.  A query for year 2012 could land on any of the
  N shards because hash(2012) is unrelated to hash(2011) or hash(2013).

  Range sharding assigns each shard a contiguous, non-overlapping key
  interval.  The shard manager holds a sorted partition map and routes each
  key in O(S) (linear scan over S shards; O(log S) with binary search for
  large S).  Range queries that filter on the shard key are *pruned*: only
  the shards whose interval overlaps the predicate are contacted.  A query
  spanning one year-range shard out of four contacts 25 % of the hardware
  instead of 100 %.

  Partition pruning is the core reason column-oriented analytical systems
  use range partitioning by date.  ClickHouse PARTITION BY toYYYYMM(),
  BigQuery PARTITION BY DATE(), and Snowflake CLUSTER BY timestamp all
  implement this pattern so that OLAP queries with time-window predicates
  skip irrelevant partitions entirely.

  Trade-offs vs. consistent hashing:
    + Partition pruning slashes cross-shard I/O for range queries.
    + Sorted iteration across shards is possible with a merge-sort step.
    − Skewed key distributions cause uneven shard load (hot partitions).
    − Adding a shard requires bisecting an existing interval rather than
      simply inserting a new point on the hash ring (coarser rebalancing).

  Hot-partition splitting mitigates skew: when a shard exceeds a row-count
  threshold the manager bisects its range, migrating the upper half to a new
  shard.  Apache HBase region splitting, Google Spanner tablet splitting, and
  TiDB region split use the same mechanism, triggered automatically when a
  tablet exceeds a size threshold (typically 96 MB – 1 GB).

  This module uses SQLite in-memory databases as shard backends so all
  behaviour — routing, pruning, scatter-gather, and splitting — can be
  exercised without any network infrastructure.

Production parallels:
  - ClickHouse: PARTITION BY toYYYYMM(event_date) — month-per-partition;
    queries with date filters only open the matching part files on disk.
  - BigQuery: PARTITION BY DATE(timestamp) — daily partitions; queries
    billed for and physically scanning only the touched partitions.
  - Apache HBase: rows sorted by rowkey; each region owns a key range.
    Regions split automatically when they exceed ~10 GB.
  - TiDB / CockroachDB: distributed SQL engines that range-shard the
    primary-key space and automatically split hot ranges to balance load.
  - Snowflake: CLUSTER BY clustering keys reorder micro-partitions so that
    range predicates overlap fewer micro-partitions (pruning).
"""

import sys
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple


_DDL = """
CREATE TABLE IF NOT EXISTS movies (
    id       TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    genre    TEXT,
    year     INTEGER,
    director TEXT
);
CREATE TABLE IF NOT EXISTS ratings (
    user_id  TEXT NOT NULL,
    movie_id TEXT NOT NULL REFERENCES movies(id),
    score    REAL NOT NULL CHECK (score BETWEEN 0.0 AND 5.0),
    review   TEXT,
    PRIMARY KEY (user_id, movie_id)
);
CREATE INDEX IF NOT EXISTS idx_ratings_score ON ratings(score);
CREATE INDEX IF NOT EXISTS idx_movies_year   ON movies(year);
"""

_OPEN_HIGH = sys.maxsize  # sentinel for an unbounded upper range


class Shard:
    """
    A single range partition backed by one SQLite connection.

    Owning interval is [low, high] (both inclusive).  high == _OPEN_HIGH
    indicates an unbounded upper range (i.e. "year >= low").  The shard
    lazily opens its database on the first query, matching the behaviour of
    a real distributed database driver that opens a network socket only when
    a request is actually routed to that node.
    """

    def __init__(
        self,
        shard_id: int,
        low: int,
        high: int,
        db_path: str = ":memory:",
    ) -> None:
        self.shard_id = shard_id
        self.low = low
        self.high = high
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.queries_served = 0

    # ------------------------------------------------------------------ #
    # Internal connection management
    # ------------------------------------------------------------------ #

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_DDL)
        return self._conn

    # ------------------------------------------------------------------ #
    # Public query helpers
    # ------------------------------------------------------------------ #

    def execute(self, sql: str, params=()):
        self.queries_served += 1
        return self._get_conn().execute(sql, params)

    def executemany(self, sql: str, seq):
        return self._get_conn().executemany(sql, seq)

    def commit(self) -> None:
        self._get_conn().commit()

    def row_count(self, table: str) -> int:
        return self._get_conn().execute(
            f"SELECT COUNT(*) FROM {table}"  # noqa: S608 — table name is internal
        ).fetchone()[0]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #

    def label(self) -> str:
        high_str = "∞" if self.high >= _OPEN_HIGH else str(self.high)
        return f"shard-{self.shard_id} [{self.low}–{high_str}]"

    def __repr__(self) -> str:
        return f"Shard(id={self.shard_id}, range=[{self.low}, {self.high}])"


# ====================================================================== #
# Range Shard Manager
# ====================================================================== #

class RangeShardManager:
    """
    Routes data to shards based on non-overlapping year ranges.

    Maintains an ordered list of Shard objects sorted by their low boundary.
    Supports:
      - Exact routing  : find the shard that owns a given year key.
      - Pruned routing : find only the shards whose range overlaps [low, high].
      - Scatter-gather : return all shards for queries without a range filter.
      - Hot detection  : find shards whose row count exceeds a threshold.
      - Splitting      : bisect a hot shard's range and migrate its rows to two
                         new child shards.
    """

    def __init__(self, shards: Optional[List[Shard]] = None) -> None:
        self._shards: List[Shard] = sorted(shards or [], key=lambda s: s.low)

    # ------------------------------------------------------------------ #
    # Shard registry mutations
    # ------------------------------------------------------------------ #

    def add_shard(self, shard: Shard) -> None:
        self._shards.append(shard)
        self._shards.sort(key=lambda s: s.low)

    def remove_shard(self, shard_id: int) -> None:
        self._shards = [s for s in self._shards if s.shard_id != shard_id]

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def shard_for_key(self, year: int) -> Optional[Shard]:
        """Return the shard that owns this year, or None if unmapped."""
        for shard in self._shards:
            if shard.low <= year <= shard.high:
                return shard
        return None

    def shards_for_range(self, low: int, high: int) -> List[Shard]:
        """
        Return only the shards whose interval overlaps [low, high].

        This is the partition-pruning step.  Two intervals [a,b] and [c,d]
        overlap iff a <= d AND c <= b.  Any shard that fails this test is
        entirely outside the query's year range and can be skipped.
        """
        return [s for s in self._shards if s.low <= high and s.high >= low]

    def all_shards(self) -> List[Shard]:
        """Return all shards — used for scatter-gather queries."""
        return list(self._shards)

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    # ------------------------------------------------------------------ #
    # Hot-shard management
    # ------------------------------------------------------------------ #

    def hot_shards(self, threshold: int) -> List[Shard]:
        """Return shards whose ratings row count exceeds threshold."""
        return [s for s in self._shards if s.row_count("ratings") > threshold]

    def split_shard(self, shard: Shard) -> Tuple[Shard, Shard]:
        """
        Bisect a shard at the midpoint of its year range.

        Creates two new child shards with non-overlapping sub-ranges and
        migrates all movies and ratings rows from the parent shard.  The
        parent is then removed from the manager and closed.

        For an open upper bound (high == _OPEN_HIGH), the effective high is
        derived from the maximum year actually stored in the shard's movies
        table, falling back to low + 9 if the shard is empty.

        Returns the (lower_child, upper_child) pair.
        """
        if shard.high >= _OPEN_HIGH:
            row = shard.execute("SELECT MAX(year) FROM movies").fetchone()
            max_year = row[0] if row and row[0] is not None else shard.low + 9
            mid = (shard.low + max_year) // 2
            effective_high = _OPEN_HIGH  # upper child still open-ended
        else:
            mid = (shard.low + shard.high) // 2
            effective_high = shard.high

        next_id = max((s.shard_id for s in self._shards), default=-1) + 1
        lower = Shard(shard_id=next_id,     low=shard.low, high=mid,            db_path=":memory:")
        upper = Shard(shard_id=next_id + 1, low=mid + 1,   high=effective_high, db_path=":memory:")

        # Migrate movies
        movies = shard.execute(
            "SELECT id, title, genre, year, director FROM movies"
        ).fetchall()
        lower_movies, upper_movies = [], []
        for m in movies:
            (lower_movies if m["year"] <= mid else upper_movies).append(
                (m["id"], m["title"], m["genre"], m["year"], m["director"])
            )
        if lower_movies:
            lower.executemany(
                "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
                lower_movies,
            )
            lower.commit()
        if upper_movies:
            upper.executemany(
                "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
                upper_movies,
            )
            upper.commit()

        # Migrate ratings
        ratings = shard.execute(
            "SELECT r.user_id, r.movie_id, r.score, r.review, m.year "
            "FROM ratings r JOIN movies m ON r.movie_id = m.id"
        ).fetchall()
        lower_ratings, upper_ratings = [], []
        for r in ratings:
            (lower_ratings if r["year"] <= mid else upper_ratings).append(
                (r["user_id"], r["movie_id"], r["score"], r["review"])
            )
        if lower_ratings:
            lower.executemany(
                "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) VALUES (?,?,?,?)",
                lower_ratings,
            )
            lower.commit()
        if upper_ratings:
            upper.executemany(
                "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) VALUES (?,?,?,?)",
                upper_ratings,
            )
            upper.commit()

        # Swap old shard for two new children
        self.remove_shard(shard.shard_id)
        self.add_shard(lower)
        self.add_shard(upper)
        shard.close()

        return lower, upper

    def close_all(self) -> None:
        for shard in self._shards:
            shard.close()


# ====================================================================== #
# Movie database sharded by release year
# ====================================================================== #

class ShardedMovieDB:
    """
    Movie database partitioned across four year-range shards.

    Default partition map
    ─────────────────────
      shard-0: ≤ 1999  (classics)
      shard-1: 2000–2009
      shard-2: 2010–2019  (typically the hottest shard given seed data)
      shard-3: 2020+

    Each shard stores only the movies and ratings that fall within its year
    interval, giving complete data locality for year-range queries.

    Two query patterns are exposed to contrast pruning vs. scatter-gather:

      query_range_pruned   — accepts [year_low, year_high] and contacts only
                             the overlapping shards.  A query for 2010–2019
                             touches shard-2 alone (1 of 4 shards = 25 % I/O).

      query_scatter_gather — filters on score (not the shard key), so every
                             shard must be contacted to find all high-rated
                             movies regardless of their release year.
    """

    def __init__(self, shards: Optional[List[Shard]] = None) -> None:
        if shards is None:
            shards = [
                Shard(0, 0,    1999,       ":memory:"),
                Shard(1, 2000, 2009,       ":memory:"),
                Shard(2, 2010, 2019,       ":memory:"),
                Shard(3, 2020, _OPEN_HIGH, ":memory:"),
            ]
        self.manager = RangeShardManager(shards)

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #

    def seed(self, movies: list, ratings: list) -> None:
        """
        Route each movie to its year-range shard, then route its ratings.

        Movies without a matching shard (year outside all defined ranges)
        are silently dropped — in production a routing error would be raised
        so the operator can extend the partition map before ingesting the
        outlier data.
        """
        movie_year: dict = {m["id"]: m["year"] for m in movies}

        for movie in movies:
            shard = self.manager.shard_for_key(movie["year"])
            if shard is None:
                continue
            shard.execute(
                "INSERT OR IGNORE INTO movies (id,title,genre,year,director) "
                "VALUES (?,?,?,?,?)",
                (movie["id"], movie["title"], movie["genre"],
                 movie["year"], movie["director"]),
            )
            shard.commit()

        for user_id, movie_id, score, review in ratings:
            year = movie_year.get(movie_id)
            if year is None:
                continue
            shard = self.manager.shard_for_key(year)
            if shard is None:
                continue
            shard.execute(
                "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) "
                "VALUES (?,?,?,?)",
                (user_id, movie_id, score, review),
            )
            shard.commit()

    def seed_synthetic(self, n: int = 500, seed_val: int = 42) -> None:
        """
        Insert n synthetic ratings distributed across all shards by
        randomly picking a movie from each shard's existing movie table.

        Uses a deterministic pseudo-random sequence for reproducibility.
        """
        import random
        rng = random.Random(seed_val)
        scores = [round(rng.uniform(1.0, 5.0), 1) for _ in range(n)]
        uid_pool = [f"ux{i:04d}" for i in range(200)]

        all_movies: List[Tuple[str, int]] = []
        for shard in self.manager.all_shards():
            rows = shard.execute("SELECT id, year FROM movies").fetchall()
            all_movies.extend((r["id"], r["year"]) for r in rows)

        if not all_movies:
            return

        for i in range(n):
            movie_id, year = rng.choice(all_movies)
            user_id = rng.choice(uid_pool)
            score = scores[i]
            shard = self.manager.shard_for_key(year)
            if shard is None:
                continue
            try:
                shard.execute(
                    "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) "
                    "VALUES (?,?,?,?)",
                    (user_id, movie_id, score, "synthetic"),
                )
                shard.commit()
            except sqlite3.IntegrityError:
                pass

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def query_range_pruned(
        self, year_low: int, year_high: int
    ) -> Tuple[List[dict], int, float]:
        """
        Range query with partition pruning.

        Returns (results, shards_touched, elapsed_ms).
        Only shards whose year interval overlaps [year_low, year_high] are
        opened; the rest are never contacted.
        """
        shards = self.manager.shards_for_range(year_low, year_high)
        t0 = time.monotonic()
        results: List[dict] = []
        for shard in shards:
            rows = shard.execute(
                "SELECT m.title, m.genre, m.year, r.user_id, r.score "
                "FROM ratings r JOIN movies m ON r.movie_id = m.id "
                "WHERE m.year BETWEEN ? AND ? "
                "ORDER BY r.score DESC",
                (year_low, year_high),
            ).fetchall()
            results.extend(dict(r) for r in rows)
        elapsed = (time.monotonic() - t0) * 1000
        return results, len(shards), elapsed

    def query_range_full_scan(
        self, year_low: int, year_high: int
    ) -> Tuple[List[dict], int, float]:
        """
        Apply the same year-range predicate as query_range_pruned but
        WITHOUT pruning — every shard is contacted.

        Used to benchmark the cost of skipping partition pruning: the
        result set is identical but the number of shards touched equals
        the total shard count.  This simulates what happens in a hash-
        sharded cluster where the optimizer cannot prune by key range.

        Returns (results, shards_touched, elapsed_ms).
        """
        shards = self.manager.all_shards()
        t0 = time.monotonic()
        results: List[dict] = []
        for shard in shards:
            rows = shard.execute(
                "SELECT m.title, m.genre, m.year, r.user_id, r.score "
                "FROM ratings r JOIN movies m ON r.movie_id = m.id "
                "WHERE m.year BETWEEN ? AND ? "
                "ORDER BY r.score DESC",
                (year_low, year_high),
            ).fetchall()
            results.extend(dict(r) for r in rows)
        elapsed = (time.monotonic() - t0) * 1000
        return results, len(shards), elapsed

    def query_scatter_gather(
        self, min_score: float
    ) -> Tuple[List[dict], int, float]:
        """
        Query on score (not the shard key) — every shard must be contacted.

        Returns (results, shards_touched, elapsed_ms).
        The results are merged in memory after fanning out to all shards.
        """
        shards = self.manager.all_shards()
        t0 = time.monotonic()
        results: List[dict] = []
        for shard in shards:
            rows = shard.execute(
                "SELECT m.title, m.genre, m.year, r.user_id, r.score "
                "FROM ratings r JOIN movies m ON r.movie_id = m.id "
                "WHERE r.score >= ? "
                "ORDER BY r.score DESC",
                (min_score,),
            ).fetchall()
            results.extend(dict(r) for r in rows)
        results.sort(key=lambda r: r["score"], reverse=True)
        elapsed = (time.monotonic() - t0) * 1000
        return results, len(shards), elapsed

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def shard_stats(self) -> List[dict]:
        """Return per-shard row counts and query counters."""
        return [
            {
                "shard_id": s.shard_id,
                "label": s.label(),
                "movies": s.row_count("movies"),
                "ratings": s.row_count("ratings"),
                "queries_served": s.queries_served,
            }
            for s in self.manager.all_shards()
        ]

    def total_ratings(self) -> int:
        return sum(s.row_count("ratings") for s in self.manager.all_shards())

    def total_movies(self) -> int:
        return sum(s.row_count("movies") for s in self.manager.all_shards())

    def close(self) -> None:
        self.manager.close_all()
