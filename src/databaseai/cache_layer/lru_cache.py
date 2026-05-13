"""
Caching Layer — LRU Cache with TTL over a relational database
=============================================================

DB Architect notes:
  - A cache sits between the application and the database, storing the results
    of expensive queries so subsequent identical requests can be served from
    fast memory instead of disk I/O.
  - LRU (Least Recently Used) eviction removes the entry that was accessed
    least recently when the cache is full.  This is optimal when queries follow
    a power-law distribution (a small subset of keys drives most traffic —
    Zipf's law).  An OrderedDict gives O(1) move_to_end and O(1) popitem(last=False).
  - TTL (Time-To-Live) bounds data staleness: cached rows are automatically
    invalidated after a configurable window.  Short TTL = fresher data, more
    cache misses.  Long TTL = stale risk, higher hit rate.  Netflix uses ~60 s
    TTL for homepage carousels; financial tickers use < 1 s.
  - The two primary cache metrics are hit rate (fraction of requests served
    from cache) and miss penalty (extra latency incurred on a cold fetch).
  - Three caching policies for writes:
    * Write-through: writes go to both cache and DB simultaneously.
    * Write-around: writes bypass the cache — only reads are cached.
    * Write-back: writes go to cache first, flushed to DB later.
    This module implements write-around with targeted invalidation.
  - Targeted invalidation is more precise than a full cache clear: only keys
    that reference the modified data are dropped.

Production parallels:
  - Netflix caches homepage carousels in Redis with a ~60 s TTL.
  - Memcached was invented at LiveJournal in 2003 to cache MySQL query results.
  - PostgreSQL's shared_buffers is an internal LRU page cache; Redis in front
    caches at the query-result level — the two are complementary and often stacked.
"""

import sqlite3
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Optional


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl if ttl > 0 else float("inf")

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class LRUCache:
    """
    Thread-unsafe in-process LRU cache with per-entry TTL.

    Implemented with an OrderedDict where the most-recently-used entry is
    always at the tail.  On every access the entry moves to the tail; on
    eviction the head (LRU) entry is removed.  Both operations are O(1).
    """

    def __init__(self, capacity: int, ttl_seconds: float = 60.0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return False, None
        if entry.is_expired():
            del self._store[key]
            self._misses += 1
            return False, None
        self._store.move_to_end(key)
        self._hits += 1
        return True, entry.value

    def put(self, key: str, value: Any, ttl_override: Optional[float] = None) -> None:
        ttl = ttl_override if ttl_override is not None else self._ttl
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = _CacheEntry(value, ttl)
        else:
            if len(self._store) >= self._cap:
                self._store.popitem(last=False)
            self._store[key] = _CacheEntry(value, ttl)

    def invalidate(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def capacity(self) -> int:
        return self._cap

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "size": self.size,
            "capacity": self._cap,
        }

    def keys(self) -> list[str]:
        return list(self._store.keys())


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

CREATE INDEX IF NOT EXISTS idx_movies_genre  ON movies  (genre);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings (movie_id);
"""


class CachedMovieDB:
    """
    SQLite movie database fronted by an LRU + TTL cache.

    Write policy: write-around with targeted invalidation.  Writes skip the
    cache but evict only the keys that aggregate over the modified rows,
    leaving unrelated hot entries intact.
    """

    def __init__(self, db_path=":memory:", cache_capacity=128,
                 ttl_seconds=60.0, query_delay=0.0) -> None:
        self._db_path = db_path
        self._query_delay = query_delay
        self.cache = LRUCache(capacity=cache_capacity, ttl_seconds=ttl_seconds)
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            self._shared_conn = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self):
        if self._shared_conn is not None:
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _simulate_latency(self) -> None:
        if self._query_delay > 0:
            time.sleep(self._query_delay)

    def seed(self, movies, ratings) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id, title, genre, year, director) "
                "VALUES (:id, :title, :genre, :year, :director)", movies)
            conn.executemany(
                "INSERT OR IGNORE INTO ratings (user_id, movie_id, score, review) "
                "VALUES (?, ?, ?, ?)", ratings)
        self.cache.clear()

    def add_rating(self, user_id, movie_id, score, review="") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ratings (user_id, movie_id, score, review) "
                "VALUES (?, ?, ?, ?)", (user_id, movie_id, score, review))
        movie = self._fetch_movie(movie_id)
        if movie:
            self.cache.invalidate(f"top_rated:{movie['genre']}")
        self.cache.invalidate(f"avg_rating:{movie_id}")
        self.cache.invalidate("genre_stats")

    def _fetch_movie(self, movie_id):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
        return dict(row) if row else None

    def top_rated_by_genre(self, genre, limit=5):
        key = f"top_rated:{genre}"
        hit, cached = self.cache.get(key)
        if hit:
            return cached
        self._simulate_latency()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT m.id, m.title, m.genre, m.year,
                          ROUND(AVG(r.score), 2) AS avg_score,
                          COUNT(r.score)          AS rating_count
                   FROM movies m
                   JOIN ratings r ON m.id = r.movie_id
                   WHERE m.genre = ?
                   GROUP BY m.id
                   HAVING rating_count >= 1
                   ORDER BY avg_score DESC, rating_count DESC
                   LIMIT ?""",
                (genre, limit),
            ).fetchall()
        result = [dict(r) for r in rows]
        self.cache.put(key, result)
        return result

    def average_rating(self, movie_id):
        key = f"avg_rating:{movie_id}"
        hit, cached = self.cache.get(key)
        if hit:
            return cached
        self._simulate_latency()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ROUND(AVG(score), 2) FROM ratings WHERE movie_id = ?",
                (movie_id,),
            ).fetchone()
        result = row[0]
        self.cache.put(key, result)
        return result

    def genre_stats(self):
        key = "genre_stats"
        hit, cached = self.cache.get(key)
        if hit:
            return cached
        self._simulate_latency()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT m.genre,
                          COUNT(DISTINCT m.id)   AS movie_count,
                          COUNT(r.score)         AS rating_count,
                          ROUND(AVG(r.score), 2) AS avg_rating
                   FROM movies m
                   LEFT JOIN ratings r ON m.id = r.movie_id
                   GROUP BY m.genre
                   ORDER BY avg_rating DESC""",
            ).fetchall()
        result = [dict(r) for r in rows]
        self.cache.put(key, result)
        return result

    def movie_count(self):
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    def rating_count(self):
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
