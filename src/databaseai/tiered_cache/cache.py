"""
DB Architect note: Production systems rarely rely on a single cache tier.  Netflix
EVCache, for example, runs Memcached at two layers — a per-rack L1 (low latency,
small capacity, short TTL) and a per-region L2 (higher latency, large capacity,
long TTL).  A cache miss cascades L1 → L2 → origin (Cassandra / MySQL), and each
promoted entry is written back toward the client so that the next request is served
from the closest tier.  This module reproduces that hierarchy using an in-process
OrderedDict as L1 and a SQLite table as L2, with the source DB in a third SQLite
connection.  Write-through keeps every tier consistent on updates; targeted
invalidation removes only the affected keys so unrelated hot entries survive.
"""

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# L1 — in-process LRU with per-entry TTL
# ---------------------------------------------------------------------------

class L1Cache:
    """In-process LRU dict with configurable capacity and TTL."""

    def __init__(self, capacity: int = 64, ttl: float = 30.0):
        if capacity < 1:
            raise ValueError("L1 capacity must be >= 1")
        self._cap = capacity
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def ttl(self) -> float:
        return self._ttl

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return False, None
            value, expires_at = self._store[key]
            if self._ttl > 0 and time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return False, None
            self._store.move_to_end(key)
            self._hits += 1
            return True, value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            expires_at = (time.monotonic() + self._ttl) if self._ttl > 0 else 1e18
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            if len(self._store) > self._cap:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "capacity": self._cap,
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate": self._hits / total if total else 0.0,
            }

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())


# ---------------------------------------------------------------------------
# L2 — SQLite-backed persistent cache (simulates a shared Memcached/Redis tier)
# ---------------------------------------------------------------------------

class L2Cache:
    """SQLite-backed cache simulating a shared warm tier (Memcached / Redis)."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS l2_cache (
            cache_key  TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_l2_expires ON l2_cache(expires_at);
    """

    def __init__(self, db_path: str = ":memory:", ttl: float = 300.0):
        self._ttl = ttl
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self._DDL)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def ttl(self) -> float:
        return self._ttl

    def get(self, key: str) -> tuple[bool, Any]:
        now = time.monotonic()
        with self._lock:
            self._conn.execute("DELETE FROM l2_cache WHERE expires_at < ?", (now,))
            row = self._conn.execute(
                "SELECT value_json FROM l2_cache WHERE cache_key = ? AND expires_at >= ?",
                (key, now),
            ).fetchone()
            if row is None:
                self._misses += 1
                return False, None
            self._hits += 1
            return True, json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        expires_at = (time.monotonic() + self._ttl) if self._ttl > 0 else 1e18
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO l2_cache (cache_key, value_json, expires_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(value), expires_at),
            )
            self._conn.commit()

    def invalidate(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM l2_cache WHERE cache_key = ?", (key,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def invalidate_prefix(self, prefix: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM l2_cache WHERE cache_key LIKE ?", (prefix + "%",)
            )
            self._conn.commit()
            return cur.rowcount

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM l2_cache")
            self._conn.commit()

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        now = time.monotonic()
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM l2_cache WHERE expires_at >= ?", (now,)
            ).fetchone()[0]
            total = self._hits + self._misses
            return {
                "size": count,
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate": self._hits / total if total else 0.0,
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# TieredCache — read-through / write-through facade over L1 + L2
# ---------------------------------------------------------------------------

class TieredCache:
    """
    Read-through, write-through two-tier cache.  Reads cascade L1 → L2 → source;
    a hit at any tier short-circuits and back-fills closer tiers.  Writes go to
    L1 and L2 immediately; source writes are the caller's responsibility.
    """

    def __init__(
        self,
        l1: L1Cache,
        l2: L2Cache,
        source_fn: Callable[[str], Any],
    ):
        self._l1 = l1
        self._l2 = l2
        self._source = source_fn
        self._l1_hits = 0
        self._l2_hits = 0
        self._source_hits = 0
        self._lock = threading.Lock()

    @property
    def l1(self) -> L1Cache:
        return self._l1

    @property
    def l2(self) -> L2Cache:
        return self._l2

    def get(self, key: str) -> tuple[str, Any]:
        """Return (tier, value) where tier is 'l1', 'l2', or 'source'."""
        hit, val = self._l1.get(key)
        if hit:
            with self._lock:
                self._l1_hits += 1
            return "l1", val

        hit, val = self._l2.get(key)
        if hit:
            self._l1.put(key, val)
            with self._lock:
                self._l2_hits += 1
            return "l2", val

        val = self._source(key)
        self._l2.put(key, val)
        self._l1.put(key, val)
        with self._lock:
            self._source_hits += 1
        return "source", val

    def put(self, key: str, value: Any) -> None:
        """Write-through: update both L1 and L2 immediately."""
        self._l1.put(key, value)
        self._l2.put(key, value)

    def invalidate(self, key: str) -> None:
        self._l1.invalidate(key)
        self._l2.invalidate(key)

    def invalidate_prefix(self, prefix: str) -> int:
        n1 = self._l1.invalidate_prefix(prefix)
        n2 = self._l2.invalidate_prefix(prefix)
        return max(n1, n2)

    def warm(self, keys: list[str]) -> int:
        """Pre-populate L1 (and L2) from source. Returns number of keys loaded."""
        loaded = 0
        for key in keys:
            hit, _ = self._l1.get(key)
            if not hit:
                self.get(key)
                loaded += 1
        return loaded

    def reset_stats(self) -> None:
        with self._lock:
            self._l1_hits = 0
            self._l2_hits = 0
            self._source_hits = 0
        self._l1.reset_stats()
        self._l2.reset_stats()

    def stats(self) -> dict:
        with self._lock:
            total = self._l1_hits + self._l2_hits + self._source_hits
            return {
                "l1_hits": self._l1_hits,
                "l2_hits": self._l2_hits,
                "source_hits": self._source_hits,
                "total": total,
                "l1_hit_rate": self._l1_hits / total if total else 0.0,
                "l2_hit_rate": self._l2_hits / total if total else 0.0,
                "source_miss_rate": self._source_hits / total if total else 0.0,
            }


# ---------------------------------------------------------------------------
# TieredMovieDB — application layer using TieredCache for recommendation queries
# ---------------------------------------------------------------------------

class TieredMovieDB:
    """
    Movie recommendation store with a two-tier write-through cache.

    DB Architect note: the source and the L2 are both SQLite connections here.
    In production you'd separate them — source in PostgreSQL / Cassandra, L2 in
    a Redis replication group, L1 in the app process.  The tiering logic is
    identical regardless of the underlying storage technology.
    """

    def __init__(
        self,
        l1_capacity: int = 32,
        l1_ttl: float = 30.0,
        l2_ttl: float = 300.0,
        query_delay: float = 0.002,
    ):
        self._delay = query_delay
        self._src_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._src_conn.row_factory = sqlite3.Row
        self._src_lock = threading.Lock()
        self._setup_source()

        l1 = L1Cache(capacity=l1_capacity, ttl=l1_ttl)
        l2 = L2Cache(ttl=l2_ttl)
        self._cache = TieredCache(l1=l1, l2=l2, source_fn=self._fetch_from_source)

    def _setup_source(self) -> None:
        with self._src_lock:
            self._src_conn.executescript("""
                CREATE TABLE IF NOT EXISTS movies (
                    id TEXT PRIMARY KEY, title TEXT, genre TEXT,
                    year INTEGER, director TEXT
                );
                CREATE TABLE IF NOT EXISTS ratings (
                    user_id TEXT, movie_id TEXT, score REAL, review TEXT,
                    PRIMARY KEY (user_id, movie_id)
                );
            """)

    def seed(self, movies: list[dict], ratings: list[tuple]) -> None:
        with self._src_lock:
            self._src_conn.executemany(
                "INSERT OR IGNORE INTO movies VALUES (:id,:title,:genre,:year,:director)",
                movies,
            )
            self._src_conn.executemany(
                "INSERT OR IGNORE INTO ratings(user_id,movie_id,score,review) "
                "VALUES (?,?,?,?)",
                ratings,
            )
            self._src_conn.commit()

    def _fetch_from_source(self, key: str) -> Any:
        """Execute SQL against the source DB; simulates I/O latency."""
        time.sleep(self._delay)
        with self._src_lock:
            if key == "genre_stats":
                rows = self._src_conn.execute("""
                    SELECT m.genre,
                           COUNT(DISTINCT m.id)  AS movie_count,
                           COUNT(r.score)        AS rating_count,
                           ROUND(AVG(r.score),2) AS avg_rating
                    FROM movies m
                    LEFT JOIN ratings r ON r.movie_id = m.id
                    GROUP BY m.genre ORDER BY m.genre
                """).fetchall()
                return [dict(r) for r in rows]

            if key.startswith("top:"):
                genre = key[4:]
                rows = self._src_conn.execute("""
                    SELECT m.title, m.genre,
                           ROUND(AVG(r.score),2) AS avg_score,
                           COUNT(r.score)        AS num_ratings
                    FROM movies m
                    JOIN ratings r ON r.movie_id = m.id
                    WHERE m.genre = ?
                    GROUP BY m.id ORDER BY avg_score DESC LIMIT 5
                """, (genre,)).fetchall()
                return [dict(r) for r in rows]

            if key.startswith("avg:"):
                mid = key[4:]
                row = self._src_conn.execute(
                    "SELECT ROUND(AVG(score),2) AS avg FROM ratings WHERE movie_id = ?",
                    (mid,),
                ).fetchone()
                return {"avg": row["avg"] if row else None}

            return None

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def genre_stats(self) -> list[dict]:
        _, val = self._cache.get("genre_stats")
        return val

    def top_rated_by_genre(self, genre: str) -> list[dict]:
        _, val = self._cache.get(f"top:{genre}")
        return val

    def average_rating(self, movie_id: str) -> Optional[float]:
        _, val = self._cache.get(f"avg:{movie_id}")
        return val["avg"] if val else None

    def get_cached(self, key: str) -> tuple[str, Any]:
        """Return (tier, value) — useful for benchmarking and testing."""
        return self._cache.get(key)

    def add_rating(
        self, user_id: str, movie_id: str, score: float, review: str = ""
    ) -> None:
        with self._src_lock:
            self._src_conn.execute(
                "INSERT OR REPLACE INTO ratings VALUES (?,?,?,?)",
                (user_id, movie_id, score, review),
            )
            self._src_conn.commit()
            row = self._src_conn.execute(
                "SELECT genre FROM movies WHERE id = ?", (movie_id,)
            ).fetchone()
            genre = row["genre"] if row else None

        self._cache.invalidate("genre_stats")
        if genre:
            self._cache.invalidate(f"top:{genre}")
        self._cache.invalidate(f"avg:{movie_id}")

    @property
    def cache(self) -> TieredCache:
        return self._cache

    def movie_count(self) -> int:
        with self._src_lock:
            return self._src_conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    def rating_count(self) -> int:
        with self._src_lock:
            return self._src_conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
