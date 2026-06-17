"""
Adaptive Connection Pool — dynamic resizing with health checks
==============================================================

DB Architect notes:
  A fixed-size pool forces a painful trade-off at provisioning time: size it
  for peak traffic and waste server-side RAM during quiet hours, or size it
  for average traffic and suffer PoolExhaustedError during spikes.  An
  adaptive pool eliminates that trade-off by observing demand and resizing
  between min_size and max_size on the fly.

  Two complementary mechanisms keep the pool healthy:

    Scale-up:   When utilisation (checked_out / current_capacity) exceeds
                scale_up_threshold, the pool creates new connections in a
                batch (grow_step) — up to max_size.  This amortises the
                creation cost across several requests rather than adding
                one connection per spike.

    Scale-down: When utilisation drops below scale_down_threshold for an
                entire cooldown window, the pool closes idle connections
                down to min_size.  The cooldown prevents thrashing: a
                momentary dip in traffic should not immediately discard
                warmed connections that will be needed seconds later.

  Connection health is maintained by two eviction policies:

    Max-age eviction:  Connections older than max_conn_age_s are closed on
                       return rather than recycled.  This limits damage from
                       slow memory leaks or stale server-side session state
                       (PostgreSQL parameter drift, MySQL timezone changes).

    Validate-on-borrow: Before handing a connection to a caller, the pool
                        runs a lightweight probe query (SELECT 1).  If it
                        fails, the connection is discarded and a fresh one
                        is created.  This catches connections dropped by
                        firewalls, load balancers, or server restarts
                        without surfacing errors to the application.

Production parallels:
  - HikariCP (Java/Spring Boot): adaptive pool sizing with
    minimumIdle / maximumPoolSize / idleTimeout / maxLifetime.
  - Aurora Serverless v2: auto-scales database capacity units (ACUs) between
    min and max based on CPU/connection demand, billed per second.
  - Pgpool-II: process-based pooler with dynamic child process management
    that scales worker processes to match connection demand.
  - Django CONN_MAX_AGE + CONN_HEALTH_CHECKS: max-age eviction and
    validate-on-borrow in Django 4.1+.
"""

import sqlite3
import threading
import time
import queue
from contextlib import contextmanager
from typing import Callable, Optional


class PoolExhaustedError(Exception):
    """Raised when acquire() cannot return a connection within the timeout."""


class _ManagedConnection:
    """Wrapper around a raw DB-API connection with creation timestamp."""

    __slots__ = ("_conn", "_pool", "conn_id", "created_at")

    def __init__(self, raw_conn, pool: "AdaptivePool", conn_id: int) -> None:
        self._conn = raw_conn
        self._pool = pool
        self.conn_id = conn_id
        self.created_at = time.monotonic()

    def release(self) -> None:
        self._pool._release(self)

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def executemany(self, sql, seq):
        return self._conn.executemany(sql, seq)

    def executescript(self, script):
        return self._conn.executescript(script)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def validate(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False


class AdaptivePool:
    """
    Connection pool that dynamically resizes between min_size and max_size
    based on observed utilisation, with validate-on-borrow and max-age
    eviction.
    """

    def __init__(
        self,
        db_factory: Callable,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 5.0,
        scale_up_threshold: float = 0.75,
        scale_down_threshold: float = 0.25,
        grow_step: int = 2,
        cooldown_s: float = 2.0,
        max_conn_age_s: float = 300.0,
        connection_overhead: float = 0.0,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if not (1 <= min_size <= max_size):
            raise ValueError("min_size must be between 1 and max_size")
        if not (0 < scale_up_threshold <= 1.0):
            raise ValueError("scale_up_threshold must be in (0, 1]")
        if not (0 <= scale_down_threshold < scale_up_threshold):
            raise ValueError("scale_down_threshold must be in [0, scale_up_threshold)")

        self._db_factory = db_factory
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._grow_step = grow_step
        self._cooldown_s = cooldown_s
        self._max_conn_age_s = max_conn_age_s
        self._connection_overhead = connection_overhead

        self._lock = threading.Lock()
        self._available: queue.Queue = queue.Queue()
        self._next_id = 0
        self._capacity = 0
        self._checked_out = 0

        self._last_scale_up = 0.0
        self._last_scale_down = 0.0

        self._stat_served = 0
        self._stat_timeouts = 0
        self._stat_total_wait_ms = 0.0
        self._stat_max_wait_ms = 0.0
        self._stat_scale_ups = 0
        self._stat_scale_downs = 0
        self._stat_health_failures = 0
        self._stat_age_evictions = 0
        self._stat_connections_created = 0

        self._utilisation_samples: list[float] = []

        for _ in range(min_size):
            self._available.put(self._new_connection())
            self._capacity += 1

    def _new_connection(self) -> _ManagedConnection:
        with self._lock:
            conn_id = self._next_id
            self._next_id += 1
            self._stat_connections_created += 1
        if self._connection_overhead > 0:
            time.sleep(self._connection_overhead)
        return _ManagedConnection(self._db_factory(), self, conn_id)

    def _is_expired(self, conn: _ManagedConnection) -> bool:
        return (time.monotonic() - conn.created_at) > self._max_conn_age_s

    def _utilisation(self) -> float:
        with self._lock:
            cap = self._capacity
            out = self._checked_out
        if cap <= 0:
            return 1.0
        return out / cap

    def _maybe_scale_up(self) -> int:
        now = time.monotonic()
        with self._lock:
            if now - self._last_scale_up < self._cooldown_s:
                return 0
            util = self._checked_out / self._capacity if self._capacity > 0 else 1.0
            if util < self._scale_up_threshold:
                return 0
            headroom = self._max_size - self._capacity
            if headroom <= 0:
                return 0
            to_add = min(self._grow_step, headroom)
            self._capacity += to_add
            self._last_scale_up = now
            self._stat_scale_ups += 1

        added = 0
        for _ in range(to_add):
            try:
                self._available.put(self._new_connection())
                added += 1
            except Exception:
                with self._lock:
                    self._capacity -= 1
        return added

    def _maybe_scale_down(self) -> int:
        now = time.monotonic()
        with self._lock:
            if now - self._last_scale_down < self._cooldown_s:
                return 0
            util = self._checked_out / self._capacity if self._capacity > 0 else 0.0
            if util > self._scale_down_threshold:
                return 0
            idle = self._available.qsize()
            removable = self._capacity - max(self._min_size, self._checked_out)
            to_remove = min(idle, removable, self._grow_step)
            if to_remove <= 0:
                return 0
            self._last_scale_down = now
            self._stat_scale_downs += 1

        removed = 0
        for _ in range(to_remove):
            try:
                conn = self._available.get_nowait()
                conn.close()
                removed += 1
                with self._lock:
                    self._capacity -= 1
            except queue.Empty:
                break
        return removed

    def _record_acquisition(self, t0: float) -> None:
        wait_ms = (time.monotonic() - t0) * 1000
        with self._lock:
            self._stat_served += 1
            self._checked_out += 1
            self._stat_total_wait_ms += wait_ms
            if wait_ms > self._stat_max_wait_ms:
                self._stat_max_wait_ms = wait_ms
            util = self._checked_out / self._capacity if self._capacity > 0 else 1.0
            self._utilisation_samples.append(util)

    def acquire(self) -> _ManagedConnection:
        t0 = time.monotonic()
        deadline = t0 + self._timeout

        while True:
            try:
                conn = self._available.get_nowait()
            except queue.Empty:
                conn = None

            if conn is not None:
                if self._is_expired(conn):
                    conn.close()
                    with self._lock:
                        self._capacity -= 1
                        self._stat_age_evictions += 1
                    self._maybe_scale_up()
                    continue
                if not conn.validate():
                    conn.close()
                    with self._lock:
                        self._capacity -= 1
                        self._stat_health_failures += 1
                    self._maybe_scale_up()
                    continue
                self._record_acquisition(t0)
                self._maybe_scale_up()
                return conn

            added = self._maybe_scale_up()
            if added > 0:
                continue

            with self._lock:
                can_grow = self._capacity < self._max_size
            if can_grow:
                with self._lock:
                    self._capacity += 1
                new_conn = self._new_connection()
                self._record_acquisition(t0)
                return new_conn

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                conn = self._available.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    break
                continue

            if self._is_expired(conn):
                conn.close()
                with self._lock:
                    self._capacity -= 1
                    self._stat_age_evictions += 1
                continue
            if not conn.validate():
                conn.close()
                with self._lock:
                    self._capacity -= 1
                    self._stat_health_failures += 1
                continue

            self._record_acquisition(t0)
            return conn

        with self._lock:
            self._stat_timeouts += 1
        raise PoolExhaustedError(
            f"No connection available after {self._timeout:.1f}s "
            f"(capacity={self._capacity}, max_size={self._max_size})"
        )

    def _release(self, conn: _ManagedConnection) -> None:
        with self._lock:
            self._checked_out = max(0, self._checked_out - 1)

        if self._is_expired(conn):
            conn.close()
            with self._lock:
                self._capacity -= 1
                self._stat_age_evictions += 1
            return

        self._available.put(conn)
        self._maybe_scale_down()

    @contextmanager
    def connection(self):
        conn = self.acquire()
        try:
            yield conn
        finally:
            conn.release()

    @property
    def capacity(self) -> int:
        with self._lock:
            return self._capacity

    @property
    def checked_out(self) -> int:
        with self._lock:
            return self._checked_out

    @property
    def idle(self) -> int:
        return self._available.qsize()

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def max_size(self) -> int:
        return self._max_size

    def utilisation(self) -> float:
        return self._utilisation()

    def stats(self) -> dict:
        with self._lock:
            served = self._stat_served
            avg = round(self._stat_total_wait_ms / served, 3) if served else 0.0
            avg_util = (
                round(sum(self._utilisation_samples) / len(self._utilisation_samples), 4)
                if self._utilisation_samples
                else 0.0
            )
            return {
                "min_size": self._min_size,
                "max_size": self._max_size,
                "capacity": self._capacity,
                "checked_out": self._checked_out,
                "idle": self._available.qsize(),
                "total_connections_created": self._stat_connections_created,
                "total_served": served,
                "total_timeouts": self._stat_timeouts,
                "avg_wait_ms": avg,
                "max_wait_ms": round(self._stat_max_wait_ms, 3),
                "scale_ups": self._stat_scale_ups,
                "scale_downs": self._stat_scale_downs,
                "health_failures": self._stat_health_failures,
                "age_evictions": self._stat_age_evictions,
                "avg_utilisation": avg_util,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._stat_served = 0
            self._stat_timeouts = 0
            self._stat_total_wait_ms = 0.0
            self._stat_max_wait_ms = 0.0
            self._stat_scale_ups = 0
            self._stat_scale_downs = 0
            self._stat_health_failures = 0
            self._stat_age_evictions = 0
            self._stat_connections_created = 0
            self._utilisation_samples.clear()

    def force_scale_up(self, count: int = 1) -> int:
        added = 0
        for _ in range(count):
            with self._lock:
                if self._capacity >= self._max_size:
                    break
                self._capacity += 1
            self._available.put(self._new_connection())
            added += 1
        return added

    def force_scale_down(self, count: int = 1) -> int:
        removed = 0
        for _ in range(count):
            with self._lock:
                if self._capacity <= self._min_size:
                    break
            try:
                conn = self._available.get_nowait()
                conn.close()
                removed += 1
                with self._lock:
                    self._capacity -= 1
            except queue.Empty:
                break
        return removed

    def close(self) -> None:
        while True:
            try:
                conn = self._available.get_nowait()
                conn.close()
            except queue.Empty:
                break
        with self._lock:
            self._capacity = 0


def sqlite_factory(path: str = ":memory:") -> Callable:
    if path == ":memory:":
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        path = tmp.name
        tmp.close()

    def _open():
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    return _open


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
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_movies_genre  ON movies(genre);
"""


class AdaptiveMovieDB:
    """Movie query runner backed by an adaptive connection pool."""

    def __init__(self, pool: AdaptivePool) -> None:
        self._pool = pool
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.executescript(_DDL)

    def seed(self, movies, ratings) -> None:
        with self._pool.connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id, title, genre, year, director) "
                "VALUES (:id, :title, :genre, :year, :director)", movies)
            conn.executemany(
                "INSERT OR IGNORE INTO ratings (user_id, movie_id, score, review) "
                "VALUES (?, ?, ?, ?)", ratings)
            conn.commit()

    def top_rated(self, limit: int = 5) -> list:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT m.title, m.genre, ROUND(AVG(r.score), 2) AS avg_score,
                          COUNT(r.score) AS votes
                   FROM movies m JOIN ratings r ON m.id = r.movie_id
                   GROUP BY m.id ORDER BY avg_score DESC, votes DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def genre_breakdown(self) -> list:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT m.genre, COUNT(DISTINCT m.id) AS movies,
                          ROUND(AVG(r.score), 2) AS avg_score
                   FROM movies m JOIN ratings r ON m.id = r.movie_id
                   GROUP BY m.genre ORDER BY avg_score DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def movie_count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    def rating_count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
