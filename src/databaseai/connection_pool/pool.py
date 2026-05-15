"""
Connection Pooling — bounded pool with FIFO queue and configurable timeout
=========================================================================

DB Architect notes:
  Opening a database connection is expensive: TCP handshake, TLS negotiation,
  authentication, and session initialisation can cost 5–50 ms on a local
  network and 100–500 ms across availability zones.  Under sustained load,
  creating a new connection per request not only adds per-request latency but
  also threatens the DB server's max_connections ceiling (PostgreSQL default:
  100; each connection consumes ~5 MB of server RAM for shared memory
  structures like pg_stat_activity rows and per-backend data structures).

  A connection pool amortises creation cost by keeping a bounded set of
  connections alive and reusing them across many requests:

    max_size  — hard cap on simultaneous open connections.  Should match the
                DB server's quota allocated to this service tier.  Exceeding it
                causes "FATAL: sorry, too many clients already" on Postgres.
    min_size  — connections pre-created at startup ("warm pool") so the first
                burst of traffic does not trigger a creation spike.
    timeout   — maximum seconds a caller waits before PoolExhaustedError.
                Bounded waiting is preferable to an infinite queue, which
                masks overload and accumulates memory without back-pressure.
    FIFO      — first-in-first-out queue prevents starvation and gives
                predictable worst-case wait times.

  Three pool modes used in production:
    Session mode     — one server connection per client for the entire session.
                       Simplest but wastes connections during idle time.
    Transaction mode — connection held only for the duration of one
                       transaction, then returned.  PgBouncer's default;
                       allows N_clients >> pool_size when idle fraction is high.
    Statement mode   — connection returned after each statement; requires the
                       app to be stateless (no SET, no temp tables).

  This module implements transaction mode: every public PooledMovieDB method
  acquires a connection, executes its query, commits if needed, and releases.

Production parallels:
  - PgBouncer: open-source PostgreSQL connection pooler used by Netflix,
    Shopify, GitHub, and Discord.  Runs as a sidecar or standalone proxy;
    fully transparent to the application.
  - HikariCP (Java/Spring Boot): "zero-overhead" pool using lock-free CAS;
    the Spring Boot default for all JDBC data sources.
  - SQLAlchemy QueuePool (Python): backs Django, FastAPI, Flask-SQLAlchemy,
    and most Python ORMs when connecting to a relational database.
  - AWS RDS Proxy: managed pooler that also handles automatic failover to a
    read replica, reducing Lambda cold-start connection storms.
"""

import sqlite3
import threading
import time
import queue
from contextlib import contextmanager
from typing import Callable, Optional


class PoolExhaustedError(Exception):
    """Raised when acquire() cannot return a connection within the timeout."""


class _PooledConnection:
    """
    Thin proxy around a raw DBAPI-2 connection with pool-return bookkeeping.

    Delegates the common sqlite3.Connection methods so callers can use this
    object transparently.  Calling .release() returns the underlying
    connection to the pool for reuse instead of closing it.
    """

    __slots__ = ("_conn", "_pool", "conn_id")

    def __init__(self, raw_conn, pool: "ConnectionPool", conn_id: int) -> None:
        self._conn = raw_conn
        self._pool = pool
        self.conn_id = conn_id

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


class ConnectionPool:
    """
    Bounded connection pool with FIFO wait queue and configurable timeout.

    Connections are created lazily on demand up to max_size.  Pre-warming
    min_size connections at startup eliminates the cold-start spike on the
    first burst of traffic.  Any caller that cannot acquire a connection
    within timeout seconds receives PoolExhaustedError — surfacing overload
    early rather than letting it cascade silently through the call stack.

    The optional connection_overhead parameter inserts a sleep() during
    connection creation to simulate TCP + TLS + auth round-trip latency,
    making the no-pool vs. pool latency difference visible without needing
    a real remote database.
    """

    def __init__(
        self,
        db_factory: Callable,
        max_size: int = 5,
        min_size: int = 0,
        timeout: float = 5.0,
        connection_overhead: float = 0.0,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if not (0 <= min_size <= max_size):
            raise ValueError("min_size must be between 0 and max_size")

        self._db_factory = db_factory
        self._max_size = max_size
        self._min_size = min_size
        self._timeout = timeout
        self._connection_overhead = connection_overhead

        self._lock = threading.Lock()
        self._available: queue.Queue = queue.Queue()
        self._next_id = 0      # monotonically increasing; equals total created
        self._checked_out = 0

        # Aggregate statistics
        self._stat_requests = 0
        self._stat_served = 0
        self._stat_timeouts = 0
        self._stat_total_wait_ms = 0.0
        self._stat_max_wait_ms = 0.0

        # Pre-warm min_size connections so first-burst requests don't pay
        # the creation cost.
        for _ in range(min_size):
            self._available.put(self._new_connection())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_connection(self) -> "_PooledConnection":
        """Create a new connection, incrementing the counter unconditionally."""
        with self._lock:
            conn_id = self._next_id
            self._next_id += 1
        if self._connection_overhead > 0:
            time.sleep(self._connection_overhead)
        return _PooledConnection(self._db_factory(), self, conn_id)

    def _try_create(self) -> Optional["_PooledConnection"]:
        """Atomically reserve a slot and create a connection, or return None."""
        with self._lock:
            if self._next_id >= self._max_size:
                return None
            conn_id = self._next_id
            self._next_id += 1
        # Simulate network latency outside the lock so we don't stall other threads.
        if self._connection_overhead > 0:
            time.sleep(self._connection_overhead)
        return _PooledConnection(self._db_factory(), self, conn_id)

    def _record_acquisition(self, t0: float) -> None:
        wait_ms = (time.monotonic() - t0) * 1000
        with self._lock:
            self._stat_served += 1
            self._checked_out += 1
            self._stat_total_wait_ms += wait_ms
            if wait_ms > self._stat_max_wait_ms:
                self._stat_max_wait_ms = wait_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> "_PooledConnection":
        """
        Return a connection from the pool.

        Tries three paths in order:
          1. Fast path  — grab an idle connection immediately (queue non-empty).
          2. Lazy path  — create a new connection if total_created < max_size.
          3. Block path — wait up to timeout seconds for a release.

        Raises PoolExhaustedError if no connection becomes available in time.
        """
        t0 = time.monotonic()
        with self._lock:
            self._stat_requests += 1

        # 1. Fast path: idle connection immediately available
        try:
            conn = self._available.get_nowait()
            self._record_acquisition(t0)
            return conn
        except queue.Empty:
            pass

        # 2. Lazy creation: pool not yet at max_size
        conn = self._try_create()
        if conn is not None:
            self._record_acquisition(t0)
            return conn

        # 3. Block: wait for an in-flight connection to be released
        remaining = self._timeout - (time.monotonic() - t0)
        if remaining > 0:
            try:
                conn = self._available.get(timeout=remaining)
                self._record_acquisition(t0)
                return conn
            except queue.Empty:
                pass

        with self._lock:
            self._stat_timeouts += 1
        raise PoolExhaustedError(
            f"No connection available after {self._timeout:.1f}s "
            f"(max_size={self._max_size})"
        )

    def _release(self, conn: "_PooledConnection") -> None:
        with self._lock:
            self._checked_out = max(0, self._checked_out - 1)
        self._available.put(conn)

    @contextmanager
    def connection(self):
        """Context manager: acquire → yield → release (even on exception)."""
        conn = self.acquire()
        try:
            yield conn
        finally:
            conn.release()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_created(self) -> int:
        with self._lock:
            return self._next_id

    @property
    def available(self) -> int:
        return self._available.qsize()

    @property
    def max_size(self) -> int:
        return self._max_size

    def stats(self) -> dict:
        with self._lock:
            served = self._stat_served
            avg = round(self._stat_total_wait_ms / served, 3) if served else 0.0
            return {
                "max_size": self._max_size,
                "min_size": self._min_size,
                "total_created": self._next_id,
                "available": self._available.qsize(),
                "checked_out": self._checked_out,
                "total_requests": self._stat_requests,
                "total_served": served,
                "total_timeouts": self._stat_timeouts,
                "avg_wait_ms": avg,
                "max_wait_ms": round(self._stat_max_wait_ms, 3),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._stat_requests = 0
            self._stat_served = 0
            self._stat_timeouts = 0
            self._stat_total_wait_ms = 0.0
            self._stat_max_wait_ms = 0.0

    def close(self) -> None:
        """Close all idle connections in the pool."""
        while True:
            try:
                conn = self._available.get_nowait()
                conn.close()
            except queue.Empty:
                break


# ------------------------------------------------------------------
# SQLite factory helper
# ------------------------------------------------------------------

def sqlite_factory(path: str) -> Callable:
    """Return a callable that opens a new SQLite connection to path."""
    def _open():
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    return _open


# ------------------------------------------------------------------
# Movie database backed by the pool (transaction-mode pattern)
# ------------------------------------------------------------------

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


class PooledMovieDB:
    """
    Movie query runner that acquires a pool connection per operation and
    releases it immediately after — equivalent to PgBouncer transaction mode.

    Every method is stateless with respect to the connection: it borrows one
    for the duration of a single query and returns it.  This maximises reuse
    even when the caller holds this object across multiple requests.
    """

    def __init__(self, pool: ConnectionPool) -> None:
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
