"""
Read-Replica Connection Pool — Primary / Replica Routing
=========================================================

DB Architect notes:
  A single database server becomes the read bottleneck long before it
  hits write saturation on most recommendation workloads.  A Netflix
  homepage load fires 10–20 reads per user session (carousels, continue-
  watching, top-picks) while generating 1–2 writes (play-start event).
  The 10:1 read/write ratio means that adding 3 read replicas can absorb
  the read load even if primary throughput stays constant.

  Read replicas work by continuously shipping the primary's WAL (Write-
  Ahead Log in PostgreSQL, binlog in MySQL) to standby servers that apply
  each change in order.  The delay between a commit on primary and its
  visibility on the replica is called *replication lag*.  A router that
  sends reads to lagging replicas returns *stale data*; the acceptability
  of staleness is application-specific:
    • Recommendation scores refreshed every 15 min: 5 s lag is fine.
    • A user's own watchlist after pressing "Add": 0 ms lag is expected.

  Three routing concerns this module demonstrates:
    1. Write routing  — all mutations go to the primary pool.
    2. Read routing   — weighted round-robin across healthy replicas.
    3. Failover       — when a replica is marked unhealthy its weight is
                        excluded; reads shift to the surviving replicas.
                        If every replica fails, reads fall back to primary
                        (at the cost of defeating the isolation benefit).

  Replication is simulated by maintaining a *write log* on the router.
  Each entry carries a `replicate_after` timestamp equal to
  time.perf_counter() + lag_s.  Calling propagate() drains entries whose
  deadline has passed, applying each SQL statement to every healthy
  replica.  This makes replication lag measurable and reproducible in
  tests without any background threads.

Production parallels:
  - PostgreSQL streaming replication + PgBouncer + Patroni HA
  - MySQL Group Replication + ProxySQL read/write splitting
  - AWS RDS Proxy — transparent primary/replica routing for Aurora
  - Vitess (used by YouTube / Slack) — per-shard tablet routing for MySQL
  - Netflix Dynomite / RigDB — local-DC reads, cross-DC write fanout
"""

import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional


# ── Schema ────────────────────────────────────────────────────────────────────

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
    movie_id TEXT NOT NULL,
    score    REAL NOT NULL CHECK(score BETWEEN 0.0 AND 5.0),
    review   TEXT,
    PRIMARY KEY (user_id, movie_id)
);
CREATE INDEX IF NOT EXISTS idx_ratings_score ON ratings(score);
CREATE INDEX IF NOT EXISTS idx_movies_year   ON movies(year);
"""


# ── Bounded connection pool ───────────────────────────────────────────────────

class PoolExhaustedError(Exception):
    pass


class BoundedPool:
    """Thread-safe, bounded pool of SQLite connections to one database.

    When db_path is ':memory:' (the default), all connections in the pool
    share the same in-process database via SQLite's shared-cache URI so that
    writes on one connection are immediately visible on the others.
    """

    def __init__(self, db_path: str = ":memory:", max_size: int = 4,
                 timeout: float = 2.0):
        self._timeout = timeout
        self._max_size = max_size
        self._q: queue.Queue = queue.Queue(maxsize=max_size)
        self._all_conns: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._checked_out = 0

        # Each :memory: pool gets a unique named database so connections share
        # data but pools for different nodes (primary, replica-1 …) remain
        # isolated from each other.
        if db_path == ":memory:":
            db_path = f"file:pool_{uuid.uuid4().hex}?mode=memory&cache=shared"
        self._db_path = db_path
        use_uri = db_path.startswith("file:")

        for _ in range(max_size):
            conn = sqlite3.connect(db_path, check_same_thread=False, uri=use_uri)
            conn.row_factory = sqlite3.Row
            conn.executescript(_DDL)
            self._q.put(conn)
            self._all_conns.append(conn)

    @contextmanager
    def connection(self):
        try:
            conn = self._q.get(timeout=self._timeout)
        except queue.Empty:
            raise PoolExhaustedError(
                f"Pool({self._db_path!r}) exhausted — all {self._max_size} "
                f"connections checked out"
            )
        with self._lock:
            self._checked_out += 1
        try:
            yield conn
        finally:
            with self._lock:
                self._checked_out -= 1
            self._q.put(conn)

    @property
    def checked_out(self) -> int:
        with self._lock:
            return self._checked_out

    def close_all(self):
        for conn in self._all_conns:
            try:
                conn.close()
            except Exception:
                pass


# ── Replica ───────────────────────────────────────────────────────────────────

@dataclass
class Replica:
    """One read replica: bounded pool + health state + query counters."""

    replica_id: int
    weight: float = 1.0
    lag_ms: float = 0.0          # simulated replication lag in milliseconds
    pool_size: int = 3
    healthy: bool = True
    queries_served: int = field(default=0, init=False)
    _total_latency_us: float = field(default=0.0, init=False)
    _pool: Optional[BoundedPool] = field(default=None, init=False)

    def __post_init__(self):
        self._pool = BoundedPool(":memory:", max_size=self.pool_size)

    @contextmanager
    def connection(self):
        t0 = time.perf_counter()
        with self._pool.connection() as conn:
            self.queries_served += 1
            yield conn
        self._total_latency_us += (time.perf_counter() - t0) * 1e6

    def apply_write(self, sql: str, params: tuple = ()):
        with self._pool.connection() as conn:
            conn.execute(sql, params)
            conn.commit()

    def seed(self, movies, ratings):
        with self._pool.connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id,title,genre,year,director) "
                "VALUES (:id,:title,:genre,:year,:director)",
                movies,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) "
                "VALUES (?,?,?,?)",
                [(r[0], r[1], r[2], r[3]) for r in ratings],
            )
            conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def avg_latency_us(self) -> float:
        if self.queries_served == 0:
            return 0.0
        return self._total_latency_us / self.queries_served

    def close(self):
        if self._pool:
            self._pool.close_all()


# ── Write log ─────────────────────────────────────────────────────────────────

class _WriteEntry(NamedTuple):
    sql: str
    params: tuple
    replicate_after: float    # perf_counter() timestamp when replication is due


# ── Primary / Replica Router ─────────────────────────────────────────────────

class PrimaryReplicaRouter:
    """
    Routes writes to primary and reads to healthy replicas using
    weighted round-robin.  Replication is simulated via a write log
    drained by propagate().
    """

    def __init__(self, replicas: List[Replica], primary_pool_size: int = 4):
        self._primary = BoundedPool(":memory:", max_size=primary_pool_size)
        self._replicas = replicas
        self._write_log: List[_WriteEntry] = []
        self._log_lock = threading.Lock()
        self._rr_index = 0
        self._rr_lock = threading.Lock()
        # Serialize writes to the primary — SQLite doesn't support concurrent
        # writers within the same process (table-level locks in shared-cache
        # mode).  Real databases (PostgreSQL, MySQL) serialize at row / page
        # level via MVCC; this mutex is the simplified equivalent.
        self._write_mutex = threading.Lock()

        # counters
        self._writes = 0
        self._reads_replica = 0
        self._reads_primary_fallback = 0

    # ── seeding ───────────────────────────────────────────────────────────────

    def seed(self, movies, ratings):
        """Seed primary and all replicas with the same initial dataset."""
        with self._primary.connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id,title,genre,year,director) "
                "VALUES (:id,:title,:genre,:year,:director)",
                movies,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) "
                "VALUES (?,?,?,?)",
                [(r[0], r[1], r[2], r[3]) for r in ratings],
            )
            conn.commit()
        for replica in self._replicas:
            replica.seed(movies, ratings)

    # ── write path ────────────────────────────────────────────────────────────

    def execute_write(self, sql: str, params: tuple = (),
                      replica_lag_ms: Optional[float] = None):
        """
        Execute a write on primary and enqueue it for replica propagation.
        replica_lag_ms overrides the per-replica lag for this specific write.
        Pass replica_lag_ms=0 to replicate immediately on next propagate().
        """
        with self._write_mutex:
            with self._primary.connection() as conn:
                conn.execute(sql, params)
                conn.commit()
        self._writes += 1

        now = time.perf_counter()
        with self._log_lock:
            for replica in self._replicas:
                lag = replica_lag_ms if replica_lag_ms is not None else replica.lag_ms
                self._write_log.append(
                    _WriteEntry(sql, params, now + lag / 1000.0)
                )

    # ── replication pump ──────────────────────────────────────────────────────

    def propagate(self, until_ts: Optional[float] = None) -> int:
        """
        Apply all write-log entries whose replicate_after <= until_ts
        to their target replicas.  Defaults to time.perf_counter() (now).
        Returns the number of entries applied.
        """
        cutoff = time.perf_counter() if until_ts is None else until_ts
        with self._log_lock:
            due = [e for e in self._write_log if e.replicate_after <= cutoff]
            self._write_log = [e for e in self._write_log if e.replicate_after > cutoff]

        applied = 0
        for entry in due:
            for replica in self._replicas:
                if replica.healthy:
                    try:
                        replica.apply_write(entry.sql, entry.params)
                        applied += 1
                    except Exception:
                        pass
        return applied

    def pending_replication_count(self) -> int:
        with self._log_lock:
            return len(self._write_log)

    # ── read path ─────────────────────────────────────────────────────────────

    def _weighted_slots(self) -> List[Replica]:
        """
        Expand healthy replicas into a flat slot list proportional to their
        weights.  Weights are normalised relative to the minimum healthy weight
        so fractional weights like [0.5, 1.0, 1.5] still produce whole-number
        slot counts.  Rebuilding on every call is O(n) but n (replica count) is
        always small in practice.
        """
        healthy = [r for r in self._replicas if r.healthy]
        if not healthy:
            return []
        min_w = min(r.weight for r in healthy)
        slots: List[Replica] = []
        for r in healthy:
            slots.extend([r] * max(1, round(r.weight / min_w)))
        return slots

    def _pick_replica(self) -> Optional[Replica]:
        """Weighted round-robin over healthy replicas."""
        slots = self._weighted_slots()
        if not slots:
            return None
        with self._rr_lock:
            self._rr_index = (self._rr_index + 1) % len(slots)
            return slots[self._rr_index]

    def execute_read(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Route a SELECT to a healthy replica; fall back to primary."""
        replica = self._pick_replica()
        if replica is not None:
            self._reads_replica += 1
            return replica.execute(sql, params)
        self._reads_primary_fallback += 1
        with self._primary.connection() as conn:
            return conn.execute(sql, params).fetchall()

    # ── health management ─────────────────────────────────────────────────────

    def mark_unhealthy(self, replica_id: int):
        for r in self._replicas:
            if r.replica_id == replica_id:
                r.healthy = False

    def mark_healthy(self, replica_id: int):
        for r in self._replicas:
            if r.replica_id == replica_id:
                r.healthy = True

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def routing_stats(self) -> dict:
        return {
            "total_writes": self._writes,
            "reads_to_replicas": self._reads_replica,
            "reads_to_primary_fallback": self._reads_primary_fallback,
            "pending_replication": self.pending_replication_count(),
        }

    @property
    def replica_stats(self) -> List[dict]:
        return [
            {
                "id": r.replica_id,
                "healthy": r.healthy,
                "weight": r.weight,
                "lag_ms": r.lag_ms,
                "queries_served": r.queries_served,
                "avg_latency_us": round(r.avg_latency_us(), 1),
            }
            for r in self._replicas
        ]

    def healthy_replica_count(self) -> int:
        return sum(1 for r in self._replicas if r.healthy)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        self._primary.close_all()
        for r in self._replicas:
            r.close()
