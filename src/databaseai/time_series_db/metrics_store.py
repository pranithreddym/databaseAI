"""
Time-Series Metrics Store — SQLite-backed
==========================================

Tracks recommendation model accuracy metrics and system events over time.

DB Architect notes:
  - Time-series data has a natural access pattern: (metric_name, recorded_at).
    The composite index on those two columns turns range scans from O(n) to O(log n + k).
  - Timestamps are stored as ISO-8601 strings ("YYYY-MM-DD HH:MM:SS").  SQLite's
    lexicographic ordering on that format is identical to chronological ordering,
    so BETWEEN clauses on an indexed column are both correct and fast.
  - Downsampling (AVG over strftime() buckets) is the standard way to trade
    fidelity for storage.  Keep raw data for recent periods, daily averages forever.
  - detect_trend() fits a least-squares line to the last N points.  A threshold
    of ±0.001 filters noise; anything steeper triggers an alert or auto-rollback.
  - For production scale: TimescaleDB hypertables auto-partition by time;
    InfluxDB's TSM engine uses columnar compression; Prometheus + Thanos handle
    multi-datacenter federation and long-term retention.

Production parallels:
  - Netflix Atlas ingests millions of time-series per second for model monitoring.
  - A/B test metrics (precision@k, NDCG) feed real-time dashboards that decide
    which model variant deserves 100% of production traffic.
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional


DDL = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_ver    TEXT    NOT NULL,
    variant      TEXT    NOT NULL DEFAULT 'control',
    metric_name  TEXT    NOT NULL,
    metric_value REAL    NOT NULL,
    recorded_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_time
    ON model_metrics (metric_name, recorded_at);

CREATE INDEX IF NOT EXISTS idx_metrics_variant
    ON model_metrics (variant, metric_name, recorded_at);

CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    description TEXT,
    metadata    TEXT,
    occurred_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_time
    ON system_events (occurred_at);
"""


class MetricsStore:
    """
    Time-series store for ML model metrics and system events.

    Real-world parallel:
      A recommendation model A/B test runs two variants simultaneously.
      Metrics (precision@10, NDCG@10) are recorded every hour.
      The team queries rolling windows, downsamples to daily averages,
      and runs trend detection to decide which variant to promote.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            self._shared_conn = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

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

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record_metric(
        self,
        model_ver: str,
        variant: str,
        metric_name: str,
        value: float,
        at: Optional[str] = None,
    ) -> int:
        ts = at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO model_metrics
                   (model_ver, variant, metric_name, metric_value, recorded_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (model_ver, variant, metric_name, value, ts),
            )
            return cur.lastrowid

    def bulk_record_metrics(
        self, rows: list[tuple]
    ) -> None:
        """
        Insert many metric rows in a single transaction.

        Each tuple: (model_ver, variant, metric_name, metric_value, recorded_at).
        One transaction for the entire batch is orders of magnitude faster than
        individual INSERTs because SQLite flushes the WAL once per commit.
        """
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO model_metrics
                   (model_ver, variant, metric_name, metric_value, recorded_at)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )

    def record_event(
        self,
        event_type: str,
        description: str = "",
        metadata: Optional[dict] = None,
        at: Optional[str] = None,
    ) -> int:
        ts = at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO system_events
                   (event_type, description, metadata, occurred_at)
                   VALUES (?, ?, ?, ?)""",
                (event_type, description, json.dumps(metadata or {}), ts),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query_window(
        self,
        metric_name: str,
        start: str,
        end: str,
        variant: Optional[str] = None,
    ) -> list[dict]:
        """Return all raw metric rows within [start, end] (inclusive)."""
        with self._conn() as conn:
            if variant:
                rows = conn.execute(
                    """SELECT * FROM model_metrics
                       WHERE metric_name = ? AND recorded_at BETWEEN ? AND ?
                         AND variant = ?
                       ORDER BY recorded_at""",
                    (metric_name, start, end, variant),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM model_metrics
                       WHERE metric_name = ? AND recorded_at BETWEEN ? AND ?
                       ORDER BY recorded_at""",
                    (metric_name, start, end),
                ).fetchall()
            return [dict(r) for r in rows]

    def downsample(
        self,
        metric_name: str,
        bucket_fmt: str,
        start: str,
        end: str,
        variant: Optional[str] = None,
    ) -> list[dict]:
        """
        Aggregate raw metrics into time buckets using AVG.

        bucket_fmt controls bucket granularity via strftime():
          '%Y-%m-%d %H:00:00'  → hourly buckets
          '%Y-%m-%d'           → daily buckets

        This is the canonical time-series storage trade-off: recent data stays
        at full resolution; older data gets rolled up to save disk space.
        """
        with self._conn() as conn:
            if variant:
                rows = conn.execute(
                    """SELECT strftime(?, recorded_at) AS bucket,
                              variant,
                              AVG(metric_value)         AS avg_value,
                              MIN(metric_value)         AS min_value,
                              MAX(metric_value)         AS max_value,
                              COUNT(*)                  AS sample_count
                       FROM model_metrics
                       WHERE metric_name = ? AND recorded_at BETWEEN ? AND ?
                         AND variant = ?
                       GROUP BY bucket, variant
                       ORDER BY bucket""",
                    (bucket_fmt, metric_name, start, end, variant),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT strftime(?, recorded_at) AS bucket,
                              variant,
                              AVG(metric_value)         AS avg_value,
                              MIN(metric_value)         AS min_value,
                              MAX(metric_value)         AS max_value,
                              COUNT(*)                  AS sample_count
                       FROM model_metrics
                       WHERE metric_name = ? AND recorded_at BETWEEN ? AND ?
                       GROUP BY bucket, variant
                       ORDER BY bucket, variant""",
                    (bucket_fmt, metric_name, start, end),
                ).fetchall()
            return [dict(r) for r in rows]

    def latest_metric(
        self, metric_name: str, variant: Optional[str] = None
    ) -> Optional[dict]:
        """Return the single most recent row for a metric (and optional variant)."""
        with self._conn() as conn:
            if variant:
                row = conn.execute(
                    """SELECT * FROM model_metrics
                       WHERE metric_name = ? AND variant = ?
                       ORDER BY recorded_at DESC LIMIT 1""",
                    (metric_name, variant),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT * FROM model_metrics
                       WHERE metric_name = ?
                       ORDER BY recorded_at DESC LIMIT 1""",
                    (metric_name,),
                ).fetchone()
            return dict(row) if row else None

    def ab_test_summary(self, metric_name: str) -> list[dict]:
        """
        Return per-variant statistics for a given metric, ordered by avg descending.

        The first row is the winning variant — the one to promote to 100% traffic.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT variant,
                          COUNT(*)                          AS sample_count,
                          AVG(metric_value)                 AS avg_value,
                          MIN(metric_value)                 AS min_value,
                          MAX(metric_value)                 AS max_value,
                          MAX(metric_value) - MIN(metric_value) AS range_value
                   FROM model_metrics
                   WHERE metric_name = ?
                   GROUP BY variant
                   ORDER BY avg_value DESC""",
                (metric_name,),
            ).fetchall()
            return [dict(r) for r in rows]

    def detect_trend(
        self,
        metric_name: str,
        variant: str,
        window_size: int = 5,
    ) -> dict:
        """
        Compute the slope of the last `window_size` data points via
        ordinary least-squares linear regression.

        A slope above +0.001 signals improvement; below -0.001 signals
        degradation and can trigger an automated rollback in production.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT metric_value FROM model_metrics
                   WHERE metric_name = ? AND variant = ?
                   ORDER BY recorded_at DESC LIMIT ?""",
                (metric_name, variant, window_size),
            ).fetchall()

        values = [r[0] for r in reversed(rows)]
        n = len(values)
        if n < 2:
            return {"slope": 0.0, "n": n, "direction": "stable", "values": values}

        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = numerator / denominator if denominator else 0.0

        if slope > 0.001:
            direction = "improving"
        elif slope < -0.001:
            direction = "degrading"
        else:
            direction = "stable"

        return {
            "slope": round(slope, 6),
            "n": n,
            "direction": direction,
            "values": values,
        }

    def query_events(self, start: str, end: str) -> list[dict]:
        """Return system events within [start, end] ordered chronologically."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM system_events
                   WHERE occurred_at BETWEEN ? AND ?
                   ORDER BY occurred_at""",
                (start, end),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["metadata"] = json.loads(d["metadata"])
                result.append(d)
            return result

    def metric_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]

    def event_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
