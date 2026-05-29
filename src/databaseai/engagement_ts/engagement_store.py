"""
Engagement Time-Series Store — SQLite-backed user engagement analytics
======================================================================

Tracks user watch sessions and interaction events with ISO-8601 timestamps.
Provides rolling aggregations, peak-hour detection, daily active user (DAU)
trends, content dropout rates, and cohort retention analysis.

DB Architect notes:
  - Watch sessions are append-only; never UPDATE or DELETE historical rows.
    Immutability removes lock contention on reads and matches the semantics
    of write-ahead log stores like Kafka → ClickHouse in production.
  - (user_id, started_at) composite index supports per-user timeline scans;
    the standalone (started_at) index covers time-bucketed global aggregations
    (hourly DAU, peak-hour detection) without touching user partition rows.
  - strftime() GROUP BY is SQLite's equivalent of TimescaleDB time_bucket()
    and ClickHouse toStartOfHour().  The index scan reduces the input set
    first; strftime() runs only over the already-filtered rows.
  - cohort_retention() uses a two-CTE pattern: first_watch computes each
    user's join cohort in one pass; daily_activity deduplicates to one row
    per user per day; a single LEFT JOIN then folds four retention milestones
    (day-0, day-1, day-3, day-7) into one aggregation scan, avoiding four
    separate subqueries.
  - completion_rate_by_movie aggregates SUM(completed)/COUNT(*) per movie_id.
    At scale this is materialised as a ClickHouse MV or a TimescaleDB
    continuous aggregate refreshed every 15 minutes.

Production parallels:
  - Netflix engagement pipeline: Kafka topics → Apache Flink → Apache Iceberg
    on S3; ClickHouse serves sub-second OLAP for dashboard queries.
  - Disney+ dropout analysis: per-timestamp completion buckets identify where
    audiences abandon content, driving re-cut and thumbnail A/B decisions.
  - Spotify streaming minutes: rolling 28-day MAU derived from session rows
    feeds licensing negotiations and investor reporting.
  - YouTube watch-time signal: completion_pct outweighs raw view counts in
    the ranking algorithm — videos watched to completion surface above click-bait.
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Optional


_DDL = """
CREATE TABLE IF NOT EXISTS watch_sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT    NOT NULL,
    movie_id     TEXT    NOT NULL,
    started_at   TEXT    NOT NULL,
    duration_sec INTEGER NOT NULL,
    completed    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_user  ON watch_sessions (user_id, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_time  ON watch_sessions (started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_movie ON watch_sessions (movie_id);

CREATE TABLE IF NOT EXISTS engagement_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_user ON engagement_events (user_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_time ON engagement_events (occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON engagement_events (event_type);
"""


class EngagementStore:
    """
    Append-only time-series store for user watch sessions and interaction events.

    Real-world parallel:
      Netflix / Disney+ every play, pause, and session-complete event lands in
      a Kafka topic then sinks to ClickHouse for near-real-time OLAP queries.
      This SQLite implementation reproduces the same schema, indexes, and
      analytics patterns in an in-process store — no infrastructure required.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path == ":memory:":
            self._shared = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared.row_factory = sqlite3.Row
        else:
            self._shared = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self):
        if self._shared is not None:
            try:
                yield self._shared
                self._shared.commit()
            except Exception:
                self._shared.rollback()
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

    # ── Write operations ─────────────────────────────────────────────────────

    def record_session(
        self,
        user_id: str,
        movie_id: str,
        started_at: str,
        duration_sec: int,
        completed: bool,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO watch_sessions
                   (user_id, movie_id, started_at, duration_sec, completed)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, movie_id, started_at, duration_sec, int(completed)),
            )

    def bulk_record_sessions(self, rows: list) -> None:
        """Each row: (user_id, movie_id, started_at, duration_sec, completed)."""
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO watch_sessions
                   (user_id, movie_id, started_at, duration_sec, completed)
                   VALUES (?, ?, ?, ?, ?)""",
                [(r[0], r[1], r[2], r[3], int(r[4])) for r in rows],
            )

    def record_event(
        self,
        user_id: str,
        event_type: str,
        occurred_at: str,
        metadata: Optional[dict] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO engagement_events
                   (user_id, event_type, occurred_at, metadata)
                   VALUES (?, ?, ?, ?)""",
                (user_id, event_type, occurred_at, json.dumps(metadata or {})),
            )

    def bulk_record_events(self, rows: list) -> None:
        """Each row: (user_id, event_type, occurred_at[, metadata_dict])."""
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO engagement_events
                   (user_id, event_type, occurred_at, metadata)
                   VALUES (?, ?, ?, ?)""",
                [(r[0], r[1], r[2], json.dumps(r[3] if len(r) > 3 else {})) for r in rows],
            )

    # ── Counts ───────────────────────────────────────────────────────────────

    def session_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM watch_sessions").fetchone()[0]

    def event_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM engagement_events").fetchone()[0]

    # ── Time-windowed aggregations ───────────────────────────────────────────

    def rolling_window_stats(self, start_ts: str, end_ts: str) -> dict:
        """Aggregate session stats within a UTC time window."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT
                       COUNT(*)                              AS session_count,
                       ROUND(AVG(duration_sec) / 60.0, 1)   AS avg_watch_min,
                       ROUND(MAX(duration_sec) / 60.0, 1)   AS max_watch_min,
                       ROUND(MIN(duration_sec) / 60.0, 1)   AS min_watch_min,
                       ROUND(AVG(completed) * 100.0, 1)      AS completion_pct,
                       COUNT(DISTINCT user_id)               AS unique_users
                   FROM watch_sessions
                   WHERE started_at BETWEEN ? AND ?""",
                (start_ts, end_ts),
            ).fetchone()
        return dict(row) if row else {}

    def hourly_activity(self, start_ts: str, end_ts: str) -> list:
        """Per-calendar-hour bucket: session count, unique users, avg watch duration."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       strftime('%Y-%m-%d %H:00', started_at) AS hour_bucket,
                       COUNT(*)                                AS sessions,
                       COUNT(DISTINCT user_id)                 AS unique_users,
                       ROUND(AVG(duration_sec) / 60.0, 1)     AS avg_watch_min,
                       ROUND(AVG(completed) * 100.0, 1)        AS completion_pct
                   FROM watch_sessions
                   WHERE started_at BETWEEN ? AND ?
                   GROUP BY hour_bucket
                   ORDER BY hour_bucket""",
                (start_ts, end_ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def peak_hours(self, start_ts: str, end_ts: str, top_n: int = 5) -> list:
        """Top-N busiest hours-of-day (0–23) aggregated across all days in the window."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       CAST(strftime('%H', started_at) AS INTEGER) AS hour_of_day,
                       COUNT(*)                                     AS session_count,
                       COUNT(DISTINCT user_id)                      AS unique_users,
                       ROUND(AVG(duration_sec) / 60.0, 1)          AS avg_watch_min
                   FROM watch_sessions
                   WHERE started_at BETWEEN ? AND ?
                   GROUP BY hour_of_day
                   ORDER BY session_count DESC
                   LIMIT ?""",
                (start_ts, end_ts, top_n),
            ).fetchall()
        return [dict(r) for r in rows]

    def daily_active_users(self, start_ts: str, end_ts: str) -> list:
        """Distinct active users (DAU), session count, and avg watch time per calendar day."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       strftime('%Y-%m-%d', started_at)     AS day,
                       COUNT(DISTINCT user_id)               AS dau,
                       COUNT(*)                              AS sessions,
                       ROUND(AVG(duration_sec) / 60.0, 1)   AS avg_watch_min
                   FROM watch_sessions
                   WHERE started_at BETWEEN ? AND ?
                   GROUP BY day
                   ORDER BY day""",
                (start_ts, end_ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def completion_rate_by_movie(self, movie_ids: Optional[list] = None) -> list:
        """Completion rate, total plays, and avg watch time per movie."""
        with self._conn() as conn:
            if movie_ids:
                placeholders = ",".join("?" * len(movie_ids))
                rows = conn.execute(
                    f"""SELECT
                            movie_id,
                            COUNT(*)                            AS total_plays,
                            SUM(completed)                      AS completions,
                            ROUND(AVG(completed) * 100.0, 1)   AS completion_pct,
                            ROUND(AVG(duration_sec) / 60.0, 1) AS avg_watch_min
                        FROM watch_sessions
                        WHERE movie_id IN ({placeholders})
                        GROUP BY movie_id
                        ORDER BY completion_pct DESC""",
                    movie_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT
                            movie_id,
                            COUNT(*)                            AS total_plays,
                            SUM(completed)                      AS completions,
                            ROUND(AVG(completed) * 100.0, 1)   AS completion_pct,
                            ROUND(AVG(duration_sec) / 60.0, 1) AS avg_watch_min
                        FROM watch_sessions
                        GROUP BY movie_id
                        ORDER BY completion_pct DESC"""
                ).fetchall()
        return [dict(r) for r in rows]

    def cohort_retention(self, start_ts: str, end_ts: str) -> list:
        """
        Group users by first-watch day; return day-0, day-1, day-3, day-7 retention counts.

        The CTE pattern avoids four separate subqueries: first_watch pins each user's
        cohort day; daily_activity deduplicates to one row per (user, day); a single
        LEFT JOIN and CASE WHEN aggregation folds all four milestones into one scan.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """WITH first_watch AS (
                       SELECT user_id, MIN(DATE(started_at)) AS cohort_day
                       FROM watch_sessions
                       WHERE started_at BETWEEN ? AND ?
                       GROUP BY user_id
                   ),
                   daily_activity AS (
                       SELECT DISTINCT user_id, DATE(started_at) AS active_day
                       FROM watch_sessions
                       WHERE started_at BETWEEN ? AND ?
                   )
                   SELECT
                       f.cohort_day,
                       COUNT(DISTINCT f.user_id)                                                    AS cohort_size,
                       COUNT(DISTINCT CASE WHEN a.active_day = f.cohort_day                   THEN f.user_id END) AS day_0,
                       COUNT(DISTINCT CASE WHEN a.active_day = DATE(f.cohort_day, '+1 day')   THEN f.user_id END) AS day_1,
                       COUNT(DISTINCT CASE WHEN a.active_day = DATE(f.cohort_day, '+3 days')  THEN f.user_id END) AS day_3,
                       COUNT(DISTINCT CASE WHEN a.active_day = DATE(f.cohort_day, '+7 days')  THEN f.user_id END) AS day_7
                   FROM first_watch f
                   LEFT JOIN daily_activity a ON f.user_id = a.user_id
                   GROUP BY f.cohort_day
                   ORDER BY f.cohort_day""",
                (start_ts, end_ts, start_ts, end_ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def event_type_breakdown(self, start_ts: str, end_ts: str) -> list:
        """Count events grouped by type within a time window."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       event_type,
                       COUNT(*)                AS event_count,
                       COUNT(DISTINCT user_id) AS unique_users
                   FROM engagement_events
                   WHERE occurred_at BETWEEN ? AND ?
                   GROUP BY event_type
                   ORDER BY event_count DESC""",
                (start_ts, end_ts),
            ).fetchall()
        return [dict(r) for r in rows]
