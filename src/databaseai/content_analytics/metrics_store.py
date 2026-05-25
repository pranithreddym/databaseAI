"""
Content Performance Analytics — Time-Series Tracking for Movie Launches
========================================================================

DB Architect notes:
  This module demonstrates a second application of time-series database
  principles, focusing on content engagement rather than system-monitoring.
  The schema design differs from sub-second infrastructure metrics in two
  important ways:

  1. Grain is daily, not sub-second.  Each row is one (movie_id, event_date)
     pair containing aggregate counters: views, completions, ratings given,
     and mean watch percentage.  Pre-aggregating at daily grain trades fine-
     grained precision for 86,400× smaller storage and dramatically faster
     GROUP BY queries.  The daily-partition pattern is identical to BigQuery
     table partitioning and PostgreSQL declarative range partitioning.

  2. The PRIMARY KEY is (movie_id, event_date).  This composite key enforces
     exactly one row per movie per day — making re-ingestion idempotent via
     INSERT OR REPLACE — and simultaneously acts as a covering index for the
     dominant access pattern: "all metrics for movie X between dates A and B."
     No separate CREATE INDEX is needed for that query.

  Time-series lessons illustrated:
    • Time-windowed aggregation  — SUM(views) BETWEEN two dates; the composite
      PK makes the range scan O(log n + k).
    • Rolling averages           — a Python sliding-window deque equivalent to
      SQL window function AVG(...) OVER (PARTITION BY movie_id ORDER BY
      event_date ROWS BETWEEN w-1 PRECEDING AND CURRENT ROW).
    • Retention curves           — normalising raw views to a launch-day
      baseline of 1.0 creates a scale-invariant shape that enables cross-
      title comparison regardless of absolute audience size.
    • Peak detection             — a simple ORDER BY metric DESC LIMIT 1;
      in production, Prometheus alerting rules and Grafana anomaly panels
      replace the argmax.
    • Genre-level aggregation    — summing a metric across all titles sharing
      a genre produces a "genre health" signal used to tune recommendation
      diversity; equivalent to a cohort rollup in analytics pipelines.

Production parallels:
  Netflix's Content Intelligence team stores daily impression, play, and
  completion counters per title in Apache Iceberg tables on S3.  Apache Druid
  powers sub-second rollup queries over those counters.  The retention-curve
  shape ("Day 1 vs Day 7 vs Day 28") is a primary signal used to decide
  whether to renew a series for a second season.

  Spotify's "Track Performance" pipeline is structurally identical — daily
  streams, skips, and saves per track — stored in BigQuery with date-based
  partitioning, which maps directly to the composite PK used here.

  TikTok uses ClickHouse with a MergeTree engine sorted by (content_id, date)
  for exactly the same query pattern at petabyte scale.
"""

import json
import sqlite3
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS content_metrics (
    movie_id      TEXT    NOT NULL,
    event_date    TEXT    NOT NULL,
    views         INTEGER NOT NULL DEFAULT 0,
    completions   INTEGER NOT NULL DEFAULT 0,
    ratings_given INTEGER NOT NULL DEFAULT 0,
    avg_watch_pct REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (movie_id, event_date)
);

CREATE INDEX IF NOT EXISTS idx_cm_date ON content_metrics(event_date);

CREATE TABLE IF NOT EXISTS content_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id    TEXT,
    event_type  TEXT    NOT NULL,
    description TEXT,
    metadata    TEXT,
    occurred_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ce_movie    ON content_events(movie_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_ce_occurred ON content_events(occurred_at);
"""


class ContentMetricsStore:
    """
    Append-oriented time-series store for per-movie daily engagement metrics.
    Backed by SQLite; designed for single-process use in demos and tests.
    Create a fresh instance per test via tmp_path to avoid state leakage.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def bulk_record_metrics(self, rows: List[Tuple]) -> None:
        """
        Insert (movie_id, event_date, views, completions, ratings_given,
        avg_watch_pct) tuples.  INSERT OR REPLACE makes re-ingestion idempotent.
        """
        self._conn.executemany(
            "INSERT OR REPLACE INTO content_metrics "
            "(movie_id, event_date, views, completions, ratings_given, avg_watch_pct) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def record_event(
        self,
        movie_id: Optional[str],
        event_type: str,
        description: str,
        metadata: Dict[str, Any],
        occurred_at: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO content_events "
            "(movie_id, event_type, description, metadata, occurred_at) "
            "VALUES (?,?,?,?,?)",
            (movie_id, event_type, description, json.dumps(metadata), occurred_at),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Counting helpers
    # ------------------------------------------------------------------

    def metric_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM content_metrics"
        ).fetchone()[0]

    def event_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM content_events"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Time-windowed queries
    # ------------------------------------------------------------------

    def query_window(
        self,
        movie_id: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """
        Return all daily rows for *movie_id* between *start_date* and
        *end_date* inclusive, ordered chronologically.
        """
        rows = self._conn.execute(
            "SELECT * FROM content_metrics "
            "WHERE movie_id = ? AND event_date BETWEEN ? AND ? "
            "ORDER BY event_date",
            (movie_id, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def aggregate_window(
        self,
        movie_id: str,
        start_date: str,
        end_date: str,
    ) -> Dict:
        """Aggregate views, completions, and ratings across a date window."""
        row = self._conn.execute(
            "SELECT "
            "  COUNT(*) AS days, "
            "  SUM(views) AS total_views, "
            "  SUM(completions) AS total_completions, "
            "  SUM(ratings_given) AS total_ratings, "
            "  AVG(avg_watch_pct) AS avg_watch_pct "
            "FROM content_metrics "
            "WHERE movie_id = ? AND event_date BETWEEN ? AND ?",
            (movie_id, start_date, end_date),
        ).fetchone()
        return dict(row)

    # ------------------------------------------------------------------
    # Rolling averages
    # ------------------------------------------------------------------

    def rolling_average(
        self,
        movie_id: str,
        metric: str,
        window_days: int,
    ) -> List[Dict]:
        """
        Compute a rolling N-day average for *metric* over all stored days for
        the given movie.  Implemented as a Python sliding-window deque so it
        works on any SQLite version (window functions require >= 3.25).

        Returns list of dicts: {event_date, raw_value, rolling_avg}.
        """
        allowed = {"views", "completions", "ratings_given", "avg_watch_pct"}
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")

        rows = self._conn.execute(
            f"SELECT event_date, {metric} AS val FROM content_metrics "
            "WHERE movie_id = ? ORDER BY event_date",
            (movie_id,),
        ).fetchall()

        result: List[Dict] = []
        window: deque = deque()
        for row in rows:
            window.append(row["val"])
            if len(window) > window_days:
                window.popleft()
            result.append({
                "event_date": row["event_date"],
                "raw_value": row["val"],
                "rolling_avg": sum(window) / len(window),
            })
        return result

    # ------------------------------------------------------------------
    # Retention curves
    # ------------------------------------------------------------------

    def retention_curve(self, movie_id: str) -> List[Dict]:
        """
        Normalise daily views relative to day-0 views (baseline = 1.0).
        Returns list of {days_since_launch, event_date, views, relative_views}.

        The normalised shape is scale-invariant: a blockbuster with 10M day-0
        views and an indie with 10K day-0 views become directly comparable on
        the same 0-1 axis.
        """
        rows = self._conn.execute(
            "SELECT event_date, views FROM content_metrics "
            "WHERE movie_id = ? ORDER BY event_date",
            (movie_id,),
        ).fetchall()
        if not rows:
            return []
        baseline = rows[0]["views"] or 1
        return [
            {
                "days_since_launch": i,
                "event_date": r["event_date"],
                "views": r["views"],
                "relative_views": round(r["views"] / baseline, 4),
            }
            for i, r in enumerate(rows)
        ]

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def peak_day(self, movie_id: str, metric: str = "views") -> Dict:
        """Return the day with the highest value for *metric*."""
        allowed = {"views", "completions", "ratings_given", "avg_watch_pct"}
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        row = self._conn.execute(
            f"SELECT event_date, {metric} AS peak_value "
            "FROM content_metrics WHERE movie_id = ? "
            f"ORDER BY {metric} DESC LIMIT 1",
            (movie_id,),
        ).fetchone()
        if row is None:
            return {}
        return {
            "movie_id": movie_id,
            "event_date": row["event_date"],
            "peak_value": row["peak_value"],
        }

    # ------------------------------------------------------------------
    # Genre-level trends
    # ------------------------------------------------------------------

    def genre_trend(
        self,
        genre: str,
        metric: str,
        start_date: str,
        end_date: str,
        movie_genre_map: Dict[str, str],
    ) -> List[Dict]:
        """
        Aggregate *metric* by date across all movies in *genre*.
        *movie_genre_map* is a {movie_id: genre} dict derived from seed data.
        Returns list of {event_date, total_value, movie_count}.
        """
        allowed = {"views", "completions", "ratings_given"}
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        movie_ids = [mid for mid, g in movie_genre_map.items() if g == genre]
        if not movie_ids:
            return []
        placeholders = ",".join("?" * len(movie_ids))
        rows = self._conn.execute(
            f"SELECT event_date, SUM({metric}) AS total_value, "
            f"COUNT(DISTINCT movie_id) AS movie_count "
            f"FROM content_metrics "
            f"WHERE movie_id IN ({placeholders}) AND event_date BETWEEN ? AND ? "
            "GROUP BY event_date ORDER BY event_date",
            (*movie_ids, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Top-N movies in a window
    # ------------------------------------------------------------------

    def top_movies_in_window(
        self,
        start_date: str,
        end_date: str,
        metric: str,
        n: int,
    ) -> List[Dict]:
        """Return top-N movies by total *metric* in the given date window."""
        allowed = {"views", "completions", "ratings_given"}
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        rows = self._conn.execute(
            f"SELECT movie_id, SUM({metric}) AS total_value "
            "FROM content_metrics WHERE event_date BETWEEN ? AND ? "
            "GROUP BY movie_id ORDER BY total_value DESC LIMIT ?",
            (start_date, end_date, n),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def query_events(
        self,
        start_date: str,
        end_date: str,
        movie_id: Optional[str] = None,
    ) -> List[Dict]:
        if movie_id:
            rows = self._conn.execute(
                "SELECT * FROM content_events "
                "WHERE movie_id = ? AND occurred_at BETWEEN ? AND ? "
                "ORDER BY occurred_at",
                (movie_id, start_date, end_date),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM content_events "
                "WHERE occurred_at BETWEEN ? AND ? ORDER BY occurred_at",
                (start_date, end_date),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
