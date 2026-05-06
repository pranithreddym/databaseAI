"""
Feature Store — SQLite-backed (Offline + Online)
==================================================

Pre-computes and serves ML features for users and movies.

DB Architect notes:
  - Offline store: historical features for model training (full history)
  - Online store: latest features for real-time inference (low-latency)
  - Point-in-time correctness: never leak future data into training features
  - Feature versioning: new feature logic = new version, old models still work
  - Training/serving skew is the #1 ML reliability bug — the feature store
    prevents it by using the same computation for both train and serve

Production equivalents:
  - Tecton, Feast, Hopsworks (open source)
  - AWS SageMaker Feature Store, Databricks Feature Store
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional
import json
from datetime import datetime, timezone


DDL = """
CREATE TABLE IF NOT EXISTS user_features (
    user_id          TEXT NOT NULL,
    feature_version  TEXT NOT NULL DEFAULT 'v1',
    avg_rating       REAL,
    total_ratings    INTEGER,
    fav_genre        TEXT,
    watch_count_7d   INTEGER,
    watch_count_30d  INTEGER,
    computed_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, feature_version)
);

CREATE TABLE IF NOT EXISTS movie_features (
    movie_id         TEXT NOT NULL,
    feature_version  TEXT NOT NULL DEFAULT 'v1',
    popularity_score REAL,
    avg_score        REAL,
    total_ratings    INTEGER,
    genre_vector     TEXT,       -- JSON list of genre weights
    computed_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (movie_id, feature_version)
);

CREATE TABLE IF NOT EXISTS feature_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type      TEXT NOT NULL,  -- 'user' or 'movie'
    entity_id        TEXT NOT NULL,
    feature_version  TEXT NOT NULL,
    event            TEXT NOT NULL,  -- 'computed', 'served'
    logged_at        TEXT DEFAULT (datetime('now'))
);
"""

GENRES = ["action", "comedy", "drama", "sci-fi", "thriller", "horror", "romance", "animation"]


class FeatureStore:
    """
    Two-tier feature store: offline (full history) + online (latest).

    Real-world parallel:
      Uber's Michelangelo feature platform computes driver/rider features
      offline for training and serves them online at sub-millisecond
      latency during trip matching.
    """

    def __init__(self, db_path: str = ":memory:", version: str = "v1"):
        self._db_path = db_path
        self.version = version
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
    # Compute + store features
    # ------------------------------------------------------------------

    def compute_user_features(
        self,
        user_id: str,
        ratings: list[dict],
        watch_count_7d: int = 0,
        watch_count_30d: int = 0,
    ) -> dict:
        """
        Derive user features from their rating history.

        In production this runs as a batch Spark/Flink job nightly.
        The same logic runs at serving time for real-time updates.
        """
        if not ratings:
            avg_rating = 0.0
            fav_genre = "unknown"
        else:
            avg_rating = round(sum(r["score"] for r in ratings) / len(ratings), 2)
            genre_counts: dict[str, int] = {}
            for r in ratings:
                g = r.get("genre", "unknown")
                genre_counts[g] = genre_counts.get(g, 0) + 1
            fav_genre = max(genre_counts, key=genre_counts.get)

        features = {
            "user_id": user_id,
            "feature_version": self.version,
            "avg_rating": avg_rating,
            "total_ratings": len(ratings),
            "fav_genre": fav_genre,
            "watch_count_7d": watch_count_7d,
            "watch_count_30d": watch_count_30d,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO user_features
                   (user_id, feature_version, avg_rating, total_ratings,
                    fav_genre, watch_count_7d, watch_count_30d, computed_at)
                   VALUES (:user_id, :feature_version, :avg_rating, :total_ratings,
                           :fav_genre, :watch_count_7d, :watch_count_30d, :computed_at)""",
                features,
            )
            conn.execute(
                "INSERT INTO feature_log (entity_type, entity_id, feature_version, event) VALUES (?,?,?,?)",
                ("user", user_id, self.version, "computed"),
            )
        return features

    def compute_movie_features(
        self, movie_id: str, genre: str, ratings: list[float]
    ) -> dict:
        """Derive movie features from its ratings."""
        avg_score = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
        total = len(ratings)
        popularity = round(min(avg_score * (total ** 0.5) / 10, 10.0), 2)

        genre_vec = {g: 0.0 for g in GENRES}
        if genre in genre_vec:
            genre_vec[genre] = 1.0

        features = {
            "movie_id": movie_id,
            "feature_version": self.version,
            "popularity_score": popularity,
            "avg_score": avg_score,
            "total_ratings": total,
            "genre_vector": json.dumps(genre_vec),
        }

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO movie_features
                   (movie_id, feature_version, popularity_score,
                    avg_score, total_ratings, genre_vector)
                   VALUES (:movie_id, :feature_version, :popularity_score,
                           :avg_score, :total_ratings, :genre_vector)""",
                features,
            )
            conn.execute(
                "INSERT INTO feature_log (entity_type, entity_id, feature_version, event) VALUES (?,?,?,?)",
                ("movie", movie_id, self.version, "computed"),
            )
        features["genre_vector"] = genre_vec
        return features

    # ------------------------------------------------------------------
    # Serve features (online path — simulated low-latency lookup)
    # ------------------------------------------------------------------

    def get_user_features(self, user_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM user_features
                   WHERE user_id = ? AND feature_version = ?
                   ORDER BY computed_at DESC LIMIT 1""",
                (user_id, self.version),
            ).fetchone()
            if not row:
                return None
            self._log_serve(conn, "user", user_id)
            return dict(row)

    def get_movie_features(self, movie_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM movie_features
                   WHERE movie_id = ? AND feature_version = ?
                   ORDER BY computed_at DESC LIMIT 1""",
                (movie_id, self.version),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["genre_vector"] = json.loads(result["genre_vector"])
            self._log_serve(conn, "movie", movie_id)
            return result

    def get_top_movies_by_popularity(self, n: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT movie_id, popularity_score, avg_score, total_ratings
                   FROM movie_features WHERE feature_version = ?
                   ORDER BY popularity_score DESC LIMIT ?""",
                (self.version, n),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Point-in-time query (training data generation)
    # ------------------------------------------------------------------

    def get_training_snapshot(self, as_of: str) -> list[dict]:
        """
        Return all user features computed before a given timestamp.

        This is point-in-time correct: no future data leaks into training.
        In production this uses a time-travel query on the offline store
        (e.g. Delta Lake TIME TRAVEL, Iceberg time-travel).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM user_features
                   WHERE computed_at <= ? AND feature_version = ?
                   ORDER BY user_id""",
                (as_of, self.version),
            ).fetchall()
            return [dict(r) for r in rows]

    def feature_count(self) -> dict:
        with self._conn() as conn:
            users = conn.execute("SELECT COUNT(*) FROM user_features").fetchone()[0]
            movies = conn.execute("SELECT COUNT(*) FROM movie_features").fetchone()[0]
            return {"user_features": users, "movie_features": movies}

    def _log_serve(self, conn: sqlite3.Connection, entity_type: str, entity_id: str) -> None:
        conn.execute(
            "INSERT INTO feature_log (entity_type, entity_id, feature_version, event) VALUES (?,?,?,?)",
            (entity_type, entity_id, self.version, "served"),
        )
