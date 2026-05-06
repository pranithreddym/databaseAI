"""
Relational Database — SQLite
=============================

Stores structured movie metadata, users, and ratings with ACID guarantees.

DB Architect notes:
  - SQLite here; swap connection string for PostgreSQL in production
  - ACID transactions prevent partial writes (e.g. rating without user update)
  - Indexes on foreign keys and common filter columns (genre, year)
  - Aggregate queries (AVG rating, total reviews) run in SQL, not Python
  - Schema migrations: in prod use Alembic or Flyway, not DROP TABLE
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional
import os


DDL = """
CREATE TABLE IF NOT EXISTS movies (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    genre       TEXT NOT NULL,
    year        INTEGER NOT NULL,
    director    TEXT NOT NULL,
    description TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ratings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(id),
    movie_id    TEXT NOT NULL REFERENCES movies(id),
    score       REAL NOT NULL CHECK (score BETWEEN 1 AND 5),
    review      TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE (user_id, movie_id)          -- one rating per user per movie
);

CREATE INDEX IF NOT EXISTS idx_ratings_movie  ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_ratings_user   ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_movies_genre   ON movies(genre);
CREATE INDEX IF NOT EXISTS idx_movies_year    ON movies(year);
"""


class MovieRegistry:
    """
    Relational store for movies, users, and ratings.

    Real-world parallel:
      Netflix's primary database stores every title, subscription,
      user account, and review in a relational system. ACID guarantees
      mean a payment never succeeds without the subscription being created.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        # Keep a single connection alive for :memory: — each new connect() creates a blank DB
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON")
        else:
            self._shared_conn = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _conn(self):
        if self._shared_conn is not None:
            # Re-use the persistent in-memory connection
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Movies
    # ------------------------------------------------------------------

    def add_movie(self, movie: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO movies
                   (id, title, genre, year, director, description)
                   VALUES (:id, :title, :genre, :year, :director, :description)""",
                movie,
            )

    def add_movies(self, movies: list[dict]) -> None:
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO movies
                   (id, title, genre, year, director, description)
                   VALUES (:id, :title, :genre, :year, :director, :description)""",
                movies,
            )

    def get_movie(self, movie_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM movies WHERE id = ?", (movie_id,)
            ).fetchone()
            return dict(row) if row else None

    def search_movies(
        self,
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
        director: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        clauses, params = [], []
        if genre:
            clauses.append("genre = ?")
            params.append(genre)
        if min_year:
            clauses.append("year >= ?")
            params.append(min_year)
        if director:
            clauses.append("director LIKE ?")
            params.append(f"%{director}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM movies {where} ORDER BY year DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def top_rated_movies(self, genre: Optional[str] = None, n: int = 10) -> list[dict]:
        """Aggregate query: average score per movie, minimum 3 ratings."""
        genre_filter = "AND m.genre = ?" if genre else ""
        params = [genre] if genre else []
        params.append(n)

        sql = f"""
            SELECT m.id, m.title, m.genre, m.year, m.director,
                   ROUND(AVG(r.score), 2) AS avg_score,
                   COUNT(r.id)            AS total_ratings
            FROM   movies m
            JOIN   ratings r ON r.movie_id = m.id
            {genre_filter}
            GROUP  BY m.id
            HAVING COUNT(r.id) >= 3
            ORDER  BY avg_score DESC, total_ratings DESC
            LIMIT  ?
        """
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def add_user(self, user: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username, email) VALUES (:id, :username, :email)",
                user,
            )

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Ratings — transactional write
    # ------------------------------------------------------------------

    def add_rating(self, user_id: str, movie_id: str, score: float, review: str = "") -> None:
        """ACID transaction: upsert rating atomically."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO ratings (user_id, movie_id, score, review)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, movie_id) DO UPDATE
                   SET score = excluded.score, review = excluded.review,
                       created_at = datetime('now')""",
                (user_id, movie_id, score, review),
            )

    def get_user_ratings(self, user_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT r.score, r.review, r.created_at,
                          m.title, m.genre, m.director, m.year
                   FROM   ratings r JOIN movies m ON m.id = r.movie_id
                   WHERE  r.user_id = ?
                   ORDER  BY r.created_at DESC""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def movie_stats(self, movie_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as total_ratings,
                          ROUND(AVG(score), 2) as avg_score,
                          MIN(score) as min_score,
                          MAX(score) as max_score
                   FROM ratings WHERE movie_id = ?""",
                (movie_id,),
            ).fetchone()
            return dict(row)

    def movie_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    def user_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def rating_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
