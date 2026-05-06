"""
Relational Database — MySQL (via PyMySQL)
==========================================

Drop-in alternative to SQLiteMovieRegistry with MySQL-native SQL.

DB Architect notes — SQLite vs MySQL key differences:
  - Parameter style: SQLite uses `?`, MySQL uses `%s`
  - Upsert: SQLite `ON CONFLICT(...) DO UPDATE`, MySQL `ON DUPLICATE KEY UPDATE`
  - DDL: `INTEGER PRIMARY KEY AUTOINCREMENT` → `INT AUTO_INCREMENT PRIMARY KEY`
  - Default timestamp: `datetime('now')` → `NOW()`
  - Foreign keys: SQLite needs `PRAGMA foreign_keys=ON`, MySQL enforces by default
  - Connection: SQLite is file/memory; MySQL is a network socket
  - Row results: SQLite Row object; MySQL DictCursor returns plain dicts
  - Schema init: SQLite `executescript()`; MySQL requires per-statement execution
  - REAL/TEXT column types → FLOAT/VARCHAR(n) in MySQL

When to choose MySQL over SQLite:
  - Multi-process / multi-host writes (MySQL has row-level locking)
  - > ~1 TB data (SQLite tops out around 100 GB in practice)
  - Replication, read replicas, HA failover
  - Connection pooling via PgBouncer-equivalent (ProxySQL)

Connection string format (for reference):
  mysql+pymysql://user:password@host:3306/dbname
"""

from contextlib import contextmanager
from typing import Optional
import pymysql
import pymysql.cursors


MYSQL_DDL = [
    """
    CREATE TABLE IF NOT EXISTS movies (
        id          VARCHAR(10)  PRIMARY KEY,
        title       VARCHAR(255) NOT NULL,
        genre       VARCHAR(50)  NOT NULL,
        year        INT          NOT NULL,
        director    VARCHAR(255) NOT NULL,
        description TEXT,
        created_at  DATETIME     DEFAULT NOW()
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id          VARCHAR(10)  PRIMARY KEY,
        username    VARCHAR(100) UNIQUE NOT NULL,
        email       VARCHAR(255) UNIQUE NOT NULL,
        created_at  DATETIME     DEFAULT NOW()
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS ratings (
        id          INT          AUTO_INCREMENT PRIMARY KEY,
        user_id     VARCHAR(10)  NOT NULL,
        movie_id    VARCHAR(10)  NOT NULL,
        score       FLOAT        NOT NULL,
        review      TEXT,
        created_at  DATETIME     DEFAULT NOW(),
        UNIQUE KEY  uq_user_movie (user_id, movie_id),
        FOREIGN KEY (user_id)  REFERENCES users(id),
        FOREIGN KEY (movie_id) REFERENCES movies(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_ratings_user  ON ratings(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_movies_genre  ON movies(genre)",
    "CREATE INDEX IF NOT EXISTS idx_movies_year   ON movies(year)",
]


class MySQLMovieRegistry:
    """
    MySQL-backed movie registry — identical interface to MovieRegistry (SQLite).

    Real-world parallel:
      Production streaming services run MySQL (or Aurora MySQL-compatible)
      for their core transactional tables. The same ACID guarantees as SQLite
      but with multi-host replication and row-level locking.

    Requires a running MySQL server. Quick local setup:
      docker run -d --name mysql-cineai \\
        -e MYSQL_ROOT_PASSWORD=cineai \\
        -e MYSQL_DATABASE=cineai \\
        -p 3306:3306 mysql:8

    Then connect:
      registry = MySQLMovieRegistry(
          host="127.0.0.1", port=3306,
          user="root", password="cineai", database="cineai"
      )
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "cineai",
    ):
        self._config = dict(
            host=host, port=port, user=user, password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
            autocommit=False,
        )
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                for stmt in MYSQL_DDL:
                    cur.execute(stmt)

    @contextmanager
    def _conn(self):
        conn = pymysql.connect(**self._config)
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
        sql = """
            INSERT INTO movies (id, title, genre, year, director, description)
            VALUES (%(id)s, %(title)s, %(genre)s, %(year)s, %(director)s, %(description)s)
            ON DUPLICATE KEY UPDATE
                title=VALUES(title), genre=VALUES(genre), year=VALUES(year),
                director=VALUES(director), description=VALUES(description)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, movie)

    def add_movies(self, movies: list[dict]) -> None:
        sql = """
            INSERT INTO movies (id, title, genre, year, director, description)
            VALUES (%(id)s, %(title)s, %(genre)s, %(year)s, %(director)s, %(description)s)
            ON DUPLICATE KEY UPDATE
                title=VALUES(title), genre=VALUES(genre), year=VALUES(year),
                director=VALUES(director), description=VALUES(description)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, movies)

    def get_movie(self, movie_id: str) -> Optional[dict]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM movies WHERE id = %s", (movie_id,))
                return cur.fetchone()

    def search_movies(
        self,
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
        director: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        clauses, params = [], []
        if genre:
            clauses.append("genre = %s")
            params.append(genre)
        if min_year:
            clauses.append("year >= %s")
            params.append(min_year)
        if director:
            clauses.append("director LIKE %s")
            params.append(f"%{director}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM movies {where} ORDER BY year DESC LIMIT %s"
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def top_rated_movies(self, genre: Optional[str] = None, n: int = 10) -> list[dict]:
        genre_filter = "AND m.genre = %s" if genre else ""
        params = [genre] if genre else []
        params.append(n)

        sql = f"""
            SELECT m.id, m.title, m.genre, m.year, m.director,
                   ROUND(AVG(r.score), 2) AS avg_score,
                   COUNT(r.id)            AS total_ratings
            FROM   movies m
            JOIN   ratings r ON r.movie_id = m.id
            {genre_filter}
            GROUP  BY m.id, m.title, m.genre, m.year, m.director
            HAVING COUNT(r.id) >= 3
            ORDER  BY avg_score DESC, total_ratings DESC
            LIMIT  %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def add_user(self, user: dict) -> None:
        sql = """
            INSERT IGNORE INTO users (id, username, email)
            VALUES (%(id)s, %(username)s, %(email)s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, user)

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                return cur.fetchone()

    # ------------------------------------------------------------------
    # Ratings
    # ------------------------------------------------------------------

    def add_rating(self, user_id: str, movie_id: str, score: float, review: str = "") -> None:
        sql = """
            INSERT INTO ratings (user_id, movie_id, score, review)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                score=VALUES(score), review=VALUES(review), created_at=NOW()
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, movie_id, score, review))

    def get_user_ratings(self, user_id: str) -> list[dict]:
        sql = """
            SELECT r.score, r.review, r.created_at,
                   m.title, m.genre, m.director, m.year
            FROM   ratings r JOIN movies m ON m.id = r.movie_id
            WHERE  r.user_id = %s
            ORDER  BY r.created_at DESC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchall()

    def movie_stats(self, movie_id: str) -> dict:
        sql = """
            SELECT COUNT(*)       AS total_ratings,
                   ROUND(AVG(score), 2) AS avg_score,
                   MIN(score)     AS min_score,
                   MAX(score)     AS max_score
            FROM ratings WHERE movie_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (movie_id,))
                return cur.fetchone()

    def movie_count(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM movies")
                return cur.fetchone()["n"]

    def user_count(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM users")
                return cur.fetchone()["n"]

    def rating_count(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM ratings")
                return cur.fetchone()["n"]

    def teardown(self) -> None:
        """Drop all tables — for test cleanup only."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SET FOREIGN_KEY_CHECKS = 0")
                for tbl in ("ratings", "users", "movies"):
                    cur.execute(f"DROP TABLE IF EXISTS {tbl}")
                cur.execute("SET FOREIGN_KEY_CHECKS = 1")
