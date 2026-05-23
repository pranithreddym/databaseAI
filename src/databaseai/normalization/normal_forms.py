"""
Database Normalization — UNF → 1NF → 2NF → 3NF
================================================

DB Architect notes:
  Normalization is a systematic process of decomposing a relational schema to
  eliminate data redundancy and prevent three classes of anomaly:

  Update anomaly — the same fact is stored in N rows.  A director's nationality
    must be corrected in every movie row they appear in.  Miss one row and the
    database now holds contradictory facts about the real world.

  Insertion anomaly — related entities are coupled in a single table.  You
    cannot record a new director (with birth year, nationality) until they have
    at least one movie row to attach the data to.

  Deletion anomaly — deleting the last row referencing an entity silently
    destroys all knowledge of that entity.  Remove a director's only movie and
    their bio vanishes from the database.

  The normal forms that eliminate these anomalies:

  UNF (Unnormalized Form):
    Non-atomic columns; a single field holds multiple values (a CSV of
    ratings, a JSON array of actors).  Queries must parse strings at runtime.

  1NF (First Normal Form):
    Every cell holds one atomic value; there are no repeating groups.  The
    ratings CSV is exploded into one row per (movie, user) pair and the table
    gains a composite primary key.

  2NF (Second Normal Form):
    Every non-key column is fully functionally dependent on the WHOLE primary
    key — no partial dependencies.  In the 1NF table the composite key is
    (movie_id, user_id).  user_name depends only on user_id, not on movie_id,
    so it is extracted to a separate Users table.  Similarly, all movie
    attributes depend only on movie_id and are split off into Movies.

  3NF (Third Normal Form):
    Every non-key column depends on the key, the whole key, and nothing but the
    key.  In the 2NF Movies table, director_birth_year and director_nationality
    are determined by director_name, not by movie_id — a transitive dependency.
    Extracting Directors eliminates it: now every non-key column in every table
    depends directly and solely on that table's primary key.

Production parallels:
  - Legacy flat-file migrations at Netflix, Spotify, and Airbnb typically start
    from a denormalized "event log" table (one row per user-action) and
    progressively normalise into dimension/fact schemas for OLAP, or into
    3NF OLTP schemas for transactional workloads.
  - Django ORM migrations follow this same progression: AddField followed by
    SeparateDatabaseAndState when extracting a new model from an existing one.
  - PostgreSQL's COPY FROM for bulk imports often loads into a staging table in
    UNF and then a set of INSERT … SELECT statements normalise the data into
    production tables inside a single transaction.
  - Snowflake schema (star schema extended) in data warehouses is effectively
    3NF applied to dimensional modelling — dimension tables hold descriptive
    attributes, the fact table holds only foreign keys and measures.
"""

import re
import sqlite3
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Director metadata — supplements the movie seed data
# ---------------------------------------------------------------------------

DIRECTOR_METADATA: Dict[str, Dict] = {
    "Christopher Nolan":    {"birth_year": 1970, "nationality": "British"},
    "Lana Wachowski":       {"birth_year": 1965, "nationality": "American"},
    "Bong Joon-ho":         {"birth_year": 1969, "nationality": "South Korean"},
    "Quentin Tarantino":    {"birth_year": 1963, "nationality": "American"},
    "Frank Darabont":       {"birth_year": 1959, "nationality": "American"},
    "Hayao Miyazaki":       {"birth_year": 1941, "nationality": "Japanese"},
    "Jordan Peele":         {"birth_year": 1979, "nationality": "American"},
    "George Miller":        {"birth_year": 1945, "nationality": "Australian"},
    "Denis Villeneuve":     {"birth_year": 1967, "nationality": "Canadian"},
    "Spike Jonze":          {"birth_year": 1969, "nationality": "American"},
    "Lee Unkrich":          {"birth_year": 1967, "nationality": "American"},
    "Rian Johnson":         {"birth_year": 1973, "nationality": "American"},
    "Francis Ford Coppola": {"birth_year": 1939, "nationality": "American"},
    "Daniel Kwan":          {"birth_year": 1988, "nationality": "American"},
    "Ari Aster":            {"birth_year": 1986, "nationality": "American"},
    "Sidney Lumet":         {"birth_year": 1924, "nationality": "American"},
    "Anthony Russo":        {"birth_year": 1970, "nationality": "American"},
}


def _director_slug(name: str) -> str:
    """'Denis Villeneuve' → 'denis_villeneuve'; 'Bong Joon-ho' → 'bong_joon_ho'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Schema DDL — all stages in one executescript call
# ---------------------------------------------------------------------------

_SCHEMA_ALL = """
PRAGMA journal_mode=WAL;

-- UNF: ratings stored as a pipe-delimited CSV in a single non-atomic column
CREATE TABLE IF NOT EXISTS movies_unf (
    movie_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    genre               TEXT,
    year                INTEGER,
    director_name       TEXT,
    director_birth_year INTEGER,
    director_nationality TEXT,
    ratings_csv         TEXT
);

-- 1NF: atomic values; composite PK (movie_id, user_id); repeating group gone
CREATE TABLE IF NOT EXISTS movies_1nf (
    movie_id            TEXT NOT NULL,
    title               TEXT NOT NULL,
    genre               TEXT,
    year                INTEGER,
    director_name       TEXT,
    director_birth_year INTEGER,
    director_nationality TEXT,
    user_id             TEXT NOT NULL,
    user_name           TEXT,
    user_rating         REAL,
    PRIMARY KEY (movie_id, user_id)
);

-- 2NF: partial dependencies removed; user_name lives in users_2nf
CREATE TABLE IF NOT EXISTS movies_2nf (
    movie_id             TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    genre                TEXT,
    year                 INTEGER,
    director_name        TEXT,
    director_birth_year  INTEGER,
    director_nationality TEXT
);
CREATE TABLE IF NOT EXISTS users_2nf (
    user_id   TEXT PRIMARY KEY,
    user_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ratings_2nf (
    movie_id TEXT NOT NULL,
    user_id  TEXT NOT NULL,
    rating   REAL NOT NULL,
    PRIMARY KEY (movie_id, user_id)
);

-- 3NF: transitive dependency removed; director metadata lives in directors_3nf
CREATE TABLE IF NOT EXISTS directors_3nf (
    director_id   TEXT PRIMARY KEY,
    director_name TEXT NOT NULL,
    birth_year    INTEGER,
    nationality   TEXT
);
CREATE TABLE IF NOT EXISTS movies_3nf (
    movie_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    genre       TEXT,
    year        INTEGER,
    director_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users_3nf (
    user_id   TEXT PRIMARY KEY,
    user_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ratings_3nf (
    movie_id TEXT NOT NULL,
    user_id  TEXT NOT NULL,
    rating   REAL NOT NULL,
    PRIMARY KEY (movie_id, user_id)
);
"""


class NormalizationDemo:
    """
    Manages four progressive schemas in a single SQLite database — UNF, 1NF,
    2NF, 3NF — each populated from the same seed data so that the schemas can
    be compared side-by-side.  Anomaly demonstrations mutate 2NF/3NF tables
    directly so the before/after counts are meaningful; tests should use a
    fresh instance per test case.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_ALL)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(
        self,
        movies: List[Dict],
        users: List[Dict],
        ratings: List[Tuple],
    ) -> None:
        """Populate all four normal-form stages from the canonical seed data."""
        user_map = {u["id"]: u["username"] for u in users}
        ratings_by_movie: Dict[str, List] = {}
        for uid, mid, score, _ in ratings:
            ratings_by_movie.setdefault(mid, []).append((uid, score))

        # ---- UNF: one row per movie, ratings packed into ratings_csv ----
        for movie in movies:
            dmeta = DIRECTOR_METADATA.get(movie["director"], {})
            pairs = ratings_by_movie.get(movie["id"], [])
            csv = "|".join(
                f"{uid}:{user_map.get(uid, uid)}:{score}"
                for uid, score in pairs
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO movies_unf VALUES (?,?,?,?,?,?,?,?)",
                (
                    movie["id"], movie["title"], movie["genre"], movie["year"],
                    movie["director"],
                    dmeta.get("birth_year"), dmeta.get("nationality"),
                    csv if csv else None,
                ),
            )

        # ---- 1NF: one row per (movie, rating) ----
        for movie in movies:
            dmeta = DIRECTOR_METADATA.get(movie["director"], {})
            for uid, score in ratings_by_movie.get(movie["id"], []):
                self._conn.execute(
                    "INSERT OR IGNORE INTO movies_1nf VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        movie["id"], movie["title"], movie["genre"], movie["year"],
                        movie["director"],
                        dmeta.get("birth_year"), dmeta.get("nationality"),
                        uid, user_map.get(uid, uid), score,
                    ),
                )

        # ---- 2NF: three tables ----
        for movie in movies:
            dmeta = DIRECTOR_METADATA.get(movie["director"], {})
            self._conn.execute(
                "INSERT OR IGNORE INTO movies_2nf VALUES (?,?,?,?,?,?,?)",
                (
                    movie["id"], movie["title"], movie["genre"], movie["year"],
                    movie["director"],
                    dmeta.get("birth_year"), dmeta.get("nationality"),
                ),
            )
        for user in users:
            self._conn.execute(
                "INSERT OR IGNORE INTO users_2nf VALUES (?,?)",
                (user["id"], user["username"]),
            )
        for uid, mid, score, _ in ratings:
            self._conn.execute(
                "INSERT OR IGNORE INTO ratings_2nf VALUES (?,?,?)",
                (mid, uid, score),
            )

        # ---- 3NF: four tables ----
        seen: set = set()
        for movie in movies:
            dname = movie["director"]
            did = _director_slug(dname)
            if did not in seen:
                dmeta = DIRECTOR_METADATA.get(dname, {})
                self._conn.execute(
                    "INSERT OR IGNORE INTO directors_3nf VALUES (?,?,?,?)",
                    (did, dname, dmeta.get("birth_year"), dmeta.get("nationality")),
                )
                seen.add(did)
            self._conn.execute(
                "INSERT OR IGNORE INTO movies_3nf VALUES (?,?,?,?,?)",
                (movie["id"], movie["title"], movie["genre"], movie["year"], did),
            )
        for user in users:
            self._conn.execute(
                "INSERT OR IGNORE INTO users_3nf VALUES (?,?)",
                (user["id"], user["username"]),
            )
        for uid, mid, score, _ in ratings:
            self._conn.execute(
                "INSERT OR IGNORE INTO ratings_3nf VALUES (?,?,?)",
                (mid, uid, score),
            )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def row_count(self, table: str) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def table_columns(self, table: str) -> List[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r["name"] for r in rows]

    def list_tables(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------
    # 1NF analysis
    # ------------------------------------------------------------------

    def count_non_atomic_values(self) -> Dict:
        """Report how many UNF rows contain multi-valued ratings_csv fields."""
        total = self.row_count("movies_unf")
        multi = self._conn.execute(
            "SELECT COUNT(*) FROM movies_unf WHERE ratings_csv LIKE '%|%'"
        ).fetchone()[0]
        has_ratings = self._conn.execute(
            "SELECT COUNT(*) FROM movies_unf WHERE ratings_csv IS NOT NULL"
        ).fetchone()[0]
        return {
            "total_unf_movies": total,
            "movies_with_ratings": has_ratings,
            "movies_with_multiple_ratings": multi,
            "rows_in_1nf": self.row_count("movies_1nf"),
        }

    # ------------------------------------------------------------------
    # 2NF analysis — partial dependency
    # ------------------------------------------------------------------

    def count_redundant_user_data_in_1nf(self) -> Dict:
        """
        Measure the partial-dependency redundancy in 1NF: user_name depends
        only on user_id (not on the composite key), so it is repeated once per
        movie the user has rated.
        """
        total_1nf = self.row_count("movies_1nf")
        unique_user_pairs = self._conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT user_id, user_name FROM movies_1nf)"
        ).fetchone()[0]
        return {
            "total_1nf_rows": total_1nf,
            "unique_user_name_pairs": unique_user_pairs,
            "redundant_user_name_rows": total_1nf - unique_user_pairs,
            "nf2_user_rows": self.row_count("users_2nf"),
        }

    # ------------------------------------------------------------------
    # 3NF analysis — transitive dependency
    # ------------------------------------------------------------------

    def count_redundant_director_data_in_2nf(self) -> Dict:
        """
        Measure the transitive-dependency redundancy in 2NF: director_birth_year
        and director_nationality depend on director_name (not on movie_id), so
        they repeat once per film by the same director.
        """
        total_movies = self.row_count("movies_2nf")
        unique_directors = self._conn.execute(
            "SELECT COUNT(DISTINCT director_name) FROM movies_2nf"
        ).fetchone()[0]
        multi = self._conn.execute(
            "SELECT director_name, COUNT(*) AS cnt FROM movies_2nf "
            "GROUP BY director_name HAVING cnt > 1"
        ).fetchall()
        return {
            "total_movies_2nf": total_movies,
            "unique_directors": unique_directors,
            "redundant_director_rows": total_movies - unique_directors,
            "multi_movie_directors": [
                {"name": r["director_name"], "movie_count": r["cnt"]} for r in multi
            ],
            "nf3_director_rows": self.row_count("directors_3nf"),
        }

    # ------------------------------------------------------------------
    # Update anomaly
    # ------------------------------------------------------------------

    def demonstrate_update_anomaly(
        self, director_name: str = "Denis Villeneuve"
    ) -> Dict:
        """
        Correct a director's nationality.  In 2NF (movies_2nf) the update touches
        one row per film by that director; in 3NF (directors_3nf) it touches
        exactly one row regardless of filmography size.
        """
        movies_in_2nf = self._conn.execute(
            "SELECT COUNT(*) FROM movies_2nf WHERE director_name = ?",
            (director_name,),
        ).fetchone()[0]

        # 2NF update — count affected rows, then undo
        cur2 = self._conn.execute(
            "UPDATE movies_2nf SET director_nationality = 'Québécois' "
            "WHERE director_name = ?",
            (director_name,),
        )
        rows_updated_2nf = cur2.rowcount
        original_nat = DIRECTOR_METADATA.get(director_name, {}).get("nationality", "Unknown")
        self._conn.execute(
            "UPDATE movies_2nf SET director_nationality = ? WHERE director_name = ?",
            (original_nat, director_name),
        )
        self._conn.commit()

        # 3NF update — count affected rows, then undo
        did = _director_slug(director_name)
        cur3 = self._conn.execute(
            "UPDATE directors_3nf SET nationality = 'Québécois' WHERE director_id = ?",
            (did,),
        )
        rows_updated_3nf = cur3.rowcount
        self._conn.execute(
            "UPDATE directors_3nf SET nationality = ? WHERE director_id = ?",
            (original_nat, did),
        )
        self._conn.commit()

        return {
            "director": director_name,
            "rows_to_update_in_2nf": movies_in_2nf,
            "rows_actually_updated_in_2nf": rows_updated_2nf,
            "rows_to_update_in_3nf": 1,
            "rows_actually_updated_in_3nf": rows_updated_3nf,
            "inconsistency_risk_in_2nf": movies_in_2nf > 1,
        }

    # ------------------------------------------------------------------
    # Insertion anomaly
    # ------------------------------------------------------------------

    def demonstrate_insertion_anomaly(self) -> Dict:
        """
        Insert a new director who has not yet made a film.  In 3NF this is a
        single INSERT into directors_3nf.  In 2NF the director's metadata is
        embedded in movies_2nf, so it cannot exist without a movie row.
        """
        new_id = "alejandro_inarritu"
        new_name = "Alejandro González Iñárritu"

        # 3NF: director row stands alone
        self._conn.execute(
            "INSERT OR IGNORE INTO directors_3nf VALUES (?,?,?,?)",
            (new_id, new_name, 1963, "Mexican"),
        )
        self._conn.commit()
        in_3nf = (
            self._conn.execute(
                "SELECT COUNT(*) FROM directors_3nf WHERE director_id = ?", (new_id,)
            ).fetchone()[0]
            == 1
        )

        # 2NF: how many movies_2nf rows carry this director? (should be zero)
        movies_in_2nf = self._conn.execute(
            "SELECT COUNT(*) FROM movies_2nf WHERE director_name = ?", (new_name,)
        ).fetchone()[0]

        # Cleanup
        self._conn.execute(
            "DELETE FROM directors_3nf WHERE director_id = ?", (new_id,)
        )
        self._conn.commit()

        return {
            "director_name": new_name,
            "can_insert_director_in_3nf_without_movie": in_3nf,
            "director_rows_in_2nf_without_movie": movies_in_2nf,
            "insertion_anomaly_in_2nf": movies_in_2nf == 0,
        }

    # ------------------------------------------------------------------
    # Deletion anomaly
    # ------------------------------------------------------------------

    def demonstrate_deletion_anomaly(self, movie_id: str = "m17") -> Dict:
        """
        Delete the sole film by Ari Aster (Hereditary, m17).  In 2NF the
        director's birth_year and nationality are stored only in movies_2nf, so
        deleting the last movie destroys all knowledge of the director.  In 3NF
        directors_3nf retains the record independently.

        The deleted rows are restored after the demonstration so the instance
        remains usable for subsequent calls.
        """
        row_2nf = self._conn.execute(
            "SELECT * FROM movies_2nf WHERE movie_id = ?", (movie_id,)
        ).fetchone()
        row_3nf = self._conn.execute(
            "SELECT * FROM movies_3nf WHERE movie_id = ?", (movie_id,)
        ).fetchone()

        if row_2nf is None:
            return {"error": f"movie {movie_id} not found in movies_2nf"}

        director_name = row_2nf["director_name"]
        director_id = _director_slug(director_name)

        movies_by_dir_before = self._conn.execute(
            "SELECT COUNT(*) FROM movies_2nf WHERE director_name = ?",
            (director_name,),
        ).fetchone()[0]

        # Delete from 2NF
        self._conn.execute("DELETE FROM movies_2nf WHERE movie_id = ?", (movie_id,))
        self._conn.commit()
        dir_rows_after_2nf = self._conn.execute(
            "SELECT COUNT(*) FROM movies_2nf WHERE director_name = ?",
            (director_name,),
        ).fetchone()[0]
        director_lost_in_2nf = dir_rows_after_2nf == 0

        # Delete from 3NF (movie row only; directors_3nf untouched)
        self._conn.execute("DELETE FROM movies_3nf WHERE movie_id = ?", (movie_id,))
        self._conn.commit()
        director_preserved_in_3nf = (
            self._conn.execute(
                "SELECT COUNT(*) FROM directors_3nf WHERE director_id = ?",
                (director_id,),
            ).fetchone()[0]
            == 1
        )

        # Restore deleted rows so the instance stays consistent
        self._conn.execute(
            "INSERT OR IGNORE INTO movies_2nf VALUES (?,?,?,?,?,?,?)",
            (
                dict(row_2nf)["movie_id"], dict(row_2nf)["title"],
                dict(row_2nf)["genre"], dict(row_2nf)["year"],
                dict(row_2nf)["director_name"],
                dict(row_2nf)["director_birth_year"],
                dict(row_2nf)["director_nationality"],
            ),
        )
        if row_3nf:
            self._conn.execute(
                "INSERT OR IGNORE INTO movies_3nf VALUES (?,?,?,?,?)",
                (
                    dict(row_3nf)["movie_id"], dict(row_3nf)["title"],
                    dict(row_3nf)["genre"], dict(row_3nf)["year"],
                    dict(row_3nf)["director_id"],
                ),
            )
        self._conn.commit()

        return {
            "director_name": director_name,
            "movie_id": movie_id,
            "movies_by_director_before_deletion": movies_by_dir_before,
            "director_info_lost_in_2nf": director_lost_in_2nf,
            "director_info_preserved_in_3nf": director_preserved_in_3nf,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def table_counts_summary(self) -> Dict[str, int]:
        """Row counts for every normalization-stage table."""
        tables = [
            "movies_unf",
            "movies_1nf",
            "movies_2nf", "users_2nf", "ratings_2nf",
            "directors_3nf", "movies_3nf", "users_3nf", "ratings_3nf",
        ]
        return {t: self.row_count(t) for t in tables}

    def close(self) -> None:
        self._conn.close()
