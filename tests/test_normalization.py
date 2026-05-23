"""Tests for the Database Normalization module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, USERS, RATINGS
from databaseai.normalization import NormalizationDemo, DIRECTOR_METADATA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo(tmp_path):
    """Fresh demo backed by a real WAL-mode file; seeded and cleaned up after each test."""
    db_path = str(tmp_path / "test_norm.db")
    d = NormalizationDemo(db_path=db_path)
    d.seed(MOVIES, USERS, RATINGS)
    yield d
    d.close()


@pytest.fixture
def empty_demo(tmp_path):
    """Demo with schema created but no seed data."""
    db_path = str(tmp_path / "empty_norm.db")
    d = NormalizationDemo(db_path=db_path)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

class TestDirectorMetadata:

    def test_director_metadata_covers_all_seed_directors(self):
        """Every director in the movie seed list should have a metadata entry."""
        seed_directors = {m["director"] for m in MOVIES}
        for director in seed_directors:
            assert director in DIRECTOR_METADATA, (
                f"Missing DIRECTOR_METADATA entry for '{director}'"
            )

    def test_each_entry_has_birth_year_and_nationality(self):
        for name, meta in DIRECTOR_METADATA.items():
            assert "birth_year" in meta, f"{name} missing birth_year"
            assert "nationality" in meta, f"{name} missing nationality"
            assert isinstance(meta["birth_year"], int)
            assert isinstance(meta["nationality"], str)

    def test_christopher_nolan_is_british(self):
        assert DIRECTOR_METADATA["Christopher Nolan"]["nationality"] == "British"

    def test_hayao_miyazaki_is_japanese(self):
        assert DIRECTOR_METADATA["Hayao Miyazaki"]["nationality"] == "Japanese"


# ---------------------------------------------------------------------------
# 2. UNF stage
# ---------------------------------------------------------------------------

class TestUNF:

    def test_unf_has_one_row_per_movie(self, demo):
        """movies_unf should contain exactly one row per movie in MOVIES."""
        assert demo.row_count("movies_unf") == len(MOVIES)

    def test_unf_ratings_csv_is_non_atomic(self, demo):
        """Movies with multiple ratings should have a pipe-separated ratings_csv."""
        info = demo.count_non_atomic_values()
        assert info["movies_with_multiple_ratings"] > 0, (
            "Expected some movies with multiple ratings; ratings_csv looks empty"
        )

    def test_unf_has_eight_columns(self, demo):
        cols = demo.table_columns("movies_unf")
        assert len(cols) == 8

    def test_unf_contains_ratings_csv_column(self, demo):
        assert "ratings_csv" in demo.table_columns("movies_unf")


# ---------------------------------------------------------------------------
# 3. 1NF stage
# ---------------------------------------------------------------------------

class TestFirstNormalForm:

    def test_1nf_expands_ratings_into_individual_rows(self, demo):
        """The 1NF table must have more rows than the UNF table (ratings exploded)."""
        assert demo.row_count("movies_1nf") > demo.row_count("movies_unf")

    def test_1nf_row_count_equals_total_ratings(self, demo):
        """Each (movie, user) rating becomes exactly one row in 1NF."""
        assert demo.row_count("movies_1nf") == len(RATINGS)

    def test_1nf_has_composite_pk_columns(self, demo):
        cols = demo.table_columns("movies_1nf")
        assert "movie_id" in cols
        assert "user_id" in cols

    def test_1nf_no_duplicate_movie_user_pairs(self, demo):
        """The composite PK constraint means no (movie_id, user_id) appears twice."""
        import sqlite3
        conn = sqlite3.connect(demo._db_path)
        dupes = conn.execute(
            "SELECT movie_id, user_id, COUNT(*) AS cnt "
            "FROM movies_1nf GROUP BY movie_id, user_id HAVING cnt > 1"
        ).fetchall()
        conn.close()
        assert len(dupes) == 0, f"Duplicate (movie_id, user_id) pairs found: {dupes}"

    def test_count_non_atomic_values_structure(self, demo):
        info = demo.count_non_atomic_values()
        assert "total_unf_movies" in info
        assert "rows_in_1nf" in info
        assert info["rows_in_1nf"] > info["total_unf_movies"]


# ---------------------------------------------------------------------------
# 4. 2NF stage — partial dependency eliminated
# ---------------------------------------------------------------------------

class TestSecondNormalForm:

    def test_2nf_movies_has_one_row_per_movie(self, demo):
        assert demo.row_count("movies_2nf") == len(MOVIES)

    def test_2nf_users_has_one_row_per_user(self, demo):
        assert demo.row_count("users_2nf") == len(USERS)

    def test_2nf_ratings_has_one_row_per_rating(self, demo):
        assert demo.row_count("ratings_2nf") == len(RATINGS)

    def test_2nf_movies_does_not_contain_user_name_column(self, demo):
        """Partial dependency removed: user_name must NOT be in movies_2nf."""
        assert "user_name" not in demo.table_columns("movies_2nf")

    def test_2nf_redundant_user_data_analysis(self, demo):
        info = demo.count_redundant_user_data_in_1nf()
        assert info["redundant_user_name_rows"] > 0, (
            "Expected redundant user_name rows in 1NF; is seed data loaded?"
        )
        assert info["nf2_user_rows"] == len(USERS)
        assert info["nf2_user_rows"] < info["total_1nf_rows"]


# ---------------------------------------------------------------------------
# 5. 3NF stage — transitive dependency eliminated
# ---------------------------------------------------------------------------

class TestThirdNormalForm:

    def test_3nf_has_separate_directors_table(self, demo):
        assert "directors_3nf" in demo.list_tables()

    def test_3nf_directors_has_one_row_per_unique_director(self, demo):
        unique_directors = len({m["director"] for m in MOVIES})
        assert demo.row_count("directors_3nf") == unique_directors

    def test_3nf_movies_does_not_contain_director_metadata(self, demo):
        """Transitive dependency removed: birth_year/nationality must not be in movies_3nf."""
        cols = demo.table_columns("movies_3nf")
        assert "director_birth_year" not in cols
        assert "director_nationality" not in cols

    def test_3nf_movies_has_director_id_foreign_key(self, demo):
        assert "director_id" in demo.table_columns("movies_3nf")

    def test_3nf_director_id_references_valid_director(self, demo):
        import sqlite3
        conn = sqlite3.connect(demo._db_path)
        orphans = conn.execute(
            "SELECT m.movie_id FROM movies_3nf m "
            "LEFT JOIN directors_3nf d ON m.director_id = d.director_id "
            "WHERE d.director_id IS NULL"
        ).fetchall()
        conn.close()
        assert len(orphans) == 0, f"Orphan director_id references: {orphans}"

    def test_3nf_redundant_director_analysis(self, demo):
        info = demo.count_redundant_director_data_in_2nf()
        assert info["redundant_director_rows"] > 0, (
            "Expected some directors with multiple movies in 2NF (e.g. Christopher Nolan)"
        )
        assert info["nf3_director_rows"] < info["total_movies_2nf"]
        assert len(info["multi_movie_directors"]) > 0


# ---------------------------------------------------------------------------
# 6. Update anomaly
# ---------------------------------------------------------------------------

class TestUpdateAnomaly:

    def test_update_anomaly_requires_more_rows_in_2nf_than_3nf(self, demo):
        result = demo.demonstrate_update_anomaly("Denis Villeneuve")
        assert result["rows_to_update_in_2nf"] > result["rows_to_update_in_3nf"]

    def test_update_anomaly_2nf_rows_equals_movie_count_for_director(self, demo):
        """Denis Villeneuve directed m11 and m18 → 2 movies_2nf rows to update."""
        result = demo.demonstrate_update_anomaly("Denis Villeneuve")
        assert result["rows_to_update_in_2nf"] == 2

    def test_update_anomaly_3nf_always_one_row(self, demo):
        result = demo.demonstrate_update_anomaly("Denis Villeneuve")
        assert result["rows_actually_updated_in_3nf"] == 1

    def test_update_anomaly_does_not_permanently_change_nationality(self, demo):
        """demonstrate_update_anomaly must restore the original value."""
        import sqlite3
        demo.demonstrate_update_anomaly("Denis Villeneuve")
        conn = sqlite3.connect(demo._db_path)
        nat = conn.execute(
            "SELECT director_nationality FROM movies_2nf "
            "WHERE director_name = 'Denis Villeneuve' LIMIT 1"
        ).fetchone()[0]
        conn.close()
        assert nat == DIRECTOR_METADATA["Denis Villeneuve"]["nationality"]

    def test_inconsistency_risk_flag_for_multi_movie_director(self, demo):
        result = demo.demonstrate_update_anomaly("Denis Villeneuve")
        assert result["inconsistency_risk_in_2nf"] is True

    def test_single_movie_director_has_no_inconsistency_risk(self, demo):
        result = demo.demonstrate_update_anomaly("Ari Aster")
        assert result["inconsistency_risk_in_2nf"] is False


# ---------------------------------------------------------------------------
# 7. Insertion anomaly
# ---------------------------------------------------------------------------

class TestInsertionAnomaly:

    def test_3nf_allows_director_without_movie(self, demo):
        result = demo.demonstrate_insertion_anomaly()
        assert result["can_insert_director_in_3nf_without_movie"] is True

    def test_2nf_has_no_rows_for_new_director(self, demo):
        """A brand-new director cannot appear in movies_2nf without a movie."""
        result = demo.demonstrate_insertion_anomaly()
        assert result["director_rows_in_2nf_without_movie"] == 0
        assert result["insertion_anomaly_in_2nf"] is True

    def test_insertion_anomaly_cleanup_leaves_3nf_intact(self, demo):
        """demonstrate_insertion_anomaly must clean up the test director."""
        count_before = demo.row_count("directors_3nf")
        demo.demonstrate_insertion_anomaly()
        count_after = demo.row_count("directors_3nf")
        assert count_before == count_after


# ---------------------------------------------------------------------------
# 8. Deletion anomaly
# ---------------------------------------------------------------------------

class TestDeletionAnomaly:

    def test_deletion_destroys_director_info_in_2nf(self, demo):
        """Deleting Hereditary (m17) from movies_2nf erases all trace of Ari Aster."""
        result = demo.demonstrate_deletion_anomaly("m17")
        assert result["director_info_lost_in_2nf"] is True

    def test_deletion_preserves_director_info_in_3nf(self, demo):
        """Deleting the movie from movies_3nf must not remove directors_3nf entry."""
        result = demo.demonstrate_deletion_anomaly("m17")
        assert result["director_info_preserved_in_3nf"] is True

    def test_deletion_anomaly_restores_movie_after_demo(self, demo):
        """demonstrate_deletion_anomaly must restore deleted rows."""
        count_before = demo.row_count("movies_2nf")
        demo.demonstrate_deletion_anomaly("m17")
        count_after = demo.row_count("movies_2nf")
        assert count_before == count_after

    def test_deletion_anomaly_reports_correct_movie_and_director(self, demo):
        result = demo.demonstrate_deletion_anomaly("m17")
        assert result["movie_id"] == "m17"
        assert result["director_name"] == "Ari Aster"


# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------

class TestSummary:

    def test_table_counts_summary_covers_all_stages(self, demo):
        summary = demo.table_counts_summary()
        required = {
            "movies_unf", "movies_1nf",
            "movies_2nf", "users_2nf", "ratings_2nf",
            "directors_3nf", "movies_3nf", "users_3nf", "ratings_3nf",
        }
        assert required.issubset(summary.keys())

    def test_empty_demo_has_zero_rows_everywhere(self, empty_demo):
        summary = empty_demo.table_counts_summary()
        for table, count in summary.items():
            assert count == 0, f"{table} has {count} rows before seed()"
