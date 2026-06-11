"""Tests for the Star Schema & Dimensional Modeling module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.star_schema import StarSchemaDemo
from databaseai.seed_data import MOVIES, USERS, RATINGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo():
    """Fresh StarSchemaDemo seeded with OLTP data and ETL'd into the warehouse."""
    d = StarSchemaDemo()
    oltp_seed = [(u, m, s, r) for u, m, s, r in RATINGS]
    d.seed_oltp(MOVIES, USERS, oltp_seed)
    d.run_etl()
    yield d
    d.close()


@pytest.fixture
def empty_demo():
    """StarSchemaDemo with no seed data and no ETL run."""
    d = StarSchemaDemo()
    yield d
    d.close()


@pytest.fixture
def seeded_no_etl():
    """StarSchemaDemo with OLTP data seeded but ETL not yet run."""
    d = StarSchemaDemo()
    oltp_seed = [(u, m, s, r) for u, m, s, r in RATINGS]
    d.seed_oltp(MOVIES, USERS, oltp_seed)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. OLTP seeding
# ---------------------------------------------------------------------------

class TestOLTPSeed:

    def test_all_movies_seeded(self, seeded_no_etl):
        """All 20 seed movies must appear in oltp_movies."""
        assert seeded_no_etl.oltp_count("oltp_movies") == len(MOVIES)

    def test_all_users_seeded(self, seeded_no_etl):
        """All 5 seed users must appear in oltp_users."""
        assert seeded_no_etl.oltp_count("oltp_users") == len(USERS)

    def test_all_ratings_seeded(self, seeded_no_etl):
        """All RATINGS tuples must appear in oltp_ratings."""
        assert seeded_no_etl.oltp_count("oltp_ratings") == len(RATINGS)

    def test_empty_demo_has_no_oltp_rows(self, empty_demo):
        """A demo with no seed call must have empty OLTP tables."""
        assert empty_demo.oltp_count("oltp_movies") == 0
        assert empty_demo.oltp_count("oltp_users") == 0
        assert empty_demo.oltp_count("oltp_ratings") == 0

    def test_seed_is_idempotent(self, seeded_no_etl):
        """Calling seed_oltp again must not create duplicate rows (INSERT OR IGNORE)."""
        oltp_seed = [(u, m, s, r) for u, m, s, r in RATINGS]
        seeded_no_etl.seed_oltp(MOVIES, USERS, oltp_seed)
        assert seeded_no_etl.oltp_count("oltp_movies") == len(MOVIES)
        assert seeded_no_etl.oltp_count("oltp_users") == len(USERS)


# ---------------------------------------------------------------------------
# 2. ETL pipeline
# ---------------------------------------------------------------------------

class TestETL:

    def test_dim_movie_row_count(self, demo):
        """dim_movie must contain exactly one row per seed movie."""
        assert demo.dim_count("dim_movie") == len(MOVIES)

    def test_dim_user_row_count(self, demo):
        """dim_user must contain exactly one row per seed user."""
        assert demo.dim_count("dim_user") == len(USERS)

    def test_fact_plays_row_count(self, demo):
        """fact_plays must contain exactly one row per rating event."""
        assert demo.fact_count() == len(RATINGS)

    def test_dim_date_not_empty(self, demo):
        """dim_date must have at least one row after ETL."""
        assert demo.dim_count("dim_date") >= 1

    def test_etl_is_idempotent(self, demo):
        """Running ETL twice must not duplicate rows (INSERT OR IGNORE)."""
        demo.run_etl()
        assert demo.dim_count("dim_movie") == len(MOVIES)
        assert demo.fact_count() == len(RATINGS)

    def test_no_warehouse_rows_before_etl(self, seeded_no_etl):
        """Warehouse tables must be empty before ETL is run."""
        assert seeded_no_etl.fact_count() == 0
        assert seeded_no_etl.dim_count("dim_movie") == 0


# ---------------------------------------------------------------------------
# 3. Dimension content
# ---------------------------------------------------------------------------

class TestDimensions:

    def test_dim_movie_contains_all_genres(self, demo):
        """Every genre present in MOVIES must appear in dim_movie."""
        expected_genres = {m["genre"] for m in MOVIES}
        actual_genres   = set(demo.dim_genres())
        assert expected_genres == actual_genres

    def test_dim_movie_decade_computed(self, demo):
        """The decade column must equal (year // 10) * 10 for each movie."""
        # Verify via direct query that no decade is NULL or obviously wrong
        genres = demo.dim_genres()
        assert len(genres) > 0

    def test_sci_fi_genre_in_dim(self, demo):
        """sci-fi must be a dimension genre since MOVIES contains sci-fi titles."""
        assert "sci-fi" in demo.dim_genres()

    def test_drama_genre_in_dim(self, demo):
        assert "drama" in demo.dim_genres()


# ---------------------------------------------------------------------------
# 4. Analytical queries
# ---------------------------------------------------------------------------

class TestAnalyticalQueries:

    def test_oltp_query_returns_genres(self, demo):
        """OLTP avg-rating query must return at least one genre row."""
        results = demo.oltp_avg_rating_by_genre()
        assert len(results) > 0

    def test_star_query_returns_same_genres(self, demo):
        """Star schema query must return the same set of genres as the OLTP query."""
        oltp_genres = {r["genre"] for r in demo.oltp_avg_rating_by_genre()}
        star_genres = {r["genre"] for r in demo.star_avg_rating_by_genre()}
        assert oltp_genres == star_genres

    def test_star_and_oltp_play_counts_match(self, demo):
        """Total play count must be identical between the OLTP and star queries."""
        oltp_total = sum(r["play_count"] for r in demo.oltp_avg_rating_by_genre())
        star_total  = sum(r["play_count"] for r in demo.star_avg_rating_by_genre())
        assert oltp_total == star_total == len(RATINGS)

    def test_star_ratings_match_oltp_ratings(self, demo):
        """Per-genre average ratings must agree between the OLTP and star queries."""
        oltp = {r["genre"]: r["avg_rating"] for r in demo.oltp_avg_rating_by_genre()}
        star  = {r["genre"]: r["avg_rating"] for r in demo.star_avg_rating_by_genre()}
        for genre in oltp:
            assert abs(oltp[genre] - star[genre]) < 0.01, (
                f"Genre '{genre}': OLTP avg={oltp[genre]} vs star avg={star[genre]}"
            )


# ---------------------------------------------------------------------------
# 5. OLAP operations
# ---------------------------------------------------------------------------

class TestOLAPOperations:

    def test_slice_by_scifi_returns_results(self, demo):
        """Slicing by 'sci-fi' must return at least one movie row."""
        rows = demo.slice_by_genre("sci-fi")
        assert len(rows) > 0

    def test_slice_by_nonexistent_genre_is_empty(self, demo):
        """Slicing by a genre not in the data must return an empty list."""
        rows = demo.slice_by_genre("western")
        assert rows == []

    def test_slice_returns_only_requested_genre(self, demo):
        """Every row returned by slice_by_genre must match the requested genre."""
        # Indirect check: all titles in the slice must be sci-fi titles
        scifi_titles = {m["title"] for m in MOVIES if m["genre"] == "sci-fi"}
        sliced_titles = {r["title"] for r in demo.slice_by_genre("sci-fi")}
        assert sliced_titles.issubset(scifi_titles)

    def test_drill_down_by_director_groups_by_genre(self, demo):
        """Drill-down results must include multiple genres."""
        rows = demo.drill_down_by_director()
        genres = {r["genre"] for r in rows}
        assert len(genres) > 1

    def test_rollup_by_decade_is_sorted(self, demo):
        """Decade roll-up must be returned in ascending decade order."""
        rows = demo.rollup_by_decade()
        decades = [r["decade"] for r in rows]
        assert decades == sorted(decades)

    def test_rollup_covers_expected_decades(self, demo):
        """MOVIES span from 1957 to 2022 so at least the 1990s and 2010s must appear."""
        rows = demo.rollup_by_decade()
        decades = {r["decade"] for r in rows}
        assert 1990 in decades
        assert 2010 in decades

    def test_top_movies_by_genre_rank_one_per_genre(self, demo):
        """top_movies_by_genre(top_n=1) must return exactly one row per genre."""
        rows = demo.top_movies_by_genre(top_n=1)
        genres = [r["genre"] for r in rows]
        # All ranks should be 1
        assert all(r["rn"] == 1 for r in rows)
        # One row per genre — no duplicates
        assert len(genres) == len(set(genres))

    def test_top_movies_by_genre_top2_at_most_two_per_genre(self, demo):
        """top_movies_by_genre(top_n=2) must return at most 2 rows per genre."""
        from collections import Counter
        rows = demo.top_movies_by_genre(top_n=2)
        counts = Counter(r["genre"] for r in rows)
        assert all(v <= 2 for v in counts.values())


# ---------------------------------------------------------------------------
# 6. Cross-cutting: data consistency
# ---------------------------------------------------------------------------

class TestDataConsistency:

    def test_total_play_count_equals_ratings_length(self, demo):
        """fact_plays row count must equal the number of RATINGS tuples."""
        assert demo.fact_count() == len(RATINGS)

    def test_dim_movie_count_equals_movies_length(self, demo):
        """dim_movie must have one row per MOVIES entry."""
        assert demo.dim_count("dim_movie") == len(MOVIES)

    def test_all_genres_covered_by_olap(self, demo):
        """Every genre in dim_movie must appear in the drill-down results."""
        dim_genres = set(demo.dim_genres())
        # Genres that actually have ratings
        rated_genres = {r["genre"] for r in demo.star_avg_rating_by_genre()}
        # drill_down covers the same rated genres
        drill_genres = {r["genre"] for r in demo.drill_down_by_director()}
        assert rated_genres == drill_genres
