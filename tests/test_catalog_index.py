"""Tests for the Catalog Browse Indexing module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.catalog_index import (
    CatalogIndexDemo,
    QUERY_BTREE_TOP_RATED,
    QUERY_COMPOSITE_BOTH,
    QUERY_COMPOSITE_LEFT,
    QUERY_COMPOSITE_RIGHT_ONLY,
    QUERY_COVERING_CAROUSEL,
    QUERY_PARTIAL_MATCH,
    QUERY_PARTIAL_NO_MATCH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo():
    """Fresh CatalogIndexDemo loaded with seed data + synthetic rows, no user indexes."""
    d = CatalogIndexDemo()
    d.seed(MOVIES, RATINGS)
    d.seed_large(n_titles=4000)
    d.analyze()
    yield d
    d.close()


@pytest.fixture
def empty_demo():
    """Demo with just seed data -- no synthetic rows, no indexes."""
    d = CatalogIndexDemo()
    d.seed(MOVIES, RATINGS)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. Full table scan baseline
# ---------------------------------------------------------------------------

class TestFullScan:

    def test_top_rated_query_uses_full_scan_without_index(self, demo):
        """Without any index on rating_avg, the shelf query must scan every row."""
        assert demo.uses_full_scan(QUERY_BTREE_TOP_RATED), (
            "Expected SCAN (no index on rating_avg) but plan showed an index"
        )

    def test_carousel_query_uses_full_scan_without_index(self, demo):
        """Without an index on genre, the carousel projection scans the whole table."""
        assert demo.uses_full_scan(QUERY_COVERING_CAROUSEL, ("sci-fi",))

    def test_partial_match_query_scans_without_index(self, demo):
        assert demo.uses_full_scan(QUERY_PARTIAL_MATCH)


# ---------------------------------------------------------------------------
# 2. B-tree index
# ---------------------------------------------------------------------------

class TestBtreeIndex:

    def test_btree_index_changes_plan_to_index_search(self, demo):
        """After adding a B-tree index on rating_avg, range queries must use it."""
        demo.create_btree_index()
        assert demo.uses_index(QUERY_BTREE_TOP_RATED), (
            "Expected index scan after CREATE INDEX on rating_avg"
        )

    def test_drop_btree_index_reverts_to_full_scan(self, demo):
        """Dropping the B-tree index must revert the query plan to a full scan."""
        demo.create_btree_index()
        demo.drop_btree_index()
        assert demo.uses_full_scan(QUERY_BTREE_TOP_RATED)

    def test_btree_index_appears_in_list(self, demo):
        demo.create_btree_index()
        names = {i["name"] for i in demo.list_indexes()}
        assert "idx_btree_rating" in names

    def test_btree_index_drop_removes_from_list(self, demo):
        demo.create_btree_index()
        demo.drop_btree_index()
        names = {i["name"] for i in demo.list_indexes()}
        assert "idx_btree_rating" not in names


# ---------------------------------------------------------------------------
# 3. Composite index -- leftmost prefix rule
# ---------------------------------------------------------------------------

class TestCompositeIndex:

    def test_composite_index_used_for_both_columns(self, demo):
        """Composite (genre, year) must accelerate genre+decade browse rows."""
        demo.create_composite_index()
        assert demo.uses_index(QUERY_COMPOSITE_BOTH, ("sci-fi", 2010))

    def test_composite_index_used_for_left_column_only(self, demo):
        """Left prefix (genre only) should still use the composite index."""
        demo.create_composite_index()
        assert demo.uses_index(QUERY_COMPOSITE_LEFT, ("sci-fi",))

    def test_composite_index_not_used_for_right_column_alone(self, demo):
        """Querying only the right column (year) bypasses the composite index."""
        demo.create_composite_index()
        assert demo.uses_full_scan(QUERY_COMPOSITE_RIGHT_ONLY, (2010,))


# ---------------------------------------------------------------------------
# 4. Covering index
# ---------------------------------------------------------------------------

class TestCoveringIndex:

    def test_covering_index_plan_shows_covering_keyword(self, demo):
        """USING COVERING INDEX must appear when all projected columns are indexed."""
        demo.create_covering_index()
        assert demo.uses_covering_index(QUERY_COVERING_CAROUSEL, ("sci-fi",)), (
            "Expected 'USING COVERING INDEX' in plan -- check that index includes "
            "all projected columns (genre, title, rating_avg)"
        )

    def test_regular_index_does_not_show_covering_keyword(self, demo):
        """A non-covering index uses USING INDEX (with table lookup), not COVERING."""
        demo.create_btree_index()
        assert not demo.uses_covering_index(QUERY_BTREE_TOP_RATED)


# ---------------------------------------------------------------------------
# 5. Partial index
# ---------------------------------------------------------------------------

class TestPartialIndex:

    def test_partial_index_used_for_matching_predicate(self, demo):
        """Partial index WHERE year>=2020 is used when the query predicate matches exactly."""
        demo.create_partial_index()
        assert demo.uses_index(QUERY_PARTIAL_MATCH), (
            "Expected optimizer to use partial index for 'year >= 2020' "
            "(syntactically identical to the index predicate)"
        )

    def test_partial_index_not_used_for_non_matching_predicate(self, demo):
        """year >= 2010 spans rows outside the partial index; expect full scan."""
        demo.create_partial_index()
        assert demo.uses_full_scan(QUERY_PARTIAL_NO_MATCH), (
            "Partial index WHERE year>=2020 must NOT be used for 'year >= 2010'"
        )

    def test_partial_index_listed_separately_from_btree(self, demo):
        """Both a B-tree and a partial index can coexist on different columns."""
        demo.create_btree_index()
        demo.create_partial_index()
        names = {i["name"] for i in demo.list_indexes()}
        assert "idx_btree_rating" in names
        assert "idx_partial_new_releases" in names
        assert len(names) == 2


# ---------------------------------------------------------------------------
# 6. Introspection and catalog
# ---------------------------------------------------------------------------

class TestIndexCatalog:

    def test_list_indexes_empty_at_start(self, demo):
        """No user-defined indexes exist before any create_* call."""
        assert demo.list_indexes() == []

    def test_list_indexes_shows_all_four_after_creation(self, demo):
        demo.create_btree_index()
        demo.create_composite_index()
        demo.create_covering_index()
        demo.create_partial_index()
        names = {i["name"] for i in demo.list_indexes()}
        expected = {
            "idx_btree_rating",
            "idx_composite_genre_year",
            "idx_covering_genre_carousel",
            "idx_partial_new_releases",
        }
        assert expected == names

    def test_list_indexes_table_name_is_correct(self, demo):
        demo.create_btree_index()
        demo.create_composite_index()
        idx_map = {i["name"]: i["table_name"] for i in demo.list_indexes()}
        assert idx_map["idx_btree_rating"] == "catalog"
        assert idx_map["idx_composite_genre_year"] == "catalog"

    def test_row_count_matches_seeded_data(self, empty_demo):
        assert empty_demo.row_count("catalog") == len(MOVIES)

    def test_seed_large_adds_rows(self, empty_demo):
        before = empty_demo.row_count("catalog")
        empty_demo.seed_large(n_titles=500)
        after = empty_demo.row_count("catalog")
        assert after == before + 500

    def test_explain_returns_non_empty_string(self, demo):
        plan = demo.explain(QUERY_BTREE_TOP_RATED)
        assert isinstance(plan, str)
        assert len(plan) > 0

    def test_time_query_returns_positive_microseconds(self, demo):
        t = demo.time_query(QUERY_BTREE_TOP_RATED, repeat=10)
        assert t > 0


# ---------------------------------------------------------------------------
# 7. Seeding from real movie data
# ---------------------------------------------------------------------------

class TestSeedFromMovies:

    def test_seeded_titles_match_movie_titles(self, empty_demo):
        rows = empty_demo._conn.execute(
            "SELECT id, title FROM catalog ORDER BY id"
        ).fetchall()
        seeded = {(r["id"], r["title"]) for r in rows}
        expected = {(m["id"], m["title"]) for m in MOVIES}
        assert seeded == expected

    def test_rating_avg_reflects_seeded_ratings(self, empty_demo):
        """A movie with ratings must have a non-null rating_avg between 1 and 5."""
        row = empty_demo._conn.execute(
            "SELECT rating_avg FROM catalog WHERE id = 'm01'"
        ).fetchone()
        assert row["rating_avg"] is not None
        assert 1.0 <= row["rating_avg"] <= 5.0
