"""Tests for the Indexing Strategies module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.indexing import (
    IndexingDemo,
    QUERY_BTREE,
    QUERY_COMPOSITE_BOTH,
    QUERY_COMPOSITE_LEFT,
    QUERY_COVERING,
    QUERY_PARTIAL_MATCH,
    QUERY_PARTIAL_NO_MATCH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo():
    """Fresh IndexingDemo loaded with seed data + synthetic rows, no user indexes."""
    d = IndexingDemo()
    d.seed(MOVIES, RATINGS)
    d.seed_large(n_ratings=4000)
    d.analyze()
    yield d
    d.close()


@pytest.fixture
def empty_demo():
    """Demo with just seed data -- no synthetic rows, no indexes."""
    d = IndexingDemo()
    d.seed(MOVIES, RATINGS)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. Full table scan baseline
# ---------------------------------------------------------------------------

class TestFullScan:

    def test_score_range_uses_full_scan_without_index(self, demo):
        """Without any index on score, the range query must scan every row."""
        assert demo.uses_full_scan(QUERY_BTREE), (
            "Expected SCAN (no index on score) but plan showed an index"
        )

    def test_covering_query_uses_full_scan_without_index(self, demo):
        """Without an index on user_id, the equality lookup scans the whole table."""
        assert demo.uses_full_scan(QUERY_COVERING, ("ux0050",))

    def test_partial_match_query_scans_without_index(self, demo):
        assert demo.uses_full_scan(QUERY_PARTIAL_MATCH)


# ---------------------------------------------------------------------------
# 2. B-tree index
# ---------------------------------------------------------------------------

class TestBtreeIndex:

    def test_btree_index_changes_plan_to_index_search(self, demo):
        """After adding a B-tree index on score, range queries must use it."""
        demo.create_btree_index()
        assert demo.uses_index(QUERY_BTREE), (
            "Expected index scan after CREATE INDEX on score"
        )

    def test_drop_btree_index_reverts_to_full_scan(self, demo):
        """Dropping the B-tree index must revert the query plan to a full scan."""
        demo.create_btree_index()
        demo.drop_btree_index()
        assert demo.uses_full_scan(QUERY_BTREE)

    def test_btree_index_appears_in_list(self, demo):
        demo.create_btree_index()
        names = {i["name"] for i in demo.list_indexes()}
        assert "idx_btree_score" in names

    def test_btree_index_drop_removes_from_list(self, demo):
        demo.create_btree_index()
        demo.drop_btree_index()
        names = {i["name"] for i in demo.list_indexes()}
        assert "idx_btree_score" not in names


# ---------------------------------------------------------------------------
# 3. Composite index -- leftmost prefix rule
# ---------------------------------------------------------------------------

class TestCompositeIndex:

    def test_composite_index_used_for_both_columns(self, demo):
        """Composite (genre, year) must accelerate queries that filter on both."""
        demo.create_composite_index()
        assert demo.uses_index(QUERY_COMPOSITE_BOTH, ("sci-fi", 2010))

    def test_composite_index_used_for_left_column_only(self, demo):
        """Left prefix (genre only) should still use the composite index."""
        demo.create_composite_index()
        assert demo.uses_index(QUERY_COMPOSITE_LEFT, ("sci-fi",))

    def test_composite_index_not_used_for_right_column_alone(self, demo):
        """Querying only the right column (year) bypasses the composite index."""
        demo.create_composite_index()
        from databaseai.indexing import QUERY_COMPOSITE_RIGHT_ONLY
        assert demo.uses_full_scan(QUERY_COMPOSITE_RIGHT_ONLY, (2010,))


# ---------------------------------------------------------------------------
# 4. Covering index
# ---------------------------------------------------------------------------

class TestCoveringIndex:

    def test_covering_index_plan_shows_covering_keyword(self, demo):
        """USING COVERING INDEX must appear when all projected columns are indexed."""
        demo.create_covering_index()
        assert demo.uses_covering_index(QUERY_COVERING, ("ux0050",)), (
            "Expected 'USING COVERING INDEX' in plan -- check that index includes "
            "all projected columns (user_id, score)"
        )

    def test_regular_index_does_not_show_covering_keyword(self, demo):
        """A non-covering index uses USING INDEX (with table lookup), not COVERING."""
        demo.create_btree_index()
        assert not demo.uses_covering_index(QUERY_BTREE)


# ---------------------------------------------------------------------------
# 5. Partial index
# ---------------------------------------------------------------------------

class TestPartialIndex:

    def test_partial_index_used_for_matching_predicate(self, demo):
        """Partial index WHERE score>=4.0 must be used when query asks for score>=4.0."""
        demo.create_partial_index()
        assert demo.uses_index(QUERY_PARTIAL_MATCH), (
            "Expected optimizer to use partial index for 'score >= 4.0'"
        )

    def test_partial_index_not_used_for_non_matching_predicate(self, demo):
        """score >= 3.0 spans rows outside the partial index; expect full scan."""
        demo.create_partial_index()
        assert demo.uses_full_scan(QUERY_PARTIAL_NO_MATCH), (
            "Partial index WHERE score>=4.0 must NOT be used for 'score >= 3.0'"
        )

    def test_partial_index_listed_separately_from_btree(self, demo):
        """Both a B-tree and a partial index on score can coexist."""
        demo.create_btree_index()
        demo.create_partial_index()
        names = {i["name"] for i in demo.list_indexes()}
        assert "idx_btree_score" in names
        assert "idx_partial_high_score" in names
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
            "idx_btree_score",
            "idx_composite_genre_year",
            "idx_covering_user_score",
            "idx_partial_high_score",
        }
        assert expected == names

    def test_list_indexes_table_name_is_correct(self, demo):
        demo.create_btree_index()
        demo.create_composite_index()
        idx_map = {i["name"]: i["table_name"] for i in demo.list_indexes()}
        assert idx_map["idx_btree_score"] == "ratings_idx"
        assert idx_map["idx_composite_genre_year"] == "movies_idx"

    def test_row_count_matches_seeded_data(self, empty_demo):
        assert empty_demo.row_count("movies_idx") == len(MOVIES)
        assert empty_demo.row_count("ratings_idx") == len(RATINGS)

    def test_seed_large_adds_rows(self, empty_demo):
        before = empty_demo.row_count("ratings_idx")
        empty_demo.seed_large(n_ratings=500)
        after = empty_demo.row_count("ratings_idx")
        assert after == before + 500

    def test_explain_returns_non_empty_string(self, demo):
        plan = demo.explain(QUERY_BTREE)
        assert isinstance(plan, str)
        assert len(plan) > 0

    def test_time_query_returns_positive_microseconds(self, demo):
        t = demo.time_query(QUERY_BTREE, repeat=10)
        assert t > 0
