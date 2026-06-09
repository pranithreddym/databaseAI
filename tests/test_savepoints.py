"""Tests for the Savepoints & Nested Rollbacks module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES
from databaseai.savepoints import SavepointDemo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo():
    """Fresh SavepointDemo loaded with seed movies."""
    d = SavepointDemo()
    d.seed(MOVIES)
    yield d
    d.close()


@pytest.fixture
def empty_demo():
    """Fresh SavepointDemo with no seed data."""
    d = SavepointDemo()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. Basic savepoint: partial rollback
# ---------------------------------------------------------------------------

class TestBasicSavepoint:

    def test_item_a_is_committed(self, demo):
        """Item inserted before the savepoint must reach disk after COMMIT."""
        result = demo.basic_savepoint(user_id="u01")
        assert result["item_a_committed"], "m01 was inserted before the savepoint and must be committed"

    def test_item_b_is_not_committed(self, demo):
        """Item inserted inside the savepoint must be absent after ROLLBACK TO."""
        result = demo.basic_savepoint(user_id="u01")
        assert not result["item_b_committed"], (
            "m02 was inserted after SAVEPOINT and should be gone after ROLLBACK TO"
        )

    def test_count_after_rollback_equals_pre_savepoint_count(self, demo):
        """Row count immediately after ROLLBACK TO must equal count before SAVEPOINT."""
        result = demo.basic_savepoint(user_id="u01")
        assert result["count_after_rollback"] == result["count_after_insert_a"]

    def test_final_committed_count_is_one(self, demo):
        """Only one row (m01) must survive COMMIT."""
        result = demo.basic_savepoint(user_id="u01")
        assert result["final_count"] == 1

    def test_count_inside_savepoint_is_two(self, demo):
        """While inside the savepoint, both rows must be visible to the same connection."""
        result = demo.basic_savepoint(user_id="u01")
        assert result["count_after_insert_b"] == 2


# ---------------------------------------------------------------------------
# 2. Nested savepoints
# ---------------------------------------------------------------------------

class TestNestedSavepoints:

    def test_only_pre_outer_savepoint_row_survives(self, demo):
        """ROLLBACK TO outer_sp must discard inner_sp's work along with outer_sp's work."""
        result = demo.nested_savepoints(user_id="u02")
        assert result["only_m01_survived"], (
            f"Expected only m01; got: {result['survived_movies']}"
        )

    def test_count_inside_nested_savepoints_is_three(self, demo):
        """With m01, m02 (outer), and m03 (inner) inserted, count must be 3."""
        result = demo.nested_savepoints(user_id="u02")
        assert result["count_with_all_three"] == 3

    def test_count_after_outer_rollback_is_one(self, demo):
        """After ROLLBACK TO outer_sp the count must revert to 1 (only m01 remains)."""
        result = demo.nested_savepoints(user_id="u02")
        assert result["count_after_outer_rollback"] == 1

    def test_survived_movies_list_is_correct(self, demo):
        """The exact set of persisted movies must be ['m01']."""
        result = demo.nested_savepoints(user_id="u02")
        assert result["survived_movies"] == ["m01"]


# ---------------------------------------------------------------------------
# 3. Purchase flow — happy path
# ---------------------------------------------------------------------------

class TestPurchaseFlowHappy:

    def test_all_steps_succeed(self, demo):
        """Every step must report a successful outcome on the happy path."""
        result = demo.purchase_flow(order_id="ord_h1", user_id="u01", plan="premium")
        assert result["all_steps_succeeded"]

    def test_order_status_is_completed(self, demo):
        """After all steps the order status must be 'completed'."""
        result = demo.purchase_flow(order_id="ord_h2", user_id="u01", plan="basic")
        assert result["order_status"] is not None
        assert result["order_status"]["status"] == "completed"

    def test_subscription_is_active(self, demo):
        """A subscription row must be created with status 'active'."""
        result = demo.purchase_flow(order_id="ord_h3", user_id="u01", plan="standard")
        assert result["subscription"] is not None
        assert result["subscription"]["status"] == "active"

    def test_four_steps_executed(self, demo):
        result = demo.purchase_flow(order_id="ord_h4", user_id="u01", plan="premium")
        assert len(result["steps"]) == 4


# ---------------------------------------------------------------------------
# 4. Purchase flow — transient failure with savepoint retry
# ---------------------------------------------------------------------------

class TestPurchaseFlowRetry:

    def test_failed_step_retried_successfully(self, demo):
        """A step that fails once must be retried and succeed, not abort the flow."""
        result = demo.purchase_flow(
            order_id="ord_r1",
            user_id="u02",
            plan="premium",
            fail_at="charge_payment",
        )
        assert result["all_steps_succeeded"]

    def test_failed_step_shows_retry_outcome(self, demo):
        """The charge_payment step must record a 'retrying' outcome before final success."""
        result = demo.purchase_flow(
            order_id="ord_r2",
            user_id="u04",
            plan="standard",
            fail_at="charge_payment",
        )
        step_map = {s["step"]: s["outcome"] for s in result["steps"]}
        assert step_map["charge_payment"] == "success_after_retry"

    def test_earlier_steps_not_rolled_back_on_failure(self, demo):
        """create_order and reserve_slots must still show 'success' when charge_payment fails."""
        result = demo.purchase_flow(
            order_id="ord_r3",
            user_id="u05",
            plan="basic",
            fail_at="charge_payment",
        )
        step_map = {s["step"]: s["outcome"] for s in result["steps"]}
        assert step_map["create_order"] == "success"
        assert step_map["reserve_slots"] == "success"

    def test_order_completed_after_retry(self, demo):
        """Even after a charge_payment retry, the order must reach 'completed'."""
        result = demo.purchase_flow(
            order_id="ord_r4",
            user_id="u03",
            plan="premium",
            fail_at="activate_sub",
        )
        assert result["order_status"]["status"] == "completed"


# ---------------------------------------------------------------------------
# 5. Batch ingestion with per-row savepoints
# ---------------------------------------------------------------------------

_BATCH = [
    {"movie_id": "b01", "title": "Film Alpha",   "genre": "drama",  "year": 2023, "rating": 4.2},
    {"movie_id": "b02", "title": "Film Beta",    "genre": "sci-fi", "year": 2022, "rating": 3.9},
    {"movie_id": "b03", "title": "Film Gamma",   "genre": "horror", "year": 2021, "rating": 4.0},
    {"movie_id": "bad", "title": "",             "genre": None,     "year": None, "rating": None},
    {"movie_id": "b04", "title": "Film Delta",   "genre": "action", "year": 2023, "rating": 4.5},
]


class TestBatchIngest:

    def test_good_rows_are_committed(self, demo):
        """All records not in fail_ids must be persisted after COMMIT."""
        result = demo.batch_ingest(_BATCH, fail_ids={"bad"})
        assert set(result["succeeded"]) == {"b01", "b02", "b03", "b04"}

    def test_bad_rows_are_skipped(self, demo):
        """Records in fail_ids must be in the 'skipped' list, not committed."""
        result = demo.batch_ingest(_BATCH, fail_ids={"bad"})
        assert result["skipped"] == ["bad"]

    def test_counts_sum_to_total(self, demo):
        """success_count + skip_count must equal total records submitted."""
        result = demo.batch_ingest(_BATCH, fail_ids={"bad"})
        assert result["success_count"] + result["skip_count"] == result["total"]

    def test_all_succeed_when_no_fail_ids(self, demo):
        """With no fail_ids the entire batch must commit with zero skips."""
        records = [
            {"movie_id": "c01", "title": "Clean Alpha", "genre": "drama",  "year": 2023, "rating": 4.0},
            {"movie_id": "c02", "title": "Clean Beta",  "genre": "sci-fi", "year": 2022, "rating": 3.8},
        ]
        result = demo.batch_ingest(records, fail_ids=set())
        assert result["skip_count"] == 0
        assert result["success_count"] == 2

    def test_all_skipped_when_all_fail(self, demo):
        """When every record is in fail_ids the batch commits zero rows."""
        records = [
            {"movie_id": "f01", "title": "", "genre": None, "year": None, "rating": None},
            {"movie_id": "f02", "title": "", "genre": None, "year": None, "rating": None},
        ]
        result = demo.batch_ingest(records, fail_ids={"f01", "f02"})
        assert result["success_count"] == 0
        assert result["skip_count"] == 2

    def test_good_rows_visible_in_catalog_after_ingest(self, demo):
        """Committed records must be queryable via row_count after the batch."""
        before = demo.row_count("ingest_catalog")
        demo.batch_ingest(
            [{"movie_id": "v01", "title": "Visible", "genre": "drama", "year": 2024, "rating": 4.0}],
            fail_ids=set(),
        )
        assert demo.row_count("ingest_catalog") == before + 1

    def test_skipped_rows_absent_from_catalog(self, demo):
        """A record in fail_ids must not appear in ingest_catalog after COMMIT."""
        before = demo.row_count("ingest_catalog")
        demo.batch_ingest(
            [{"movie_id": "skip01", "title": "", "genre": None, "year": None, "rating": None}],
            fail_ids={"skip01"},
        )
        assert demo.row_count("ingest_catalog") == before


# ---------------------------------------------------------------------------
# 6. ROLLBACK TO savepoint vs full ROLLBACK
# ---------------------------------------------------------------------------

class TestSavepointVsFullRollback:

    def test_scenario_a_persists_two_movies(self, demo):
        """Savepoint partial rollback must leave exactly m04 and m06 committed."""
        result = demo.savepoint_vs_full_rollback(user_id="u04")
        assert result["scenario_a_persisted"] == ["m04", "m06"]

    def test_scenario_a_excludes_rolled_back_movie(self, demo):
        """m05 — inserted inside the savepoint then rolled back — must not be present."""
        result = demo.savepoint_vs_full_rollback(user_id="u04")
        assert "m05" not in result["scenario_a_persisted"]

    def test_scenario_b_persists_nothing(self, demo):
        """A full ROLLBACK must leave no rows committed."""
        result = demo.savepoint_vs_full_rollback(user_id="u04")
        assert result["scenario_b_persisted"] == []
        assert result["scenario_b_count"] == 0

    def test_scenario_a_count_is_two(self, demo):
        result = demo.savepoint_vs_full_rollback(user_id="u04")
        assert result["scenario_a_count"] == 2


# ---------------------------------------------------------------------------
# 7. Seed data and row_count introspection
# ---------------------------------------------------------------------------

class TestSeedAndIntrospection:

    def test_seed_populates_ingest_catalog(self, demo):
        """Seeding from MOVIES must insert all 20 records into ingest_catalog."""
        assert demo.row_count("ingest_catalog") == len(MOVIES)

    def test_empty_demo_has_zero_rows(self, empty_demo):
        """A demo with no seed call must have an empty ingest_catalog."""
        assert empty_demo.row_count("ingest_catalog") == 0

    def test_batch_ingest_accumulates_on_top_of_seed(self, demo):
        """Batch ingesting N new records on top of seeded data adds exactly N rows."""
        before = demo.row_count("ingest_catalog")
        new_records = [
            {"movie_id": f"acc{i}", "title": f"Acc {i}", "genre": "drama",
             "year": 2020 + i, "rating": 3.5}
            for i in range(5)
        ]
        demo.batch_ingest(new_records, fail_ids=set())
        assert demo.row_count("ingest_catalog") == before + 5
