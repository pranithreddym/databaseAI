"""Tests for the Transaction Isolation Levels module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, USERS
from databaseai.transactions import (
    TransactionIsolationDemo,
    ISOLATION_MATRIX,
    LEVEL_READ_UNCOMMITTED,
    LEVEL_READ_COMMITTED,
    LEVEL_REPEATABLE_READ,
    LEVEL_SERIALIZABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo(tmp_path):
    """Fresh demo backed by a real WAL-mode file; cleaned up after each test."""
    db_path = str(tmp_path / "test_tx.db")
    d = TransactionIsolationDemo(db_path=db_path)
    d.seed(MOVIES, USERS)
    yield d
    d.close()


@pytest.fixture
def bare_demo(tmp_path):
    """Demo with no seed data — for testing schema setup only."""
    db_path = str(tmp_path / "bare_tx.db")
    d = TransactionIsolationDemo(db_path=db_path)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

class TestIsolationMatrix:

    def test_matrix_has_four_levels(self):
        assert len(ISOLATION_MATRIX) == 4

    def test_level_names_are_correct(self):
        names = [r["level"] for r in ISOLATION_MATRIX]
        assert LEVEL_READ_UNCOMMITTED in names
        assert LEVEL_READ_COMMITTED   in names
        assert LEVEL_REPEATABLE_READ  in names
        assert LEVEL_SERIALIZABLE     in names

    def test_serializable_prevents_all_anomalies(self):
        row = next(r for r in ISOLATION_MATRIX if r["level"] == LEVEL_SERIALIZABLE)
        assert not row["dirty_read"]
        assert not row["nonrepeatable_read"]
        assert not row["phantom_read"]

    def test_read_committed_allows_nonrepeatable_and_phantom(self):
        row = next(r for r in ISOLATION_MATRIX if r["level"] == LEVEL_READ_COMMITTED)
        assert not row["dirty_read"],          "READ COMMITTED must prevent dirty reads"
        assert row["nonrepeatable_read"],      "READ COMMITTED allows non-repeatable reads"
        assert row["phantom_read"],            "READ COMMITTED allows phantom reads"

    def test_repeatable_read_prevents_dirty_and_nrr(self):
        row = next(r for r in ISOLATION_MATRIX if r["level"] == LEVEL_REPEATABLE_READ)
        assert not row["dirty_read"]
        assert not row["nonrepeatable_read"]


# ---------------------------------------------------------------------------
# 2. Seed and introspection
# ---------------------------------------------------------------------------

class TestSeedAndSetup:

    def test_seed_populates_movies(self, demo):
        """Movies from MOVIES are queryable after seed()."""
        import sqlite3
        conn = sqlite3.connect(demo._db_path)
        count = conn.execute("SELECT COUNT(*) FROM movies_tx").fetchone()[0]
        conn.close()
        assert count == len(MOVIES)

    def test_seed_creates_one_account_per_user(self, demo):
        """Each USERS entry gets an account_tx row with balance 1000."""
        import sqlite3
        conn = sqlite3.connect(demo._db_path)
        rows = conn.execute("SELECT * FROM accounts_tx").fetchall()
        conn.close()
        assert len(rows) == len(USERS)

    def test_get_balance_returns_initial_value(self, demo):
        first_user = USERS[0]
        account_id = f"acc_{first_user['id']}"
        assert demo.get_balance(account_id) == pytest.approx(1000.0)

    def test_reset_balance_updates_value(self, demo):
        account_id = f"acc_{USERS[0]['id']}"
        demo.reset_balance(account_id, 42.5)
        assert demo.get_balance(account_id) == pytest.approx(42.5)

    def test_get_balance_returns_none_for_unknown_account(self, demo):
        assert demo.get_balance("nonexistent_acc") is None


# ---------------------------------------------------------------------------
# 3. Dirty Read — always prevented in SQLite
# ---------------------------------------------------------------------------

class TestDirtyRead:

    def test_dirty_read_is_prevented(self, demo):
        """SQLite WAL mode must never expose an uncommitted row to another connection."""
        result = demo.demonstrate_dirty_read()
        assert not result["anomaly_occurred"], (
            "Dirty read occurred — SQLite should never expose uncommitted data"
        )

    def test_t1_did_write_the_uncommitted_row(self, demo):
        """Confirm the writer actually executed the INSERT before we checked."""
        result = demo.demonstrate_dirty_read()
        assert result["t1_wrote_uncommitted"], (
            "Writer never inserted the row — test precondition not met"
        )

    def test_t2_did_not_see_uncommitted_row(self, demo):
        result = demo.demonstrate_dirty_read()
        assert not result["t2_saw_uncommitted_row"]

    def test_result_contains_prevention_message(self, demo):
        result = demo.demonstrate_dirty_read()
        assert "prevention" in result
        assert len(result["prevention"]) > 0


# ---------------------------------------------------------------------------
# 4. Non-Repeatable Read — occurs with autocommit
# ---------------------------------------------------------------------------

class TestNonRepeatableRead:

    def test_anomaly_is_observed_with_autocommit(self, demo):
        """READ COMMITTED (autocommit) must produce a non-repeatable read."""
        result = demo.demonstrate_nonrepeatable_read()
        assert result["anomaly_occurred"], (
            "Non-repeatable read not observed — verify T2 committed before T1's second read"
        )

    def test_first_and_second_reads_differ(self, demo):
        result = demo.demonstrate_nonrepeatable_read()
        assert result["t1_first_balance"] != result["t1_second_balance"]

    def test_first_balance_is_initial_value(self, demo):
        """T1's first read should see 1 000.0 (the seeded balance)."""
        result = demo.demonstrate_nonrepeatable_read()
        assert result["t1_first_balance"] == pytest.approx(1000.0)

    def test_second_balance_reflects_t2_write(self, demo):
        """T1's second read should see 250.0 (what T2 wrote)."""
        result = demo.demonstrate_nonrepeatable_read()
        assert result["t1_second_balance"] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# 5. Repeatable Read Prevention — snapshot transaction
# ---------------------------------------------------------------------------

class TestRepeatableReadPrevention:

    def test_snapshot_transaction_gives_consistent_reads(self, demo):
        """Both reads inside a BEGIN/COMMIT block must return the same balance."""
        result = demo.demonstrate_repeatable_read_prevention()
        assert result["consistent"], (
            "Non-repeatable read inside a snapshot transaction — WAL snapshot broken?"
        )

    def test_both_reads_return_the_original_balance(self, demo):
        """The snapshot must reflect the pre-update value (1 000.0) for both reads."""
        result = demo.demonstrate_repeatable_read_prevention()
        assert result["t1_first_balance"]  == pytest.approx(1000.0)
        assert result["t1_second_balance"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# 6. Phantom Read — occurs with autocommit
# ---------------------------------------------------------------------------

class TestPhantomRead:

    def test_phantom_read_occurs_with_autocommit(self, demo):
        """Autocommit mode must produce a phantom read when T2 inserts between counts."""
        result = demo.demonstrate_phantom_read()
        assert result["anomaly_occurred"], (
            "Phantom read not observed — verify T2 committed the INSERT before T1's second COUNT"
        )

    def test_second_count_is_greater_than_first(self, demo):
        result = demo.demonstrate_phantom_read()
        assert result["t1_second_count"] == result["t1_first_count"] + 1

    def test_first_count_matches_seeded_scifi_movies(self, demo):
        """First COUNT should equal the number of sci-fi movies in MOVIES seed data."""
        from databaseai.seed_data import MOVIES
        expected = sum(1 for m in MOVIES if m["genre"] == "sci-fi")
        result = demo.demonstrate_phantom_read()
        assert result["t1_first_count"] == expected


# ---------------------------------------------------------------------------
# 7. Phantom Prevention — snapshot transaction
# ---------------------------------------------------------------------------

class TestPhantomPrevention:

    def test_snapshot_prevents_phantom_read(self, demo):
        """Both COUNTs inside a BEGIN/COMMIT block must return the same value."""
        result = demo.demonstrate_phantom_prevention()
        assert result["consistent"], (
            "Phantom read inside a snapshot transaction — WAL snapshot not working as expected"
        )

    def test_both_counts_are_equal(self, demo):
        result = demo.demonstrate_phantom_prevention()
        assert result["t1_first_count"] == result["t1_second_count"]


# ---------------------------------------------------------------------------
# 8. Exclusive Lock (SERIALIZABLE)
# ---------------------------------------------------------------------------

class TestExclusiveLock:

    def test_t2_was_blocked_by_exclusive_lock(self, demo):
        """T2's BEGIN EXCLUSIVE must be delayed by T1's 120 ms exclusive hold."""
        result = demo.demonstrate_exclusive_lock()
        assert result["t2_was_blocked"], (
            f"T2 waited only {result['t2_blocked_ms']:.1f} ms — expected > 50 ms "
            "while T1 holds BEGIN EXCLUSIVE for 120 ms"
        )

    def test_t1_acquired_exclusive_lock(self, demo):
        result = demo.demonstrate_exclusive_lock()
        assert result["t1_lock_acquired"]

    def test_t2_reads_t1_committed_value_after_unblock(self, demo):
        """After T1 commits, T2's EXCLUSIVE transaction must see the updated balance (800.0)."""
        result = demo.demonstrate_exclusive_lock()
        assert result["t2_final_balance"] == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# 9. Payment Scenario — Lost Update
# ---------------------------------------------------------------------------

class TestPaymentScenario:

    def test_unserialized_produces_lost_update(self, demo):
        """Without isolation, two concurrent withdrawals of 300 from 500 only
        produce one effective deduction (lost update anomaly)."""
        pay = demo.simulate_payment_scenario(initial_balance=500.0, withdrawal=300.0)
        unser = pay["unserialized"]
        # Both threads read the same original balance
        assert unser["t1_read"] == pytest.approx(500.0)
        assert unser["t2_read"] == pytest.approx(500.0)
        # Both wrote the same value (lost update)
        assert unser["t1_wrote"] == pytest.approx(200.0)
        assert unser["t2_wrote"] == pytest.approx(200.0)
        # Only one withdrawal is reflected in the final balance
        assert unser["withdrawals_applied"] == 1

    def test_serialized_prevents_lost_update(self, demo):
        """With BEGIN EXCLUSIVE, one withdrawal succeeds and the other is
        rejected — the final balance is correct."""
        pay = demo.simulate_payment_scenario(initial_balance=500.0, withdrawal=300.0)
        ser = pay["serialized"]
        assert ser["withdrawals_succeeded"] == 1
        assert ser["withdrawals_rejected"]  == 1
        assert ser["final_balance"] == pytest.approx(200.0)

    def test_both_scenarios_use_same_initial_balance(self, demo):
        """Both scenario branches start from the configured initial_balance."""
        pay = demo.simulate_payment_scenario(initial_balance=800.0, withdrawal=100.0)
        assert pay["unserialized"]["t1_read"] == pytest.approx(800.0)
        assert pay["unserialized"]["t2_read"] == pytest.approx(800.0)
