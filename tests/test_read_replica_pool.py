"""Tests for the Read-Replica Connection Pool module."""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.read_replica_pool import (
    BoundedPool,
    PoolExhaustedError,
    Replica,
    PrimaryReplicaRouter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_router(weights=(1.0, 1.0, 1.0), lags_ms=(0.0, 0.0, 0.0)) -> PrimaryReplicaRouter:
    replicas = [
        Replica(replica_id=i + 1, weight=weights[i], lag_ms=lags_ms[i])
        for i in range(3)
    ]
    return PrimaryReplicaRouter(replicas=replicas, primary_pool_size=4)


@pytest.fixture
def seeded_router():
    r = _make_router()
    r.seed(MOVIES, RATINGS)
    yield r
    r.close()


@pytest.fixture
def empty_router():
    r = _make_router()
    yield r
    r.close()


# ─────────────────────────────────────────────────────────────────────────────
# BoundedPool
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedPool:

    def test_connection_yields_sqlite_connection(self):
        pool = BoundedPool(max_size=2)
        with pool.connection() as conn:
            row = conn.execute("SELECT 1 AS x").fetchone()
            assert row["x"] == 1
        pool.close_all()

    def test_pool_exhausted_raises(self):
        pool = BoundedPool(max_size=1, timeout=0.05)
        with pytest.raises(PoolExhaustedError):
            with pool.connection():
                with pool.connection():  # second borrow should time out
                    pass
        pool.close_all()

    def test_checked_out_counter(self):
        pool = BoundedPool(max_size=3)
        assert pool.checked_out == 0
        with pool.connection():
            assert pool.checked_out == 1
        assert pool.checked_out == 0
        pool.close_all()

    def test_ddl_tables_created(self):
        pool = BoundedPool(max_size=1)
        with pool.connection() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "movies" in tables
        assert "ratings" in tables
        pool.close_all()


# ─────────────────────────────────────────────────────────────────────────────
# Replica
# ─────────────────────────────────────────────────────────────────────────────

class TestReplica:

    def test_replica_seed_inserts_movies(self):
        r = Replica(replica_id=1)
        r.seed(MOVIES, RATINGS)
        rows = r.execute("SELECT COUNT(*) AS cnt FROM movies")
        assert rows[0]["cnt"] == len(MOVIES)
        r.close()

    def test_replica_seed_inserts_ratings(self):
        r = Replica(replica_id=1)
        r.seed(MOVIES, RATINGS)
        rows = r.execute("SELECT COUNT(*) AS cnt FROM ratings")
        assert rows[0]["cnt"] == len(RATINGS)
        r.close()

    def test_queries_served_increments(self):
        r = Replica(replica_id=1)
        r.seed(MOVIES, RATINGS)
        assert r.queries_served == 0
        r.execute("SELECT 1")
        r.execute("SELECT 1")
        assert r.queries_served == 2
        r.close()

    def test_avg_latency_zero_before_queries(self):
        r = Replica(replica_id=1)
        assert r.avg_latency_us() == 0.0
        r.close()

    def test_avg_latency_positive_after_queries(self):
        r = Replica(replica_id=1)
        r.seed(MOVIES, RATINGS)
        r.execute("SELECT COUNT(*) FROM movies")
        assert r.avg_latency_us() > 0.0
        r.close()

    def test_apply_write_visible_on_execute(self):
        r = Replica(replica_id=1)
        r.seed(MOVIES, RATINGS)
        r.apply_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) "
            "VALUES (?,?,?,?,?)",
            ("z99", "Test Movie", "drama", 2000, "Director"),
        )
        rows = r.execute("SELECT id FROM movies WHERE id = 'z99'")
        assert len(rows) == 1
        r.close()


# ─────────────────────────────────────────────────────────────────────────────
# PrimaryReplicaRouter — seeding
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterSeeding:

    def test_seed_movies_on_primary(self, seeded_router):
        with seeded_router._primary.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        assert count == len(MOVIES)

    def test_seed_movies_on_all_replicas(self, seeded_router):
        for replica in seeded_router._replicas:
            rows = replica.execute("SELECT COUNT(*) AS cnt FROM movies")
            assert rows[0]["cnt"] == len(MOVIES)

    def test_seed_ratings_on_all_replicas(self, seeded_router):
        for replica in seeded_router._replicas:
            rows = replica.execute("SELECT COUNT(*) AS cnt FROM ratings")
            assert rows[0]["cnt"] == len(RATINGS)


# ─────────────────────────────────────────────────────────────────────────────
# PrimaryReplicaRouter — write routing
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterWriteRouting:

    def test_execute_write_increments_write_counter(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("w01", "Write Test", "drama", 2020, "D"),
            replica_lag_ms=0,
        )
        assert seeded_router.routing_stats["total_writes"] == 1

    def test_write_appears_on_primary_immediately(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("w02", "Primary Immediate", "sci-fi", 2023, "D"),
            replica_lag_ms=0,
        )
        with seeded_router._primary.connection() as conn:
            row = conn.execute(
                "SELECT id FROM movies WHERE id = 'w02'"
            ).fetchone()
        assert row is not None

    def test_write_not_on_replica_before_propagate(self, seeded_router):
        """With a large lag, replicas should not see the write until propagate is called."""
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("w03", "Lag Test", "drama", 2023, "D"),
            replica_lag_ms=60_000,   # 60-second lag — never naturally expires in this test
        )
        # Replica should NOT see it yet
        for replica in seeded_router._replicas:
            rows = replica.execute("SELECT id FROM movies WHERE id = 'w03'")
            assert len(rows) == 0

    def test_write_on_replica_after_propagate(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("w04", "After Propagate", "drama", 2023, "D"),
            replica_lag_ms=0,
        )
        seeded_router.propagate()
        for replica in seeded_router._replicas:
            rows = replica.execute("SELECT id FROM movies WHERE id = 'w04'")
            assert len(rows) == 1

    def test_pending_count_decreases_after_propagate(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("w05", "Pending Test", "drama", 2023, "D"),
            replica_lag_ms=0,
        )
        assert seeded_router.pending_replication_count() > 0
        seeded_router.propagate()
        assert seeded_router.pending_replication_count() == 0

    def test_propagate_with_future_timestamp_applies_lagged_writes(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("w06", "Future TS", "drama", 2023, "D"),
            replica_lag_ms=500,   # 500 ms lag
        )
        # Propagate with a future timestamp 1 s ahead
        applied = seeded_router.propagate(until_ts=time.perf_counter() + 1.0)
        assert applied > 0
        for replica in seeded_router._replicas:
            rows = replica.execute("SELECT id FROM movies WHERE id = 'w06'")
            assert len(rows) == 1


# ─────────────────────────────────────────────────────────────────────────────
# PrimaryReplicaRouter — read routing
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterReadRouting:

    def test_execute_read_returns_results(self, seeded_router):
        rows = seeded_router.execute_read("SELECT title FROM movies LIMIT 5")
        assert len(rows) == 5

    def test_read_counter_increments(self, seeded_router):
        seeded_router.execute_read("SELECT 1")
        seeded_router.execute_read("SELECT 1")
        assert seeded_router.routing_stats["reads_to_replicas"] == 2

    def test_reads_distributed_across_replicas(self, seeded_router):
        for _ in range(90):
            seeded_router.execute_read("SELECT title FROM movies LIMIT 1")
        # Each replica should have received at least some queries
        served = [r.queries_served for r in seeded_router._replicas]
        assert all(s > 0 for s in served), f"Some replica got no reads: {served}"

    def test_weighted_replica_gets_more_reads(self):
        router = _make_router(weights=(1.0, 3.0, 1.0))
        router.seed(MOVIES, RATINGS)
        for _ in range(100):
            router.execute_read("SELECT 1")
        stats = router.replica_stats
        replica_2 = next(r for r in stats if r["id"] == 2)
        replica_1 = next(r for r in stats if r["id"] == 1)
        assert replica_2["queries_served"] > replica_1["queries_served"]
        router.close()


# ─────────────────────────────────────────────────────────────────────────────
# PrimaryReplicaRouter — health / failover
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterFailover:

    def test_mark_unhealthy_reduces_healthy_count(self, seeded_router):
        seeded_router.mark_unhealthy(1)
        assert seeded_router.healthy_replica_count() == 2

    def test_mark_healthy_restores_count(self, seeded_router):
        seeded_router.mark_unhealthy(1)
        seeded_router.mark_healthy(1)
        assert seeded_router.healthy_replica_count() == 3

    def test_unhealthy_replica_not_served_reads(self, seeded_router):
        seeded_router.mark_unhealthy(1)
        for _ in range(30):
            seeded_router.execute_read("SELECT 1")
        stats = seeded_router.replica_stats
        replica_1 = next(r for r in stats if r["id"] == 1)
        assert replica_1["queries_served"] == 0

    def test_all_unhealthy_falls_back_to_primary(self, seeded_router):
        for rid in (1, 2, 3):
            seeded_router.mark_unhealthy(rid)
        seeded_router.execute_read("SELECT 1")
        rs = seeded_router.routing_stats
        assert rs["reads_to_primary_fallback"] == 1
        assert rs["reads_to_replicas"] == 0

    def test_reads_continue_after_one_replica_down(self, seeded_router):
        seeded_router.mark_unhealthy(2)
        rows = seeded_router.execute_read("SELECT title FROM movies LIMIT 3")
        assert len(rows) == 3


# ─────────────────────────────────────────────────────────────────────────────
# PrimaryReplicaRouter — consistency (stale read detection)
# ─────────────────────────────────────────────────────────────────────────────

class TestStaleReads:

    def test_stale_read_before_propagate(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) VALUES (?,?,?,?)",
            ("u_stale", "m01", 4.5, "stale test"),
            replica_lag_ms=60_000,
        )
        rows = seeded_router.execute_read(
            "SELECT COUNT(*) AS cnt FROM ratings WHERE user_id='u_stale'"
        )
        assert rows[0]["cnt"] == 0, "Replica should not see un-propagated write"

    def test_fresh_read_after_propagate(self, seeded_router):
        seeded_router.execute_write(
            "INSERT OR IGNORE INTO ratings (user_id,movie_id,score,review) VALUES (?,?,?,?)",
            ("u_fresh", "m01", 4.5, "fresh test"),
            replica_lag_ms=0,
        )
        seeded_router.propagate()
        rows = seeded_router.execute_read(
            "SELECT COUNT(*) AS cnt FROM ratings WHERE user_id='u_fresh'"
        )
        assert rows[0]["cnt"] == 1, "Replica should see write after propagate"


# ─────────────────────────────────────────────────────────────────────────────
# PrimaryReplicaRouter — routing stats structure
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingStats:

    def test_routing_stats_keys_present(self, seeded_router):
        rs = seeded_router.routing_stats
        assert "total_writes" in rs
        assert "reads_to_replicas" in rs
        assert "reads_to_primary_fallback" in rs
        assert "pending_replication" in rs

    def test_replica_stats_structure(self, seeded_router):
        for r in seeded_router.replica_stats:
            assert "id" in r
            assert "healthy" in r
            assert "weight" in r
            assert "lag_ms" in r
            assert "queries_served" in r
            assert "avg_latency_us" in r

    def test_initial_counters_zero(self, empty_router):
        rs = empty_router.routing_stats
        assert rs["total_writes"] == 0
        assert rs["reads_to_replicas"] == 0
        assert rs["reads_to_primary_fallback"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_reads_no_error(self, seeded_router):
        errors = []

        def reader():
            try:
                for _ in range(10):
                    seeded_router.execute_read("SELECT title FROM movies LIMIT 5")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent read errors: {errors}"

    def test_concurrent_writes_and_reads_no_error(self, seeded_router):
        errors = []

        def writer():
            try:
                for i in range(5):
                    seeded_router.execute_write(
                        "INSERT OR REPLACE INTO ratings "
                        "(user_id,movie_id,score,review) VALUES (?,?,?,?)",
                        (f"uc{i}", "m01", 3.0, "concurrent"),
                        replica_lag_ms=0,
                    )
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(10):
                    seeded_router.execute_read("SELECT COUNT(*) FROM movies")
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=writer) for _ in range(3)]
            + [threading.Thread(target=reader) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent R/W errors: {errors}"
