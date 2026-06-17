"""Tests for the Adaptive Connection Pool module."""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.adaptive_pool import AdaptivePool, PoolExhaustedError, AdaptiveMovieDB
from databaseai.adaptive_pool.adaptive import sqlite_factory


# ────────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ────────────────────────────────────────────────────────────────────────────────

def _make_pool(
    min_size=2,
    max_size=8,
    timeout=5.0,
    cooldown_s=0.05,
    max_conn_age_s=300.0,
    connection_overhead=0.0,
) -> AdaptivePool:
    return AdaptivePool(
        db_factory=sqlite_factory(":memory:"),
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        scale_up_threshold=0.75,
        scale_down_threshold=0.25,
        grow_step=2,
        cooldown_s=cooldown_s,
        max_conn_age_s=max_conn_age_s,
        connection_overhead=connection_overhead,
    )


@pytest.fixture
def pool():
    p = _make_pool()
    yield p
    p.close()


@pytest.fixture
def seeded_db():
    p = _make_pool()
    db = AdaptiveMovieDB(p)
    db.seed(MOVIES, RATINGS)
    yield db, p
    p.close()


# ────────────────────────────────────────────────────────────────────────────────
# Basic pool operations
# ────────────────────────────────────────────────────────────────────────────────

class TestBasicPool:

    def test_initial_capacity_equals_min_size(self, pool):
        assert pool.capacity == pool.min_size

    def test_connection_context_manager_yields_usable_conn(self, pool):
        with pool.connection() as conn:
            row = conn.execute("SELECT 1 AS x").fetchone()
            assert row["x"] == 1

    def test_checked_out_increments_and_decrements(self, pool):
        assert pool.checked_out == 0
        with pool.connection():
            assert pool.checked_out == 1
        assert pool.checked_out == 0

    def test_idle_count_reflects_available_connections(self, pool):
        initial_idle = pool.idle
        with pool.connection():
            assert pool.idle < initial_idle
        assert pool.idle >= initial_idle

    def test_stats_keys_present(self, pool):
        s = pool.stats()
        expected = {
            "min_size", "max_size", "capacity", "checked_out", "idle",
            "total_connections_created", "total_served", "total_timeouts",
            "avg_wait_ms", "max_wait_ms", "scale_ups", "scale_downs",
            "health_failures", "age_evictions", "avg_utilisation",
        }
        assert expected.issubset(s.keys())

    def test_pool_exhaustion_raises_on_timeout(self):
        tiny = _make_pool(min_size=1, max_size=1, timeout=0.1)
        with pytest.raises(PoolExhaustedError):
            with tiny.connection():
                with tiny.connection():
                    pass
        tiny.close()

    def test_invalid_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            _make_pool(min_size=10, max_size=5)

    def test_invalid_max_size_zero_raises(self):
        with pytest.raises(ValueError):
            AdaptivePool(
                db_factory=sqlite_factory(":memory:"),
                min_size=1, max_size=0,
            )


# ────────────────────────────────────────────────────────────────────────────────
# Scale-up behaviour
# ────────────────────────────────────────────────────────────────────────────────

class TestScaleUp:

    def test_force_scale_up_increases_capacity(self, pool):
        before = pool.capacity
        added = pool.force_scale_up(2)
        assert added == 2
        assert pool.capacity == before + 2

    def test_force_scale_up_respects_max_size(self):
        p = _make_pool(min_size=1, max_size=3)
        p.force_scale_up(10)
        assert p.capacity <= p.max_size
        p.close()

    def test_concurrent_burst_triggers_growth(self):
        p = _make_pool(min_size=1, max_size=6, cooldown_s=0.01)
        errors = []

        def worker():
            try:
                with p.connection() as conn:
                    conn.execute("SELECT 1")
                    time.sleep(0.03)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert p.stats()["total_connections_created"] > 1
        p.close()


# ────────────────────────────────────────────────────────────────────────────────
# Scale-down behaviour
# ────────────────────────────────────────────────────────────────────────────────

class TestScaleDown:

    def test_force_scale_down_decreases_capacity(self, pool):
        pool.force_scale_up(3)
        cap_before = pool.capacity
        removed = pool.force_scale_down(2)
        assert removed == 2
        assert pool.capacity == cap_before - 2

    def test_force_scale_down_respects_min_size(self):
        p = _make_pool(min_size=2, max_size=6)
        p.force_scale_down(10)
        assert p.capacity >= p.min_size
        p.close()

    def test_scale_down_after_burst_idle(self):
        p = _make_pool(min_size=1, max_size=6, cooldown_s=0.01)
        p.force_scale_up(4)
        high_cap = p.capacity

        while p.capacity > p.min_size and p.idle > 0:
            p.force_scale_down(1)

        assert p.capacity < high_cap
        assert p.capacity >= p.min_size
        p.close()


# ────────────────────────────────────────────────────────────────────────────────
# Health checks and eviction
# ────────────────────────────────────────────────────────────────────────────────

class TestHealthAndEviction:

    def test_validate_returns_true_for_healthy_conn(self, pool):
        with pool.connection() as conn:
            assert conn.validate() is True

    def test_max_age_eviction_replaces_old_connection(self):
        p = _make_pool(min_size=1, max_size=4, max_conn_age_s=0.05)
        with p.connection() as conn:
            first_id = conn.conn_id

        time.sleep(0.1)

        with p.connection() as conn:
            second_id = conn.conn_id

        assert first_id != second_id
        s = p.stats()
        assert s["age_evictions"] > 0
        p.close()

    def test_age_eviction_counter_increments(self):
        p = _make_pool(min_size=1, max_size=4, max_conn_age_s=0.03)
        with p.connection():
            pass
        time.sleep(0.06)
        with p.connection():
            pass
        s = p.stats()
        assert s["age_evictions"] >= 1
        p.close()


# ────────────────────────────────────────────────────────────────────────────────
# AdaptiveMovieDB
# ────────────────────────────────────────────────────────────────────────────────

class TestAdaptiveMovieDB:

    def test_movie_count_matches_seed(self, seeded_db):
        db, _ = seeded_db
        assert db.movie_count() == len(MOVIES)

    def test_rating_count_matches_seed(self, seeded_db):
        db, _ = seeded_db
        assert db.rating_count() == len(RATINGS)

    def test_top_rated_returns_ordered_results(self, seeded_db):
        db, _ = seeded_db
        top = db.top_rated(limit=5)
        assert len(top) == 5
        scores = [r["avg_score"] for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_genre_breakdown_returns_all_rated_genres(self, seeded_db):
        db, _ = seeded_db
        genres = db.genre_breakdown()
        genre_names = {g["genre"] for g in genres}
        assert len(genre_names) >= 3


# ────────────────────────────────────────────────────────────────────────────────
# Concurrency
# ────────────────────────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_reads_no_errors(self, seeded_db):
        db, pool = seeded_db
        errors = []

        def reader():
            try:
                for _ in range(5):
                    with pool.connection() as conn:
                        conn.execute("SELECT COUNT(*) FROM movies").fetchone()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent read errors: {errors}"

    def test_concurrent_mixed_operations(self, seeded_db):
        db, pool = seeded_db
        errors = []

        def query_worker():
            try:
                db.top_rated(3)
                db.genre_breakdown()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=query_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent mixed errors: {errors}"


# ────────────────────────────────────────────────────────────────────────────────
# Stats and reset
# ────────────────────────────────────────────────────────────────────────────────

class TestStats:

    def test_served_counter_increments(self, pool):
        assert pool.stats()["total_served"] == 0
        with pool.connection():
            pass
        assert pool.stats()["total_served"] == 1

    def test_reset_stats_zeroes_counters(self, pool):
        with pool.connection():
            pass
        pool.reset_stats()
        s = pool.stats()
        assert s["total_served"] == 0
        assert s["avg_wait_ms"] == 0.0

    def test_utilisation_between_zero_and_one(self, pool):
        util = pool.utilisation()
        assert 0.0 <= util <= 1.0
