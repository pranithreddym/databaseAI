"""Tests for the Connection Pooling module (ConnectionPool + PooledMovieDB)."""

import sys
import os
import time
import threading
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.connection_pool import (
    ConnectionPool, PoolExhaustedError, PooledMovieDB, sqlite_factory
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = tmp.name
    tmp.close()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def pool(db_path):
    p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                       min_size=0, timeout=2.0)
    yield p
    p.close()


@pytest.fixture
def movie_db(db_path):
    p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                       min_size=0, timeout=5.0)
    db = PooledMovieDB(p)
    db.seed(MOVIES, RATINGS)
    yield db
    p.close()


# ---------------------------------------------------------------------------
# Basic pool operations
# ---------------------------------------------------------------------------

class TestConnectionPoolBasics:

    def test_acquire_returns_a_connection(self, pool):
        conn = pool.acquire()
        assert conn is not None
        conn.release()

    def test_acquire_increments_total_created(self, pool):
        assert pool.total_created == 0
        conn = pool.acquire()
        assert pool.total_created == 1
        conn.release()

    def test_pool_does_not_exceed_max_size(self, pool):
        conns = [pool.acquire() for _ in range(pool.max_size)]
        assert pool.total_created == pool.max_size
        for c in conns:
            c.release()

    def test_acquire_reuses_released_connection(self, pool):
        conn1 = pool.acquire()
        id1 = conn1.conn_id
        conn1.release()
        conn2 = pool.acquire()
        assert conn2.conn_id == id1
        conn2.release()

    def test_checked_out_count_tracks_held_connections(self, pool):
        assert pool.stats()["checked_out"] == 0
        conn = pool.acquire()
        assert pool.stats()["checked_out"] == 1
        conn.release()
        assert pool.stats()["checked_out"] == 0

    def test_available_count_decrements_on_acquire(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=2,
                           min_size=2, timeout=1.0)
        assert p.available == 2
        conn = p.acquire()
        assert p.available == 1
        conn.release()
        p.close()

    def test_connection_executes_sql(self, pool):
        conn = pool.acquire()
        row = conn.execute("SELECT 1 AS val").fetchone()
        assert row["val"] == 1
        conn.release()


# ---------------------------------------------------------------------------
# Pre-warming
# ---------------------------------------------------------------------------

class TestPoolPreWarming:

    def test_min_size_creates_connections_at_startup(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                           min_size=2, timeout=1.0)
        assert p.total_created == 2
        assert p.available == 2
        p.close()

    def test_zero_min_size_creates_no_connections_at_startup(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                           min_size=0, timeout=1.0)
        assert p.total_created == 0
        p.close()

    def test_min_size_stat_reported_correctly(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=4,
                           min_size=2, timeout=1.0)
        assert p.stats()["min_size"] == 2
        p.close()


# ---------------------------------------------------------------------------
# Pool exhaustion & timeout
# ---------------------------------------------------------------------------

class TestPoolExhaustion:

    def test_exhausted_pool_raises_pool_exhausted_error(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=1,
                           timeout=0.05)
        conn = p.acquire()
        with pytest.raises(PoolExhaustedError):
            p.acquire()
        conn.release()
        p.close()

    def test_timeout_increments_stat_counter(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=1,
                           timeout=0.05)
        conn = p.acquire()
        try:
            p.acquire()
        except PoolExhaustedError:
            pass
        conn.release()
        assert p.stats()["total_timeouts"] == 1
        p.close()

    def test_released_connection_unblocks_waiting_caller(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=1,
                           timeout=2.0)
        c1 = p.acquire()
        acquired = []

        def waiter():
            c = p.acquire()
            acquired.append(c.conn_id)
            c.release()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.02)
        c1.release()
        t.join(timeout=3.0)
        assert len(acquired) == 1
        p.close()

    def test_pool_exhausted_error_message_contains_max_size(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=2,
                           timeout=0.05)
        c1 = p.acquire()
        c2 = p.acquire()
        try:
            p.acquire()
        except PoolExhaustedError as exc:
            assert "max_size=2" in str(exc)
        finally:
            c1.release()
            c2.release()
            p.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_context_manager_releases_on_normal_exit(self, pool):
        with pool.connection():
            assert pool.stats()["checked_out"] == 1
        assert pool.stats()["checked_out"] == 0

    def test_context_manager_releases_on_exception(self, pool):
        try:
            with pool.connection():
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        assert pool.stats()["checked_out"] == 0

    def test_context_manager_yields_usable_connection(self, pool):
        with pool.connection() as conn:
            row = conn.execute("SELECT 42 AS answer").fetchone()
            assert row["answer"] == 42


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestPoolStats:

    def test_stats_initial_state(self, pool):
        s = pool.stats()
        assert s["total_requests"] == 0
        assert s["total_served"] == 0
        assert s["total_timeouts"] == 0

    def test_stats_track_total_requests(self, pool):
        pool.reset_stats()
        pool.acquire().release()
        pool.acquire().release()
        assert pool.stats()["total_requests"] == 2
        assert pool.stats()["total_served"] == 2

    def test_avg_wait_ms_is_non_negative(self, pool):
        pool.acquire().release()
        assert pool.stats()["avg_wait_ms"] >= 0.0

    def test_reset_stats_clears_all_counters(self, pool):
        pool.acquire().release()
        pool.reset_stats()
        s = pool.stats()
        assert s["total_requests"] == 0
        assert s["total_served"] == 0
        assert s["avg_wait_ms"] == 0.0

    def test_max_size_stat_matches_constructor_argument(self, pool):
        assert pool.stats()["max_size"] == 3

    def test_invalid_max_size_zero_raises_value_error(self, db_path):
        with pytest.raises(ValueError):
            ConnectionPool(db_factory=sqlite_factory(db_path), max_size=0)

    def test_invalid_min_size_exceeds_max_raises_value_error(self, db_path):
        with pytest.raises(ValueError):
            ConnectionPool(db_factory=sqlite_factory(db_path), max_size=2,
                           min_size=3)

    def test_invalid_negative_min_size_raises_value_error(self, db_path):
        with pytest.raises(ValueError):
            ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                           min_size=-1)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_concurrent_workers_all_succeed(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                           timeout=10.0)
        errors = []
        successes = []

        def worker():
            try:
                with p.connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                    time.sleep(0.01)
                successes.append(1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == []
        assert len(successes) == 8
        p.close()

    def test_connection_count_bounded_under_concurrent_load(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                           timeout=10.0)

        def worker():
            with p.connection() as conn:
                conn.execute("SELECT 1").fetchone()
                time.sleep(0.01)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert p.total_created <= 3
        p.close()

    def test_concurrent_timeouts_counted_correctly(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=1,
                           timeout=0.05)
        barrier = threading.Barrier(4)
        timeouts = []

        def worker():
            barrier.wait()
            try:
                with p.connection():
                    time.sleep(0.20)
            except PoolExhaustedError:
                timeouts.append(1)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(timeouts) == p.stats()["total_timeouts"]
        p.close()


# ---------------------------------------------------------------------------
# PooledMovieDB integration
# ---------------------------------------------------------------------------

class TestPooledMovieDB:

    def test_seed_populates_correct_movie_count(self, movie_db):
        assert movie_db.movie_count() == len(MOVIES)

    def test_seed_populates_correct_rating_count(self, movie_db):
        assert movie_db.rating_count() == len(RATINGS)

    def test_top_rated_returns_descending_scores(self, movie_db):
        results = movie_db.top_rated(limit=5)
        scores = [r["avg_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_rated_limit_respected(self, movie_db):
        assert len(movie_db.top_rated(limit=3)) <= 3

    def test_genre_breakdown_covers_all_seeded_genres(self, movie_db):
        expected = {m["genre"] for m in MOVIES}
        actual = {r["genre"] for r in movie_db.genre_breakdown()}
        assert expected == actual

    def test_genre_breakdown_scores_in_range(self, movie_db):
        for row in movie_db.genre_breakdown():
            assert 0.0 <= row["avg_score"] <= 5.0

    def test_concurrent_queries_return_consistent_counts(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=3,
                           timeout=5.0)
        db = PooledMovieDB(p)
        db.seed(MOVIES, RATINGS)

        counts = []
        lock = threading.Lock()

        def worker():
            c = db.movie_count()
            with lock:
                counts.append(c)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert all(c == len(MOVIES) for c in counts)
        p.close()

    def test_pool_stats_reflect_movie_db_queries(self, db_path):
        p = ConnectionPool(db_factory=sqlite_factory(db_path), max_size=2,
                           timeout=5.0)
        db = PooledMovieDB(p)
        db.seed(MOVIES, RATINGS)
        p.reset_stats()

        db.movie_count()
        db.rating_count()
        db.top_rated()

        assert p.stats()["total_served"] == 3
        p.close()
