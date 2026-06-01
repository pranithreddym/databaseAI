"""Tests for the Tiered Caching module (L1Cache, L2Cache, TieredCache, TieredMovieDB)."""

import sys
import os
import time
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.tiered_cache import L1Cache, L2Cache, TieredCache, TieredMovieDB


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _simple_source(key: str):
    return f"value_for_{key}"


def _make_tiered(l1_capacity=8, l1_ttl=60.0, l2_ttl=300.0) -> TieredCache:
    l1 = L1Cache(capacity=l1_capacity, ttl=l1_ttl)
    l2 = L2Cache(ttl=l2_ttl)
    return TieredCache(l1=l1, l2=l2, source_fn=_simple_source)


@pytest.fixture
def tc():
    return _make_tiered()


@pytest.fixture
def db():
    d = TieredMovieDB(l1_capacity=32, l1_ttl=60.0, l2_ttl=300.0, query_delay=0.0)
    d.seed(MOVIES, RATINGS)
    return d


# ---------------------------------------------------------------------------
# L1Cache tests
# ---------------------------------------------------------------------------

class TestL1Cache:

    def test_capacity_below_one_raises(self):
        with pytest.raises(ValueError):
            L1Cache(capacity=0)

    def test_miss_on_empty_cache(self):
        c = L1Cache(capacity=4, ttl=60.0)
        hit, val = c.get("missing")
        assert hit is False and val is None

    def test_put_then_get_returns_value(self):
        c = L1Cache(capacity=4, ttl=60.0)
        c.put("k", 42)
        hit, val = c.get("k")
        assert hit is True and val == 42

    def test_lru_eviction_removes_least_recently_used(self):
        c = L1Cache(capacity=2, ttl=0)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        hit_a, _ = c.get("a")
        hit_c, _ = c.get("c")
        assert hit_a is False and hit_c is True

    def test_accessed_entry_survives_eviction(self):
        c = L1Cache(capacity=2, ttl=0)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.put("c", 3)
        hit_a, _ = c.get("a")
        hit_b, _ = c.get("b")
        assert hit_a is True and hit_b is False

    def test_ttl_expiry_causes_miss(self):
        c = L1Cache(capacity=4, ttl=0.05)
        c.put("k", "v")
        time.sleep(0.08)
        hit, _ = c.get("k")
        assert hit is False

    def test_invalidate_removes_key(self):
        c = L1Cache(capacity=4, ttl=60.0)
        c.put("x", 99)
        c.invalidate("x")
        hit, _ = c.get("x")
        assert hit is False

    def test_invalidate_prefix_removes_matching_keys(self):
        c = L1Cache(capacity=8, ttl=60.0)
        c.put("top:sci-fi", 1)
        c.put("top:action", 2)
        c.put("genre_stats", 3)
        removed = c.invalidate_prefix("top:")
        assert removed == 2
        hit_sci, _ = c.get("top:sci-fi")
        hit_gs,  _ = c.get("genre_stats")
        assert hit_sci is False and hit_gs is True

    def test_stats_hit_rate(self):
        c = L1Cache(capacity=4, ttl=60.0)
        c.put("a", 1)
        c.get("a"); c.get("a"); c.get("missing")
        s = c.stats()
        assert s["hits"] == 2 and s["misses"] == 1
        assert abs(s["hit_rate"] - 2 / 3) < 1e-4

    def test_clear_empties_entries_not_stats(self):
        c = L1Cache(capacity=4, ttl=60.0)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.clear()
        assert c.stats()["size"] == 0
        hit, _ = c.get("a")
        assert hit is False


# ---------------------------------------------------------------------------
# L2Cache tests
# ---------------------------------------------------------------------------

class TestL2Cache:

    def test_put_then_get_returns_value(self):
        c = L2Cache(ttl=60.0)
        c.put("k", {"data": [1, 2, 3]})
        hit, val = c.get("k")
        assert hit is True and val == {"data": [1, 2, 3]}

    def test_miss_on_empty_cache(self):
        c = L2Cache(ttl=60.0)
        hit, val = c.get("no_such_key")
        assert hit is False and val is None

    def test_ttl_expiry_causes_miss(self):
        c = L2Cache(ttl=0.05)
        c.put("k", "temp")
        time.sleep(0.08)
        hit, _ = c.get("k")
        assert hit is False

    def test_invalidate_removes_key(self):
        c = L2Cache(ttl=60.0)
        c.put("x", 42)
        c.invalidate("x")
        hit, _ = c.get("x")
        assert hit is False

    def test_invalidate_prefix(self):
        c = L2Cache(ttl=60.0)
        c.put("top:sci-fi", [1])
        c.put("top:action", [2])
        c.put("genre_stats", [3])
        removed = c.invalidate_prefix("top:")
        assert removed == 2
        hit_top, _ = c.get("top:sci-fi")
        hit_gs,  _ = c.get("genre_stats")
        assert hit_top is False and hit_gs is True

    def test_stats_size_counts_live_entries(self):
        c = L2Cache(ttl=60.0)
        c.put("a", 1)
        c.put("b", 2)
        assert c.stats()["size"] == 2


# ---------------------------------------------------------------------------
# TieredCache tests
# ---------------------------------------------------------------------------

class TestTieredCache:

    def test_cold_start_served_from_source(self, tc):
        tier, val = tc.get("hello")
        assert tier == "source" and val == "value_for_hello"

    def test_second_get_served_from_l1(self, tc):
        tc.get("hello")
        tier, _ = tc.get("hello")
        assert tier == "l1"

    def test_l2_hit_promotes_to_l1(self, tc):
        tc.get("hello")
        tc.l1.invalidate("hello")
        tier, _ = tc.get("hello")
        assert tier == "l2"
        # After promotion, next call is L1
        tier2, _ = tc.get("hello")
        assert tier2 == "l1"

    def test_l1_eviction_does_not_evict_l2(self):
        tc = _make_tiered(l1_capacity=2)
        tc.get("a")
        tc.get("b")
        tc.get("c")
        hit_a, _ = tc.l1.get("a")
        assert hit_a is False
        hit_a_l2, _ = tc.l2.get("a")
        assert hit_a_l2 is True

    def test_put_writes_both_tiers(self, tc):
        tc.put("manual", "overridden")
        hit_l1, v1 = tc.l1.get("manual")
        hit_l2, v2 = tc.l2.get("manual")
        assert hit_l1 and v1 == "overridden"
        assert hit_l2 and v2 == "overridden"

    def test_invalidate_removes_from_both_tiers(self, tc):
        tc.get("x")
        tc.invalidate("x")
        hit_l1, _ = tc.l1.get("x")
        hit_l2, _ = tc.l2.get("x")
        assert hit_l1 is False and hit_l2 is False

    def test_warm_pre_populates_l1(self, tc):
        keys = ["k1", "k2", "k3"]
        loaded = tc.warm(keys)
        assert loaded == len(keys)
        for k in keys:
            hit, _ = tc.l1.get(k)
            assert hit is True

    def test_warm_skips_already_cached_keys(self, tc):
        tc.get("k1")
        loaded = tc.warm(["k1", "k2"])
        assert loaded == 1

    def test_stats_account_for_all_tiers(self, tc):
        tc.get("a")
        tc.get("a")
        tc.l1.invalidate("a")
        tc.get("a")
        s = tc.stats()
        assert s["source_hits"] == 1
        assert s["l1_hits"] == 1
        assert s["l2_hits"] == 1
        assert s["total"] == 3

    def test_reset_stats_zeroes_counters(self, tc):
        tc.get("a"); tc.get("a")
        tc.reset_stats()
        s = tc.stats()
        assert s["l1_hits"] == 0 and s["l2_hits"] == 0 and s["source_hits"] == 0

    def test_concurrent_gets_consistent(self):
        tc = _make_tiered()
        errors = []

        def worker():
            try:
                for _ in range(20):
                    tier, val = tc.get("concurrent_key")
                    assert val == "value_for_concurrent_key"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert errors == [], errors


# ---------------------------------------------------------------------------
# TieredMovieDB tests
# ---------------------------------------------------------------------------

class TestTieredMovieDB:

    def test_seed_populates_movies_and_ratings(self, db):
        assert db.movie_count() == len(MOVIES)
        assert db.rating_count() == len(RATINGS)

    def test_genre_stats_returns_all_genres(self, db):
        genres_seed  = {m["genre"] for m in MOVIES}
        genres_stats = {row["genre"] for row in db.genre_stats()}
        assert genres_seed == genres_stats

    def test_top_rated_by_genre_correct_genre(self, db):
        results = db.top_rated_by_genre("sci-fi")
        assert all(r["genre"] == "sci-fi" for r in results)

    def test_top_rated_sorted_descending(self, db):
        scores = [r["avg_score"] for r in db.top_rated_by_genre("sci-fi")]
        assert scores == sorted(scores, reverse=True)

    def test_average_rating_in_valid_range(self, db):
        avg = db.average_rating("m01")
        assert avg is not None and 0.0 <= avg <= 5.0

    def test_first_call_served_from_source(self, db):
        db.cache.reset_stats()
        tier, _ = db.get_cached("genre_stats")
        assert tier == "source"
        s = db.cache.stats()
        assert s["source_hits"] == 1

    def test_second_call_served_from_l1(self, db):
        db.get_cached("genre_stats")
        tier, _ = db.get_cached("genre_stats")
        assert tier == "l1"

    def test_add_rating_invalidates_both_tiers(self, db):
        db.genre_stats()
        db.top_rated_by_genre("sci-fi")
        db.add_rating("u05", "m01", 3.8, "Test review")
        hit_l1_gs,  _ = db.cache.l1.get("genre_stats")
        hit_l2_gs,  _ = db.cache.l2.get("genre_stats")
        hit_l1_top, _ = db.cache.l1.get("top:sci-fi")
        hit_l2_top, _ = db.cache.l2.get("top:sci-fi")
        assert hit_l1_gs is False and hit_l2_gs is False
        assert hit_l1_top is False and hit_l2_top is False

    def test_add_rating_retains_unrelated_genre(self, db):
        db.genre_stats()
        db.top_rated_by_genre("action")
        db.add_rating("u05", "m01", 3.8, "sci-fi movie rated")
        hit_action_l1, _ = db.cache.l1.get("top:action")
        assert hit_action_l1 is True

    def test_l2_hit_after_l1_clear(self, db):
        db.genre_stats()
        db.cache.l1.clear()
        tier, _ = db.get_cached("genre_stats")
        assert tier == "l2"

    def test_cache_warming_achieves_full_l1_hit_rate(self, db):
        warm_keys = ["genre_stats"] + [f"top:{g}" for g in ["sci-fi", "action", "drama"]]
        db.cache.l1.clear()
        db.cache.l2.clear()
        db.cache.warm(warm_keys)
        db.cache.reset_stats()
        for key in warm_keys:
            db.get_cached(key)
        s = db.cache.stats()
        assert s["l1_hit_rate"] == 1.0

    def test_stats_sum_equals_total_requests(self, db):
        db.cache.reset_stats()
        for key in ["genre_stats", "top:sci-fi", "top:action"]:
            db.get_cached(key)
            db.get_cached(key)
        s = db.cache.stats()
        assert s["l1_hits"] + s["l2_hits"] + s["source_hits"] == s["total"]
