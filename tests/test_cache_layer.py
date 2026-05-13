"""Tests for the Caching Layer module (LRUCache + CachedMovieDB)."""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.cache_layer import LRUCache, CachedMovieDB


@pytest.fixture
def cache():
    return LRUCache(capacity=4, ttl_seconds=60.0)


@pytest.fixture
def tiny_cache():
    return LRUCache(capacity=3, ttl_seconds=0)


@pytest.fixture
def db():
    d = CachedMovieDB(cache_capacity=32, ttl_seconds=60.0, query_delay=0.0)
    d.seed(MOVIES, RATINGS)
    return d


class TestLRUCacheBasics:

    def test_get_miss_on_empty_cache(self, cache):
        hit, val = cache.get("nonexistent")
        assert hit is False and val is None

    def test_put_then_get_returns_hit(self, cache):
        cache.put("k1", "hello")
        hit, val = cache.get("k1")
        assert hit is True and val == "hello"

    def test_put_updates_existing_key(self, cache):
        cache.put("k1", "first")
        cache.put("k1", "second")
        _, val = cache.get("k1")
        assert val == "second"

    def test_size_tracks_insertions(self, cache):
        assert cache.size == 0
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size == 2

    def test_clear_resets_size_and_stats(self, cache):
        cache.put("a", 1)
        cache.get("a")
        cache.get("missing")
        cache.clear()
        assert cache.size == 0
        assert cache.stats()["hits"] == 0 and cache.stats()["misses"] == 0

    def test_invalidate_removes_key(self, cache):
        cache.put("x", 42)
        assert cache.invalidate("x") is True
        hit, _ = cache.get("x")
        assert hit is False

    def test_invalidate_absent_key_returns_false(self, cache):
        assert cache.invalidate("ghost") is False

    def test_capacity_property(self, cache):
        assert cache.capacity == 4


class TestLRUEviction:

    def test_lru_entry_evicted_when_full(self, tiny_cache):
        tiny_cache.put("a", 1)
        tiny_cache.put("b", 2)
        tiny_cache.put("c", 3)
        tiny_cache.put("d", 4)
        hit_a, _ = tiny_cache.get("a")
        hit_d, _ = tiny_cache.get("d")
        assert hit_a is False and hit_d is True

    def test_accessed_entry_not_evicted(self, tiny_cache):
        tiny_cache.put("a", 1)
        tiny_cache.put("b", 2)
        tiny_cache.put("c", 3)
        tiny_cache.get("a")
        tiny_cache.put("d", 4)
        hit_a, _ = tiny_cache.get("a")
        hit_b, _ = tiny_cache.get("b")
        assert hit_a is True and hit_b is False

    def test_keys_order_lru_to_mru(self, tiny_cache):
        tiny_cache.put("x", 10)
        tiny_cache.put("y", 20)
        tiny_cache.put("z", 30)
        tiny_cache.get("x")
        keys = tiny_cache.keys()
        assert keys[-1] == "x" and keys[0] != "x"

    def test_size_never_exceeds_capacity(self, tiny_cache):
        for i in range(10):
            tiny_cache.put(f"key{i}", i)
        assert tiny_cache.size <= tiny_cache.capacity


class TestLRUCacheTTL:

    def test_entry_valid_before_ttl_expires(self):
        c = LRUCache(capacity=4, ttl_seconds=10.0)
        c.put("k", "value")
        hit, val = c.get("k")
        assert hit is True and val == "value"

    def test_entry_expired_after_ttl(self):
        c = LRUCache(capacity=4, ttl_seconds=0.05)
        c.put("k", "value")
        time.sleep(0.07)
        hit, val = c.get("k")
        assert hit is False and val is None

    def test_expired_entry_removed_from_size(self):
        c = LRUCache(capacity=4, ttl_seconds=0.05)
        c.put("k", "v")
        time.sleep(0.07)
        c.get("k")
        assert c.size == 0

    def test_ttl_override_per_key(self):
        c = LRUCache(capacity=4, ttl_seconds=60.0)
        c.put("short", "bye",  ttl_override=0.05)
        c.put("long",  "stay", ttl_override=60.0)
        time.sleep(0.07)
        hit_short, _ = c.get("short")
        hit_long,  _ = c.get("long")
        assert hit_short is False and hit_long is True

    def test_ttl_zero_means_never_expire(self):
        c = LRUCache(capacity=4, ttl_seconds=0)
        c.put("k", "immortal")
        time.sleep(0.01)
        hit, val = c.get("k")
        assert hit is True and val == "immortal"


class TestLRUCacheStats:

    def test_stats_initial_state(self, cache):
        s = cache.stats()
        assert s["hits"] == 0 and s["misses"] == 0 and s["hit_rate"] == 0.0

    def test_stats_after_hits_and_misses(self, cache):
        cache.put("a", 1)
        cache.get("a"); cache.get("a"); cache.get("b")
        s = cache.stats()
        assert s["hits"] == 2 and s["misses"] == 1
        assert abs(s["hit_rate"] - 2/3) < 1e-4

    def test_reset_stats_keeps_entries(self, cache):
        cache.put("x", 99)
        cache.get("x")
        cache.reset_stats()
        s = cache.stats()
        assert s["hits"] == 0 and s["misses"] == 0
        hit, val = cache.get("x")
        assert hit is True and val == 99


class TestCachedMovieDB:

    def test_seed_populates_movies_and_ratings(self, db):
        assert db.movie_count() == len(MOVIES) and db.rating_count() == len(RATINGS)

    def test_first_call_is_cache_miss(self, db):
        db.cache.clear(); db.cache.reset_stats()
        db.genre_stats()
        s = db.cache.stats()
        assert s["misses"] == 1 and s["hits"] == 0

    def test_second_call_is_cache_hit(self, db):
        db.cache.clear(); db.cache.reset_stats()
        db.genre_stats(); db.genre_stats()
        s = db.cache.stats()
        assert s["hits"] == 1 and s["misses"] == 1

    def test_genre_stats_returns_all_genres(self, db):
        genres_in_seed  = {m["genre"] for m in MOVIES}
        genres_in_stats = {row["genre"] for row in db.genre_stats()}
        assert genres_in_seed == genres_in_stats

    def test_top_rated_by_genre_correct_genre(self, db):
        assert all(r["genre"] == "sci-fi" for r in db.top_rated_by_genre("sci-fi", limit=5))

    def test_top_rated_by_genre_sorted_desc(self, db):
        scores = [r["avg_score"] for r in db.top_rated_by_genre("sci-fi", limit=5)]
        assert scores == sorted(scores, reverse=True)

    def test_average_rating_within_range(self, db):
        avg = db.average_rating("m01")
        assert avg is not None and 0.0 <= avg <= 5.0

    def test_average_rating_cached_on_second_call(self, db):
        db.cache.clear(); db.cache.reset_stats()
        db.average_rating("m01"); db.average_rating("m01")
        assert db.cache.stats()["hits"] == 1

    def test_add_rating_invalidates_genre_cache(self, db):
        db.genre_stats(); db.top_rated_by_genre("sci-fi")
        db.add_rating("u05", "m01", 3.5, "Fine")
        db.cache.reset_stats()
        db.genre_stats()
        assert db.cache.stats()["misses"] == 1

    def test_add_rating_does_not_evict_unrelated_genre(self, db):
        db.genre_stats(); db.top_rated_by_genre("action")
        db.add_rating("u05", "m01", 3.5, "Fine")
        db.cache.reset_stats()
        db.top_rated_by_genre("action")
        assert db.cache.stats()["hits"] == 1

    def test_cache_size_bounded_by_capacity(self, db):
        small_db = CachedMovieDB(cache_capacity=3, ttl_seconds=60.0)
        small_db.seed(MOVIES, RATINGS)
        for genre in ["sci-fi", "action", "thriller", "drama", "animation"]:
            small_db.top_rated_by_genre(genre)
        assert small_db.cache.size <= 3

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)
